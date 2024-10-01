

import tqdm, trimesh
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import os, json, scipy, cv2
import numpy as np

def _align_meshes_with_optim(mesh1, mesh2, translation = None): # we translate mesh2 to align with mesh1

    def chamfer_distance(translation, mesh1, mesh2):
        translated_mesh1 = mesh1.copy()
        translated_mesh1.vertices += translation
        distance_matrix = cdist(translated_mesh1.vertices, mesh2.vertices)
        chamfer_dist = np.sum(np.min(distance_matrix, axis=1)) + np.sum(np.min(distance_matrix, axis=0))

        return chamfer_dist

    if translation is None:
        initial_translation = np.array([0.0, 0.0, 0.0])
        result = minimize(chamfer_distance, initial_translation, args=(mesh1.simplify_quadric_decimation(face_count = min(5000, len(mesh1.faces))), mesh2.simplify_quadric_decimation(face_count =  min(5000, len(mesh2.faces)))), method='Nelder-Mead')
        translation = -result.x
        mesh2.apply_translation(translation)
    else:
        mesh2.apply_translation(translation)

    return mesh2, translation

def create_point_label(input_mesh, deform_mesh, eps = 0.1):
    pnts2 = trimesh.sample.sample_surface(input_mesh, 50000)[0]
    pnts1 =trimesh.sample.sample_surface(deform_mesh, 50000)[0]
    distances = scipy.spatial.distance.cdist(pnts1, pnts2)
    min_distances_pt1 = np.min(distances, axis=1)
    diff_pt1_indices = np.where(min_distances_pt1 > eps)[0]
    point_cloud = trimesh.PointCloud(pnts1, colors = [255,255,255,255])
    point_cloud.colors[diff_pt1_indices] = [0,0,0,0]
    return point_cloud


def generate_localization_maps(anomaly_path, ref_normal_path):

    os.makedirs(f'{anomaly_path}/location_maps', exist_ok=True)

    mesh = trimesh.load(f'{anomaly_path}/transformed.obj')
    mesh_ref = trimesh.load(f'{ref_normal_path}/transformed.obj')
    info = json.load(open(f'{anomaly_path}/scene3d.metadata.json', 'r'))
    info_ref = json.load(open(f'{ref_normal_path}/scene3d.metadata.json', 'r'))
    # Compute the relative transformation matrix to align A to B
    relative_transform = np.dot(np.array(info['objects'][0]['rotation_matrix']), np.linalg.inv(np.array(info_ref['objects'][0]['rotation_matrix'])))
    mesh_ref.apply_transform(relative_transform)


    mesh_ref_2,_  = _align_meshes_with_optim(mesh, mesh_ref)
    pnts = create_point_label(mesh_ref_2, mesh, eps = 0.005)

    sampleidx = np.random.choice(len(pnts.vertices), 10000, replace = False)

    points_3d = pnts.vertices[sampleidx]
    labels_3d = (pnts.colors[:,0]==0)[sampleidx]

    ray_tracer = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)

    for num_, file_name in tqdm.tqdm(enumerate(sorted(os.listdir(f'{anomaly_path}/RGB/')))):

        RT = np.array(info['camera']['poses'][num_]['rotation'])
        K = np.array(info['camera']['K'])

        size = int(K[0,-1]*2)

        R = RT[:3, :3]  # The rotation matrix (first 3 columns)
        T = RT[:3, 3]   # The translation vector (last column)
        # Compute the camera position C in world coordinates
        camera_position = -np.dot(R.T, T)
        ray_origins = camera_position - points_3d+points_3d
        ray_directions = points_3d - camera_position
        intersections,_,_ = ray_tracer.intersects_location(
                        ray_origins=ray_origins,
                        ray_directions=ray_directions,
                        multiple_hits = False
                    )
        is_visible = scipy.spatial.distance.cdist(points_3d, intersections).min(1)<1e-5

        labels_visible= labels_3d[is_visible]
        ones = np.ones((points_3d[is_visible].shape[0], 1))
        points_3d_homogeneous = np.hstack((points_3d[is_visible], ones))  # Shape becomes N x 4

        P = np.dot(K, RT[:3])  # Shape (3x4)

        # Project the points
        points_2d_homogeneous = np.dot(P, points_3d_homogeneous.T)  # Shape (3, N)
        points_2d = points_2d_homogeneous[:2] / points_2d_homogeneous[2]  # (2, N) / (1, N)
        points_2d = points_2d.T
        

        zero_image = np.zeros((size,size))
        for idx, xy in enumerate(points_2d):
            if int(xy[1])<size and int(xy[0])<size:
                zero_image[int(xy[1]), int(xy[0])] = labels_visible[idx]
            
        cv2.imwrite(f'{anomaly_path}/location_maps/{file_name}', zero_image*255)
