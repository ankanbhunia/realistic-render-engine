conda create -n moad python=3.10
conda activate moad

python -m pip install -r requirements.txt
conda install -c conda-forge igl scikit-sparse tetgen
conda install -c mosek mosek ## ad mosek key

pip install https://files.pythonhosted.org/packages/9c/f3/732e7c6d6c0932b80df488a24e02ac34e8eae14a3c893eb97dcbf6e9c93c/bpy-3.4.0-cp310-cp310-manylinux_2_17_x86_64.whl
pip install trimesh[all] connected-components-3d pymeshfix scikit-learn open3d bounding_box
bpy_post_install