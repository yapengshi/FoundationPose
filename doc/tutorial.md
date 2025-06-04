# FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects
[[Paper]](https://arxiv.org/abs/2312.08344) [[Website]](https://nvlabs.github.io/FoundationPose/)

## 前期准备
### Conda环境配置
```bash
# create conda environment
conda create -n foundationpose python=3.9

# activate conda environment
conda activate foundationpose

# Install Eigen3 3.4.0 under conda environment
conda install conda-forge::eigen=3.4.0
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:/eigen/path/under/conda"

# install dependencies
python -m pip install -r requirements.txt

# Install NVDiffRast
python -m pip install --quiet --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git

# Kaolin (Optional, needed if running model-free setup)
python -m pip install --quiet --no-cache-dir kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.0_cu118.html

# PyTorch3D
python -m pip install --quiet --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt200/download.html

# Build extensions
CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/pybind11/share/cmake/pybind11 bash build_all_conda.sh
```
### 模型权重-数据准备
- 下载权重文件到`weights/`目录下
[权重文件下载](https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i?usp=sharing)
1） refiner: `2023-10-28-18-33-37`
2） scorer: `2024-01-11-20-02-45`

- 下载测试数据到`demo_data/`目录下
[kinect_driller_seq|mustard0 下载](https://drive.google.com/drive/folders/1pRyFmxYXmAnpku7nGRioZaKrVJtIsroP?usp=sharing)解压到`demo_data/`目录下

- [Model Free] 下载经过预处理的参考视图到`Datasets/`目录下
运行model-free的少样本学习版本，请[下载linemod.zip | ycbv.zip](https://drive.google.com/drive/folders/1PXXCOJqHXwQTbwPwPbGDN9_vLVe0XpFS?usp=sharing)

- [可选] 下载训练数据到`Datasets/`目录下
[下载 gso | objaverse_path_tracing](https://drive.google.com/drive/folders/1s4pB6p4ApfWMiMjmTXOFco8dHbNXikp-?usp=sharing)

- [可选] 下载YCB-Video数据集
如果需要YCB-Video数据集，这是一个200G+的数据集，BOP版做了筛选，在100G左右，前往此处下载，解压到demo_data/目录下。


## 代码运行
1. 运行 `Model-based demo`  
默认情况下，路径已在`argparse`中设置。如果需要更改场景，可以相应地传递`args`,生成的可视化结果将保存到`argparse`中指定的`debug_dir`中。
```bash
conda activate foundationpose
python run_demo.py
```
<img src="assets/demo.jpg" width="50%"> 

尝试其他物体(**无需再训练**) 比如`driller`，通过改变`argparse`路径，
```bash
python run_demo.py --mesh_file ./demo_data/kinect_driller_seq/mesh/textured_mesh.obj --test_scene_dir ./demo_data/kinect_driller_seq
```
<img src="assets/demo_driller.jpg" width="50%">


2. 公开数据集`(LINEMOD, YCB-Video)`运行
运行结果会被存储到`debug`文件夹中
```bash
python run_linemod.py --linemod_dir /home/lab/yapeng/github/FoundationPose/Datasets/linemod --use_reconstructed_mesh 0

python run_ycb_video.py --ycbv_dir /home/lab/yapeng/github/FoundationPose/Datasets/ycbv --use_reconstructed_mesh 0
```


To run model-based version on these two datasets respectively, set the paths based on where you download. 
```bash
python run_linemod.py --linemod_dir /mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/LINEMOD --use_reconstructed_mesh 0

python run_ycb_video.py --ycbv_dir /mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/YCB_Video --use_reconstructed_mesh 0
```


3. 运行 `Model-free demo`
要运行model-free的少样本版本，首先需要训练`Neural Object Field`。 `ref_view_dir` 是下载预处理的参考视图的路径，这里是`Datasets/`
```bash
python bundlesdf/run_nerf.py --ref_view_dir /mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/YCB_Video/bowen_addon/ref_views_16 --dataset ycbv
```

Then run the similar command as the model-based version with some small modifications. Here we are using YCB-Video as example:
```
python run_ycb_video.py --ycbv_dir /mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/YCB_Video --use_reconstructed_mesh 1 --ref_view_dir /mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/YCB_Video/bowen_addon/ref_views_16
```
