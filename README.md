## Smart Explorer
## Installation

1\) Environment requirements

* Python 3.x
* Pytorch 1.11
* CUDA 9.2 or higher

The following installation guild suppose ``python=3.7`` ``pytorch=1.11`` and ``cuda=10.2``. You may change them according to your system.

Create a conda virtual environment and activate it.
```
conda create -n realsense python=3.7
conda activate realsense
```

2\) Clone the following project.
```
git clone https://github.com/dbolya/yolact.git
```
note: Please put this project in our directory.

3\) Install the dependencies.
```
conda install pytorch cudatoolkit=10.2 -c pytorch
pip install cython
pip install pillow pycocotools matplotlib 
pip install opencv-python
pip install pycocotools
pip install PyQt5
pip install opencv-contrib-python==4.5.2.52
pip install pybullet
pip install open3D
pip install trimesh
```
4\)Setup
```
cd chamfer3D
python setup.py install
```

## Prepare Data
Prepare your own 3D model files (urdf format)

1\) Create sense
```
python create_dataset.py
```

2\) Prepare 2D instance segmentation label
```
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install
pip install git+git://github.com/waspinator/coco.git@2.1.0
```
Put `create_json.py` to `./cocoapi/PythonAPI`

Put `name_list_train.txt` and `name_list_val.txt` to `./cocoapi/PythonAPI`
```
python create_json.py
```
note: 
After running the script, you should get two files `instances_train2017.json` and `instances_val2017.json`.

Put the above two files to `./data/coco/annotations`.

Put the images under the `./dataset/rbg_img` folder into `./data/coco/images`.

## Train and Test

1\) Train 2D instance segmentation model

```
python train.py --config=yolact_resnet50_config
```
Move the trained model to the `weights` file and rename it to `yolact_resnet50.pth`

2\) Test
```
python test_push.py
```