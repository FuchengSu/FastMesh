# Optimization From Normal Map
It is a simplified version of the mesh optimized with normal map which comments out color and texture optimizations due to GPU RAM(If you want to use the texture section, you need to uncomment this section)

First we use MVSNet-based method(specifically RA-MVSNet) to get normal map of every view. Then the optimizing program uses normal maps, images and corresponding masks as input to get smooth and correct mesh topologies with fine details. 

## ⚙ Setup
#### 1. Recommended environment
- PyTorch 1.2+
- Python 3.6+
- [nvdiffrast](https://nvlabs.github.io/nvdiffrast/)
- [Kaolin](https://github.com/NVIDIAGameWorks/kaolin)
- GCC(7.5.0+)
- opencv-python
- trimesh
- sklearn

#### 2. Testing Dataset for MVSNet
You can test with [DTU testing data](https://drive.google.com/file/d/135oKPefcPTsdtLRzoDAQtPpHuoIrpRI_/view), [BlendedMVS](https://drive.google.com/file/d/1ilxls-VJNvJnB7IaFj7P0ehMPr7ikRCb/view), [ETH3D](https://www.eth3d.net/datasets) and your own data. [Tanks and Temples](https://drive.google.com/file/d/1YArOJaX9WVLJh4757uE8AEREYkgszrCo/view) are temporarily unavailable due to lack of mask and we will solve this problem later.

And the data organization should be as follows
```
dtu_testing                                          
       ├── scan1   
            ├── cams
            ├── images
            pair.txt
       ├── scan2
       ├── ...
```
or 
```
your_own_data                                          
       ├── scan1_name
            ├── cams
            ├── images
            pair.txt
       ├── scan2_name
       ├── ...
```
We provide the DTU dataset and the script files for your own dataset testing at `<scripts/dtu_test.sh>` and `<scripts/human_test.sh>`.

Finally, you will get the result of RA-MVSNet as follows
```
dtu_result                                          
       ├── scan1   
            ├── cams
            ├── images
            ├── confidences
            ├── depth_est
            ├── normal

       ├── scan2
       ├── ...
```
or
```
human_result                                          
       ├── scan1   
            ├── cams
            ├── images
            ├── confidences
            ├── depth_est
            ├── normal
       ├── scan2
       ├── ...
```

## Optimization using normal map
The data organization and parameter settings in this part are very messy. We briefly introduce important parameters and data formats here and will simplify them later.

The input data organization should be as follows
```
lego_test                                          
    ├── scan65
       ├── CAM
            ├── 00000000_cam.txt
            ├── ...
       ├── NORMAL
            ├── 00000000_normal.png
            ├── ...
       ├── MASK
            ├── 00000000.png
            ├── ...
       ├── RENDER
            ├── 00000000.png
            ├── ...
    ├── nhr
       ├── ...
```
The camera parameter file `<CAM/*.txt>`, normal map file `<NORMAL/*.png>` and images file `<RENDER/*.png>` can be simply copied from RA-MVSNet output file. And the mask file `<MASK/*.png>` requires additional input to label reconstructed objects.

The parameters required for optimization are specified in the  `<config.py>` file and the important parameters are described as follows
- proj_name = "man0" # the project name
- SIZE = [1024, 1280] # image resolution
- loop_num = 200 # epoch, if only for normal 200 is enough
- grid_num = 512 # the grid resolution for visual hull, the initial mesh is sphere with radius 6. Since the subdivide operation will be performed later, the grid resolution does not need to be very large 
- subdivide_order = 1 # the order of subdivided operation. Each order corresponds to the mesh resolution multiplied by 4
- xy_num # input number of images
- batch_size = 2 # the number of images input per batch
- ps_normal = False # whether to use the photometric stereo normal map format
- change_weight = True # whether the parameters are variable in the optimization process
- load_mode = 'colmap' # the data load mode, and colmap corresponds to MVSNet format input
- uniform_flag = False # whether the object uses the uniform format

Besides, we use c++ to calculate Laplace Adjacency Matrix for regular term. Therefore, we provide source code `<computeK.cpp>` and executable file in `<build/>`.

#### 4. Scripts for RA-MVSNet and Optimization
To get the normal map, you first need to run 
```
bash <scripts/human_test.sh>
```
And we provide three pretrained models `<log/model_000040.ckpt>`, `<log/model_000032_dtu.ckpt>` and `<log/model_000009_blendedmvs.ckpt>` which You can specify to use the corresponding pre-trained model in the resume parameter in the bash file.

As for optimization, you are supposed to run
```
g++ -shared -fPIC `python -m pybind11 --includes` computeK.cpp -o py2cpp.so -I /opt/conda/include/python3.8/
```
to generate `py2cpp.so` file for generate sparse matrix for regularization.

Then run 
```
python <demo_deform.py> -n 5 # for scan65
python <demo_deform.py> -n 15 # for nhr data
python <demo_deform.py> -n 16 # for 101_gan data
```
to generate result of optimization. The result will be saved in `<result/scan/final_result.obj>`