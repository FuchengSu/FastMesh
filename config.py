import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# proj_name = "scan114"
proj_name_list = ["scan24", "scan37", "scan40", "scan55", "scan63", "scan65", "scan69", "scan83", "scan97", "scan105", "scan106", "scan110", "scan114", "scan118", "scan122", "nhr", "101", "scan65_test"]
# nhr 15

# for DTU
SIZE = [512, 640]
# # for nhr
# SIZE = [1024, 1216]
# # for 101
# SIZE = [1024, 1280]

loop_num = 500
displace_num = 2000

shape_num = loop_num
subdivide_num = 4000
grid_num = 50
subdivide_order = 0
focal = 560
views = [1,3,4,5]
xy_num = 49
z_num = 1
total_num = int(xy_num * z_num)
batch_size = 4
gif_view_num = 0
f0 = 0.04
eps = 1e-6
light_h = 1
light_radius = 6 # 5 for human, 8 for horse

load_render_net = True
ps_normal = False
change_weight = True
use_adam = True
camlist_flag = False
only_tex = True
load_K = only_tex
only_shape = not only_tex
two_step = True 
remesh_step_flag = True
subdivide_flag = True
remesh_step = loop_num
save_flag = True
novel_flag = False
own_flag = True
lookat_flag = True
load_mode = "colmap"
uniform_flag = False

if only_tex:
    sphere_path = "/workspace/data/obj/sphere/sphere_642.obj"
else:
    if load_mode == 'colmap':
        # obj_path = "/workspace/data/obj/sphere/sphere_642.obj"
        obj_path = "/workspace/data/obj/sphere/sphere_642_1000.obj"
    else:
        obj_path = "/workspace/data/obj/sphere/sphere_1352.obj"

vertex_rot90 = np.array([
    [ 1, 0, 0],
    [ 0, 0, 1],
    [ 0,-1, 0]
])



proj_vertex_rot180 = np.array([
    [-1, 0, 0],
    [ 0, 1, 0],
    [ 0, 0,-1]
])
