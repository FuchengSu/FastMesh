import os
from pkg_resources import resource_filename
import trimesh
from utils import *
from config import *
import torch
import argparse

class Model(nn.Module):
    def __init__(self, template_mesh, total_num):
        super(Model, self).__init__()

        self.template_mesh = template_mesh
        self.register_parameter('vertices', nn.Parameter(torch.from_numpy(np.array(self.template_mesh.vertices).astype(np.float32)).cuda()))
        print(self.vertices.device)
        # self.vertices = nn.Parameter(torch.from_numpy(np.array(self.template_mesh.vertices).astype(np.float32)).cuda())
        # self.register_buffer('vertices', torch.from_numpy(np.array(self.template_mesh.vertices).astype(np.float32)).cuda())
        self.register_buffer('faces', torch.from_numpy(np.array(self.template_mesh.faces).astype(np.int32)).cuda())
        self.register_buffer('v2f', get_v2f(self.vertices.unsqueeze(0), self.faces))

        # lxyz, lareas = gen_light_xyz(light_h, 2*light_h, light_radius)
        # lxyz = torch.from_numpy(lxyz.reshape(-1, 3).astype(np.float32)).cuda()
        # lareas = torch.from_numpy(lareas.reshape(1, -1, 1).astype(np.float32)).cuda()
        
        # self.register_buffer('lxyz', lxyz)
        # self.register_buffer('lareas', lareas)

        # # optimize for displacement map and center
        # self.register_parameter('displace', nn.Parameter(torch.zeros_like(torch.from_numpy(self.template_mesh.vertices), dtype=torch.float32)))
        # self.register_parameter('center', nn.Parameter(torch.zeros(1, 1, 3)))
        # self.register_parameter('texture', nn.Parameter(torch.ones((self.vertices.shape[0], 7), dtype=torch.float32)))
        # self.register_parameter('light', nn.Parameter(torch.from_numpy(np.ones((total_num, light_h*2*light_h, 3)).astype(np.float32)*0.1)))

        if load_K:
            K_path = os.path.join(output, "K.txt")
            self.K = loadK(K_path, self.template_mesh.vertices.shape[0])
            # self.K = np.loadtxt(K_path, dtype=np.int8).reshape(self.template_mesh.vertices.shape[0], -1)
            # self.K = np.array(self.K, dtype=np.int8)
            # print(self.K.shape)
        else:
            self.K = compute_K(self.template_mesh)

        self.K = torch.from_numpy(self.K.astype(np.float32)).cuda()

        if only_tex:
            self.laplacian_loss = 0
            # self.Flattenloss = MyFlatten(self.faces)
            # self.Normalloss = NormalConsistency(self.faces)

            self.laplacian_L = compute_laplacian(self.vertices, self.faces.long())
            # self.flatten_loss = 0
        else:
            self.laplacian_L = compute_laplacian(self.vertices, self.faces.long())
            



    def forward(self, total_num, delta):
        if uniform_flag:
            base = torch.log(self.vertices.abs() / (4 - self.vertices.abs()))
            centroid = torch.tanh(self.center)
            vertices = torch.sigmoid(base + self.displace) * torch.sign(self.vertices)
            vertices = F.relu(vertices) * (1 - centroid) - F.relu(-vertices) * (centroid + 1)
            vertices = vertices + centroid
        else:
            vertices = self.vertices.unsqueeze(0)
        # apply Laplacian and flatten geometry constraints
        
        # if load_colmap:
        #     vertices = vertices + delta[0]

        if only_tex:
            laplacian_loss = 0
            # albedo_flatten_loss = compute_hexagon_loss(self.texture[:, 3:6].unsqueeze(0), self.K) + compute_hexagon_loss(self.texture[:, 0:3].unsqueeze(0), self.K)
            hexagon_loss = compute_hexagon_loss(vertices, self.K)
            albedo_flatten_loss = 0
            vertices_flatten_loss = 0
            # flatten_loss = 0
        else:
            laplacian_loss = laplacian_smoothing(vertices.squeeze(0), self.faces, method='uniform', laplacian=self.laplacian_L)
            hexagon_loss = compute_hexagon_loss(vertices, self.K)
            albedo_flatten_loss = vertices_flatten_loss = 0
            # flatten_loss = self.flatten_loss(vertices)

        # texture = self.tex

        return vertices.repeat(total_num, 1, 1), self.faces.repeat(total_num, 1, 1), laplacian_loss, hexagon_loss, albedo_flatten_loss, vertices_flatten_loss

    def save_obj(self, savepath, method='uniform'):
        if method == 'uniform':
            base = torch.log(self.vertices.abs() / (1 - self.vertices.abs()))
            centroid = torch.tanh(self.center)
            result_vertices = torch.sigmoid(base + self.displace) * torch.sign(self.vertices)
            result_vertices = F.relu(result_vertices) * (1 - centroid) - F.relu(-result_vertices) * (centroid + 1)
            result_vertices = result_vertices + centroid
            result_faces = self.faces
        else:
            result_vertices, result_faces = self.vertices, self.faces
        
        result_vertices = result_vertices.detach().cpu()
        result_faces = result_faces.detach().cpu()
        result_mesh = trimesh.base.Trimesh(result_vertices, result_faces)
        result_mesh.export(savepath)

        return 1

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num', type=int, default=0)
args = parser.parse_args()

proj_name = proj_name_list[args.num]

if load_mode == "colmap":
    images, cameras, mvps, w2cs, normals_gt, masks_gt, confs, depths, specs = load_from_colmap(os.path.join("/workspace/lego_test", proj_name))
elif load_mode == "tanks":
    images, cameras, mvps, w2cs, normals_gt, depth_gt, masks_gt = load_from_tanks(os.path.join("/workspace/lego_test", proj_name))
else:
    images, cameras, mvps, w2cs, normals_gt = load_own_blender(os.path.join("/workspace/lego_test", proj_name))

tmp_mesh = trimesh.load(sphere_path)
rm_binvox_cmd = 'rm test/*binvox'
os.system(rm_binvox_cmd)



os.makedirs(os.path.join('/workspace/nvrender/results/', proj_name), exist_ok=True)

savepath = os.path.join('/workspace/nvrender/results/', proj_name, 'plane.obj')
vertices = tmp_mesh.vertices
faces = tmp_mesh.faces

# space_carving_list = [9,21,22,23]
# mvps = mvps[space_carving_list]
# masks_gt = masks_gt[space_carving_list]
new_vertices, new_faces = remesh_test(vertices, faces, mvps, masks_gt, grid_num)
# new_vertices, new_faces = remesh(vertices, faces, mvps, masks_gt, grid_num)
result_mesh = trimesh.base.Trimesh(new_vertices, new_faces)
print(result_mesh.vertices.shape, result_mesh.faces.shape, savepath)
result_mesh.export(savepath)