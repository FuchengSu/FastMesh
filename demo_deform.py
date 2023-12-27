"""
Demo deform.
Deform template mesh based on input silhouettes and camera pose
"""
from dis import dis
import os
import torch
torch.cuda.set_device(1)
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch.nn as nn
import torch.nn.functional as F
# import matplotlib.pyplot as plt
import tqdm
import numpy as np
import imageio
import argparse
import json
import math
import trimesh
from trimesh.util import submesh
from utils import *
from skimage.transform import resize
from sklearn.metrics.pairwise import cosine_similarity
from config import *
from Unet import UNet
import py2cpp
import time
    
import nvdiffrast.torch as dr

np_proj = proj
proj = torch.from_numpy(np_proj).cuda()

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, '/workspace/data')
       


class Model(nn.Module):
    def __init__(self, template_mesh, total_num, proj_idx):
        super(Model, self).__init__()

        self.template_mesh = template_mesh
        self.register_parameter('vertices', nn.Parameter(torch.from_numpy(np.array(self.template_mesh.vertices).astype(np.float32)).cuda()))
        self.register_parameter('center', nn.Parameter(torch.zeros((1, 3), dtype=torch.float32)))
        print(self.vertices.device)
        self.register_buffer('faces', torch.from_numpy(np.array(self.template_mesh.faces).astype(np.int32)).cuda())
        self.register_buffer('v2f', get_v2f(self.vertices.unsqueeze(0), self.faces))

        # self.position_encoder = PositionalEncoding_nerf(L=10)
        self.position_encoder = PostionalEncoding(max_deg=3)
        # self.position_encoder = lambda x:x
        # self.position_encoder.embedding_size = 3
        
        # self.position_encoder_ray = PositionalEncoding_nerf(L=10)
        # self.position_encoder_ray = PostionalEncoding(max_deg=3)
        self.position_encoder_ray = lambda x:x
        self.position_encoder_ray.embedding_size = 3
        
        self.feature_channel = 7
        # self.position_encoder_feature = lambda x:x
        # self.position_encoder_feature.embedding_size = self.feature_channel
        self.position_encoder_feature = PostionalEncoding(max_deg=3, input_channle=self.feature_channel)
        self.input_channel = self.position_encoder.embedding_size + self.position_encoder_feature.embedding_size + self.position_encoder_ray.embedding_size + 3
        # self.input_channel = 13

        # if use_high_feature:
        #     self.position_encoder_feature = PostionalEncoding(max_deg=3, input_channle=9)
        #     self.feature_net = FeatureNet(base_channels=9, num_stage=3, mode='fpn')
        #     if load_high_feature:
        #         self.feature_net.load_state_dict(torch.load("/workspace/nvrender/results/"+proj_name_list[proj_idx]+"/feature_net.pth"))
        #     # self.input_channel = self.position_encoder.embedding_size + self.feature_net.out_channels[2] + 6
        #     self.input_channel = self.position_encoder.embedding_size + self.position_encoder_feature.embedding_size + 6

        self.register_parameter('texture', nn.Parameter(torch.ones((self.vertices.shape[0], self.feature_channel), dtype=torch.float32)))
        # bilinear or deconv
        self.render_net = UNet(num_input_channels=self.input_channel, num_output_channels=3, feature_scale=8, more_layers=0, upsample_mode='bilinear', last_act='sigmoid').cuda()
        if load_render_net:
            self.render_net.load_state_dict(torch.load("/workspace/nvrender/results/"+proj_name_list[proj_idx]+"/render_net.pth", map_location='cpu'))
            # self.texture = nn.Parameter(torch.from_numpy(np.load("/workspace/nvrender/results/"+proj_name_list[proj_idx]+"/feature.npy")))
        
        self.render_net.requires_grad_(True)


        

        K_sparse = py2cpp.compute_k(proj_name_list[proj_idx])
        idx = K_sparse[:2]
        v = K_sparse[2]
        verts_num = self.template_mesh.vertices.shape[0]
        self.K = torch.sparse_coo_tensor(idx, v, (verts_num, verts_num), dtype=torch.float32).cuda()

        # self.K = torch.from_numpy(self.K.astype(np.float32)).cuda()

        if only_tex:
            self.laplacian_loss = 0
            # self.Flattenloss = MyFlatten(self.faces)
            # self.Normalloss = NormalConsistency(self.faces)

            self.laplacian_L = compute_laplacian(self.vertices, self.faces.long())
            # self.flatten_loss = 0
        else:
            self.laplacian_L = compute_laplacian(self.vertices, self.faces.long())
            



    def forward(self, total_num):
        if uniform_flag:
            base = torch.log(self.vertices.abs() / (4 - self.vertices.abs()))
            centroid = torch.tanh(self.center)
            vertices = torch.sigmoid(base + self.displace) * torch.sign(self.vertices)
            vertices = F.relu(vertices) * (1 - centroid) - F.relu(-vertices) * (centroid + 1)
            vertices = vertices + centroid
        else:
            vertices = self.vertices.unsqueeze(0)
            vertices = vertices + self.center.unsqueeze(0)
        # apply Laplacian and flatten geometry constraints
        
        if only_tex:
            laplacian_loss = 0
            # albedo_flatten_loss = compute_hexagon_loss(self.texture[:, 3:6].unsqueeze(0), self.K) + compute_hexagon_loss(self.texture[:, 0:3].unsqueeze(0), self.K)
            hexagon_loss = compute_hexagon_loss(vertices, self.K)
            # hexagon_loss = laplacian_smoothing(vertices.squeeze(0), self.faces, method='uniform', laplacian=self.laplacian_L)
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



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--filename-input', type=str,
                        default=os.path.join(data_dir, 'source.npy'))
    parser.add_argument('-c', '--camera-input', type=str,
                        default=os.path.join(data_dir, 'camera.npy'))
    parser.add_argument('-t', '--template-mesh', type=str)
    parser.add_argument('-o', '--output-dir', type=str)
    parser.add_argument('-n', '--num', type=int,
                        default=6)
    args = parser.parse_args()

    proj_name = proj_name_list[args.num]
    args.output_dir = os.path.join('/workspace/nvrender/results/', proj_name)
    os.makedirs(args.output_dir, exist_ok=True)

    rm_binvox_cmd = 'rm test/*binvox'
    os.system(rm_binvox_cmd)
    # read training images and camera poses
    if load_mode == "colmap":
        images_np, intrinsic_gt_np, mvps_np, w2cs_np, normals_gt_np, masks_gt_np, conf_gt_np, depth_gt_np, spec_gt_np, images_np_test, intrinsic_gt_np_test, mvps_np_test, w2cs_np_test, normals_gt_np_test, masks_gt_np_test, conf_gt_np_test, depth_gt_np_test, spec_gt_np_test = load_from_colmap(os.path.join("/workspace/lego_test", proj_name))
    elif load_mode == "tanks":
        images_np, cameras, mvps_np, w2cs_np, normals_gt_np, depth_gt_np, masks_gt_np = load_from_tanks(os.path.join("/workspace/lego_test", proj_name))
    else:
        images_np, cameras, mvps_np, w2cs_np, normals_gt_np = load_own_blender(os.path.join("/workspace/lego_test", proj_name))
    

    if isinstance(mvps_np, np.ndarray):
        mvps = torch.from_numpy(np.array(mvps_np).astype(np.float32)).cuda()
    if isinstance(w2cs_np, np.ndarray):
        w2cs = torch.from_numpy(np.array(w2cs_np).astype(np.float32)).cuda()

    args.template_mesh = "/workspace/nvrender/results/" + proj_name + "/plane2.obj"
    # tmp_mesh = trimesh.load(args.template_mesh)
    tmp_mesh = trimesh.load(args.template_mesh, process=False, maintain_order=True)

    model = Model(tmp_mesh, total_num, args.num).cuda()    
    lr_render = 1e-4
    lr_featur = 1e-3
    if use_adam:
        optimizer = torch.optim.Adam([
            {'params': model.render_net.parameters(), 'lr': lr_render, 'momentum': 0.9},
            {'params': model.center, 'lr': 5e-4, 'momentum': 0.9},
            {'params': [model.texture, model.vertices], 'lr': lr_featur, 'momentum': 0.9}], 1e-4, betas=(0.5, 0.99))
        # optimizer = torch.optim.Adam([{'params': model.parameters()}], 1e-3, betas=(0.5, 0.99))
    else:
        optimizer = torch.optim.SGD([
            {'params': [model.vertices, model.displace, model.center], 'lr': 2e-4, 'momentum': 0.9},
            {'params': [model.light], 'lr': 1, 'momentum': 0.9},
            {'params': [model.texture], 'lr': 1, 'momentum': 0.9}
            ])
    glctx = dr.RasterizeGLContext()


    if only_tex:
        for name, param in model.named_parameters():
            if name == 'displace':
                param.requires_grad = True
            elif name == 'center':
                param.requires_grad = True
    if only_shape:
        for name, param in model.named_parameters():
            if name == 'texture':
                param.requires_grad = False
    for name, param in model.named_parameters():
        if name == 'vertices':
            param.requires_grad = False
        if param.requires_grad:
            print(name)
    

    loop = tqdm.tqdm(list(range(0, loop_num)))
    writer = imageio.get_writer(os.path.join(args.output_dir, 'deform.gif'), mode='I')
    images_gt_total = torch.from_numpy(images_np)
    normals_gt_total = torch.from_numpy(normals_gt_np)
    confs_gt_total = torch.from_numpy(conf_gt_np)
    intrinsics_gt_total = torch.from_numpy(intrinsic_gt_np)
    if load_mode == 'colmap':
        masks_gt_total = torch.from_numpy(masks_gt_np)
        depth_gt_total = torch.from_numpy(depth_gt_np)
        spec_gt_total = torch.from_numpy(spec_gt_np)
    for i in loop:
        if i == (loop_num-1):
            images_gt_total = torch.from_numpy(images_np)
            normals_gt_total = torch.from_numpy(normals_gt_np)
            confs_gt_total = torch.from_numpy(conf_gt_np)
            intrinsics_gt_total = torch.from_numpy(intrinsic_gt_np)
            masks_gt_total = torch.from_numpy(masks_gt_np)
            depth_gt_total = torch.from_numpy(depth_gt_np)
            spec_gt_total = torch.from_numpy(spec_gt_np)
            mvps = torch.from_numpy(np.array(mvps_np).astype(np.float32)).cuda()
            mvps_test = torch.from_numpy(np.array(mvps_np_test).astype(np.float32)).cuda()
            # mvps = torch.cat((mvps, mvps_test), dim=0)
            w2cs = torch.from_numpy(np.array(w2cs_np).astype(np.float32)).cuda()
            w2cs_test = torch.from_numpy(np.array(w2cs_np_test).astype(np.float32)).cuda()
            # w2cs = torch.cat((w2cs, w2cs_test), dim=0)
        else:
            # shuffle data
            shuffle_idx = torch.randperm(images_gt_total.size()[0])
            images_gt_total = images_gt_total[shuffle_idx]
            normals_gt_total = normals_gt_total[shuffle_idx]
            confs_gt_total = confs_gt_total[shuffle_idx]
            intrinsics_gt_total = intrinsics_gt_total[shuffle_idx]
            masks_gt_total = masks_gt_total[shuffle_idx]
            depth_gt_total = depth_gt_total[shuffle_idx]
            spec_gt_total = spec_gt_total[shuffle_idx]
            mvps = mvps[shuffle_idx]
            w2cs = w2cs[shuffle_idx]
        # torch.cuda.empty_cache()
        # model.vertices = batch_vertices[0]
        for batch in range(0, total_num, batch_size):
            cur_batch_size = min(total_num, (batch + batch_size)) - batch
            images_gt = images_gt_total[batch:(batch + cur_batch_size)].cuda()
            normals_gt = normals_gt_total[batch:(batch + cur_batch_size)].cuda()
            confs_gt = confs_gt_total[batch:(batch + cur_batch_size)].cuda()
            intrinsic_gt = intrinsics_gt_total[batch:(batch + cur_batch_size)].cuda()
            if load_mode == 'colmap':
                depth_gt = depth_gt_total[batch:(batch + cur_batch_size)].cuda()
                masks_gt = masks_gt_total[batch:(batch + cur_batch_size)].cuda()
                spec_gt = spec_gt_total[batch:(batch + cur_batch_size)].cuda()

            normals_gt = torch.where((masks_gt.unsqueeze(3)>0), normals_gt, torch.zeros_like(normals_gt))

            batch_vertices, batch_faces, laplacian_loss, hexagon_loss, albedo_flatten_loss, vertices_flatten_loss = model(cur_batch_size)

            vertsw = torch.cat([batch_vertices, torch.ones([batch_vertices.shape[0], batch_vertices.shape[1], 1], device=batch_vertices.device)], axis=2)
            rot_verts = torch.einsum('bvk, blk->bvl', vertsw, w2cs[batch:(batch + cur_batch_size)])
            normals = get_vertex_normals(rot_verts[:,:,:3], model.faces.long(), model.v2f)
            normals_world = get_vertex_normals(vertsw[:,:,:3], model.faces.long(), model.v2f)
            # for test
            # mm = model.template_mesh
            # test_normals = trimesh.geometry.weighted_vertex_normals(mm.vertices.shape[0], mm.faces, mm.face_normals, mm.face_angles)
            # test_normals = torch.from_numpy(test_normals.astype(np.float32)).cuda().unsqueeze(0)

            #TODO: for ray direction
            H, W = SIZE
            fx, fy, cx, cy = intrinsic_gt[:,0,0], intrinsic_gt[:,1,1], intrinsic_gt[:,0,2], intrinsic_gt[:,1,2]
            ray_direction = torch.ones((fx.shape[0], H, W, 3), device=vertsw.device, requires_grad=False)
            for bi in range(fx.shape[0]):
                cam_ray_direction = get_ray_directions(H, W, fx[bi], fy[bi], cx[bi], cy[bi])
                c2w = torch.inverse(w2cs[bi])
                tmp_ray_direction = get_rays(cam_ray_direction, c2w)
                ray_direction[bi] = tmp_ray_direction


            # if load_mode == 'colmap':
            proj_verts = torch.einsum('bvk, blk->bvl', vertsw, mvps[batch:(batch + cur_batch_size)])
            # else:
            #     proj_verts = torch.einsum('bvk, blk->bvl', rot_verts, proj.expand(cur_batch_size, -1, -1))
            rast_out, rast_out_db = dr.rasterize(glctx, proj_verts, model.faces, resolution=SIZE)



            # use texture
            feat = torch.cat([normals_world, normals, torch.ones_like(normals), proj_verts[:,:,3].unsqueeze(2), rot_verts[:,:,:3], model.texture.unsqueeze(0).expand(cur_batch_size, -1, -1)], dim=2)
            feat, _ = dr.interpolate(feat, rast_out, model.faces)
            normals_world_map = feat[:,:,:,:3].contiguous()
            normal_map = feat[:,:,:,3:6].contiguous()
            if ps_normal:
                normal_map[..., 1] = -normal_map[..., 1]
                normal_map[..., 2] = -normal_map[..., 2]
            pred_mask = feat[:,:,:,6:9].contiguous()
            depth_map = feat[:,:,:,9].contiguous()
            verts_map = feat[:,:,:,10:13].contiguous()
            tex = feat[:,:,:,13:].contiguous()
            

            # feat = torch.cat([rot_verts[:,:,:3], normals, torch.ones_like(normals), proj_verts[:,:,3].unsqueeze(2)], dim=2)
            # feat, _ = dr.interpolate(feat, rast_out, model.faces)
            # verts_map = feat[:,:,:,:3].contiguous()
            # normal_map = feat[:,:,:,3:6].contiguous()
            # if ps_normal:
            #     normal_map[..., 1] = -normal_map[..., 1]
            #     normal_map[..., 2] = -normal_map[..., 2]
            # pred_mask = feat[:,:,:,6:9].contiguous()
            # depth_map = feat[:,:,:,9].contiguous()

            valid_idx = torch.where(rast_out[:,:,:,3] > 0)

            # valid_verts = verts_map[valid_idx]
            # valid_normal = F.normalize(normal_map[valid_idx], p=2, dim=1)
            
            normal_map = (normal_map+1) / 2
            # # for nhr
            # normal_map = 1-normal_map
            if len(confs_gt.shape) == 4:
                confs_gt = confs_gt[:,:,:,0]
            normal_valid_idx = torch.where(((rast_out[:,:,:,3] > 0) & (confs_gt > 0)))
            normal_loss = F.l1_loss(normal_map[normal_valid_idx], normals_gt[normal_valid_idx])
            # normal_loss = F.l1_loss(normal_map, normals_gt)
            
            depth_loss = F.l1_loss(depth_map[normal_valid_idx], depth_gt[normal_valid_idx])
            
            pred_mask = dr.antialias(pred_mask, rast_out, proj_verts, model.faces)

            # valid = (pred_mask > 0) & (masks[k:k+batch] > 0)
            # TODO: specluar using SUV space
            # TODO: queding specluar xiang
            # TODO: change lr
            # TODO: add photometric loss
            
            # BUG sparse matrix
            # BUG c++ to so for high mesh resolution
            # BUG render net
            # BUG PSNR high and good Chamfer distance
            # BUG coarse to fine
            # BUG all data
            if change_weight:
                if use_adam:
                    # for adam
                    if (i == 0):
                        w_h = 1
                        w_n = 1e2
                        w_c = 0
                        w_d = 1
                        w_i = 1e2
                        for name, param in model.named_parameters():
                            if name == 'light' or name == 'texture' or name == 'vertices':
                                param.requires_grad = False
                            if name == 'displace' or name == 'center':
                                param.requires_grad = True
                    elif (i == 1):
                        w_h = 2
                        w_n = 1
                        w_c = 1
                        w_d = 4
                        w_i = 1e2
                        for name, param in model.named_parameters():
                            if name == 'light' or name == 'texture' or name == 'center':
                                param.requires_grad = True
                            if name == 'displace' or name == 'vertices':
                                param.requires_grad = False
                    elif (i == 5):
                        w_h = 8
                        w_n = 40
                        w_c = 0
                        w_d = 0
                        w_i = 1e2
                        for name, param in model.named_parameters():
                            if name == 'light' or name == 'texture':
                                param.requires_grad = True
                            if name == 'displace' or name == 'center' or name == 'vertices':
                                param.requires_grad = True
                    # elif (i >= 100 and i < 200 and (i % 100 == 0) and batch == 0):
                    #     w_d = w_d / 2
                    #     w_h = w_h - 0.3
                    #     w_n = 10
                    #     for name, param in model.named_parameters():
                    #         if name == 'light' or name == 'texture':
                    #             param.requires_grad = True
                    #         if name == 'displace' or name == 'center' or name == 'vertices':
                    #             param.requires_grad = False
                    elif (i >= 20 and i < 2600 and (i % 200 == 0) and batch == 0):
                        if w_c > 2e4:
                            w_c = 2e4
                        else:
                            w_c = w_c / 1.2
                        w_h /= 1.2
                        w_n = 2
                        w_d /= 1.5
                        lr_render /= 2
                        lr_featur /= 2
                        optimizer = torch.optim.Adam([
                            {'params': model.render_net.parameters(), 'lr': lr_render, 'momentum': 0.9},
                            {'params': [model.texture, model.vertices], 'lr': lr_featur, 'momentum': 0.9}], 1e-3, betas=(0.5, 0.99))
                        for name, param in model.named_parameters():
                            if name == 'light' or name == 'texture':
                                param.requires_grad = True
                            if name == 'displace' or name == 'center' or name == 'vertices':
                                param.requires_grad = True
                    elif (i == 3000):
                        w_c = 10
                        w_h = 1
                        print(w_h)
                        w_a = 0
                        w_n = 0
                        w_d = 1e-1
                        for name, param in model.named_parameters():
                            if name == 'light' or name == 'texture':
                                param.requires_grad = False
                            if name == 'displace' or name == 'center' or name == 'vertices':
                                param.requires_grad = True
                    # elif (i == 1340):
                    #     w_c = 200
                    #     w_h = 0
                    #     print(w_h)
                    #     w_a = 0
                    #     w_n = 0
                    #     for name, param in model.named_parameters():
                    #         if name == 'light' or name == 'texture':
                    #             param.requires_grad = True
                    #         if name == 'displace' or name == 'center' or name == 'vertices':
                    #             param.requires_grad = False

            
            #TODO: like volume rendering, RGB = f(pos_en(x), pos_en(view), world_normal, feature)
            # # add positional encoding on feature
            # feature = model.feature_net(images_gt.permute(0,3,1,2)) # B C H W
            # render_tensor = torch.cat((model.position_encoder(verts_map.view(-1,3)).view(-1, H, W, (model.position_encoder.embedding_size)), model.position_encoder_feature(feature['stage3'].permute(0,2,3,1).reshape(-1,9)).view(-1, H, W, (model.position_encoder_feature.embedding_size)), ray_direction, normals_world_map), dim=3).permute(0,3,1,2)
            # # use high-level feature from img replace tex[:7]
            # feature = model.feature_net(images_gt.permute(0,3,1,2)) # B C H W
            # render_tensor = torch.cat((model.position_encoder(verts_map.view(-1,3)).view(-1, H, W, (model.position_encoder.embedding_size)), feature['stage3'].permute(0,2,3,1), ray_direction, normals_world_map), dim=3).permute(0,3,1,2)

            # positional encoding verts and tex[:]
            render_tensor = torch.cat((model.position_encoder(verts_map.view(-1,3)).view(-1, H, W, model.position_encoder.embedding_size), model.position_encoder_feature(tex[:,:,:,:].view(-1,model.feature_channel)).view(-1, H, W, model.position_encoder_feature.embedding_size), model.position_encoder_ray(ray_direction.view(-1,3)).view(-1, H, W, model.position_encoder_ray.embedding_size), normals_world_map), dim=3).permute(0,3,1,2)
            
            # # only positional encoding verts_position
            # render_tensor = torch.cat((model.position_encoder(verts_map.view(-1,3)).view(-1, H, W, model.position_encoder.embedding_size), tex[:,:,:,:], ray_direction, normals_world_map), dim=3).permute(0,3,1,2)            
            # # PE only for albedo, add world normal
            # render_tensor = torch.cat((model.position_encoder(tex[:,:,:,:3].view(-1,3)).view(-1, H, W, model.position_encoder.embedding_size), tex[:,:,:,3:], ray_direction, normals_world_map), dim=3).permute(0,3,1,2)
            # # PE only for albedo, no normal
            # render_tensor = torch.cat((model.position_encoder(tex[:,:,:,:3].view(-1,3)).view(-1, H, W, model.input_channel-7), tex[:,:,:,3:], ray_direction), dim=3).permute(0,3,1,2)
            # # PE for albedo and spec, no normal
            # render_tensor = torch.cat((model.position_encoder(tex.view(-1,7)).view(-1, H, W, model.input_channel-3), ray_direction), dim=3).permute(0,3,1,2)
            # # original, no PE, no spec, no normal
            # render_tensor = torch.cat((tex, ray_direction), dim=3).permute(0,3,1,2)
            if i < 1:
                images_pred = images_gt
                spec_pred = spec_gt
                spec_loss = 0
            elif i < 100:
                images_pred = tex[:,:,:,:3]
                # images_pred = model.render_net(render_tensor).permute(0,2,3,1)
                spec_pred = tex[:,:,:,3:]
                spec_loss = F.smooth_l1_loss(spec_pred[:, :, :, :3][valid_idx], spec_gt[:, :, :, :3][valid_idx])
            elif i < 200:
                for name, param in model.named_parameters():
                    if name == 'texture':
                        param.requires_grad = False
                images_pred = model.render_net(render_tensor).permute(0,2,3,1)
                spec_loss = 0
            else:
                for name, param in model.named_parameters():
                    if name == 'texture':
                        param.requires_grad = True
                images_pred = model.render_net(render_tensor).permute(0,2,3,1)
                spec_loss = 0

            # color_loss = F.l1_loss(images_pred[:, :, :, :3][valid_idx], images_gt[:, :, :, :3][valid_idx])
            color_loss = F.smooth_l1_loss(images_pred[:, :, :, :3][valid_idx], images_gt[:, :, :, :3][valid_idx])
            # color_loss = F.mse_loss(images_pred[:, :, :, :3][valid_idx], images_gt[:, :, :, :3][valid_idx])
            
            # average_edge = get_average_edge(vertsw[:,:,:3], model.faces.long())
            edge_loss = get_edge_loss(vertsw[:,:,:3], model.faces.long())
            
            
            if load_mode == 'colmap':
                mask_loss = F.mse_loss(pred_mask, masks_gt.unsqueeze(3).expand(-1,-1,-1,3))
                IOU_loss = neg_iou_loss(pred_mask, masks_gt.unsqueeze(3).expand(-1,-1,-1,3))
            else:
                mask_loss = F.mse_loss(pred_mask, images_gt[:, :, :, 3].unsqueeze(3).expand(-1,-1,-1,3))
                IOU_loss = neg_iou_loss(pred_mask, images_gt[:, :, :, 3].unsqueeze(3).expand(-1,-1,-1,3))
            # mask_loss = F.mse_loss(pred_mask, images_gt[:, :, :, 3])
            # IOU_loss = neg_iou_loss(pred_mask, images_gt[:, :, :, 3])
            
            if only_tex:
                loss = w_c*color_loss + spec_loss + w_i*IOU_loss + w_h*hexagon_loss + w_n*normal_loss + w_d*depth_loss + 1e5*edge_loss
                # loss = 1e2*color_loss
                print("normal: ", w_n*normal_loss)
                print("hexagon: ", w_h*hexagon_loss)
                print("depth: ", w_d*depth_loss)
                print("color: ", color_loss)
                print("edge: ", 5e5*edge_loss)
                # print("displace: ", w_d*displace_loss)
                # print(1e2*color_loss/(1e-4*rgb_flatten_loss))
            elif only_shape:
                loss = 100*IOU_loss + 0.5 * hexagon_loss
            else:
                loss = 10*color_loss + IOU_loss + 0.03 * laplacian_loss + 0.01 * hexagon_loss
            
            # img_save = pred_mask.clone()
            # img_save = images_gt[:, :, :, 3].unsqueeze(3).expand(-1,-1,-1,3).clone()
            # img_save = img_save[0].cpu().detach().numpy()
            # imageio.imsave("./zysresult/img_mask%d.png"%0, img_save[:,:,:])
            
            loop.set_description('Loss: %.4f' % (loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                if save_flag and i == (loop_num-1):
                    for j in range(cur_batch_size):
                        img_save = images_pred.detach().cpu().numpy()[j]
                        img_gt_save = images_gt.clone()
                        img_gt_save = img_gt_save[j].cpu().detach().numpy()
                        
                        # img_save = normal_map.clone()
                        # img_save = img_save[j].cpu().detach().numpy()
                        # img_gt_save = normals_gt.clone()
                        # img_gt_save = img_gt_save[j].cpu().detach().numpy()
                        imageio.imsave(os.path.join(args.output_dir, 'img_gt%05d.png' % (j+batch)), img_gt_save[:,:,:])
                        imageio.imsave(os.path.join(args.output_dir, 'img_pred%05d.png' % (j+batch)), img_save[:,:,:])
                if novel_flag and i == (loop_num-1) and batch == 0:
                    # novel_mvps, novel_extrinsics = gen_novel_view(w2cs, intrinsic_gt[0])
                    novel_mvps = mvps_test
                    novel_extrinsics = w2cs_test
                    
                    novel_ray_direction = torch.ones((novel_mvps.shape[0], H, W, 3), device=vertsw.device, requires_grad=False)
                    for bi in range(novel_mvps.shape[0]):
                        novel_cam_ray_direction = get_ray_directions(H, W, fx[0], fy[0], cx[0], cy[0])
                        c2w = torch.inverse(novel_extrinsics[bi])
                        novel_tmp_ray_direction = get_rays(novel_cam_ray_direction, c2w)
                        novel_ray_direction[bi] = novel_tmp_ray_direction
                    
                    for ni in range(0, novel_mvps.shape[0], batch_size):
                        start_time = time.time()
                        
                        novel_proj_verts = torch.einsum('bvk, blk->bvl', vertsw, novel_mvps[ni:ni+batch_size])
                        novel_rot_verts = torch.einsum('bvk, blk->bvl', vertsw, novel_extrinsics[ni:ni+batch_size])
                        novel_rast_out, rast_out_db = dr.rasterize(glctx, novel_proj_verts, model.faces, resolution=SIZE)
                        novel_feat = torch.cat([normals_world, torch.ones_like(normals), novel_rot_verts[:,:,:3], model.texture.unsqueeze(0).expand(batch_size, -1, -1)], dim=2)
                        novel_feat, _ = dr.interpolate(novel_feat, novel_rast_out, model.faces)
                        normals_world_map = novel_feat[:,:,:,:3].contiguous()
                        pred_mask = novel_feat[:,:,:,3:6].contiguous()
                        novel_verts_map = novel_feat[:,:,:,6:9].contiguous()
                        novel_tex = novel_feat[:,:,:,9:].contiguous()
                        pred_mask = dr.antialias(pred_mask, novel_rast_out, novel_proj_verts, model.faces.clone())
                        # # only positional encoding verts_position
                        # novel_render_tensor = torch.cat((model.position_encoder(novel_verts_map.view(-1,3)).view(-1, H, W, model.position_encoder.embedding_size), novel_tex, novel_ray_direction[ni:ni+batch_size], normals_world_map), dim=3).permute(0,3,1,2)
                        
                        # PE for verts and feature
                        novel_render_tensor = torch.cat((model.position_encoder(novel_verts_map.view(-1,3)).view(-1, H, W, model.position_encoder.embedding_size), model.position_encoder_feature(novel_tex.view(-1,model.feature_channel)).view(-1, H, W, model.position_encoder_feature.embedding_size), model.position_encoder_ray(novel_ray_direction[ni:ni+batch_size].view(-1,3)).view(-1, H, W, model.position_encoder_ray.embedding_size), normals_world_map), dim=3).permute(0,3,1,2)
                        images_pred = model.render_net(novel_render_tensor).permute(0,2,3,1)
                        
                        end_time = time.time()
                        print("time: ", end_time-start_time)
                        # images_pred = torch.where(pred_mask > 0, images_pred, torch.zeros_like(images_pred, dtype=torch.float32))
                        # imageio.imsave(os.path.join(args.output_dir, 'novel%05d.png'%ni), images_pred[0].detach().cpu().numpy())
                        for k in range(batch_size):
                            imageio.imsave(os.path.join(args.output_dir, 'novel_mask%05d.png'%(ni+k)), images_pred[k].detach().cpu().numpy())
                            tex_map = novel_tex[k][:,:,3:6].detach().cpu().numpy()
                            imageio.imsave(os.path.join(args.output_dir, 'novel_tex%05d.png'%(ni+k)), tex_map)
                

                if (i % 20 == 0) and (batch == 0):
                    # cast_normal_gt = normals_gt[2].squeeze(0).cpu().detach().numpy()
                    # cast_normal = normal_map[2].squeeze(0).cpu().detach().numpy()
                    # print(cast_normal.shape)
                    # imageio.imsave('/workspace/nvrender/normal_map.png', cast_normal[:,:,:])
                    # imageio.imsave('/workspace/nvrender/normal_map_gt.png', cast_normal_gt[:,:,:])
                    image = images_gt[:, 3].detach().cpu().numpy()[gif_view_num]
                    if i < 400:
                        image = normal_map[gif_view_num].squeeze(0).cpu().detach().numpy()
                        # image = images_pred.detach().cpu().numpy()[gif_view_num]
                    else:
                        image = images_pred.detach().cpu().numpy()[gif_view_num]
                        
                    writer.append_data((255*image).astype(np.uint8))
                    imageio.imsave(os.path.join(args.output_dir, 'deform_%05d.png' % i), (255*image[:,:,:]).astype(np.uint8))
   
        
    # save optimized mesh
    if only_tex:
        savepath = os.path.join(args.output_dir, 'plane_tex.obj')
        torch.save(model.render_net.state_dict(), os.path.join(args.output_dir, 'render_net.pth'))
        np.save(os.path.join(args.output_dir, 'feature.npy'), model.texture.detach().cpu().numpy())
        if use_high_feature:
            torch.save(model.feature_net.state_dict(), os.path.join(args.output_dir, 'feature_net.pth'))
    else:
        savepath = os.path.join(args.output_dir, 'plane.obj')
    
    if uniform_flag:
        model.save_obj(savepath, 'uniform')
    else:
        model.save_obj(savepath, 'vertices')

    # base = torch.log(model.vertices.abs() / (1 - model.vertices.abs()))
    # centroid = torch.tanh(model.center)
    # result_vertices = torch.sigmoid(base + model.displace) * torch.sign(model.vertices)
    # result_vertices = F.relu(result_vertices) * (1 - centroid) - F.relu(-result_vertices) * (centroid + 1)
    # result_vertices = result_vertices + centroid
    # result_faces = model.faces

    # # result_vertices , result_faces = model.vertices, model.faces


    # result_mesh = trimesh.base.Trimesh(result_vertices, result_faces)
    # result_mesh.export(os.path.join(args.output_dir, 'plane.obj'))



if __name__ == '__main__':
    main()


# python compute_K.py -n 0; python demo_deform.py -n 0; python compute_K.py -n 1; python demo_deform.py -n 1; python compute_K.py -n 3; python demo_deform.py -n 3; python compute_K.py -n 4; python demo_deform.py -n 4; python compute_K.py -n 5; python demo_deform.py -n 5; python compute_K.py -n 6; python demo_deform.py -n 6;     python compute_K.py -n 7; python demo_deform.py -n 7; python compute_K.py -n 8; python demo_deform.py -n 8; python compute_K.py -n 9; python demo_deform.py -n 9; python compute_K.py -n 10; python demo_deform.py -n 10; python compute_K.py -n 11; python demo_deform.py -n 11; python compute_K.py -n 12; python demo_deform.py -n 12; python compute_K.py -n 13; python demo_deform.py -n 13; python compute_K.py -n 14; python demo_deform.py -n 14; python compute_K.py -n 1; python demo_deform.py -n 1;
# cp results/scan69_3/render_net.pth results/scan24; cp results/scan69_3/render_net.pth results/scan37; cp results/scan69_3/render_net.pth results/scan40; cp results/scan69_3/render_net.pth results/scan55; cp results/scan69_3/render_net.pth results/scan63; cp results/scan69_3/render_net.pth results/scan65; cp results/scan69_3/render_net.pth results/scan69; cp results/scan69_3/render_net.pth results/scan83; cp results/scan69_3/render_net.pth results/scan97; cp results/scan69_3/render_net.pth results/scan105; cp results/scan69_3/render_net.pth results/scan106; cp results/scan69_3/render_net.pth results/scan110; cp results/scan69_3/render_net.pth results/scan114; cp results/scan69_3/render_net.pth results/scan118; cp results/scan69_3/render_net.pth results/scan122;