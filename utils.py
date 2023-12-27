import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
import os
import numpy as np
import imageio
import json
import math
from skimage.transform import resize
import skimage
from config import *
import tqdm
import cv2
import re

K = np.array([
    [focal, 0, 0.5*800],
    [0, focal, 0.5*SIZE[1]],
    [0, 0, 1]
], dtype='float32')
proj = np.eye(4, dtype='float32')
proj[:3,:3] = K
proj[0,0] = proj[0,0] / (0.5*float(SIZE[0]))
proj[1,1] = proj[1,1] / (0.5*float(SIZE[1]))
proj[0,2] = proj[0,2] / (0.5*float(SIZE[0])) - 1.
proj[1,2] = proj[1,2] / (0.5*float(SIZE[1])) - 1.
proj[2,2] = 0.
proj[2,3] = -1.
proj[3,2] = 1.0
proj[3,3] = 0.0

def add_color(inputfile, outputfile, color=None):
    if color is None:
        with open(outputfile, 'w') as file:
            with open(inputfile) as f:
                lines = f.readlines()

            for line in lines:
                if len(line.split()) == 0:
                    continue
                if line.split()[0] == '#':
                    file.write(line)
                if line.split()[0] == 'v':
                    file.write(line[:-1])
                    file.write(" 1.00000 1.00000 1.00000\n")
                if line.split()[0] == 'f':
                    file.write(line)
    else:
        with open(outputfile, 'w') as file:
            with open(inputfile) as f:
                lines = f.readlines()
            idx = 0
            for line in lines:
                if len(line.split()) == 0:
                    continue
                if line.split()[0] == '#':
                    file.write(line)
                if line.split()[0] == 'v':
                    file.write(line[:-1])
                    file.write(" %f %f %f\n"%(color[idx][0], color[idx][1], color[idx][2]))
                    idx += 1
                if line.split()[0] == 'f':
                    file.write(line)    

def get_radiance(coff, normal):
    '''
    coff 9
    normal n 3
    '''

    radiance = coff[0]
    radiance = radiance + coff[1] * normal[:,1]
    radiance = radiance + coff[2] * normal[:,2]
    radiance = radiance + coff[3] * normal[:,0]
    radiance = radiance + coff[4] * normal[:,0] * normal[:,1]
    radiance = radiance + coff[5] * normal[:,1] * normal[:,2]
    radiance = radiance + coff[6] * (2*normal[:,2]*normal[:,2]-normal[:,0]*normal[:,0]-normal[:,1]*normal[:,1])
    radiance = radiance + coff[7] * normal[:,2] * normal[:,0]
    radiance = radiance + coff[8] * (normal[:,0]*normal[:,0]-normal[:,1]*normal[:,1])

    return radiance

def get_v2f(vertices, faces):
    n = vertices.shape[1]
    f = faces.shape[0]
    u = []
    v = []
    val = []
    for i in range(f):
        for j in range(3):
            u.append(faces[i,j].item())
            v.append(i)
            val.append(1.)

    return torch.sparse_coo_tensor([u,v], val, size=(n,f), dtype=torch.float, device=vertices.device)

def get_average_edge(vertices, faces):
    v1 = vertices[:, faces[:,0]]
    v2 = vertices[:, faces[:,1]]
    v3 = vertices[:, faces[:,2]]
    
    e1 = v1 - v2
    e2 = v2 - v3
    e3 = v3 - v1

    average_e = torch.norm(e1, p=2, dim=2).mean() + torch.norm(e2, p=2, dim=2).mean() + torch.norm(e3, p=2, dim=2).mean()
    
    return average_e

def get_edge_loss(vertices, faces):
    v1 = vertices[:, faces[:,0]]
    v2 = vertices[:, faces[:,1]]
    v3 = vertices[:, faces[:,2]]
    
    e1 = torch.norm(v1 - v2, p=2, dim=2)
    e2 = torch.norm(v2 - v3, p=2, dim=2)
    e3 = torch.norm(v3 - v1, p=2, dim=2)
    average_edge = e1.mean() + e2.mean() + e3.mean()
    edges = torch.cat([e1, e2, e3], axis=1)
    edge_loss = F.smooth_l1_loss(edges, average_edge)
    return edge_loss

def get_vertex_normals(vertices, faces, v2f=None):
    '''
    :param vertices: [B, N, 3]
    :param faces: [F, 3]
    :param v2f: [N, F] 
    '''

    if v2f is None:
        v2f = get_v2f(vertices, faces)

    v1 = vertices[:, faces[:,1]] - vertices[:, faces[:,0]] # B F 3
    v1 = F.normalize(v1, p=2, dim=2)
    v2 = vertices[:, faces[:,2]] - vertices[:, faces[:,0]]
    v2 = F.normalize(v2, p=2, dim=2)

    face_normal = torch.cross(v1, v2, dim=2)
    vertices_normal = torch.stack([torch.matmul(v2f, item) for item in face_normal], dim=0) / torch.sparse.sum(v2f, dim=1).to_dense().reshape(1,-1,1)

    # vertices_normal = torch.zeros_like(vertices)
    # vertices_normal = vertices_normal.index_add(0, faces[:, 0], face_normal)
    # vertices_normal = vertices_normal.index_add(0, faces[:, 1], face_normal)
    # vertices_normal = vertices_normal.index_add(0, faces[:, 2], face_normal)
    
    vertices_normal = F.normalize(vertices_normal, p=2, dim=2)
    return vertices_normal

def laplacian_cot(verts, faces):
    '''
    verts n 3
    faces f 3
    '''
    V, F = verts.shape[0], faces.shape[0]

    face_verts = verts[faces.long()] # f 3 3
    v0, v1, v2 = face_verts[:,0], face_verts[:,1], face_verts[:,2]

    A = (v1 - v2).norm(dim=1)
    B = (v0 - v2).norm(dim=1)
    C = (v0 - v1).norm(dim=1)

    s = 0.5 * (A + B + C)
    area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=1e-12).sqrt()

    A2, B2, C2 = A * A, B * B, C * C
    cota = (B2 + C2 - A2) / area
    cotb = (A2 + C2 - B2) / area
    cotc = (A2 + B2 - C2) / area
    cot = torch.stack([cota, cotb, cotc], dim=1)
    cot = cot / 4.0

    ii = faces[:, [1, 2, 0]]
    jj = faces[:, [2, 0, 1]]

    idx = torch.stack([ii, jj], dim=0).view(2, F * 3)
    L = torch.sparse.FloatTensor(idx.long(), cot.view(-1), (V, V))

    L = L + L.t()

    idx = faces.long().view(-1)
    inv_areas = torch.zeros(V, dtype=torch.float32, device=verts.device)
    val = torch.stack([area] * 3, dim=1).view(-1)
    inv_areas.scatter_add_(0, idx, val)
    idx = inv_areas > 0
    inv_areas[idx] = 1.0 / inv_areas[idx]
    inv_areas = inv_areas.view(-1, 1)

    return L, inv_areas

def compute_laplacian(verts, faces):
    n = verts.shape[0]
    device = verts.device
    u = []
    v = []
    val = []
    v0, v1, v2 = faces.chunk(3,dim=1)
    e1 = torch.cat([v0, v1], dim=1)
    e2 = torch.cat([v1, v2], dim=1)
    e3 = torch.cat([v2, v0], dim=1)
    edges = torch.cat([e1, e2, e3], dim=0)
    for i in range(edges.shape[0]):
        u.append(edges[i,0].item())
        v.append(edges[i,1].item())
        val.append(1.)
        u.append(edges[i,1].item())
        v.append(edges[i,0].item())
        val.append(1.)

    A = torch.sparse_coo_tensor([u,v], val, size=(n,n), dtype=torch.float, device=device)
    deg = torch.sparse.sum(A, dim=1).to_dense()
    deg = torch.where(deg > 0, 1.0 / deg, deg)
    for i in range(len(u)):
        idx = u[i]
        val[i] = deg[idx].item()
    L = torch.sparse_coo_tensor([u,v], val, size=(n,n), dtype=torch.float, device=device)

    idx = torch.arange(n, device=device)
    idx = torch.stack([idx, idx], dim=0)
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=device)
    L -= torch.sparse.FloatTensor(idx, ones, (n, n))

    return L

def laplacian_smoothing(verts, faces, method="uniform", laplacian=None):
    weights = 1.0 / verts.shape[0]

    with torch.no_grad():
        if method == "uniform":
            assert laplacian is not None
            L = laplacian
        else:
            L, inv_areas = laplacian_cot(verts, faces)
            if method == "cot":
                norm_w = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
                idx = norm_w > 0
                norm_w[idx] = 1.0 / norm_w[idx]
            else:
                L_sum = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
                norm_w = 0.25 * inv_areas

    if method == "uniform":
        loss = L.mm(verts)
    elif method == "cot":
        loss = L.mm(verts) * norm_w - verts
    else:
        loss = (L.mm(verts) - L_sum * verts) * norm_w
    
    loss = loss.norm(dim=1)

    loss = loss * weights
    return loss.sum()

def get_neighbors(edges, point_index):
    neighbor_index = np.argwhere(edges == point_index)
    neighbor_0_index = np.argwhere(neighbor_index[:, 1] == 0)
    neighbor_index = np.delete(neighbor_index[:, 0], neighbor_0_index[:, 0])
    neighbors = edges[neighbor_index][:, 0]
    return neighbors

def compute_K(mesh):
    edges = mesh.edges
    vertices_num = mesh.vertices.shape[0]
    v6 = 0
    K = np.zeros((vertices_num, vertices_num), np.int8)
    loop = tqdm.tqdm(list(range(0, vertices_num)))
    for vertice in loop:
        # print(vertice)
        vertice_neighbors = get_neighbors(edges, vertice)
        sorted_vertice_neighbors = vertice_neighbors.copy()
        if vertice_neighbors.shape[0] == 6:
            try:
                tmp = vertice_neighbors[0]
                sorted_vertice_neighbors[0] = tmp
                neighbor0_neighbors = get_neighbors(edges, sorted_vertice_neighbors[0])
                sorted_vertice_neighbors[1], sorted_vertice_neighbors[2] = np.intersect1d(vertice_neighbors, neighbor0_neighbors)
                neighbor1_neighbors = get_neighbors(edges, sorted_vertice_neighbors[1])
                neighbor1_neighbors = np.setdiff1d(neighbor1_neighbors, sorted_vertice_neighbors[0], True)
                sorted_vertice_neighbors[3] = np.intersect1d(vertice_neighbors, neighbor1_neighbors, True)

                neighbor2_neighbors = get_neighbors(edges, sorted_vertice_neighbors[2])
                neighbor2_neighbors = np.setdiff1d(neighbor2_neighbors, sorted_vertice_neighbors[0], True)
                sorted_vertice_neighbors[4] = np.intersect1d(vertice_neighbors, neighbor2_neighbors, True)

                neighbor3_neighbors = get_neighbors(edges, sorted_vertice_neighbors[3])
                neighbor3_neighbors = np.setdiff1d(neighbor3_neighbors, sorted_vertice_neighbors[1], True)
                sorted_vertice_neighbors[5] = np.intersect1d(vertice_neighbors, neighbor3_neighbors, True)
            except ValueError:
                continue

            for idx in range(3):
                A = np.zeros((vertices_num, 1), np.int8)
                A[sorted_vertice_neighbors[2*idx]] = A[sorted_vertice_neighbors[5-2*idx]] = 2
                A[vertice] = -4
                tmpK = np.dot(A, A.T)
                K = K + tmpK
            v6 += 1
        # elif vertice_neighbors.shape[0] == 4:
        #     try:
        #         tmp = vertice_neighbors[0]
        #         sorted_vertice_neighbors[0] = tmp
        #         neighbor0_neighbors = get_neighbors(edges, sorted_vertice_neighbors[0])
        #         sorted_vertice_neighbors[1], sorted_vertice_neighbors[2] = np.intersect1d(vertice_neighbors, neighbor0_neighbors, True)
        #         neighbor1_neighbors = get_neighbors(edges, sorted_vertice_neighbors[1])
        #         neighbor1_neighbors = np.setdiff1d(neighbor1_neighbors, sorted_vertice_neighbors[0], True)
        #         sorted_vertice_neighbors[3] = np.intersect1d(vertice_neighbors, neighbor1_neighbors, True)
        #     except ValueError:
        #         continue

        #     for idx in range(2):
        #         A = np.zeros((vertices_num, 1), np.int8)
        #         A[sorted_vertice_neighbors[2*idx]] = A[sorted_vertice_neighbors[3-2*idx]] = 2
        #         A[vertice] = -4
        #         tmpK = np.dot(A, A.T)
        #         K = K + tmpK

        # elif vertice_neighbors.shape[0] == 7:
        #     tmp = vertice_neighbors[0]
        #     sorted_vertice_neighbors[0] = tmp
        #     neighbor0_neighbors = get_neighbors(edges, sorted_vertice_neighbors[0])
        #     try:
        #         sorted_vertice_neighbors[1], sorted_vertice_neighbors[2] = np.intersect1d(vertice_neighbors, neighbor0_neighbors, True)
        #     except ValueError:
        #         continue
        #     neighbor1_neighbors = get_neighbors(edges, sorted_vertice_neighbors[1])
        #     neighbor1_neighbors = np.setdiff1d(neighbor1_neighbors, sorted_vertice_neighbors[0], True)
        #     sorted_vertice_neighbors[3] = np.intersect1d(vertice_neighbors, neighbor1_neighbors, True)

        #     neighbor2_neighbors = get_neighbors(edges, sorted_vertice_neighbors[2])
        #     neighbor2_neighbors = np.setdiff1d(neighbor2_neighbors, sorted_vertice_neighbors[0], True)
        #     sorted_vertice_neighbors[4] = np.intersect1d(vertice_neighbors, neighbor2_neighbors, True)

        #     neighbor3_neighbors = get_neighbors(edges, sorted_vertice_neighbors[3])
        #     neighbor3_neighbors = np.setdiff1d(neighbor3_neighbors, sorted_vertice_neighbors[0], True)
        #     print(np.intersect1d(vertice_neighbors, neighbor3_neighbors, True))
        #     sorted_vertice_neighbors[5] = np.intersect1d(vertice_neighbors, neighbor3_neighbors, True)

        #     neighbor4_neighbors = get_neighbors(edges, sorted_vertice_neighbors[4])
        #     neighbor4_neighbors = np.setdiff1d(neighbor4_neighbors, sorted_vertice_neighbors[0], True)
        #     sorted_vertice_neighbors[6] = np.intersect1d(vertice_neighbors, neighbor4_neighbors, True)

        #     A05 = np.zeros((vertices_num, 1), np.int8)
        #     A05[sorted_vertice_neighbors[0]] = A05[sorted_vertice_neighbors[5]] = 1
        #     A05[vertice] = -2
        #     tmpK05 = np.dot(A05, A05.T)
        #     A06 = np.zeros((vertices_num, 1), np.int8)
        #     A06[sorted_vertice_neighbors[0]] = A06[sorted_vertice_neighbors[6]] = 1
        #     A06[vertice] = -2
        #     tmpK06 = np.dot(A06, A06.T)
        #     A14 = np.zeros((vertices_num, 1), np.int8)
        #     A14[sorted_vertice_neighbors[1]] = A14[sorted_vertice_neighbors[4]] = 1
        #     A14[vertice] = -2
        #     tmpK14 = np.dot(A14, A14.T)
        #     A16 = np.zeros((vertices_num, 1), np.int8)
        #     A16[sorted_vertice_neighbors[1]] = A16[sorted_vertice_neighbors[6]] = 1
        #     A16[vertice] = -2
        #     tmpK16 = np.dot(A16, A16.T)
        #     A25 = np.zeros((vertices_num, 1), np.int8)
        #     A25[sorted_vertice_neighbors[2]] = A25[sorted_vertice_neighbors[5]] = 1
        #     A25[vertice] = -2
        #     tmpK25 = np.dot(A25, A25.T)
        #     A23 = np.zeros((vertices_num, 1), np.int8)
        #     A23[sorted_vertice_neighbors[2]] = A23[sorted_vertice_neighbors[3]] = 1
        #     A23[vertice] = -2
        #     tmpK23 = np.dot(A23, A23.T)
        #     A34 = np.zeros((vertices_num, 1), np.int8)
        #     A34[sorted_vertice_neighbors[3]] = A34[sorted_vertice_neighbors[4]] = 1
        #     A34[vertice] = -2
        #     tmpK34 = np.dot(A34, A34.T)
        #     K = K + tmpK05 + tmpK06 + tmpK14 + tmpK16 + tmpK25 + tmpK23 + tmpK34
        elif vertice_neighbors.shape[0] == 5:
            try:
                tmp = vertice_neighbors[0]
                sorted_vertice_neighbors[0] = tmp
                neighbor0_neighbors = get_neighbors(edges, sorted_vertice_neighbors[0])
                sorted_vertice_neighbors[1], sorted_vertice_neighbors[2] = np.intersect1d(vertice_neighbors, neighbor0_neighbors, True)
                neighbor1_neighbors = get_neighbors(edges, sorted_vertice_neighbors[1])
                neighbor1_neighbors = np.setdiff1d(neighbor1_neighbors, sorted_vertice_neighbors[0], True)
                sorted_vertice_neighbors[3] = np.intersect1d(vertice_neighbors, neighbor1_neighbors, True)

                neighbor2_neighbors = get_neighbors(edges, sorted_vertice_neighbors[2])
                neighbor2_neighbors = np.setdiff1d(neighbor2_neighbors, sorted_vertice_neighbors[0], True)
                sorted_vertice_neighbors[4] = np.intersect1d(vertice_neighbors, neighbor2_neighbors, True)
            except ValueError:
                continue

            A03 = np.zeros((vertices_num, 1), np.int8)
            A03[sorted_vertice_neighbors[0]] = A03[sorted_vertice_neighbors[3]] = 1
            A03[vertice] = -2
            tmpK03 = np.dot(A03, A03.T)
            A04 = np.zeros((vertices_num, 1), np.int8)
            A04[sorted_vertice_neighbors[0]] = A04[sorted_vertice_neighbors[4]] = 1
            A04[vertice] = -2
            tmpK04 = np.dot(A04, A04.T)
            A12 = np.zeros((vertices_num, 1), np.int8)
            A12[sorted_vertice_neighbors[1]] = A12[sorted_vertice_neighbors[2]] = 1
            A12[vertice] = -2
            tmpK12 = np.dot(A12, A12.T)
            A14 = np.zeros((vertices_num, 1), np.int8)
            A14[sorted_vertice_neighbors[1]] = A14[sorted_vertice_neighbors[4]] = 1
            A14[vertice] = -2
            tmpK14 = np.dot(A14, A14.T)
            A23 = np.zeros((vertices_num, 1), np.int8)
            A23[sorted_vertice_neighbors[2]] = A23[sorted_vertice_neighbors[3]] = 1
            A23[vertice] = -2
            tmpK23 = np.dot(A23, A23.T)
            K = K + tmpK03 + tmpK04 + tmpK12 + tmpK14 + tmpK23
            v6 += 1
        # elif vertice_neighbors.shape[0] == 3:
        #     try:
        #         tmp = vertice_neighbors[0]
        #         sorted_vertice_neighbors[0] = tmp
        #         neighbor0_neighbors = get_neighbors(edges, sorted_vertice_neighbors[0])
        #         sorted_vertice_neighbors[1], sorted_vertice_neighbors[2] = np.intersect1d(vertice_neighbors, neighbor0_neighbors, True)
        #     except ValueError:
        #         continue

        #     A01 = np.zeros((vertices_num, 1), np.int8)
        #     A01[sorted_vertice_neighbors[0]] = A01[sorted_vertice_neighbors[1]] = 1
        #     A01[vertice] = -2
        #     tmpK01 = np.dot(A01, A01.T)
        #     A02 = np.zeros((vertices_num, 1), np.int8)
        #     A02[sorted_vertice_neighbors[0]] = A02[sorted_vertice_neighbors[2]] = 1
        #     A02[vertice] = -2
        #     tmpK02 = np.dot(A02, A02.T)
        #     A12 = np.zeros((vertices_num, 1), np.int8)
        #     A12[sorted_vertice_neighbors[1]] = A12[sorted_vertice_neighbors[2]] = 1
        #     A12[vertice] = -2
        #     tmpK12 = np.dot(A12, A12.T)
        #     K = K + tmpK01 + tmpK02 + tmpK12
    print(v6)
    return K

def compute_hexagon_loss(vertices, K):
    # K3 = np.block([
    #     [K, np.zeros_like(K), np.zeros_like(K)],
    #     [np.zeros_like(K), K, np.zeros_like(K)],
    #     [np.zeros_like(K), np.zeros_like(K), K]
    # ])
    # K = torch.from_numpy(K.astype(np.int8)).cuda()
    
    hexagon_loss = torch.mm(vertices[:,:,0].flatten().reshape(1,-1), torch.mm(K, vertices[:,:,0].flatten().reshape(-1,1))).squeeze() + torch.mm(vertices[:,:,1].flatten().reshape(1,-1), torch.mm(K, vertices[:,:,1].flatten().reshape(-1,1))).squeeze() + torch.mm(vertices[:,:,2].flatten().reshape(1,-1), torch.mm(K, vertices[:,:,2].flatten().reshape(-1,1))).squeeze()
    # hexagon_loss = torch.mm(torch.mm(vertices[:,:,0].flatten().reshape(1,-1), K), vertices[:,:,0].flatten().reshape(-1,1)).squeeze() + torch.mm(torch.mm(vertices[:,:,1].flatten().reshape(1,-1), K), vertices[:,:,1].flatten().reshape(-1,1)).squeeze() + torch.mm(torch.mm(vertices[:,:,2].flatten().reshape(1,-1), K), vertices[:,:,2].flatten().reshape(-1,1)).squeeze()
    
    # hexagon_loss = np.dot(np.dot(mesh.vertices.flatten('F').reshape(1,-1), K3), mesh.vertices.flatten('F').reshape(-1,1)).squeeze()

    return hexagon_loss

def neg_iou_loss(predict, target):
    dims = tuple(range(predict.ndimension())[1:])
    intersect = (predict * target).sum(dims)
    union = (predict + target - predict * target).sum(dims) + 1e-6
    return 1. - (intersect / union).sum() / intersect.nelement()

def remesh_test(verts, faces, mvps, masks, N, res=SIZE):

    num = len(masks)

    imgH, imgW = res
    # for DTU
    voxel_origin = [-1.1, -1.1, -1.1]
    x_size = 2.2 / (N - 1)
    y_size = 2.2 / (N - 1)
    z_size = 2.2 / (N - 1)
    # # for NHR
    # voxel_origin = [-0.3, -1.6, -4.5]
    # x_size = 0.6 / (N - 1)
    # y_size = 1.7 / (N - 1)
    # z_size = 0.8 / (N - 1)
    # # for 101
    # voxel_origin = [-1.1, -1.1, -1.1]
    # x_size = 4 / (N - 1)
    # y_size = 4/ (N - 1)
    # z_size = 4 / (N - 1)

    overall_index = np.arange(0, N ** 3, 1, dtype=np.int64)
    pts = np.zeros([N ** 3, 3], dtype=np.float32)

    pts[:, 2] = overall_index % N
    pts[:, 1] = (overall_index // N) % N
    pts[:, 0] = ((overall_index // N) // N) % N

    pts[:, 0] = (pts[:, 0] * x_size) + voxel_origin[0]
    pts[:, 1] = (pts[:, 1] * y_size) + voxel_origin[1]
    pts[:, 2] = (pts[:, 2] * z_size) + voxel_origin[2]

    pts = np.vstack((pts.T, np.ones((1, N**3))))

    # sum the silhouette or minus the black
    use_sum = False
    if use_sum:
        filled = []
    else:
        filled = np.ones(pts.shape[1], dtype=bool)

    for mvp, mask in (zip(mvps, masks)):
        uvs = mvp @ pts
        uvs[0] = (uvs[0] / uvs[3] + 1) * (imgW//2)
        uvs[1] = (uvs[1] / uvs[3] + 1) * (imgH//2)
        x_good = np.logical_and(uvs[0] >= 0, uvs[0] < imgW)
        y_good = np.logical_and(uvs[1] >= 0, uvs[1] < imgH)
        good = np.logical_and(x_good, y_good)
        indices = np.where(good)[0]
        fill = np.zeros(uvs.shape[1])
        sub_uvs = (uvs[:2, indices]).astype(np.int32)
        res = mask[sub_uvs[1, :], sub_uvs[0, :]]
        fill[indices] = res 
        if use_sum:
            filled.append(fill)
        else:
            filled = filled & fill.astype(bool)

    if use_sum:
        filled = np.vstack(filled)
        occupancy = -np.sum(filled, axis=0)
        level = -(num-8)
        level -= 0.5
    else:
        occupancy = -filled.astype(np.float32)
        level = -0.5

    occupancy = occupancy.reshape(N,N,N)

    verts, faces, normals, values = skimage.measure.marching_cubes(volume=occupancy, level=level, spacing=[x_size, y_size, z_size], allow_degenerate=True, method="lewiner")
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_origin[2] + verts[:, 2]
    
    return mesh_points, faces


def remesh(verts, faces, mvps, masks, N=256, res=SIZE):
    if not isinstance(verts, np.ndarray):
        verts = verts.detach().cpu().numpy()
    if not isinstance(faces, np.ndarray):
        faces = faces.detach().cpu().numpy()
    mesh = trimesh.Trimesh(verts, faces)
    mesh.export('./test/tmp.obj')
    os.system('xvfb-run -a binvox -d %d /workspace/nvrender/test/tmp.obj'%N)
    f = open('./test/tmp.binvox','rb')
    ret = trimesh.exchange.binvox.load_binvox(f)
    f.close()

    matrix = ~ret.matrix

    f = open('./test/tmp.binvox', 'rb')
    f.readline() # binvox
    f.readline() # dim
    trans = f.readline().decode().strip()
    scale = f.readline().decode().strip()
    f.close()

    trans = trans.split(' ')[1:]
    scale = scale.split(' ')[1:]
    print("scale: ", trans)
    trans = [float(item) for item in trans]
    scale = [float(item) for item in scale]

    assert len(trans) == 3 and len(scale) == 1
    voxel_origin = [float(item) for item in trans]
    scale = float(scale[0]) / (N-1)

    voxel_size = [scale, scale, scale]

    overall_index = np.arange(0, N ** 3, 1).astype(np.int)
    samples = np.zeros([N ** 3, 3])

    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index // N) % N
    samples[:, 0] = ((overall_index // N) // N) % N

    samples[:, 0] = (samples[:, 0] * scale) + voxel_origin[0]
    samples[:, 1] = (samples[:, 1] * scale) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * scale) + voxel_origin[2]

    samples = samples.reshape(N,N,N,3)

    for mvp, mask in zip(mvps, masks):
        idx = np.where(matrix==0)
        if not isinstance(mvp, np.ndarray):
            mvp = mvp.detach().cpu().numpy()
        if not isinstance(mask, np.ndarray):
            mask = mask.detach().cpu().numpy()
        sample = samples[idx[0], idx[1], idx[2]]
        verts = np.concatenate([sample, np.ones_like(sample[:,0:1])], axis=1)
        # verts = np.matmul(verts, rotation.transpose())
        proj_verts = np.matmul(verts, mvp.transpose(), dtype='float32')
        u = np.round((proj_verts[:,0] / proj_verts[:,3] + 1) * (SIZE[1]//2)).astype(np.int)
        v = np.round((proj_verts[:,1] / proj_verts[:,3] + 1) * (SIZE[0]//2)).astype(np.int)
        u = np.minimum(u, SIZE[1]-1)
        u = np.maximum(u, 0)
        # u = 799 - u
        v = np.minimum(v, SIZE[0]-1)
        v = np.maximum(v, 0)

        # imageio.imsave("./test/mask.png", mask)
        # print("U:")
        # print(u.max())
        # print(u.min())
        # print("V:")
        # print(v.max())
        # print(v.min())

        remove_idx = np.where(mask[v,u]==0)
        xx = idx[0][remove_idx]
        yy = idx[1][remove_idx]
        zz = idx[2][remove_idx]
        matrix[xx,yy,zz] = 1

    verts, faces, normals, values = skimage.measure.marching_cubes(volume=matrix, level=0.5, spacing=voxel_size, allow_degenerate=True, method="lorensen")
    # verts, faces, normals, values = skimage.measure.marching_cubes(volume=matrix, spacing=voxel_size, allow_degenerate=True, method="lorensen")

    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_origin[2] + verts[:, 2]

    os.system('rm ./test/tmp.obj')
    os.system('rm ./test/tmp.binvox')

    return mesh_points, faces

def mask(img):
    if img.shape[2] == 4:
        alpha_channel = img[:, :, 3] # 0 for black, 255 for white
        # img[:, :, 3] = np.where(img[:, :, 3]>0, 1., 0)
        img_rgbm = np.zeros(img.shape)
        img_rgbm = img
        img_rgbm[:, :, 3] = np.where(img_rgbm[:, :, 3]>0, 255., 0)
    else:
        alpha = np.where(((img[:,:,0]==0) & (img[:,:,1]==0) & (img[:,:,2]==0)), 0, 255)
        # alpha = cv2.blur(alpha, (7, 7))
        alpha = np.expand_dims(alpha, axis=2)
        print(alpha.shape)
        img_rgbm = np.concatenate((img, alpha), axis=2)
    return img_rgbm

def cal_position(position):
    x, y, z = position
    result = np.zeros(position.shape)
    distances = math.sqrt(x**2 + y**2 + z**2)
    # elevations = math.degrees(math.acos(z/distances))
    elevations = math.degrees(math.asin(z/distances))
    if x == 0:
        if y > 0:
            azimuths = math.degrees(math.pi/2)
        else:
            azimuths = math.degrees(math.pi/2) + 180
    elif x > 0 :
        azimuths = math.degrees(math.atan(y/x))
    else:
        azimuths = math.degrees(math.atan(y/x)) + 180
    result[0], result[1], result[2] = distances, elevations, azimuths

    return result

def load_own_blender(basedir):
    xyz = np.load(os.path.join(basedir, "position.npy"))
    RT_all = np.load(os.path.join(basedir, "RT.npy"))
    step_size = int(360/xy_num)
    imgs = []
    positions = []
    mvps = []
    mvp34s = []
    normals_path = os.path.join(basedir, "NORMAL")
    normals = []
    if camlist_flag:
        for step_z in range(z_num):
            for i in range(xy_num):
                img_num = i * 1
                img_name = "%d_%d_00.png"%(img_num, step_z)
                img = imageio.imread(os.path.join(basedir, "RENDER", img_name))
                img = mask(img)
                img = img / 255. 
                imgs.append(img)
                normal_name = "%d_%d_00_synthesized_image.jpg"%(img_num, step_z)
                normal_img = imageio.imread(os.path.join(normals_path, normal_name))
                normal_img = normal_img / 255.
                normals.append(normal_img)
                cam_position = xyz[i+step_z*xy_num]
                RT = RT_all[i+step_z*xy_num]
                mvp = np.matmul(proj, RT)
                mvp34 = RT
                # print(mvp, mvp34)
                mvp34s.append(mvp34)
                mvps.append(mvp)
                cam_position = cal_position(cam_position)
                positions.append(cam_position)
    else:
        for step_z in range(z_num):
            for i in range(xy_num):
                img_num = i * step_size
                img_name = "%d_%d_00.png"%(img_num, step_z)
                img = imageio.imread(os.path.join(basedir, "RENDER", img_name))
                img = mask(img)
                img = img / 255. 
                imgs.append(img)
                normal_name = "%d_%d_00_synthesized_image.jpg"%(img_num, step_z)
                normal_img = imageio.imread(os.path.join(normals_path, normal_name))
                normal_img = normal_img / 255.
                normals.append(normal_img)
                cam_position = xyz[i+step_z*xy_num]
                RT = RT_all[i+step_z*xy_num]
                mvp = np.matmul(proj, RT)
                mvp34 = RT
                # print(mvp, mvp34)
                mvp34s.append(mvp34)
                mvps.append(mvp)
                cam_position = cal_position(cam_position)
                positions.append(cam_position)
    imgs = (np.array(imgs)).astype(np.float32) # keep all 4 channels (RGBM)
    normals = (np.array(normals)).astype(np.float32)
    positions = np.array(positions).astype(np.float32)
    mvps = np.array(mvps).astype(np.float32)
    mvp34s = np.array(mvp34s).astype(np.float32)
    # print(positions)

    return imgs, positions, mvps, mvp34s, normals

def read_cam_file(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    # intrinsics = np.identity(4, dtype=np.float32)
    # intrinsics[:3, :3] = tmp

    # depth_min & depth_interval: line 11
    depth_min = float(lines[11].split()[0])
    depth_interval = float(lines[11].split()[1])
    return intrinsics, extrinsics, depth_min, depth_interval

def load_from_colmap(basedir):
    
    imgs = []
    all_intrinsics = []
    mvps = []
    mvp34s = []
    masks = []
    normals = []
    confs = []
    depths = []
    specs = []
    
    imgs_test = []
    all_intrinsics_test = []
    mvps_test = []
    mvp34s_test = []
    masks_test = []
    normals_test = []
    confs_test = []
    depths_test = []
    specs_test = []
    
    
    img_name_list = os.listdir(os.path.join(basedir, "RENDER"))
    img_name_list.sort()
    normals_path = os.path.join(basedir, "NORMAL")
    cam_path = os.path.join(basedir, "CAM")
    mask_path = os.path.join(basedir, "MASK")
    conf_path = os.path.join(basedir, "CONF")
    depth_path = os.path.join(basedir, "DEPTH")
    spec_path = os.path.join(basedir, "SPEC")

    for img_name,idx in zip(img_name_list[:49], range(49)):
        img = imageio.imread(os.path.join(basedir, "RENDER", img_name))
        img = img / 255. 
        h, w = img.shape[:2]

        normal_name = img_name[:-4] + "_normal.png"
        normal_img = imageio.imread(os.path.join(normals_path, normal_name))
        normal_img = normal_img / 255.
        # normal_name = img_name[:-4] + "_normal.npy"
        # normal_img = np.load(os.path.join(normals_path, normal_name))
        # normals.append(normal_img)
        
        depth_name = img_name[:-4] + ".pfm"
        depth = np.array(read_pfm(os.path.join(depth_path, depth_name)), dtype=np.float32)
        
        spec_name = img_name
        spec = imageio.imread(os.path.join(spec_path, spec_name))
        spec = spec / 255.

        conf_name = img_name[:-4] + "_conf.png"
        conf_img = imageio.imread(os.path.join(conf_path, conf_name))

        mask = imageio.imread(os.path.join(mask_path, img_name))
        if len(mask.shape) == 3:
            mask = mask[:,:,0]
        mask = np.where(mask>0, 1, 0)
        
        cam_name =  img_name[:-4] + "_cam.txt"
        intrinsics, extrinsics, _, _ = read_cam_file(os.path.join(cam_path, cam_name))
        RT = extrinsics
        proj = np.eye(4, dtype='float32')
        proj[:3,:3] = intrinsics
        proj[0,0] = proj[0,0] / (0.5*float(w))
        proj[0,1] = 0
        proj[1,1] = proj[1,1] / (0.5*float(h))
        proj[0,2] = proj[0,2] / (0.5*float(w)) - 1.
        proj[1,2] = proj[1,2] / (0.5*float(h)) - 1.
        proj[2,2] = 0.
        proj[2,3] = -0.2
        proj[3,2] = 1.0
        proj[3,3] = 0.0

        mvp = np.matmul(proj, RT)
        mvp34 = RT
        # print(mvp, mvp34)
        if not all_for_train:
            if idx in select_number:
                imgs_test.append(img)
                normals_test.append(normal_img)
                depths_test.append(depth)
                specs_test.append(spec)
                confs_test.append(conf_img)
                masks_test.append(mask)
                mvp34s_test.append(mvp34)
                mvps_test.append(mvp)
                all_intrinsics_test.append(intrinsics)
            else:
                imgs.append(img)
                normals.append(normal_img)
                depths.append(depth)
                specs.append(spec)
                confs.append(conf_img)
                masks.append(mask)
                mvp34s.append(mvp34)
                mvps.append(mvp)
                all_intrinsics.append(intrinsics)
        else:
            if idx in select_number:
                imgs_test.append(img)
                normals_test.append(normal_img)
                depths_test.append(depth)
                specs_test.append(spec)
                confs_test.append(conf_img)
                masks_test.append(mask)
                mvp34s_test.append(mvp34)
                mvps_test.append(mvp)
                all_intrinsics_test.append(intrinsics)
                
                imgs.append(img)
                normals.append(normal_img)
                depths.append(depth)
                specs.append(spec)
                confs.append(conf_img)
                masks.append(mask)
                mvp34s.append(mvp34)
                mvps.append(mvp)
                all_intrinsics.append(intrinsics)
            else:
                imgs.append(img)
                normals.append(normal_img)
                depths.append(depth)
                specs.append(spec)
                confs.append(conf_img)
                masks.append(mask)
                mvp34s.append(mvp34)
                mvps.append(mvp)
                all_intrinsics.append(intrinsics)

    imgs = (np.array(imgs)).astype(np.float32) # keep all 4 channels (RGBM)
    masks = (np.array(masks)).astype(np.float32)
    normals = (np.array(normals)).astype(np.float32)
    confs = (np.array(confs)).astype(np.float32)
    depths = (np.array(depths)).astype(np.float32)
    specs = (np.array(specs)).astype(np.float32)
    all_intrinsics = (np.array(all_intrinsics)).astype(np.float32)
    mvps = np.array(mvps).astype(np.float32)
    mvp34s = np.array(mvp34s).astype(np.float32)
    
    imgs_test = (np.array(imgs_test)).astype(np.float32) # keep all 4 channels (RGBM)
    masks_test = (np.array(masks_test)).astype(np.float32)
    normals_test = (np.array(normals_test)).astype(np.float32)
    confs_test = (np.array(confs_test)).astype(np.float32)
    depths_test = (np.array(depths_test)).astype(np.float32)
    specs_test = (np.array(specs_test)).astype(np.float32)
    all_intrinsics_test = (np.array(all_intrinsics_test)).astype(np.float32)
    mvps_test = np.array(mvps_test).astype(np.float32)
    mvp34s_test = np.array(mvp34s_test).astype(np.float32)

    # return imgs, all_intrinsics, mvps, mvp34s, normals, masks, confs
    return imgs, all_intrinsics, mvps, mvp34s, normals, masks, confs, depths, specs, imgs_test, all_intrinsics_test, mvps_test, mvp34s_test, normals_test, masks_test, confs_test, depths_test, specs_test

def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    # TODO: assume the feature is 1/4 of the original image size
    # intrinsics[:2, :] /= 4
    return intrinsics, extrinsics

def campose_to_extrinsic(camposes):
    if camposes.shape[1]!=12:
        raise Exception(" wrong campose data structure!")
        return
    
    res = np.zeros((camposes.shape[0],4,4))
    
    res[:,0:3,2] = camposes[:,0:3]
    res[:,0:3,0] = camposes[:,3:6]
    res[:,0:3,1] = camposes[:,6:9]
    res[:,0:3,3] = camposes[:,9:12]
    res[:,3,3] = 1.0
    
    return res

def read_intrinsics(fn_instrinsic):
    fo = open(fn_instrinsic)
    data= fo.readlines()
    i = 0
    Ks = []
    while i<len(data):
        if len(data[i])>5:
            tmp = data[i].split()
            tmp = [float(i) for i in tmp]
            a = np.array(tmp)
            i = i+1
            tmp = data[i].split()
            tmp = [float(i) for i in tmp]
            b = np.array(tmp)
            i = i+1
            tmp = data[i].split()
            tmp = [float(i) for i in tmp]
            c = np.array(tmp)
            res = np.vstack([a,b,c])
            Ks.append(res)

        i = i+1
    Ks = np.stack(Ks)
    fo.close()

    return Ks

def load_from_nhr(basedir, scan):
    imgs = []
    positions = []
    mvps = []
    mvp34s = []
    masks = []
    normals = []
    
    extrinsics_path = os.path.join(basedir, 'CamPose.inf')
    intrinsics_path= os.path.join(basedir, 'Intrinsic.inf')
    all_extrinsics = campose_to_extrinsic(np.loadtxt(extrinsics_path))
    all_intrinsics = read_intrinsics(intrinsics_path)

    img_path = os.path.join(basedir, 'img', scan)
    normal_path = os.path.join(basedir, 'normal', scan)
    mask_path = os.path.join(basedir, 'img', scan, 'mask')
    for i in range(56):
        img = imageio.imread(os.path.join(img_path, "img_%04d.jpg"%i))
        img /= 255.
        imgs.append(img)    
        h, w = img.shape[:2]
        
        normal_img = imageio.imread(os.path.join(normal_path, "%08d.jpg"%i))
        normal_img /= 255.
        normals.append(normal_img)

        mask = imageio.imread(os.path.join(mask_path, "img_%04d.jpg"%i))
        mask[mask>0] = 1
        masks.append(mask)

        RT = all_extrinsics[i]
        intrinsics = all_intrinsics[i]
        proj = np.eye(4, dtype='float32')
        proj[:3,:3] = intrinsics
        proj[0,0] = proj[0,0] / (0.5*float(w))
        proj[1,1] = proj[1,1] / (0.5*float(h))
        proj[0,2] = proj[0,2] / (0.5*float(w)) - 1.
        proj[1,2] = proj[1,2] / (0.5*float(h)) - 1.
        proj[2,2] = 0.
        proj[2,3] = -1.
        proj[3,2] = 1.0
        proj[3,3] = 0.0
        
        mvp = np.matmul(proj, RT)
        mvp34 = RT
        mvp34s.append(mvp34)
        mvps.append(mvp)
        
    imgs = (np.array(imgs)).astype(np.float32)
    masks = (np.array(masks)).astype(np.float32)
    normals = (np.array(normals)).astype(np.float32)
    positions = 0
    mvps = np.array(mvps).astype(np.float32)
    mvp34s = np.array(mvp34s).astype(np.float32)

    return imgs, positions, mvps, mvp34s, normals, masks

def load_from_tanks(basedir):
    masks = []
    imgs = []
    positions = []
    mvps = []
    mvp34s = []
    normals_path = os.path.join(basedir, "NORMAL")
    normals = []
    depths = []
    cameras_path = os.path.join(basedir, "CAM")

    img_name_list = os.listdir(os.path.join(basedir, "RENDER"))
    for img_name in img_name_list[:total_num]:
        img = imageio.imread(os.path.join(basedir, "RENDER", img_name))
        # img = mask(img)
        img = img / 255. 
        imgs.append(img)

        # mask0 = imageio.imread(os.path.join(basedir, "MASK", "000" + img_name[:-4] + "_geo.png"))
        # mask1 = imageio.imread(os.path.join(basedir, "MASK", "000" + img_name[:-4] + "_photo.png"))
        # mask2 = imageio.imread(os.path.join(basedir, "MASK", "000" + img_name[:-4] + "_final.png"))
        # mask = np.logical_or(mask0, mask1)
        # mask = np.logical_or(mask, mask2)
        # mask = np.where(mask==True, 1, 0)

        # save_mask = np.zeros((mask.shape[0], mask.shape[1], 3))
        # save_mask[:,:,0] = save_mask[:,:,1] = save_mask[:,:,2] = mask
        # save = cv2.addWeighted(save_mask, 0.5, img, 0.5, 0)
        # imageio.imwrite(os.path.join(basedir, "000" + img_name[:-4] + "_mask.png"), save)


        mask = imageio.imread(os.path.join(basedir, "MASK", "000" + img_name[:-4] + "_mask.png"))
        mask = np.where(mask==0, 0, 1)
        mask = mask[:,:,0]
        masks.append(mask)

        normal_name = img_name[:-4] + "_synthesized_image.png"
        normal_img = imageio.imread(os.path.join(normals_path, normal_name))
        # if normal_img.shape[:2] != SIZE:
        #     normal_img = cv2.resize(normal_img, (SIZE[1], SIZE[0]), interpolation=cv2.INTER_CUBIC)
        normal_img = normal_img / 255.
        normals.append(normal_img)

        camera_name = "000" + img_name[:-4] + "_cam.txt"
        K, RT = read_camera_parameters(os.path.join(cameras_path, camera_name))
        proj = np.eye(4, dtype='float32')
        proj[:3,:3] = K
        proj[0,0] = proj[0,0] / (0.5*float(SIZE[1]))
        proj[1,1] = proj[1,1] / (0.5*float(SIZE[0]))
        proj[0,2] = proj[0,2] / (0.5*float(SIZE[1])) - 1.
        proj[1,2] = proj[1,2] / (0.5*float(SIZE[0])) - 1.
        proj[2,2] = 0.
        proj[2,3] = -0.2
        proj[3,2] = 1.0
        proj[3,3] = 0.0

        depth = read_pfm(os.path.join(basedir, "DEPTH", "000"+img_name[:-4]+".pfm"))
        depths.append(depth)

        mvp = np.matmul(proj, RT)
        # print(mvp, mvp34)
        mvp34s.append(RT)
        mvps.append(mvp)

    imgs = (np.array(imgs)).astype(np.float32) # keep all 4 channels (RGBM)
    masks = (np.array(masks)).astype(np.float32)
    depths = (np.array(depths)).astype(np.float32)
    normals = (np.array(normals)).astype(np.float32)
    positions = 0
    mvps = np.array(mvps).astype(np.float32)
    mvp34s = np.array(mvp34s).astype(np.float32)
    # print(positions)

    return imgs, positions, mvps, mvp34s, normals, depths, masks

def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data


def sph2cart(pts_sph, convention='lat-lng'):
    """Inverse of :func:`cart2sph`.

    See :func:`cart2sph`.
    """
    pts_sph = np.array(pts_sph)

    # Validate inputs
    is_one_point = False
    if pts_sph.shape == (3,):
        is_one_point = True
        pts_sph = pts_sph.reshape(1, 3)
    elif pts_sph.ndim != 2 or pts_sph.shape[1] != 3:
        raise ValueError("Shape of input must be either (3,) or (n, 3)")

    # Convert to latitude-longitude convention, if necessary
    if convention == 'lat-lng':
        pts_r_lat_lng = pts_sph
    else:
        raise NotImplementedError(convention)

    # Compute x, y and z
    r = pts_r_lat_lng[:, 0]
    lat = pts_r_lat_lng[:, 1]
    lng = pts_r_lat_lng[:, 2]
    z = r * np.sin(lat)
    x = r * np.cos(lat) * np.cos(lng)
    y = r * np.cos(lat) * np.sin(lng)

    # Assemble and return
    pts_cart = np.stack((x, y, z), axis=-1)

    if is_one_point:
        pts_cart = pts_cart.reshape(3)

    return pts_cart

def gen_light_xyz(envmap_h, envmap_w, envmap_radius=1e2):
    """Additionally returns the associated solid angles, for integration.
    """
    # OpenEXR "latlong" format
    # lat = pi/2
    # lng = pi
    #     +--------------------+
    #     |                    |
    #     |                    |
    #     +--------------------+
    #                      lat = -pi/2
    #                      lng = -pi
    lat_step_size = np.pi / (envmap_h + 2)
    lng_step_size = 2 * np.pi / (envmap_w + 2)
    # Try to exclude the problematic polar points
    lats = np.linspace(
        np.pi / 2 - lat_step_size, -np.pi / 2 + lat_step_size, envmap_h)
    lngs = np.linspace(
        np.pi - lng_step_size, -np.pi + lng_step_size, envmap_w)
    lngs, lats = np.meshgrid(lngs, lats)

    # To Cartesian
    rlatlngs = np.dstack((envmap_radius * np.ones_like(lats), lats, lngs))
    rlatlngs = rlatlngs.reshape(-1, 3)
    xyz = sph2cart(rlatlngs)
    xyz = xyz.reshape(envmap_h, envmap_w, 3)

    # Calculate the area of each pixel on the unit sphere (useful for
    # integration over the sphere)
    sin_colat = np.sin(np.pi / 2 - lats)
    areas = 4 * np.pi * sin_colat / np.sum(sin_colat)

    assert 0 not in areas, \
        "There shouldn't be light pixel that doesn't contribute"

    return xyz, areas

def linear2srgb(tensor_0to1):
    if isinstance(tensor_0to1, torch.Tensor):
        pow_func = torch.pow
        where_func = torch.where
    else:
        pow_func = np.power
        where_func = np.where

    srgb_linear_thres = 0.0031308
    srgb_linear_coeff = 12.92
    srgb_exponential_coeff = 1.055
    srgb_exponent = 2.4

    tensor_linear = tensor_0to1 * srgb_linear_coeff
    tensor_nonlinear = srgb_exponential_coeff * (pow_func(tensor_0to1+1e-5, 1 / srgb_exponent)) - (srgb_exponential_coeff - 1)

    is_linear = tensor_0to1 <= srgb_linear_thres
    tensor_srgb = where_func(is_linear, tensor_linear, tensor_nonlinear)

    return tensor_srgb

class MyFlatten(nn.Module):
    def __init__(self, faces, average=False):
        super(MyFlatten, self).__init__()

        self.nf = faces.shape[0]
        self.average = average
        if torch.is_tensor(faces):
            faces = faces.detach().cpu().numpy()
        vertices = list(set([tuple(v) for v in np.sort(np.concatenate((faces[:, 0:2], faces[:, 1:3]), axis=0))]))

        tmp = dict()
        for face in faces:
            f1 = np.sort(face[:2])
            f2 = np.sort(face[1:])
            f3 = np.sort(face[::2])
            k1 = int(f1[0] * self.nf + f1[1])
            k2 = int(f2[0] * self.nf + f2[1])
            k3 = int(f3[0] * self.nf + f3[1])

            if k1 not in tmp.keys():
                tmp[k1] = []
            tmp[k1].append(face[2])

            if k2 not in tmp.keys():
                tmp[k2] = []
            tmp[k2].append(face[0])

            if k3 not in tmp.keys():
                tmp[k3] = []
            tmp[k3].append(face[1])

        v0s = np.array([v[0] for v in vertices], 'int32')
        v1s = np.array([v[1] for v in vertices], 'int32')
        v2s = []
        v3s = []
        for v0, v1 in zip(v0s, v1s):
            k = int(v0 * self.nf + v1)
            v2s.append(tmp[k][0])
            v3s.append(tmp[k][1])

        v2s = np.array(v2s, 'int32')
        v3s = np.array(v3s, 'int32')

        self.register_buffer('v0s', torch.from_numpy(v0s).long())
        self.register_buffer('v1s', torch.from_numpy(v1s).long())
        self.register_buffer('v2s', torch.from_numpy(v2s).long())
        self.register_buffer('v3s', torch.from_numpy(v3s).long())

    def forward(self, vertices, eps=1e-6):
        # make v0s, v1s, v2s, v3s
        batch_size = vertices.size(0)

        v0s = vertices[:, self.v0s, :]
        v1s = vertices[:, self.v1s, :]
        v2s = vertices[:, self.v2s, :]
        v3s = vertices[:, self.v3s, :]

        a1 = v1s - v0s
        b1 = v2s - v0s
        a1l2 = a1.pow(2).sum(-1)
        b1l2 = b1.pow(2).sum(-1)
        a1l1 = (a1l2 + eps).sqrt()
        b1l1 = (b1l2 + eps).sqrt()
        ab1 = (a1 * b1).sum(-1)
        cos1 = ab1 / (a1l1 * b1l1 + eps)
        sin1 = (1 - cos1.pow(2) + eps).sqrt()
        c1 = a1 * (ab1 / (a1l2 + eps))[:, :, None]
        cb1 = b1 - c1
        cb1l1 = b1l1 * sin1

        a2 = v1s - v0s
        b2 = v3s - v0s
        a2l2 = a2.pow(2).sum(-1)
        b2l2 = b2.pow(2).sum(-1)
        a2l1 = (a2l2 + eps).sqrt()
        b2l1 = (b2l2 + eps).sqrt()
        ab2 = (a2 * b2).sum(-1)
        cos2 = ab2 / (a2l1 * b2l1 + eps)
        sin2 = (1 - cos2.pow(2) + eps).sqrt()
        c2 = a2 * (ab2 / (a2l2 + eps))[:, :, None]
        cb2 = b2 - c2
        cb2l1 = b2l1 * sin2

        cos = (cb1 * cb2).sum(-1) / (cb1l1 * cb2l1 + eps)

        dims = tuple(range(cos.ndimension())[1:])
        loss = (cos + 1).pow(2).sum(dims)
        if self.average:
            return loss.sum() / batch_size
        else:
            return loss

class NormalConsistency(nn.Module):
    def __init__(self, faces, average=False):
        super(NormalConsistency, self).__init__()

        self.nf = faces.shape[0]
        self.average = average
        if torch.is_tensor(faces):
            faces = faces.detach().cpu().numpy()
        vertices = list(set([tuple(v) for v in np.sort(np.concatenate((faces[:, 0:2], faces[:, 1:3]), axis=0))]))

        tmp = dict()
        for face in faces:
            f1 = np.sort(face[:2])
            f2 = np.sort(face[1:])
            f3 = np.sort(face[::2])
            k1 = int(f1[0] * self.nf + f1[1])
            k2 = int(f2[0] * self.nf + f2[1])
            k3 = int(f3[0] * self.nf + f3[1])

            if k1 not in tmp.keys():
                tmp[k1] = []
            tmp[k1].append(face[2])

            if k2 not in tmp.keys():
                tmp[k2] = []
            tmp[k2].append(face[0])

            if k3 not in tmp.keys():
                tmp[k3] = []
            tmp[k3].append(face[1])

        v0s = np.array([v[0] for v in vertices], 'int32')
        v1s = np.array([v[1] for v in vertices], 'int32')
        v2s = []
        v3s = []
        for v0, v1 in zip(v0s, v1s):
            k = int(v0 * self.nf + v1)
            v2s.append(tmp[k][0])
            v3s.append(tmp[k][1])

        v2s = np.array(v2s, 'int32')
        v3s = np.array(v3s, 'int32')

        self.register_buffer('v0s', torch.from_numpy(v0s).long())
        self.register_buffer('v1s', torch.from_numpy(v1s).long())
        self.register_buffer('v2s', torch.from_numpy(v2s).long())
        self.register_buffer('v3s', torch.from_numpy(v3s).long())

    def forward(self, vertices):
        batch_size = vertices.size(0)

        v0s = vertices[:, self.v0s, :]
        v1s = vertices[:, self.v1s, :]
        v2s = vertices[:, self.v2s, :]
        v3s = vertices[:, self.v3s, :]

        a1 = v1s - v0s
        b1 = v2s - v0s
        b2 = v3s - v0s

        n1 = torch.cross(b1, a1)
        n2 = torch.cross(a1, b2)

        loss = 1. - F.cosine_similarity(n1, n2)

        return loss.sum()

def loadK(path, vertices_num):
    ls = []
    with open(path, "r") as f:
        data = f.readline().strip('\n').strip(' ')
        ls.append(data.split(' '))
        K = np.array(ls, dtype=np.int8).reshape(vertices_num, -1)
        print(K.shape)
    f.close()
    
    return K

class descriptor(object):
    def __init__(self):
        self.eps = 0.00001

    def creatDes(self, features, imarr):
        """
        return pixels and their descriptor together as a dictionary
        the coordinates of pixels are the keys, the descriptors are 
        the value
        """ 
        desDict = {}
        arr = imarr.astype(np.float64)
        desNum = len(features)

        for i in range(desNum):
            desDict[(features[i][0],features[i][1])] = self.allocate(features[i][0],features[i][1],arr)

        return desDict


    def direction(self,i,j,imarr):
        """
        computes each pixel's gradient magnitude and orientation
        """
        mij = math.sqrt((imarr[i+1,j]-imarr[i-1,j])**2 + (imarr[i,j+1]-imarr[i,j-1])**2)
        theta = math.atan((imarr[i,j+1]-imarr[i,j-1]) / (imarr[i+1,j]-imarr[i-1,j]+self.eps))

        return mij,theta


    def allocate(self,i,j,imarr):
        """
        computes the 16 local area's gradient magnitude and 
        orientation around the current pixel,
        each local area contains 8 pixels
        """
        # vec = [0]*16
        vec = torch.zeors(16)
        vec[0] = self.localdir(i-8,j-8,imarr)
        vec[1] = self.localdir(i-8,j,imarr)
        vec[2] = self.localdir(i-8,j+8,imarr)
        vec[3] = self.localdir(i-8,j+16,imarr)

        vec[4] = self.localdir(i,j-8,imarr)
        vec[5] = self.localdir(i,j,imarr)
        vec[6] = self.localdir(i,j+8,imarr)
        vec[7] = self.localdir(i,j+16,imarr)

        vec[8] = self.localdir(i+8,j-8,imarr)
        vec[9] = self.localdir(i+8,j,imarr)
        vec[10] = self.localdir(i+8,j+8,imarr)
        vec[11] = self.localdir(i+8,j+16,imarr)

        vec[12] = self.localdir(i+16,j-8,imarr)
        vec[13] = self.localdir(i+16,j,imarr)
        vec[14] = self.localdir(i+16,j+8,imarr)
        vec[15] = self.localdir(i+16,j+16,imarr)

        return [val for subl in vec for val in subl]

    def localdir(self,i,j,imarr):
        """
        return singal pixel's direction histogram
        the histogram has 18 region
        """
        P = math.pi
        # localDir = [0]*18
        localDir = torch.zeros(18)

        for b in range(i-8,i):
            for c in range(j-8,j):
                m,t = self.direction(b,c,imarr)
                if t>=P*-9/18 and t<=P*-8/18:
                    localDir[0]+=m
                if t>P*-8/18 and t<=P*-7/18:
                    localDir[1]+=m
                if t>P*-7/18 and t<=P*-6/18: 
                    localDir[2]+=m
                if t>P*-6/18 and t<=P*-5/18:
                    localDir[3]+=m	
                if t>P*-5/18 and t<=P*-4/18:
                    localDir[4]+=m
                if t>P*-4/18 and t<=P*-3/18:
                    localDir[5]+=m
                if t>P*-3/18 and t<=P*-2/18:
                    localDir[6]+=m	
                if t>P*-2/18and t<=P*-1/18:
                    localDir[7]+=m
                if t>P*-1/18 and t<=0:
                    localDir[8]+=m
                if t>0 and t<=P*1/18: 
                    localDir[9]+=m
                if t>P*1/18 and t<=P*2/18:
                    localDir[10]+=m	
                if t>P*2/18 and t<=P*3/18:
                    localDir[11]+=m
                if t>P*3/18 and t<=P*4/18:
                    localDir[12]+=m
                if t>P*4/18 and t<=P*5/18:
                    localDir[13]+=m	
                if t>P*5/18 and t<=P*6/18:
                    localDir[14]+=m
                if t>P*6/18 and t<=P*7/18:
                    localDir[15]+=m
                if t>P*7/18 and t<=P*8/18:
                    localDir[16]+=m
                if t>P*8/18 and t<=P*9/18:
                    localDir[17]+=m

        return localDir

def get_ray_directions(H, W, fx, fy, cx, cy):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        H, W, focal: image height, width and focal length
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    #grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    #i, j = grid.unbind(-1)

    O = 0.5
    x_coords = torch.linspace(O, W - 1 + O, W)
    y_coords = torch.linspace(O, H - 1 + O, H)
    j, i = torch.meshgrid([y_coords, x_coords])
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    j = j.cuda()
    i = i.cuda()
    directions = \
        torch.stack([(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1) # (H, W, 3)

    return directions

def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate
    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:3, :3] # (H, W, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:3,3].expand(rays_d.shape) # (H, W, 3)

    # rays_d = rays_d.view(-1, 3)
    # rays_o = rays_o.view(-1, 3)

    return rays_d


def transform_3D_grid(grid_3d, transform=None, scale=None):
    if scale is not None:
        grid_3d = grid_3d * scale
    if transform is not None:
        R1 = transform[None, None, None, 0, :3]
        R2 = transform[None, None, None, 1, :3]
        R3 = transform[None, None, None, 2, :3]

        grid1 = (R1 * grid_3d).sum(-1, keepdim=True)
        grid2 = (R2 * grid_3d).sum(-1, keepdim=True)
        grid3 = (R3 * grid_3d).sum(-1, keepdim=True)

        grid_3d = torch.cat([grid1, grid2, grid3], dim=-1)

        trans = transform[None, None, None, :3, 3]
        grid_3d = grid_3d + trans

    return grid_3d

def scale_input(tensor, transform=None, scale=None):
    if transform is not None:
        t_shape = tensor.shape
        tensor = transform_3D_grid(
            tensor.view(-1, 3), transform=transform)
        tensor = tensor.view(t_shape)

    if scale is not None:
        tensor = tensor * scale

    return tensor

class PostionalEncoding(torch.nn.Module):
    def __init__(
        self,
        min_deg=0,
        max_deg=6,
        scale=0.1,
        transform=None,
        input_channle=3,
    ):
        super(PostionalEncoding, self).__init__()
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.n_freqs = max_deg - min_deg + 1
        self.scale = scale
        self.transform = transform

        self.input_channle = input_channle
        self.dirs = torch.tensor([
            0.8506508, 0, 0.5257311,
            0.809017, 0.5, 0.309017,
            0.5257311, 0.8506508, 0,
            1, 0, 0,
            0.809017, 0.5, -0.309017,
            0.8506508, 0, -0.5257311,
            0.309017, 0.809017, -0.5,
            0, 0.5257311, -0.8506508,
            0.5, 0.309017, -0.809017,
            0, 1, 0,
            -0.5257311, 0.8506508, 0,
            -0.309017, 0.809017, -0.5,
            0, 0.5257311, 0.8506508,
            -0.309017, 0.809017, 0.5,
            0.309017, 0.809017, 0.5,
            0.5, 0.309017, 0.809017,
            0.5, -0.309017, 0.809017,
            0, 0, 1,
            -0.5, 0.309017, 0.809017,
            -0.809017, 0.5, 0.309017,
            -0.809017, 0.5, -0.309017
        ]).reshape(-1, self.input_channle).T

        frequency_bands = 2.0 ** np.linspace(
            self.min_deg, self.max_deg, self.n_freqs)
        self.embedding_size = 2 * self.dirs.shape[1] * self.n_freqs + self.input_channle

        print(
            "Icosahedron embedding with periods:",
            (2 * np.pi) / (frequency_bands * self.scale),
            " -- embedding size:", self.embedding_size
        )

    def vis_embedding(self):
        x = torch.linspace(0, 5, 640)
        embd = x * self.scale
        if self.gauss_embed:
            frequency_bands = torch.norm(self.B_layer.weight, dim=1)
            frequency_bands = torch.sort(frequency_bands)[0]
        else:
            frequency_bands = 2.0 ** torch.linspace(
                self.min_deg, self.max_deg, self.n_freqs)

        embd = embd[..., None] * frequency_bands
        embd = torch.sin(embd)

        import matplotlib.pylab as plt
        plt.imshow(embd.T, cmap='hot', interpolation='nearest',
                   aspect='auto', extent=[0, 5, 0, embd.shape[1]])
        plt.colorbar()
        plt.xlabel("x values")
        plt.ylabel("embedings")
        plt.show()

    def forward(self, tensor):
        frequency_bands = 2.0 ** torch.linspace(
            self.min_deg, self.max_deg, self.n_freqs,
            dtype=tensor.dtype, device=tensor.device)

        tensor = scale_input(
            tensor, transform=self.transform, scale=self.scale)

        proj = torch.matmul(tensor, self.dirs.to(tensor.device))
        xb = torch.reshape(
            proj[..., None] * frequency_bands,
            list(proj.shape[:-1]) + [-1]
        )
        embedding = torch.sin(torch.cat([xb, xb + 0.5 * np.pi], dim=-1))
        embedding = torch.cat([tensor] + [embedding], dim=-1)

        return embedding

from scipy.interpolate import CubicSpline
def gen_novel_view(extrinsics, intrinsic):
    indices_all = [11, 16, 34, 28, 11]
    pose = extrinsics[indices_all, :]
    t_in = np.array([0, 2, 3, 5, 6])
    n_inter = 5
    t_out = np.linspace(t_in[0], t_in[-1], n_inter * t_in[-1]).astype(np.float32)
    scales = np.array([4.2, 4.2, 3.8, 3.8, 4.2]).astype(np.float32)
    s_new = CubicSpline(t_in, scales, bc_type='periodic')
    s_new = s_new(t_out)
    q_new = CubicSpline(t_in, pose[:, :4].detach().cpu().numpy(), bc_type='periodic')
    q_new = q_new(t_out)
    q_new = torch.from_numpy(q_new).cuda().float()
    
    proj = torch.eye(4).cuda()
    proj[0,0] = intrinsic[0,0] / (0.5*float(SIZE[1]))
    proj[0,1] = 0
    proj[1,1] = intrinsic[1,1] / (0.5*float(SIZE[0]))
    proj[0,2] = intrinsic[0,2] / (0.5*float(SIZE[0])) - 1.
    proj[1,2] = intrinsic[1,2] / (0.5*float(SIZE[1])) - 1.
    proj[2,2] = 0.
    proj[2,3] = -1.
    proj[3,2] = 1.0
    proj[3,3] = 0.0
    novel_mvps = torch.matmul(proj, q_new)
    
    return novel_mvps, q_new


def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return

def init_uniform(module, init_method):
    if module.weight is not None:
        if init_method == "kaiming":
            nn.init.kaiming_uniform_(module.weight)
        elif init_method == "xavier":
            nn.init.xavier_uniform_(module.weight)
    return

class Conv2d(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", groups=1, **kwargs):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, groups=groups, bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        # self.bn = nn.GroupNorm(8, out_channels) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)

class Deconv2d(nn.Module):
    """Applies a 2D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv2d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        # self.bn = nn.GroupNorm(8, out_channels) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        y = self.conv(x)
        if self.stride == 2:
            h, w = list(x.size())[2:]
            y = y[:, :, :2 * h, :2 * w].contiguous()
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)

class DeConv2dFuse(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, relu=True, bn=True,
                 bn_momentum=0.1):
        super(DeConv2dFuse, self).__init__()

        self.deconv = Deconv2d(in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=1,
                               bn=True, relu=relu, bn_momentum=bn_momentum)

        self.conv = Conv2d(2 * out_channels, out_channels, kernel_size, stride=1, padding=1,
                           bn=bn, relu=relu, bn_momentum=bn_momentum)

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x_pre, x):
        x = self.deconv(x)
        x = torch.cat((x, x_pre), dim=1)
        x = self.conv(x)
        return x

class FeatureNet(nn.Module):
    def __init__(self, base_channels, num_stage=3, stride=4, mode="fpn"):
        super(FeatureNet, self).__init__()
        assert mode in ["unet", "fpn", "pvt", "rfp"], print("mode must be in 'unet', 'fpn', 'pvt', 'rfp', but get:{}".format(mode))
        self.mode = mode
        self.stride = stride
        self.base_channels = base_channels
        self.num_stage = num_stage

        if self.mode != "pvt" and self.mode != 'rfp':
            self.conv0 = nn.Sequential(
                Conv2d(3, base_channels, 3, 1, padding=1),
                Conv2d(base_channels, base_channels, 3, 1, padding=1),
            )

            self.conv1 = nn.Sequential(
                Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
                Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
                Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            )

            self.conv2 = nn.Sequential(
                Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
                Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
                Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            )

            if self.mode == 'unet':
                self.out1 = nn.Conv2d(base_channels * 4, base_channels * 4, 1, bias=False)
                self.out_channels = [4 * base_channels]
                if num_stage == 3:
                    self.deconv1 = DeConv2dFuse(base_channels * 4, base_channels * 2, 3)
                    self.deconv2 = DeConv2dFuse(base_channels * 2, base_channels, 3)

                    self.out2 = nn.Conv2d(base_channels * 2, base_channels * 2, 1, bias=False)
                    self.out3 = nn.Conv2d(base_channels, base_channels, 1, bias=False)
                    self.out_channels.append(2 * base_channels)
                    self.out_channels.append(base_channels)

                elif num_stage == 2:
                    self.deconv1 = DeConv2dFuse(base_channels * 4, base_channels * 2, 3)

                    self.out2 = nn.Conv2d(base_channels * 2, base_channels * 2, 1, bias=False)
                    self.out_channels.append(2 * base_channels)
            elif self.mode == "fpn":
                # self.out1 = nn.Conv2d(base_channels * 4, base_channels * 4, 1, bias=False)
                self.out_channels = [4 * base_channels]
                final_chs = base_channels * 4
                if num_stage == 3:
                    self.inner1 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)
                    self.inner2 = nn.Conv2d(base_channels * 1, final_chs, 1, bias=True)

                    # self.out2 = nn.Conv2d(final_chs, base_channels * 2, 3, padding=1, bias=False)
                    self.out3 = nn.Conv2d(final_chs, base_channels, 3, padding=1, bias=False)
                    self.out_channels.append(base_channels * 2)
                    self.out_channels.append(base_channels)

                elif num_stage == 2:
                    self.inner1 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)
                    self.out2 = nn.Conv2d(final_chs, base_channels, 3, padding=1, bias=False)
                    self.out_channels.append(base_channels)
        elif self.mode == 'rfp':
            self.rfp_conv0 = nn.Sequential(
                Conv2d(3, base_channels, 3, 1, padding=1),
                Conv2d(base_channels, base_channels, 3, 1, padding=1),
            )
            self.rfp_conv1 = nn.Sequential(
                Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
                Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
                Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            )
            self.rfp_conv2 = nn.Sequential(
                Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
                Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
                Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            )
            
            self.rfp_stage1_feat_conv = nn.Sequential(
                nn.Conv2d(4 * base_channels, 4 * base_channels, kernel_size=1, stride=1, bias=True),
                nn.ReLU(inplace=True),
            )
            self.rfp_stage2_feat_conv = nn.Sequential(
                nn.Conv2d(2 * base_channels, 2 * base_channels, kernel_size=1, stride=1, bias=True),
                nn.ReLU(inplace=True),
            )
            self.rfp_stage3_feat_conv = nn.Sequential(
                nn.Conv2d(base_channels, base_channels, kernel_size=1, stride=1, bias=True),
                nn.ReLU(inplace=True),
            )
            
            self.rfp_weight_stage1 = torch.nn.Conv2d(4 * base_channels, 1, kernel_size=1, stride=1, padding=0, bias=True)
            self.rfp_weight_stage2 = torch.nn.Conv2d(2 * base_channels, 1, kernel_size=1, stride=1, padding=0, bias=True)
            self.rfp_weight_stage3 = torch.nn.Conv2d(base_channels, 1, kernel_size=1, stride=1, padding=0, bias=True)

            self.conv0 = nn.Sequential(
                Conv2d(3, base_channels, 3, 1, padding=1),
                Conv2d(base_channels, base_channels, 3, 1, padding=1),
            )

            self.conv1 = nn.Sequential(
                Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
                Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
                Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            )

            self.conv2 = nn.Sequential(
                Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
                Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
                Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            )
            self.out1 = nn.Conv2d(base_channels * 4, base_channels * 4, 1, bias=False)
            self.out_channels = [4 * base_channels]
            self.aspp_stage1 = ASPP(4 * base_channels, base_channels)
            final_chs = base_channels * 4
            if num_stage == 3:
                self.inner1 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)
                self.inner2 = nn.Conv2d(base_channels * 1, final_chs, 1, bias=True)

                self.out2 = nn.Conv2d(final_chs, base_channels * 2, 3, padding=1, bias=False)
                self.out3 = nn.Conv2d(final_chs, base_channels, 3, padding=1, bias=False)
                self.out_channels.append(base_channels * 2)
                self.out_channels.append(base_channels)
                self.aspp_stage2 = ASPP(2 * base_channels, base_channels // 2)
                self.aspp_stage3 = ASPP(base_channels, base_channels // 4)

            elif num_stage == 2:
                self.inner1 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)
                self.out2 = nn.Conv2d(final_chs, base_channels, 3, padding=1, bias=False)
                self.out_channels.append(base_channels)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            import math
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def rfp_forward(self, x, rfp_feat):
        conv0 = self.rfp_conv0(x)
        conv1 = self.rfp_conv1(conv0)
        conv2 = self.rfp_conv2(conv1)

        conv = []
        rfp_conv0 = conv0 + self.rfp_stage3_feat_conv(rfp_feat[2])
        rfp_conv1 = conv1 + self.rfp_stage2_feat_conv(rfp_feat[1])
        rfp_conv2 = conv2 + self.rfp_stage1_feat_conv(rfp_feat[0])
        conv = [rfp_conv0, rfp_conv1, rfp_conv2]

        return conv
        
        
    def forward(self, x):
        if self.mode != "pvt" and self.mode != "rfp":
            conv0 = self.conv0(x)
            conv1 = self.conv1(conv0)
            conv2 = self.conv2(conv1)

            intra_feat = conv2
            outputs = {}
            if self.mode == "unet":
                # out = self.out1(intra_feat)
                # out = self.cam1(out)
                # outputs["stage1"] = out
                if self.num_stage == 3:
                    # intra_feat = self.deconv1(conv1, intra_feat)
                    # out = self.out2(intra_feat)
                    # outputs["stage2"] = out

                    intra_feat = self.deconv2(conv0, intra_feat)
                    out = self.out3(intra_feat)
                    outputs["stage3"] = out

                elif self.num_stage == 2:
                    intra_feat = self.deconv1(conv1, intra_feat)
                    out = self.out2(intra_feat)
                    # outputs["stage2"] = out

            elif self.mode == "fpn":
                # out = self.out1(intra_feat)
                # outputs["stage1"] = out
                if self.num_stage == 3:
                    # intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(conv1)
                    # out = self.out2(intra_feat)
                    # outputs["stage2"] = out

                    intra_feat = F.interpolate(intra_feat, scale_factor=4, mode="nearest") + self.inner2(conv0)
                    out = self.out3(intra_feat)
                    outputs["stage3"] = out

                elif self.num_stage == 2:
                    intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(conv1)
                    out = self.out2(intra_feat)
                    outputs["stage2"] = out
        elif self.mode == "rfp":
            # step1
            conv0 = self.conv0(x)
            conv1 = self.conv1(conv0)
            conv2 = self.conv2(conv1)

            intra_feat = conv2
            outputs = {}
            rfp_step1_out = []
            rfp_step2_out = []
            rfp_feat = []
            
            out = self.out1(intra_feat)
            rfp_feat.append(self.aspp_stage1(out))
            rfp_step1_out.append(out)
            if self.num_stage == 3:
                intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(conv1)
                out = self.out2(intra_feat)
                rfp_feat.append(self.aspp_stage2(out))
                rfp_step1_out.append(out)

                intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner2(conv0)
                out = self.out3(intra_feat)
                rfp_feat.append(self.aspp_stage3(out))
                rfp_step1_out.append(out)
            # step2
            rfp_conv = self.rfp_forward(x, rfp_feat)
            intra_feat = rfp_conv[2]
            out = self.out1(intra_feat)
            rfp_step2_out.append(out)
            if self.num_stage == 3:
                intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(rfp_conv[1])
                out = self.out2(intra_feat)
                rfp_step2_out.append(out)

                intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner2(rfp_conv[0])
                out = self.out3(intra_feat)
                rfp_step2_out.append(out)
                
            add_weight_stage1 = torch.sigmoid(self.rfp_weight_stage1(rfp_step2_out[0]))
            add_weight_stage2 = torch.sigmoid(self.rfp_weight_stage2(rfp_step2_out[1]))
            add_weight_stage3 = torch.sigmoid(self.rfp_weight_stage3(rfp_step2_out[2]))
            
            outputs["stage1"] = add_weight_stage1 * rfp_step2_out[0] + (1 - add_weight_stage1) * rfp_step1_out[0]
            outputs["stage2"] = add_weight_stage2 * rfp_step2_out[1] + (1 - add_weight_stage2) * rfp_step1_out[1]
            outputs["stage3"] = add_weight_stage3 * rfp_step2_out[2] + (1 - add_weight_stage3) * rfp_step1_out[2]
            
            # outputs["stage1"] = 0.5 * rfp_step2_out[0] + (1 - 0.5) * rfp_step1_out[0]
            # outputs["stage2"] = 0.5 * rfp_step2_out[1] + (1 - 0.5) * rfp_step1_out[1]
            # outputs["stage3"] = 0.5 * rfp_step2_out[2] + (1 - 0.5) * rfp_step1_out[2]
            

        return outputs


class PositionalEncoding_nerf(object):
    def __init__(self, L=10, dim=3):
        self.L = L
        self.embedding_size = dim * L * 2 + dim
    def __call__(self, p):
        pi = 1.0
        p_transformed = torch.cat([torch.cat(
            [torch.sin((2 ** i) * pi * p), 
             torch.cos((2 ** i) * pi * p)],
             dim=-1) for i in range(self.L)], dim=-1)
        return torch.cat([p, p_transformed], dim=-1)