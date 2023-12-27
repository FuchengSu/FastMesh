import numpy as np
import trimesh
from config import *
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num', type=int, default=0)
args = parser.parse_args()

proj_name = proj_name_list[args.num]
workpath = os.path.join('/workspace/nvrender/results/', proj_name)

tmp_path = os.path.join(workpath, 'plane.obj')
tmp_mesh = trimesh.load(tmp_path)
vertices, faces = tmp_mesh.vertices, tmp_mesh.faces
print("before: ", vertices.shape)
for nn in range(subdivide_order):
    vertices, faces = trimesh.remesh.subdivide(vertices, faces)
tmp_mesh = trimesh.base.Trimesh(vertices, faces)
tmp_path = os.path.join(workpath, 'plane2.obj')
print("after: ", vertices.shape)
tmp_mesh.export(tmp_path)
mesh_path = os.path.join(workpath, "plane2.obj")
mesh = trimesh.load(mesh_path)
edges_path = os.path.join(workpath, "edges.txt")
vertices_path = os.path.join(workpath, "vertices.txt")
# save sorted edges
edges = mesh.edges
edges = np.array(list(map(tuple, edges)), dtype=[('e1', int), ('e2', int)])
edges.sort(order=['e1'])
np.savetxt(edges_path, edges, fmt="%i")
np.savetxt(vertices_path, mesh.vertices.shape, fmt='%i')