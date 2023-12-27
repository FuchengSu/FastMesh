from math import log10, sqrt
import cv2
# import cv2
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
import numpy as np
from config import proj_name_list
import os
import py2cpp
import torch
import argparse
import math

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num', type=int, default=0)
    args = parser.parse_args()
    proj_name = proj_name_list[args.num]
    output = os.path.join('/workspace/nvrender/results/', proj_name)
    
    psnr_sum = 0
    ssim_sum = 0
    img_num = 49
    for i in range(img_num):
        # ori_path = os.path.join(output, "%06d.png"%i)
        # com_path = os.path.join(output, "eval_%03d.png"%i)
        # mask_path = os.path.join(output, "%03d.png"%i)
        
        ori_path = os.path.join(output, "img_pred%05d.png"%i)
        com_path = os.path.join(output, "img_gt%05d.png"%i)
        mask_path = os.path.join("/workspace/lego_test", proj_name, "MASK", "%08d.png"%i)
        original = cv2.imread(ori_path)
        compressed = cv2.imread(com_path)
        # print(original.shape, compressed.shape)
        compressed = compressed[:,:,:3]
        mask = cv2.imread(mask_path)
        mask[mask>0] = 1
        # mask = mask > 0
        original = original[mask>0]
        compressed = compressed[mask>0]
        # original = (original * mask).astype(np.float64)
        # compressed = (compressed * mask).astype(np.float64)
        # mse = np.mean((original - compressed)**2) * (compressed.shape[0] * compressed.shape[1]) / mask.sum()
        # psnr = 20 * math.log10(1.0 / math.sqrt(mse))
        # original = np.where(mask>0, original, 0)
        # compressed = np.where(mask>0, compressed, 0)
        psnr = PSNR(original, compressed)
        # ssim = SSIM(original, compressed, multichannel=True)
        psnr_sum += psnr
        # ssim_sum += ssim
        print(f"PSNR value is {psnr} dB")
        # print(psnr, ssim)
    print("ave: ", psnr_sum/img_num, ssim_sum/img_num)


def cal_distance(a, b, img, p_size):
    patch_a = img[a[0]:a[0]+p_size, a[1]:a[1]+p_size, :]
    patch_b = img[b[0]:b[0]+p_size, b[1]:b[1]+p_size, :]
    print(patch_a.shape)
    print(patch_b)
    temp = patch_b - patch_a
    # print(temp)
    num = np.sum(1 - np.int32(np.isnan(temp)))
    dist = np.sum(np.square(np.nan_to_num(temp))) / num
    return dist

if __name__ == "__main__":
    main()
    # img = imageio.imread("/workspace/nvrender/0_0_00.png")
    # img = np.array(img)[:,:,:3]
    # pos1 = np.array([4, 4])
    # pos2 = np.array([3, 3])
    # dis = cal_distance(pos1, pos2, img, 4)
    # print(dis)
    
    # import trimesh
    # scan = "scan114"
    # mesh = trimesh.load("/workspace/nvrender/results/" + scan + "/plane2.obj")
    # num = mesh.vertices.shape[0]
    # K_sparse = py2cpp.compute_k(scan)
    # idx = K_sparse[:2]
    # v = K_sparse[2]
    # s = torch.sparse_coo_tensor(idx, v, (num, num), dtype=torch.float32).cuda()
    # x = torch.ones(1, num, dtype=torch.float32).cuda()
    # result = torch.mm(x, torch.sparse.mm(s, x.T))
    # print(s, result, np.array(v).sum())
    # # K_sparse = torch.from_numpy(np.array(K_sparse))
    
    
    