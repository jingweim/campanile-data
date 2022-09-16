import math
import os
import cv2
import numpy as np

################################## Paths ################################## 
# Root data folder (Change to your own path)
root = '/home/jingweim/movies/campanile-data'

# # Train PSNR paths
# # Multi-NeRF
# gt_folder = '{root}/models/mip-nerf-360/images_4'
# im_folder = '{root}/models/mip-nerf-360/train_renders_'
# # Plenoxels
# gt_folder = '{root}/models/plenoxels/images_train'
# im_folder = '{root}/models/plenoxels/train_renders_'
# # Instant-NGP
# gt_folder = '{root}/drone-video/images-undistorted'
# im_folder = '{root}/models/instant-ngp/train_renders_'

# Test PSNR paths
gt_folder = '{root}/campanile-movie/images'
# # Mip-NeRF 360
# im_folder = '{root}/models/mip-nerf-360/test_renders_colorcorrect'
# # Plenoxels
# im_folder = '{root}/models/plenoxels/test_renders_colorcorrect'
# Instant-NGP
im_folder = '{root}/models/instant-ngp/test_renders_colorcorrect'


################################## Code ################################## 
im_fnames = sorted(os.listdir(im_folder))
gt_fnames = sorted(os.listdir(gt_folder))

def get_psnr(im, im_gt):
	mse = (im - im_gt) ** 2
	mse_num: float = mse.mean().item()
	psnr = -10.0 * math.log10(mse_num)
	return psnr

psnrs = []
for (im_fname, gt_fname) in zip(im_fnames, gt_fnames):
	im = (cv2.imread(os.path.join(im_folder, im_fname))).astype(float) / 255
	im_gt = (cv2.imread(os.path.join(gt_folder, gt_fname))).astype(float) / 255
	psnrs.append(get_psnr(im, im_gt))

print('Average PSNR: ', np.mean(psnrs))