import torch
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import imageio
from argparse import ArgumentParser

from preprocessing_utils import *
from volumetric_rendering import *
from NeRF_model import *


##################################################################################################################
'''
This code is based on https://github.com/kwea123/nerf_pl/
It is a simplified version of this repo implementation on pytorch lightning.
'''
def extract_model_state_dict(ckpt_path, model_name='model', prefixes_to_ignore=[]):
	checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
	checkpoint_ = {}
	if 'state_dict' in checkpoint: # if it's a pytorch-lightning checkpoint
		checkpoint = checkpoint['state_dict']
	for k, v in checkpoint.items():
		if not k.startswith(model_name):
			continue
		k = k[len(model_name)+1:]
		for prefix in prefixes_to_ignore:
			if k.startswith(prefix):
				print('ignore', k)
				break
		else:
			checkpoint_[k] = v
	return checkpoint_

def load_ckpt(model, ckpt_path, model_name='model', prefixes_to_ignore=[]):
	model_dict = model.state_dict()
	checkpoint_ = extract_model_state_dict(ckpt_path, model_name, prefixes_to_ignore)
	model_dict.update(checkpoint_)
	model.load_state_dict(model_dict)

torch.backends.cudnn.benchmark = True

@torch.no_grad()
def batched_inference(coarse_model, fine_model, embedder_pos,
					  embedder_dir,
					  rays, N_samples, N_importance,
					  chunk):
	"""Do batched inference on rays using chunk."""
	B = rays.shape[0]
	chunk = 1024*32
	results = defaultdict(list)
	for i in range(0, B, chunk):
		rendered_ray_chunks = \
			hierarchical_sampling(rays[i : i + chunk],
			coarse_model,
			fine_model,
			embedder_pos,
			embedder_dir,
			chunk,
			num_fine_samples=N_importance,
			num_samples = N_samples)

		for k, v in rendered_ray_chunks.items():
			results[k] += [v]

	for k, v in results.items():
		results[k] = torch.cat(v, 0)
	return results
	
def mse(image_pred, image_gt, reduction='mean'):
	value = (image_pred-image_gt)**2
	if reduction == 'mean':
		return np.mean(value)
	return value

def psnr(image_pred, image_gt, reduction='mean'):
	return -10*np.log10(mse(image_pred, image_gt,reduction))


if __name__ == "__main__":
	
	img_wh = (400, 400)
	w, h = img_wh
	root_dir = '../../nerf_synthetic/lego'
	split = 'test'
	ckpt_path = './logs/exp1/version_1/checkpoints/epoch=8-step=140625.ckpt'
	dataset_name = 'nerf_synthetic'
	scene_name = 'lego'
	N_samples = 64
	N_importance = 64
	use_disp = False,
	chunk = 32 * 1024
	kwargs = {'base_path': root_dir,
			  'img_dimension': img_wh,
			  'dataset_type' : split }

	dataset = load_blender(**kwargs)

	embedding_xyz = positional_encoding(3, 10)
	embedding_dir = positional_encoding(3, 4)
	nerf_coarse = NeRF()
	nerf_fine = NeRF()
	load_ckpt(nerf_coarse, ckpt_path, model_name='coarse_model')
	load_ckpt(nerf_fine, ckpt_path, model_name='fine_model')
	nerf_coarse.cuda().eval()
	nerf_fine.cuda().eval()

	imgs = []
	psnrs = []
	depth_maps = []
	dir_name = f'results/{dataset_name}/{scene_name}'
	os.makedirs(dir_name, exist_ok=True)

	for i in tqdm(range(len(dataset))):
		sample = dataset[i]
		rays = sample['rays'].cuda()
		results = batched_inference(nerf_coarse, nerf_fine, embedding_xyz,
									embedding_dir, rays,
									N_samples, N_importance,
									chunk)

		img_pred = results['color_fine'].view(h, w, 3).cpu().numpy()

		img_pred_ = (img_pred*255).astype(np.uint8)
		imgs += [img_pred_]
		imageio.imwrite(os.path.join(dir_name, f'{i:03d}.png'), img_pred_)
		
		depth_pred = results['depth_fine'].view(h, w).cpu().numpy()
		depth_maps += [depth_pred]

		if 'imgs' in sample:
			rgbs = sample['imgs']
			img_gt = rgbs.view(h, w, 3)
			psnrs += [psnr(img_pred, img_gt.cpu().numpy()).item()]
		
	imageio.mimsave(os.path.join(dir_name, f'{scene_name}.gif'), imgs, fps=30)
	
	min_depth = np.min(depth_maps)
	max_depth = np.max(depth_maps)
	
	depth_imgs = (depth_maps - np.min(depth_maps)) / (max(np.max(depth_maps) - np.min(depth_maps), 1e-8))
	depth_imgs_ = [cv2.applyColorMap((img * 255).astype(np.uint8), cv2.COLORMAP_JET) for img in depth_imgs]
	imageio.mimsave(os.path.join(dir_name, f'{scene_name}_depth.gif'), depth_imgs_, fps=30)
	
	if psnrs:
		mean_psnr = np.mean(psnrs)
		print(f'Mean PSNR : {mean_psnr:.2f}')

#########################################################################################################################33333
