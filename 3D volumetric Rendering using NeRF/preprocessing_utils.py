import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T

import torch
from kornia import create_meshgrid

######################################################################################################
'''
This code is based on https://github.com/kwea123/nerf_pl/
It is a simplified version of this repo implementation on pytorch lightning.
'''

def calculate_ray_directions(H, W, focal_length):
    
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        H, W, focal_length: image height, width and focal length
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """


    (x, y) = torch.meshgrid(torch.arange(W, dtype = torch.float32),
                               torch.arange(H, dtype = torch.float32))
    directions = torch.stack([
        (x.T - W / 2) / focal_length,
        -(y.T - H / 2) / focal_length,
        -torch.ones_like(x)
    ], dim = -1)

    return directions



def trace_rays(directions, camera2world_transformation):

    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate
    Outputs:
        rays_o: (H, W, 3), the origin of the rays in world coordinate
        rays_d: (H, W, 3), the normalized direction of the rays in world coordinate
    """

    
    # Rotate camera coordinates with respect to world coordinates to get direction.
    ray_directions = directions @ camera2world_transformation[:, :3].T
    ray_directions = ray_directions / torch.norm(ray_directions, dim = -1, keepdim = True)
    
    # Get the origin with respect to camera2world reference frame.
    ray_origin = camera2world_transformation[:, 3].expand(ray_directions.shape)
    return ray_directions, ray_origin

class load_blender(Dataset):

    '''
    Load the blender dataset and return the image and rays.

    Inputs:
        base_path = path of the root_dir which contains the scene object e.g. nerf_synthetic/lego
        img_dimension = Image dimension for resizing the image, default = (800, 800)
        dataset_type = type of the dataset('train', 'test', 'val')

    Outputs:
        imgs : image of a shape : (H*W, 3)
        rays : rays of a shape : (H*W, 8)


    
    '''
    
    def __init__(self, base_path, img_dimension = (800, 800), dataset_type = 'train'):
        
        self.base_path = base_path # Path of the root dir
        self.img_dimension = img_dimension # Image dimension
        self.dataset_type = dataset_type # Dataset type
        
        # Bounds for generating rays on blender dataset.
        self.near = 2
        self.far = 6
        
        # Read a json file containing all the information of dataset
        with open(os.path.join(self.base_path, f'transforms_{self.dataset_type}.json'), 'r') as f:
            self.dataset_file = json.load(f)
            
        self.W, self.H = self.img_dimension
        self.focal_len = 0.5 * self.W / np.tan(0.5 * self.dataset_file['camera_angle_x'])

        self.imgs = []
        self.poses = []
        self.to_tensor = T.ToTensor()
        self.directions = calculate_ray_directions(self.W, self.H, self.focal_len)

        self.rays_list = []
        if dataset_type == 'train':
            self.img_paths, self.poses = [], []
            temp_imgs = []
            for frame in self.dataset_file['frames']:
                img_path = os.path.join(self.base_path, f"{frame['file_path']}.png")
                self.img_paths.append(img_path)
                img = Image.open(img_path)
                #img = imageio.imread(img_path)
                img = img.resize(self.img_dimension, Image.LANCZOS)
                #img = (np.array(img) / 255.0).astype(np.float32)
                #img = cv2.resize(img, dsize=(self.W, self.H), interpolation = cv2.INTER_AREA) 
                #img = torch.from_numpy(img)
                #img = img.view(self.H * self.W, -1)
                img = self.to_tensor(img)
                img = img.view(4, -1).permute(1, 0)
                img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])
                temp_imgs.append(img)
                pose = np.array(frame['transform_matrix'])[:3, :4].astype(np.float32)
                self.poses.append(pose)
                pose = torch.FloatTensor(pose)
                ray_dir, ray_origin = trace_rays(self.directions, torch.FloatTensor(pose))
                ray_dir = ray_dir.view(-1, 3)
                ray_origin = ray_origin.view(-1, 3)
                near_trace = self.near * torch.ones_like(ray_dir[:, :1])
                far_trace = self.far * torch.ones_like(ray_dir[:, :1])
                rays = torch.cat([ray_origin, ray_dir,
                                 near_trace, far_trace], 1)
                self.rays_list.append(rays)
                
        
            self.imgs = torch.cat(temp_imgs, 0)
            self.rays_list = torch.cat(self.rays_list, 0)
            
            
    def __len__(self):
        if self.dataset_type == 'train':
            return len(self.rays_list)
        elif self.dataset_type == 'val':
            return 8
        else:
            return len(self.dataset_file['frames'])
        
    def __getitem__(self, idx):
        
        if self.dataset_type == 'train':
            return {'imgs' : self.imgs[idx],
                   'rays' : self.rays_list[idx]}
        else:
            frame = self.dataset_file['frames'][idx]
            img_path = os.path.join(self.base_path, f"{frame['file_path']}.png")
            #img = (np.array(img) / 255.).astype(np.float32)
            img = Image.open(img_path)
            #img = (np.array(img) / 255.0).astype(np.float32)
            img = img.resize(self.img_dimension, Image.LANCZOS)
            #img = cv2.resize(img, dsize=(self.W, self.H), interpolation = cv2.INTER_AREA)
            #img = torch.from_numpy(img)
            #img = img.view(self.H * self.W, -1)
            img = self.to_tensor(img)
            img = img.view(4, -1).permute(1, 0)
            img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])
            #transformation_matrix = torch.FloatTensor(frame['transform_matrix'][:3, :4])
            pose = np.array(frame['transform_matrix'])[:3, :4].astype(np.float32)
            pose = torch.FloatTensor(pose)
            ray_dir, ray_origin = trace_rays(self.directions, pose)
            ray_dir = ray_dir.view(-1, 3)
            ray_origin = ray_origin.view(-1, 3)
            near_trace = self.near * torch.ones_like(ray_dir[:, :1])
            far_trace = self.far * torch.ones_like(ray_dir[:, :1])
            rays = torch.cat([ray_origin, ray_dir,
                             near_trace, far_trace], 1)
            return {'imgs' : img,
                   'rays' : rays}

################################################################################################
    
