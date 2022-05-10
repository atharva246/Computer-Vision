
import os, sys
import torch
from collections import defaultdict
import torch.nn as nn
import json
import numpy as np
import math
from PIL import Image
import cv2
from torch import searchsorted


#########################################################################################
'''
This code is based on https://github.com/kwea123/nerf_pl/
It is a simplified version of this repo implementation on pytorch lightning.
'''

def fine_sampling(sample_bins, weights, num_fine_samples):

    '''
    Sample the fine sampling points from the sample_bins for the fine_nerf model, 
    distribuion for the random sampling is defined by the weights or volumetric density
    that has been predicted by the coarse_model.

    Inputs:
            sample_bins : where N_rays is number of rays per chunk and
                    N_samples are number of coarse samples.
            weights : distribution of the default sampling points.
            num_fine_samples : number of fine samples for final prediction.    
    
    '''

    
    
    # Calculate the CDF to calculate the inverse sampling
    eps = 1e-5
    num_rays = weights.shape[0]
    num_samples = weights.shape[1]
    weights = weights + eps
    pdf = weights / torch.sum(weights, -1, keepdim = True)
    cdf = torch.cumsum(pdf, -1)
    cdf_padded = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)
    
    # Inverse sampling
    random_points = torch.rand(num_rays, num_fine_samples).cuda()
    # Sort each random point with respect to the cdf for that point
    indexes = torch.searchsorted(cdf_padded.contiguous(), random_points.contiguous(), right = True)
    # clamp the points to deal with out of bounds sample points.
    below = torch.max(torch.zeros_like(indexes-1), indexes-1)
    above = torch.min((cdf_padded.shape[-1] - 1) * torch.ones_like(indexes), indexes)
    
    # Calculation of fine samples.
    indexes_stacked = torch.stack([below, above], -1).view(num_rays, 2 * num_fine_samples)
    cdf_g = torch.gather(cdf_padded, 1, indexes_stacked).view(num_rays, num_fine_samples, 2)
    sample_points_filtered = torch.gather(sample_bins, 1, indexes_stacked).view(num_rays, num_fine_samples, 2)
    denominator = cdf_g[..., 1] - cdf_g[..., 0]
    denominator[denominator < eps] = 1
    t = (random_points - cdf_g[..., 0]) / denominator
    final_weighted_samples = sample_points_filtered[..., 0] + t * (sample_points_filtered[..., 1] - sample_points_filtered[..., 0])
    return final_weighted_samples

def volumetric_rendering(num_rays, ray_direction, viewing_dir_embedded, model, embedder_pos,
                        transformed_samples, sample_points, chunk):
    
    '''
        Volumetric rendering function performs the inference and finally rendered
        the input image with respect to the NeRF model using classical rendering technique.

        Inputs:
                num_rays : number of rays per chunk.
                ray_direction : ray_direction tensor for rendering.
                viewing_dir_embedded : viewing direction tensor.
                model : NeRF model(whether coarse or fine)
                embedder_pos : xyz positional encoder.
                transformed_samples : transformed sampling points wrt camera.
                sample_points : bin of sampling points along the ray.
                chunk : chunk size, default : 32*1024.

        Output:
                rendered_image : final rendered image 
                depth_image : depth of the rendered image
                transmittance : transmittance or volumetric density.
    '''
    

    num_samples = transformed_samples.shape[1]
    transformed_samples = transformed_samples.reshape(-1, 3)
    dir_broadcasted = torch.repeat_interleave(viewing_dir_embedded, repeats = num_samples,
                                             dim = 0)
    batch_size = transformed_samples.shape[0]
    predicted_view = []
    
    # Inference mode
    for idx in range(0, batch_size, chunk):
        
        coordinates_encoded = embedder_pos(transformed_samples[idx : idx + chunk])
        dir_embedded_chunk = dir_broadcasted[idx : idx + chunk]
        prediction = model(coordinates_encoded, dir_embedded_chunk)
        predicted_view.append(prediction)
        
    predicted_view_tensor = torch.cat(predicted_view, 0)
    output_ = predicted_view_tensor.view(num_rays, num_samples, 4)
    color = output_[..., 1:]
    density = output_[..., 0]
    
    # Adjust the sampling points.
    delta = sample_points[:, 1:] - sample_points[:, :-1]
    delta_infinity = 1e10 * torch.ones_like(delta[:, :1])
    final_delta = torch.cat([delta, delta_infinity], -1)
    
    final_delta = final_delta * torch.norm(ray_direction.unsqueeze(1), dim=-1)
    
    # Adding some noise
    noise = torch.randn(density.shape).cuda() * 0
    
    alphas = 1 - torch.exp(-final_delta * torch.nn.Softplus()(density + noise))
    exp_term = 1-alphas + 1e-10
    transmittance_ip = torch.cat([torch.ones_like(alphas[:, :1]), exp_term], -1)
    transmittance = alphas * torch.cumprod(transmittance_ip, -1)[:, :-1]
    
    rendered_image = torch.sum(transmittance.unsqueeze(-1) * color, -2)
    rendered_image = rendered_image + (1 - transmittance.sum(1).unsqueeze(-1))
    depth_image = torch.sum(transmittance * sample_points, -1)
    
    return rendered_image, depth_image, transmittance
    
    
    
def hierarchical_sampling(rays, coarse_model, fine_model, embedder_pos, embedder_dir,
                         chunk_size,
                         num_fine_samples, num_samples = 64):

    '''
    Take the inputs and do the heirarchical sampling to generate the rendered image.
    Inputs:
            rays : rays tensor, which is provided by DataLoader.
            fine_model : model type
            embedder_pos : xyz positional encoder
            embedder_dir : pose positional encoder
            chunk_size : chunk size
            num_fine_samples : number of fine samples.
            num_samples : number of coarse samples.

    Outputs:
            final_result : final_result dictionary to cache output of model.
    '''
    
    rays = rays.squeeze(0)
    ray_origin, ray_direction = rays[:, 0:3], rays[:, 3:6]
    num_rays = rays.shape[0]
    near = rays[:, 6:7]
    far = rays[:, 7:8]
    final_result = {}
    
    viewing_dir_embedded = embedder_dir(ray_direction)
    points = torch.linspace(0, 1, num_samples).cuda()
    bounded_points = near * (1 - points) + (far) * points
    bounded_points_reshape = bounded_points.expand(num_rays, num_samples)

    bounded_points_mid = 0.5 * (bounded_points_reshape[:, :-1] + bounded_points_reshape[:, 1:])
    upper_bound = torch.cat([bounded_points_mid, bounded_points_reshape[:, -1:]], -1)
    lower_bound = torch.cat([bounded_points_reshape[:, :1], bounded_points_mid], -1)
    perurbation = 1 * torch.rand(bounded_points_reshape.shape).cuda()
    bounded_points_reshape = lower_bound + (upper_bound - lower_bound) * perurbation
    
    final_coarse_samples = ray_origin.unsqueeze(1) + ray_direction.unsqueeze(1) * bounded_points_reshape.unsqueeze(2)
    
    image, depth, weights = volumetric_rendering(num_rays, ray_direction, viewing_dir_embedded, coarse_model, embedder_pos, final_coarse_samples,
                                                bounded_points_reshape, chunk_size)
    
    # Mid points of final_coarse_samples
    bounded_points_mid = 0.5 * (bounded_points_reshape[:, :-1] + bounded_points_reshape[:, 1:])
    fine_sample_points = fine_sampling(bounded_points_mid, weights[:, 1:-1],
                                          num_fine_samples).detach()
    
    # Merge the coarse sample points and fine sample points and sort them
    sorted_fine_points, _ = torch.sort(torch.cat([bounded_points_reshape,
                                             fine_sample_points], -1), -1)
    
    final_sampling_points = ray_origin.unsqueeze(1) + ray_direction.unsqueeze(1) * sorted_fine_points.unsqueeze(2)
    final_img, final_depth, final_weight = volumetric_rendering(num_rays, ray_direction, viewing_dir_embedded, fine_model, embedder_pos, final_sampling_points,
                                             sorted_fine_points, chunk_size)
    final_result['color_coarse'] = image
    final_result['depth_coarse'] = depth
    final_result['weights_coarse'] = weights.sum(1)
    final_result['color_fine'] = final_img
    final_result['depth_fine'] = final_depth
    final_result['weights_fine'] = final_weight.sum(1)
    return final_result

##########################################################################################################################################
    
    
