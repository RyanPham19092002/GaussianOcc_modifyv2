#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import matplotlib.pyplot as plt
import torch
import math, pdb
import numpy as np

# no semantic
# import diff_gaussian_rasterization as diff_3d_gaussian_rasterization
# # semantic
# import diff_gaussian_rasterization_semantic as diff_3d_gaussian_rasterization_semantic
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from simple_knn._C import distCUDA2
from .sh_utils import eval_sh
from .point_utils import depth_to_normal


#--------------------visualize---------------------------
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_matrix(matrix, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix.cpu().numpy(), annot=True, fmt=".2f", cmap='viridis')
    plt.title(title)

    # Lưu hình ảnh vào thư mục 'test'
    plt.savefig(f'test/{title}.png', bbox_inches='tight')  # Lưu hình ảnh
    plt.close()


def DistCUDA2(fused_point_cloud):

    dist2 = torch.clamp_min(distCUDA2(fused_point_cloud)[0], 0.0000001)
    scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)

    return scales


def camera_intrinsic_fov(intrinsic):

    #计算FOV
    w, h = intrinsic[0][2]*2, intrinsic[1][2]*2
    fx, fy = intrinsic[0][0], intrinsic[1][1]

    # Go
    fov_x = np.rad2deg(2 * np.arctan2(w, 2 * fx))
    fov_y = np.rad2deg(2 * np.arctan2(h, 2 * fy))

    return fov_x, fov_y


def splatting_render(viewpoint_camera, pc, scaling_modifier = 1.0,
            override_color = None, white_bg = False, opt = None):

    """
    Render the scene.
    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc['get_xyz'], dtype=pc['get_xyz'].dtype, requires_grad=True, device="cuda") + 0

    try:
        screenspace_points.retain_grad()

    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera['FoVx'] * 0.5)
    tanfovy = math.tan(viewpoint_camera['FoVx'] * 0.5)

    #if min(pc.bg_color.shape) != 0:
    bg_color = torch.tensor([0., 0., 0.]).cuda()

    raster_settings = GaussianRasterizationSettings(
                image_height=int(viewpoint_camera['image_height']),
                image_width=int(viewpoint_camera['image_width']),
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg = bg_color, #torch.tensor([1., 1., 1.]).cuda() if white_bg else torch.tensor([0., 0., 0.]).cuda(), #bg_color,
                scale_modifier=scaling_modifier,
                viewmatrix=viewpoint_camera['world_view_transform'], # 外参
                projmatrix=viewpoint_camera['full_proj_transform'], # 3D -> 2D
                sh_degree=3,
                campos=viewpoint_camera['camera_center'],
                prefiltered=False,
                debug=False
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc['get_xyz'].reshape(-1, 3)
    means2D = screenspace_points
    if pc['get_opacity'].dim() == 3:
        opacity = pc['get_opacity'].permute(1, 2, 0).view(-1, 1)
        colors_precomp = pc['colors_precomp'].permute(1, 2, 0).view(-1, 3)
        scales = pc['get_scaling'].permute(1, 2, 0).view(-1, 3)
        rotations = pc['get_rotation'].permute(1, 2, 0).view(-1, 4)
    elif pc['get_opacity'].dim() == 4:
        opacity = pc['get_opacity'].permute(0, 2, 3, 1).reshape(-1, 1)
        scales = pc['get_scaling'].permute(0, 2, 3, 1).reshape(-1, 3)
        rotations = pc['get_rotation'].permute(0, 2, 3, 1).reshape(-1, 4)
        colors_precomp = pc['colors_precomp'].permute(0, 2, 3, 1).reshape(-1, 3)

    cov3D_precomp = None


    shs = None

    

    # rendered_image, radii, depth, alpha = rasterizer(
    rendered_image, _ = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    # print("---------------------------------------------------------------------------------")
    # print("rendered_image shape", rendered_image.shape, rendered_image.min(), rendered_image.max())
    # print("depth shape", depth.shape, depth.min(), depth.max())
    # print("Image Height:", int(viewpoint_camera['image_height']))
    # print("Image Width:", int(viewpoint_camera['image_width']))
    # print("Tangent of FOV X (tanfovx):", tanfovx)
    # print("Camera Position (campos):", viewpoint_camera['camera_center'])
    # print("Camera Position Shape:", viewpoint_camera['camera_center'].shape)
    # print("View Matrix (world_view_transform):", viewpoint_camera['world_view_transform'])
    # print("View Matrix Shape:", viewpoint_camera['world_view_transform'].shape)
    # print("Projection Matrix (full_proj_transform):", viewpoint_camera['full_proj_transform'])
    # print("Projection Matrix Shape:", viewpoint_camera['full_proj_transform'].shape)    
    # print("means3D shape:", means3D.shape, "min:", means3D.min(), "max:", means3D.max())
    # print("means2D shape:", means2D.shape, "min:", means2D.min(), "max:", means2D.max())
    # print("colors_precomp shape:", colors_precomp.shape, "min:", colors_precomp.min(), "max:", colors_precomp.max())
    # print("opacity shape:", opacity.shape, "min:", opacity.min(), "max:", opacity.max())
    # print("scales shape:", scales.shape, "min:", scales.min(), "max:", scales.max())
    # print("rotations shape:", rotations.shape, "min:", rotations.min(), "max:", rotations.max())
    return {"render": rendered_image}
            # "viewspace_points": screenspace_points,
            # "visibility_filter" : radii > 0,
            # "radii": radii,
            # "depth": depth}
            # 'rend_normal': render_normal,
            # 'rend_dist': render_dist,
            # 'surf_depth': depth,
            # 'surf_normal': surf_normal}


if __name__ == "__main__":

    pass