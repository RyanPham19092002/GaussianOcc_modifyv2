# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import pdb
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_efficient_distloss import eff_distloss, eff_distloss_native
from simple_knn._C import distCUDA2

from utils import geom
from utils import vox
from utils import basic
from utils import render
from ._3DCNN import S3DCNN
from sys import path
from PIL import Image
from lib.gaussian_renderer import splatting_render, DistCUDA2

from mmdet.models.builder import build_loss

# for gt occ loss
from utils.losses.semkitti_loss import sem_scal_loss, geo_scal_loss
from utils.losses.lovasz_softmax import lovasz_softmax
from nerfstudio.cameras.camera_utils import get_interpolated_poses
import open3d as o3d
import cv2
from tqdm import tqdm

nusc_class_frequencies = np.array([
    944004,
    1897170,
    152386,
    2391677,
    16957802,
    724139,
    189027,
    2074468,
    413451,
    2384460,
    5916653,
    175883646,
    4275424,
    51393615,
    61411620,
    105975596,
    116424404,
    1892500630
])



class VolumeDecoder(nn.Module):

    def __init__(self, opt):
        super(VolumeDecoder, self).__init__()

        self.opt = opt
        self.use_semantic = self.opt.use_semantic
        self.semantic_classes = self.opt.semantic_classes
        self.batch = self.opt.batch_size // self.opt.cam_N

        self.near = self.opt.min_depth
        self.far = self.opt.max_depth

        loss_occ=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            ignore_index=255,
            loss_weight=1.0)

        self.loss_occ = build_loss(loss_occ)

        num_classes = self.opt.semantic_classes

        class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:num_classes] + 0.001))
        self.cls_weights = class_weights


        self.register_buffer('xyz_min', torch.from_numpy(
            np.array([self.opt.real_size[0], self.opt.real_size[2], self.opt.real_size[4]])))
        self.register_buffer('xyz_max', torch.from_numpy(
            np.array([self.opt.real_size[1], self.opt.real_size[3], self.opt.real_size[5]])))

        self.ZMAX = self.opt.real_size[1]

        self.Z = self.opt.voxels_size[0]
        self.Y = self.opt.voxels_size[1]
        self.X = self.opt.voxels_size[2]

        self.Z_final = self.Z
        self.Y_final = self.Y
        self.X_final = self.X


        self.stepsize = self.opt.stepsize  # voxel
        self.num_voxels = self.Z_final * self.Y_final * self.X_final
        self.stepsize_log = self.stepsize
        self.interval = self.stepsize

        if self.opt.contracted_coord:
            # Sampling strategy for contracted coordinate
            contracted_rate = self.opt.contracted_ratio
            num_id_voxels = int(self.num_voxels * (contracted_rate)**3)
            self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_id_voxels).pow(1 / 3)
            diagonal = (self.xyz_max - self.xyz_min).pow(2).sum().pow(1 / 2)
            self.N_samples = int(diagonal / 2 / self.stepsize / self.voxel_size / contracted_rate)

            if self.opt.infinite_range:
                # depth_roi = [-self.far] * 3 + [self.far] * 3
                zval_roi = [-diagonal] * 3 + [diagonal] * 3
                fc = 1 - 0.5 / self.X  # avoid NaN
                zs_contracted = torch.linspace(0.0, fc, steps=self.N_samples)
                zs_world = vox.contracted2world(
                    zs_contracted[None, :, None].repeat(1, 1, 3),
                    # pc_range_roi=depth_roi,
                    pc_range_roi=zval_roi,
                    ratio=self.opt.contracted_ratio)[:, :, 0]
            else:
                zs_world = torch.linspace(0.0, self.N_samples - 1, steps=self.N_samples)[None] * self.stepsize * self.voxel_size
            self.register_buffer('Zval', zs_world)

            pc_range_roi = self.xyz_min.tolist() + self.xyz_max.tolist()

            self.norm_func = lambda xyz: vox.world2contracted(xyz, pc_range_roi=pc_range_roi, ratio=self.opt.contracted_ratio)

        else:
            self.N_samples = int(np.linalg.norm(np.array([self.Z_final // 2, self.Y_final // 2, self.X_final // 2]) + 1) / self.stepsize) + 1
            self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels).pow(1 / 3)
            zs_world = torch.linspace(0.0, self.N_samples - 1, steps=self.N_samples)[None] * self.stepsize * self.voxel_size
            self.register_buffer('Zval', zs_world)
            self.norm_func = lambda xyz: (xyz - self.xyz_min.to(xyz)) / (self.xyz_max.to(xyz) - self.xyz_min.to(xyz)) * 2.0 - 1.0

        length_pose_encoding = 3

        self.pos_embedding = None
        self.pos_embedding1 = None
        input_channel = self.opt.input_channel

        scene_centroid_x = 0.0
        scene_centroid_y = 0.0
        scene_centroid_z = 0.0

        scene_centroid = np.array([scene_centroid_x,
                                   scene_centroid_y,
                                   scene_centroid_z]).reshape([1, 3])

        self.register_buffer('scene_centroid', torch.from_numpy(scene_centroid).float())

        self.bounds = (self.opt.real_size[0], self.opt.real_size[1],
                       self.opt.real_size[2], self.opt.real_size[3],
                       self.opt.real_size[4], self.opt.real_size[5])
        #  bounds = (-40, 40, -40, 40, -1, 5.4)

        self.vox_util = vox.Vox_util(
            self.Z, self.Y, self.X,
            scene_centroid=self.scene_centroid,
            bounds=self.bounds, position = self.opt.position, length_pose_encoding = length_pose_encoding, opt = self.opt,
            assert_cube=False)


        activate_fun = nn.ReLU(inplace=True)
        if self.opt.aggregation == '3dcnn':
            out_channel = self.opt.out_channel
            self._3DCNN = S3DCNN(input_planes=input_channel, out_planes=out_channel, planes=self.opt.con_channel,
                                 activate_fun=activate_fun, opt=opt)

        else:
            print('please define the aggregation')
            exit()


        if 'gs' in self.opt.render_type:
            self.gs_vox_util = vox.Vox_util(
                self.Z_final, self.Y_final, self.X_final,
                scene_centroid = self.scene_centroid,
                bounds=self.bounds, position = self.opt.position,
                length_pose_encoding = length_pose_encoding,
                opt = self.opt, assert_cube=False)

    def visualize_point_cloud(self, points, filename):
        """
        Visualize a 3D point cloud using Open3D.

        Args:
            points (numpy.ndarray): Array of shape (N, 3) where N is the number of points.
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        o3d.io.write_point_cloud(filename, pcd)
        # print(f"Point cloud saved to {filename}")

    def warp_color_from_rgb_to_world(self, point_cloud_world, rgb_image, intrinsic, cam2world):
        """
        Warp color from RGB image based on world coordinates of point cloud.

        Args:
            point_cloud_world (torch.Tensor): Point cloud in world coordinates of shape [B, N, 3].
            rgb_image (torch.Tensor): RGB image tensor of shape [B, 3, H, W].
            intrinsic (torch.Tensor): Intrinsic matrix tensor of shape [B, 4, 4].
            cam2world (torch.Tensor): Camera to world matrix tensor of shape [B, 4, 4].

        Returns:
            torch.Tensor: Warped color of shape [B, 3, H, W].
        """
        B, N, _ = point_cloud_world.shape
        _, _, H, W = rgb_image.shape
        # print("point_cloud_world", point_cloud_world)
        # 1. Chuyển từ world coordinate về camera coordinate
        homogeneous_world_points = torch.cat((point_cloud_world, torch.ones(B, N, 1, device=point_cloud_world.device)), dim=2)  # [B, N, 4]
        # Chuyển từ world về camera bằng ma trận cam2world nghịch đảo
        camera_coordinates = torch.bmm(homogeneous_world_points, cam2world.transpose(1, 2).inverse())  # [B, N, 4]
        # print("camera_coordinates before", camera_coordinates)
        # 2. Tính tọa độ pixel trong ảnh
        camera_coordinates[:, :, :3] /= camera_coordinates[:, :, 2:3]  # Normalize với chiều z
        # print("camera_coordinates after", camera_coordinates)
        pixel_coordinates = torch.bmm(camera_coordinates[:, :, :3], intrinsic[:, :3, :3].transpose(1, 2))  # [B, N, 3]
        pixel_coordinates = torch.round(pixel_coordinates[:, :, :2]).long()  # [B, N, 2]
        # print("pixel_coordinates", pixel_coordinates)
        # 3. Tạo tensor để lưu kết quả màu
        warped_colors = torch.zeros((B, 3, H, W), device=rgb_image.device)

        # 4. Lặp qua từng điểm và gán màu từ rgb_image
        for b in range(B):
            for n in range(N):
                x, y = pixel_coordinates[b, n, 0].item(), pixel_coordinates[b, n, 1].item()  # Lấy tọa độ (x, y)
                if 0 <= x < W and 0 <= y < H:  # Kiểm tra xem tọa độ có nằm trong phạm vi ảnh không
                    warped_colors[b, :, y, x] = rgb_image[b, :, y, x]  # Gán màu từ rgb_image cho warped_colors

        return warped_colors
    def depth2pc(self, depth, extrinsic, intrinsic):
        B, C, H, W = depth.shape
        depth = depth[:, 0, :, :]
        # print("depth shape", depth.shape)
        rot = extrinsic[:, :3, :3]
        trans = extrinsic[:, :3, 3:]

        y, x = torch.meshgrid(torch.linspace(0.5, H-0.5, H, device=depth.device), torch.linspace(0.5, W-0.5, W, device=depth.device))
        pts_2d = torch.stack([x, y, torch.ones_like(x)], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)  # B H W 3
        # print("pts_2d", pts_2d)
        # depth = depth.to('cpu').detach().numpy()
        # pts_2d[..., 2] = 1.0 / (depth + 1e-8)
        pts_2d[..., 2] = depth
        pts_2d[:, :, :, 0] -= intrinsic[:, None, None, 0, 2]
        pts_2d[:, :, :, 1] -= intrinsic[:, None, None, 1, 2]
        pts_2d_xy = pts_2d[:, :, :, :2] * pts_2d[:, :, :, 2:]
        pts_2d = torch.cat([pts_2d_xy, pts_2d[..., 2:]], dim=-1)

        pts_2d[..., 0] /= intrinsic[:, 0, 0][:, None, None]
        pts_2d[..., 1] /= intrinsic[:, 1, 1][:, None, None]

        pts_2d = pts_2d.view(B, -1, 3).permute(0, 2, 1)
        rot_t = rot.permute(0, 2, 1)
        pts = torch.bmm(rot_t, pts_2d) - torch.bmm(rot_t, trans)
        # print("point cloud", pts.permute(0, 2, 1))
        return pts.permute(0, 2, 1)
    def pc_cam2pc_world(self, point_cloud, extrinsic):
        B,N,_ = point_cloud.shape
        pts_2d = point_cloud.view(B, -1, 3).permute(0, 2, 1)
        rot = extrinsic[:, :3, :3]
        trans = extrinsic[:, :3, 3:]
        rot_t = rot.permute(0, 2, 1)
        pts = torch.bmm(rot_t, pts_2d) - torch.bmm(rot_t, trans)
        return pts.permute(0, 2, 1)

    def transform_point_cloud_to_world(self, point_cloud_camera, cam2world):
        """
        Transform point cloud from camera coordinates to world coordinates.

        Args:
            point_cloud (torch.Tensor): Point cloud in camera coordinates of shape [B, 3, H, W].
            extrinsic (torch.Tensor): c2w matrix of shape [B, 4, 4].

        Returns:
            torch.Tensor: Point cloud in world coordinates of shape [B, N, 3]
        """
        # 1. Chuyển đổi point cloud thành dạng đồng nhất
        B, C, H, W = point_cloud_camera.shape

        # 1. Thêm tọa độ đồng nhất (homogeneous coordinates) [B, 4, H, W]
        ones = torch.ones(B, 1, H, W, device=point_cloud_camera.device)
        homogeneous_camera_points = torch.cat((point_cloud_camera, ones), dim=1)  # [B, 4, H, W]

        # 2. Chuyển thành dạng vector hàng để dễ nhân ma trận [B, 4, H*W]
        homogeneous_camera_points_flat = homogeneous_camera_points.view(B, 4, -1)  # [B, 4, H*W]

        # 3. Nhân với ma trận cam2world để chuyển về world coordinates [B, 4, H*W]
        point_cloud_world_flat = torch.bmm(cam2world, homogeneous_camera_points_flat)  # [B, 4, H*W]

        # 4. Reshape lại về dạng ban đầu và loại bỏ chiều thứ 4 (w)
        point_cloud_world = point_cloud_world_flat.permute(0,2,1)[:, :, :3]  # [B, N, 3]

        return point_cloud_world
    def update_z_min_for_overlap_points(self, point_cloud):
        """
        Cập nhật giá trị z của các điểm có cùng tọa độ (x, y) trong point cloud thành z min.

        Parameters:
        - point_cloud: numpy array với shape (6, N, 3), chứa tọa độ (x, y, z) của các điểm.

        Returns:
        - updated_point_cloud: numpy array với shape (6, N, 3), point cloud đã cập nhật với z min.
        """
        device = point_cloud.device

        # Reshape thành (6 * N, 3) để dễ dàng thao tác
        flat_points = point_cloud.reshape(-1, 3)

        # Sử dụng dictionary để lưu z min cho mỗi (x, y)
        z_min_dict = {}

        # Tìm z min cho mỗi cặp (x, y)
        for point in flat_points:
            x, y, z = point.tolist()
            key = (x, y)
            if key not in z_min_dict:
                z_min_dict[key] = z
            else:
                z_min_dict[key] = min(z_min_dict[key], z)

        # Chuyển z_min_dict thành tensor để cập nhật
        for i, point in enumerate(flat_points):
            x, y, _ = point.tolist()
            key = (x, y)
            flat_points[i, 2] = torch.tensor(z_min_dict[key], device=device)

        # Trả về tensor đã cập nhật với shape (6, N, 3)
        return point_cloud.reshape(6, -1, 3)
    def save_point_cloud_to_ply(self, point_cloud, rgb_colors, ply_file_path):
        """
        Save the point cloud and its colors to a PLY file.

        Args:
            point_cloud (torch.Tensor): Point cloud tensor of shape [Batch, N, 3].
            rgb_colors (torch.Tensor): RGB colors tensor of shape [Batch, N, 3].
            ply_file_path (str): Path to save the PLY file.
        """
        # Chuyển đổi point cloud và màu sắc từ tensor sang numpy
        point_cloud_np = point_cloud.detach().cpu().numpy().reshape(-1, 3)  # Chuyển đổi thành (B*N, 3)
        rgb_colors_np = rgb_colors.detach().cpu().numpy().reshape(-1, 3)  # Chuyển đổi thành (B*N, 3)

        # Tạo open3d.geometry.PointCloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud_np)  # Thêm điểm
        pcd.colors = o3d.utility.Vector3dVector(rgb_colors_np)  # Chuyển đổi màu sắc từ [0, 255] sang [0, 1]

        # Lưu point cloud vào file PLY
        o3d.io.write_point_cloud(ply_file_path, pcd)

    # Ví dụ sử dụng
    # Giả sử bạn đã có point_cloud đã warping màu và rgb_image
    # point_cloud: [B, N, 3]
    # warped_image: [B, 3, H, W] (RGB color đã được gán từ rgb_image)

    #----------------------------------------sample point---------------------------------------
    def calculate_ray_direction(self, pixel_x, pixel_y, intrinsic_matrix):
        """Tính toán ray direction từ tọa độ pixel và ma trận nội của camera."""
        fx, fy, cx, cy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1], intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
        ray_direction = torch.tensor([(pixel_x - cx) / fx,
                                    (pixel_y - cy) / fy,
                                    1.0])
        ray_direction = ray_direction / torch.norm(ray_direction)  # Chuẩn hóa vector
        return ray_direction
    def sample_points_along_ray(self, ray_origin, ray_direction, max_depth, step_size, cam2world_view1):
        """Lấy các điểm mẫu cách đều trên ray từ gốc của ray (vị trí camera)."""
        
        # ray_direction = ray_direction.to(device) 
        # cam2world_view1 = torch.tensor(cam2world_view1).to(device)
        depths = torch.arange(0, max_depth + step_size, step_size, device=ray_origin.device)
        sampled_points_camera = (ray_origin.unsqueeze(0) + depths.unsqueeze(1) * ray_direction.unsqueeze(0)).float() 
        sampled_points_camera_homogeneous = torch.cat((sampled_points_camera, torch.ones(sampled_points_camera.size(0), 1, device=sampled_points_camera.device)), dim=1)
        sampled_points_world = (cam2world_view1 @ sampled_points_camera_homogeneous.T).T

        return sampled_points_world
    def project_point_to_view(self, point_3d, world2cam_view2, intrinsic_view2):
        """Chuyển đổi điểm từ world coordinate về pixel coordinate của view 2."""
        # point_h = torch.cat((point_3d, torch.ones(point_3d.size(0), 1, device=point_3d.device)), dim=1)
        point_h = point_3d
        point_cam = world2cam_view2 @ point_h.T  # Chuyển về tọa độ camera
        point_2d = (intrinsic_view2 @ point_cam[:3, :]).clone()  # Chuyển về tọa độ ảnh
        point_2d = point_2d.clone()
        # print("point_2d", point_2d.shape, point_2d)
        point_2d_normalized = point_2d.clone()  # Clone before modification to avoid in-place operation
        point_2d_normalized /= point_2d[2, :]  # Chuẩn hóa để lấy tọa độ (x, y)
    # def project_point_to_view(self, point_3d, world2cam_view2, intrinsic_view2):
    #     """Chuyển đổi điểm từ world coordinate về pixel coordinate của view 2."""
    #     # Convert point_3d to homogeneous coordinates (N, 4)
    #     # point_h = torch.cat((point_3d, torch.ones(point_3d.size(0), 1, device=point_3d.device)), dim=1)

    #     # Reshape for batch multiplication: (1, N, 4) for world2cam_view2
    #     point_h = point_3d.unsqueeze(0)  # Shape: (1, N, 4)
    #     world2cam_view2 = world2cam_view2.unsqueeze(0)  # Shape: (1, 4, 4)

    #     # Convert to camera coordinates
    #     point_cam = torch.bmm(world2cam_view2, point_h.transpose(1, 2))  # Shape: (1, 4, N)
    #     point_cam = point_cam.squeeze(0).transpose(0, 1)  # Shape: (N, 4)

    #     # Project to pixel coordinates
    #     point_2d = (intrinsic_view2 @ point_cam[:, :3].T)  # Shape: (3, N)
        
    #     # Normalize to get pixel coordinates
    #     point_2d_normalized = point_2d.clone()
    #     point_2d_normalized /= point_2d[2, :]  # Normalize to (x, y)
        
        # print("point_2d_normalized", point_2d_normalized)
        
        
        return point_2d_normalized[:2, :].T
    def is_overlap_pixel(self, ray_origin, ray_direction, max_depth, step_size, cam2world_view1, world2cam_view2, intrinsic_view2, image_width, image_height):
        """Xác định liệu pixel có nằm trong vùng overlap giữa hai view."""
        overlap_count = 0
        # print("create sampled_points")
        sampled_points = self.sample_points_along_ray(ray_origin, ray_direction, max_depth, step_size, cam2world_view1)

        for point_3d in sampled_points:
            
            point_2d = self.project_point_to_view(point_3d.unsqueeze(0), world2cam_view2, intrinsic_view2).squeeze(0)
            
            if 0 <= point_2d[0] < image_width and 0 <= point_2d[1] < image_height:
                overlap_count += 1
                
                if overlap_count > 1:
                    return True
        # exit()
        # print("project_point_to_view done")
        return False
    
    def create_overlap_mask(self, image_width, image_height, ray_origin, max_depth, step_size, cam2world_view1, intrinsic_view1, world2cam_view2, intrinsic_view2):
        """
        Tạo mask cho ảnh view 1 với overlap màu trắng và không overlap màu đen.
        """
        # Khởi tạo mask với kích thước ảnh
        mask = torch.zeros((image_height, image_width), dtype=torch.uint8)
        device=ray_origin.device
        cam2world_view1 = torch.tensor(cam2world_view1).to(device)
        world2cam_view2 = torch.tensor(world2cam_view2).to(device)
        intrinsic_view1 = torch.tensor(intrinsic_view1).to(device)
        intrinsic_view2 = torch.tensor(intrinsic_view2).to(device)

        # Lặp qua từng pixel trong ảnh view 1
        for pixel_y in tqdm(range(image_height), desc="Processing rows"):
            for pixel_x in tqdm(range(int(image_width / 6)), desc="Processing pixels in row", leave=False):
                # Tính toán ray direction cho mỗi pixel
                # print("create dir")
                ray_dir = self.calculate_ray_direction(pixel_x, pixel_y, intrinsic_view1).to(device)
                # Kiểm tra xem pixel có nằm trong vùng overlap không
                
                if self.is_overlap_pixel(ray_origin, ray_dir, max_depth, step_size, cam2world_view1, world2cam_view2, intrinsic_view2, image_width, image_height):
                    # Cập nhật mask thành màu trắng (1)
                    mask[pixel_y, pixel_x] = 1
        return mask
    def create_overlap_mask_1(self, point_cloud_view1, cam_intr2, cam2world2, image_height, image_width):
        """
        Generate an overlap mask for view 2 based on point cloud of view 1.
        
        Parameters:
        - point_cloud_view1 (torch.Tensor): Point cloud of view 1 in world coordinates (N x 3).
        - cam_intr2 (torch.Tensor): Intrinsic matrix of view 2 (3 x 3).
        - cam2world2 (torch.Tensor): Extrinsic matrix (camera to world) of view 2 (4 x 4).
        - image_height (int): Height of the image in view 2.
        - image_width (int): Width of the image in view 2.
        
        Returns:
        - overlap_mask (torch.Tensor): Binary mask for overlapping areas (image_height x image_width).
        """
        overlap_mask = torch.zeros((image_height, image_width), dtype=torch.uint8, device=point_cloud_view1.device)
        cam_intr2 = torch.tensor(cam_intr2, device=point_cloud_view1.device)
        cam2world2 = torch.tensor(cam2world2, device=point_cloud_view1.device)
        # Convert the camera to world matrix to world to camera by inverting it
        world2cam_view2 = torch.inverse(cam2world2)

        # Convert the world points to homogeneous coordinates by adding 1s in the fourth dimension
        ones = torch.ones((point_cloud_view1.size(0), 1), device=point_cloud_view1.device)
        # print("point_cloud_view1.size(0)", point_cloud_view1.size(0))
        # print("point_cloud_view1 shape", point_cloud_view1.shape)
        # print("ones shape", ones.shape)
        point_cloud_h = torch.cat((point_cloud_view1, ones), dim=1)

        # Project points to view 2 camera coordinates using the inverted matrix
        point_cam2 = world2cam_view2 @ point_cloud_h.T
        point_2d = cam_intr2 @ point_cam2[:3, :]
        
        # Normalize to get pixel coordinates
        point_2d = (point_2d[:2, :] / point_2d[2, :]).T  # (N x 2)

        # For each point, mark the corresponding pixel in the mask if within bounds
        for pixel in point_2d:
            x, y = int(pixel[0]), int(pixel[1])
            if 0 <= x < image_width and 0 <= y < image_height:
                overlap_mask[y, x] = 1

        return overlap_mask
    #----------------------------------------------------------------
    def feature2vox_simple(self, features, pix_T_cams, cam0_T_camXs, __p, __u):

        pix_T_cams_ = pix_T_cams
        camXs_T_cam0_ = geom.safe_inverse(cam0_T_camXs)

        _, C, Hf, Wf = features.shape

        sy = Hf / float(self.opt.height)
        sx = Wf / float(self.opt.width)

        # unproject image feature to 3d grid
        featpix_T_cams_ = geom.scale_intrinsics(pix_T_cams_, sx, sy)
        # pix_T_cams_ shape: [6,4,4]  feature down sample -> featpix_T_cams_

        feat_mems_ = self.vox_util.unproject_image_to_mem(
            features,
            basic.matmul2(featpix_T_cams_, camXs_T_cam0_),
            camXs_T_cam0_, self.Z, self.Y, self.X)

        # feat_mems_ shape： torch.Size([6, 128, 200, 8, 200])
        feat_mems = __u(feat_mems_)  # B, S, C, Z, Y, X # torch.Size([1, 6, 128, 200, 8, 200])
        # feat_mems = feat_mems_
        mask_mems = (torch.abs(feat_mems) > 0).float()
        print("mask_mems shape", mask_mems.shape)
        # print("mask_mems", mask_mems)
        # print("====================================================")
        feat_mem = basic.reduce_masked_mean(feat_mems, mask_mems, dim=1)  # B, C, Z, Y, X
        # print("feat_mem shape", feat_mem.shape)
        # print("feat_mem", feat_mem)
        # print("====================================================")
        feat_mem = feat_mem.permute(0, 1, 4, 3, 2) # [0, ...].unsqueeze(0) # ZYX -> XYZ
        # exit()
        return feat_mem


    def grid_sampler(self, xyz, *grids, align_corners=True, avail_mask=None, vis=False):
        '''Wrapper for the interp operation'''

        # pdb.set_trace()

        if self.opt.semantic_sample_ratio < 1.0 and self.use_semantic and not vis:
            group_size = int(1.0 / self.opt.semantic_sample_ratio)
            group_num = xyz.shape[1] // group_size
            xyz_sem = xyz[:, :group_size * group_num].reshape(xyz.shape[0], group_num, group_size, 3).mean(dim=2)
        else:
            xyz_sem = None

        # pdb.set_trace()

        if avail_mask is not None:
            if self.opt.contracted_coord:
                ind_norm = self.norm_func(xyz)
                avail_mask = self.effective_points_mask(ind_norm)
                ind_norm = ind_norm[avail_mask]
                if xyz_sem is not None:
                    avail_mask_sem = avail_mask[:, :group_size * group_num].reshape(avail_mask.shape[0], group_num, group_size).any(dim=-1)
                    ind_norm_sem = self.norm_func(xyz_sem[avail_mask_sem])
            else:
                xyz_masked = xyz[avail_mask]
                ind_norm = self.norm_func(xyz_masked)
                if xyz_sem is not None:
                    avail_mask_sem = avail_mask[:, :group_size * group_num].reshape(avail_mask.shape[0], group_num, group_size).any(dim=-1)
                    ind_norm_sem = self.norm_func(xyz_sem[avail_mask_sem])

        else:

            ind_norm = self.norm_func(xyz)

            if xyz_sem is not None:
                ind_norm_sem = self.norm_func(xyz_sem)
                avail_mask_sem = None

        ind_norm = ind_norm.flip((-1,)) # value range: [-1, 1]
        shape = ind_norm.shape[:-1]
        ind_norm = ind_norm.reshape(1, 1, 1, -1, 3)

        if xyz_sem is None:
            grid = grids[0] # BCXYZ # torch.Size([1, C, 256, 256, 16])
            ret_lst = F.grid_sample(grid, ind_norm, mode='bilinear', align_corners=align_corners).reshape(grid.shape[1], -1).T.reshape(*shape, grid.shape[1])
            if self.use_semantic:
                semantic, feats = ret_lst[..., :self.semantic_classes], ret_lst[..., -1]
                return feats, avail_mask, semantic
            else:
                return ret_lst.squeeze(), avail_mask

        else:

            ind_norm_sem = ind_norm_sem.flip((-1,))
            shape_sem = ind_norm_sem.shape[:-1]
            ind_norm_sem = ind_norm_sem.reshape(1, 1, 1, -1, 3)
            grid_sem = grids[0][:, :self.semantic_classes] # BCXYZ # torch.Size([1, semantic_classes, H, W, Z])
            grid_geo = grids[0][:, -1:] # BCXYZ # torch.Size([1, 1, H, W, Z])
            ret_sem = F.grid_sample(grid_sem, ind_norm_sem, mode='bilinear', align_corners=align_corners).reshape(grid_sem.shape[1], -1).T.reshape(*shape_sem, grid_sem.shape[1])
            ret_geo = F.grid_sample(grid_geo, ind_norm, mode='bilinear', align_corners=align_corners).reshape(grid_geo.shape[1], -1).T.reshape(*shape, grid_geo.shape[1])

            # pdb.set_trace()

            return ret_geo.squeeze(), avail_mask, ret_sem, avail_mask_sem, group_num, group_size


    def sample_ray(self, rays_o, rays_d, is_train):
        '''Sample query points on rays'''
        Zval = self.Zval.to(rays_o)
        if is_train:
            Zval = Zval.repeat(rays_d.shape[-2], 1)
            Zval += (torch.rand_like(Zval[:, [0]]) * 0.2 - 0.1) * self.stepsize_log * self.voxel_size
            Zval = Zval.clamp(min=0.0)

        Zval = Zval + self.near
        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * Zval[..., None]
        rays_pts_depth = (rays_o[..., None, :] - rays_pts).norm(dim=-1)

        if self.opt.contracted_coord:
            # contracted coordiante has infinite perception range
            mask_outbbox = torch.zeros_like(rays_pts[..., 0]).bool()
        else:
            mask_outbbox = ((self.xyz_min > rays_pts) | (rays_pts > self.xyz_max)).any(dim=-1)

        return rays_pts, mask_outbbox, Zval, rays_pts_depth


    def effective_points_mask(self, points):
        '''Mask out points that are too close to each other in the contracted coordinate'''
        dist = torch.diff(points, dim=-2, prepend=torch.zeros_like(points[..., :1, :])).abs()
        xyz_thresh = 0.4 / torch.tensor([self.X, self.Y, self.Z]).to(points)
        mask = (dist > xyz_thresh).bool().any(dim=-1)
        return mask

    def activate_density(self, density, dists):
        return 1 - torch.exp(-F.relu(density) * dists)


    def get_density(self, is_train, inputs, rot_map, scale_map, depth_map, opacity_map, point_cloud, cam_num):

        dtype = torch.float16 if self.opt.use_fp16 else torch.float32

    
        if 'gs' in self.opt.render_type:
            semantic = None
            reg_loss = {}

            K, C2W, pc = self.prepare_gs_attribute(inputs, rot_map, scale_map, depth_map, opacity_map, point_cloud)

            # pdb.set_trace()
            rgb_marched = self.get_splatting_rendering(K, C2W, pc, inputs)
            # rgb_marched = self.get_splatting_rendering(K, C2W, pc, inputs)
            # print("rgb_marched shape", rgb_marched.shape)
            # exit()
            depth = depth_map

            # if self.opt.use_semantic:
            #     semantic = torch.cat(rgb_marched, dim=0).permute(0, 2, 3, 1).contiguous()

            # if self.opt.infinite_range:
            #     depth = depth.clamp(min=self.near, max=200)
            # else:
            # depth = depth.clamp(min=self.near, max=self.far)

            # if self.opt.weight_distortion > 0:
            #     loss_distortion = total_variation(Voxel_feat)
            #     reg_loss['loss_distortion'] = self.opt.weight_distortion * loss_distortion

            # return depth.float(), rgb_marched, semantic, reg_loss
            return depth.float(), rgb_marched

        # print("Continue-------------------------")
        __p = lambda x: basic.pack_seqdim(x, self.batch)  # merge batch and number of cameras
        __u = lambda x: basic.unpack_seqdim(x, self.batch)

         # rendering
        rays_o = __u(inputs['rays_o', 0])
        rays_d = __u(inputs['rays_d', 0])

        device = rays_o.device

        rays_o, rays_d, Voxel_feat = rays_o.to(dtype), rays_d.to(dtype), Voxel_feat.to(dtype)

        reg_loss = {}
        eps_time = time.time()

        with torch.no_grad():
            rays_o_i = rays_o[0, ...].flatten(0, 2)  # HXWX3
            rays_d_i = rays_d[0, ...].flatten(0, 2)  # HXWX3
            rays_pts, mask_outbbox, z_vals, rays_pts_depth = self.sample_ray(rays_o_i, rays_d_i, is_train=is_train)

        dists = rays_pts_depth[..., 1:] - rays_pts_depth[..., :-1]  # [num pixels, num points - 1]
        dists = torch.cat([dists, 1e4 * torch.ones_like(dists[..., :1])], dim=-1)  # [num pixels, num points]

        sample_ret = self.grid_sampler(rays_pts, Voxel_feat, avail_mask=~mask_outbbox)


        # false
        if self.use_semantic:
            if self.opt.semantic_sample_ratio < 1.0:
                geo_feats, mask, semantic, mask_sem, group_num, group_size = sample_ret

            else:
                geo_feats, mask, semantic = sample_ret

        else:
            geo_feats, mask = sample_ret
        
        print("mask", mask.shape)
        print("geo_feats",geo_feats.shape)     


        if self.opt.render_type == 'prob':
            weights = torch.zeros_like(rays_pts[..., 0])
            weights[:, -1] = 1
            geo_feats = torch.sigmoid(geo_feats)


            if self.opt.last_free:
                geo_feats = 1.0 - geo_feats
                # the last channel is the probability of being free

            weights[mask] = geo_feats

            # accumulate
            weights = weights.cumsum(dim=1).clamp(max=1)
            alphainv_fin = weights[..., -1]
            weights = weights.diff(dim=1, prepend=torch.zeros((rays_pts.shape[:1])).unsqueeze(1).to(device=device, dtype=dtype))
            depth = (weights * z_vals).sum(-1)
            rgb_marched = 0


        elif self.opt.render_type == 'density':

            alpha = torch.zeros_like(rays_pts[..., 0])  # [num pixels, num points]
            alpha[mask] = self.activate_density(geo_feats, dists[mask])

            weights, alphainv_cum = render.get_ray_marching_ray(alpha)
            alphainv_fin = alphainv_cum[..., -1]
            depth = (weights * z_vals).sum(-1)
            rgb_marched = 0

        else:
            raise NotImplementedError

        # pdb.set_trace()
        if self.use_semantic:
            if self.opt.semantic_sample_ratio < 1.0:
                semantic_out = torch.zeros(mask_sem.shape + (self.semantic_classes, )).to(device=device, dtype=dtype)
                semantic_out[mask_sem] = semantic
                weights_sem = weights[:, :group_num * group_size].reshape(weights.shape[0], group_num, group_size).sum(dim=-1)
                semantic_out = (semantic_out * weights_sem[..., None]).sum(dim=-2)

            else:
                semantic_out = torch.ones(rays_pts.shape[:-1] + (self.semantic_classes, )).to(device=device, dtype=dtype)
                semantic_out[mask] = semantic
                semantic_out = (semantic_out * weights[..., None]).sum(dim=-2)

            semantic_out = semantic_out.reshape(cam_num, self.opt.render_h, self.opt.render_w, self.semantic_classes)

        else:
            semantic_out = None

        if is_train:
            if self.opt.weight_entropy_last > 0:
                pout = alphainv_fin.float().clamp(1e-6, 1-1e-6)
                entropy_last_loss = -(pout*torch.log(pout) + (1-pout)*torch.log(1-pout)).mean()
                reg_loss["loss_entropy_last"] = self.opt.weight_entropy_last * entropy_last_loss

            if self.opt.weight_distortion > 0:
                loss_distortion = eff_distloss(weights.float(), z_vals.float(), dists.float())
                reg_loss['loss_distortion'] =  self.opt.weight_distortion * loss_distortion

            if self.opt.weight_sparse_reg > 0:
                geo_f = Voxel_feat[..., -1].float().flatten()
                if self.opt.last_free:
                    geo_f = - geo_f
                loss_sparse_reg = F.binary_cross_entropy_with_logits(geo_f, torch.zeros_like(geo_f), reduction='mean')
                reg_loss['loss_sparse_reg'] = self.opt.weight_sparse_reg * loss_sparse_reg


        depth = depth.reshape(cam_num, self.opt.render_h, self.opt.render_w).unsqueeze(1)

        if self.opt.infinite_range:
            depth = depth.clamp(min=self.near, max=200)
        else:
            depth = depth.clamp(min=self.near, max=self.far)


        return depth.float(), rgb_marched, semantic_out, reg_loss


    def get_ego2pix_rt(self, features, pix_T_cams, cam0_T_camXs):

        pix_T_cams_ = pix_T_cams
        camXs_T_cam0_ = geom.safe_inverse(cam0_T_camXs)
        Hf, Wf = features.shape[-2], features.shape[-1]

        if self.opt.view_trans == 'query' or self.opt.view_trans == 'query1':
            featpix_T_cams_ = pix_T_cams_
        else:

            sy = Hf / float(self.opt.height)
            sx = Wf / float(self.opt.width)
            # unproject image feature to 3d grid
            featpix_T_cams_ = geom.scale_intrinsics(pix_T_cams_, sx, sy)


        # pix_T_cams_ shape: [6,4,4]  feature down sample -> featpix_T_cams_
        ego2featpix = basic.matmul2(featpix_T_cams_, camXs_T_cam0_)


        return ego2featpix, camXs_T_cam0_, Hf, Wf
    def compute_scaling_map(self, intrinsic, depth_map):
        B, C, H, W = depth_map.shape
        depth = depth_map[:, 0, :, :]  # Giả sử depth map có dạng [B, 1, H, W]

        # Lấy các hệ số tỉ lệ từ intrinsic matrix
        f_x = intrinsic[:, 0, 0]  # [B]
        f_y = intrinsic[:, 1, 1]  # [B]

        # Tạo meshgrid để biểu diễn pixel tọa độ
        y, x = torch.meshgrid(torch.arange(H, device=depth.device), torch.arange(W, device=depth.device))
        x = x.float().unsqueeze(0).repeat(B, 1, 1)  # [B, H, W]
        y = y.float().unsqueeze(0).repeat(B, 1, 1)  # [B, H, W]

        # Tính scale theo trục X và Y (dựa trên f_x, f_y và khoảng cách từ camera)
        scale_x = depth / f_x[:, None, None]  # [B, H, W]
        scale_y = depth / f_y[:, None, None]  # [B, H, W]
        
        # Scale theo Z chính là chính depth map, vì Z là khoảng cách theo chiều sâu
        scale_z = depth  # [B, H, W]

        # Ghép các scale x, y, z lại thành scaling map
        scaling_map = torch.stack([scale_x, scale_y, scale_z], dim=1)  # [B, 3, H, W]
        
        return scaling_map
    def get_voxel(self, features, inputs):
        # print("features", features.shape)
        print("features[0]", features[0].shape)
        print('inputs["color", 0, 0].shape' , inputs["color", 0, 0].shape)
        __p = lambda x: basic.pack_seqdim(x, self.batch)  # merge batch and number of cameras
        __u = lambda x: basic.unpack_seqdim(x, self.batch)

        meta_similarity = None
        meta_feature = None
        curcar2precar = None
        nextcam2curego = None

        # input_channel=64
        feature_size = self.opt.input_channel

        Extrix_RT = inputs['pose_spatial'][:6]
        Intrix_K = inputs[('K', 0, 0)][:6]

        Voxel_feat = self.feature2vox_simple(inputs["color", 0, 0], Intrix_K, Extrix_RT, __p, __u)


        return Voxel_feat, meta_similarity, meta_feature, nextcam2curego, feature_size
    
    def inverse_sigmoid(self,x):
        return torch.log(x/(1-x))


    def prepare_gs_attribute(self, inputs, rot_map, scale_map, depth_map, opacity_map, point_cloud, is_train = None, index = 0):
        # prepare gaussian
        pc = {}
        bs = inputs["color", 0, 0].shape[0]
        
        # render depth map of each view
        K = inputs[('K', 0, 0)]
        K_inv = inputs[('inv_K', 0, 0)]

        B, _, H, W = inputs["color", 0, 0].shape
        # if self.opt.surround_view:
        #     C2W = inputs['surround_pose'].to('cpu').numpy()
        # else:
        pc['colors_precomp'] = inputs["color", 0, 0]
        # print("pc['colors_precomp']", pc['colors_precomp'].shape)
        C2W = inputs['pose_spatial']

        pc['get_xyz'] = self.transform_point_cloud_to_world(point_cloud, C2W)
        # pc['get_xyz'] = self.update_z_min_for_overlap_points(pc['get_xyz'])
        # print("point cloud", point_cloud.permute(0,2,3,1).view(6, -1, 3))
        # print("'depth map", self.depth2pc(depth_map, C2W, K).view(bs, -1, 3))
        # exit()
        # warp_img = self.warp_color_from_rgb_to_world(pc['get_xyz'], pc['colors_precomp'], K, C2W)
        # for i in range(B):
        #     # Chuyển đổi hình ảnh từ tensor sang numpy array
        #     print(f"warp img {i}")
        #     img_np = warp_img[i].permute(1, 2, 0).cpu().numpy()  # Chuyển từ [C, H, W] sang [H, W, C]
        #     img_np = (img_np * 255).astype(np.uint8)  # Chuyển đổi về định dạng uint8

        #     # Chuyển đổi numpy array thành PIL Image
        #     img_pil = Image.fromarray(img_np)

        #     # Lưu hình ảnh
        #     img_pil.save(f"test_img/rgb_warp_{i}.png")
        # ply_file_path = 'output_point_cloud_from_depth_5.ply'
        # pc['colors_precomp'] = warp_img
        # self.save_point_cloud_to_ply(pc['get_xyz'], warp_img.permute(0,2,3,1).view(B,-1,3), ply_file_path)
        # print(f"Point cloud saved to {ply_file_path}")
        

        # dist2_list = []

        # # Dùng vòng lặp qua từng batch
        # for b in range(B):
        #     # Tính khoảng cách cho từng batch b
        #     dist_b = torch.clamp_min(distCUDA2(pc['get_xyz'][b].float().cuda()), 0.0000001)
        #     dist2_list.append(dist_b)

        # dist2 = torch.stack(dist2_list, dim=0)

        # scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 1, 3)
        # # print("dist2 shape", dist2.shape)
        # # print("scales shape", scales.shape)
        # # exit()
        # rots = torch.zeros((pc['get_xyz'].shape[0], pc['get_xyz'].shape[1], 4), device="cuda")
        # rots[:, :, 0] = 1

        # opacities = self.inverse_sigmoid(0.1 * torch.ones((pc['get_xyz'].shape[0], pc['get_xyz'].shape[1], 1), dtype=torch.float, device="cuda"))
        
        # self._scaling = torch.exp(nn.Parameter(scales.requires_grad_(True)))
        # self._rotation = torch.nn.functional.normalize(nn.Parameter(rots.requires_grad_(True)), dim=1)
        # self._opacity = torch.sigmoid(nn.Parameter(opacities.requires_grad_(True)))


        # print("self._scaling shape", self._scaling.shape, self._scaling)
        # print("self._rotation shape", self._rotation.shape, self._rotation)
        # print("self._opacity shape", self._opacity.shape, self._opacity)
        # # exit()
        # pc['get_scaling'] = self._scaling
        # pc['get_opacity'] = self._opacity

        # pc['get_rotation'] = self._rotation

        pc['get_scaling'] = scale_map
        pc['get_opacity'] = opacity_map
        pc['get_rotation'] = rot_map

        # pc['confidence'] = torch.ones_like(pc['get_opacity'])
        pc['active_sh_degree'] = torch.full((6,), 3)
        
        # print("max color, min", pc['colors_precomp'].max(), pc['colors_precomp'].min())
        # exit()
        return K.to('cpu').numpy(), C2W.to('cpu').numpy(), pc


    def get_splatting_rendering(self, K, C2W, pc, inputs, flow_index = (0, 0)):
        # if self.opt.flow != 'No':
        rgb_spaltting = []
        depth = []

        R_only = inputs['pose_spatial'].clone()
        R_only = geom.safe_inverse(R_only)
        R_only[:, :3, 3] = 0

        rgb_marched = None
        C2W_NVS = []
        for j in range(C2W.shape[0]):
            pose_a = C2W[j]
            if j == C2W.shape[0] - 1:
                pose_b = C2W[0]
            else:
                pose_b = C2W[j+1]
            pose_ab = get_interpolated_poses(pose_a, pose_b, 3)
            pose_NVS = torch.tensor(pose_ab[1])
            one_row = torch.tensor([[0, 0, 0, 1]], dtype = torch.float32)
            pose_NVS = torch.cat([pose_NVS, one_row], dim=0)
            C2W_NVS.append(pose_NVS)
            # print("pose_NVS shape", pose_NVS.shape, pose_NVS)
        C2W_NVS = torch.stack(C2W_NVS)
        C2W_NVS = C2W_NVS.cpu().numpy().astype(np.float32)
        
        if self.opt.render_novel_view:
            C2W = C2W_NVS
        
        # print("C2W_NVS shape", C2W_NVS.shape)
        # exit()

        # depth
        viewpoint_camera_list = []
        for j in range (C2W.shape[0]):

            
            all_cam_center = inputs['all_cam_center']
            # print("intrinsic", K[j])
            # print("c2w", C2W[j])
            # print("=======================================================")
            viewpoint_camera = geom.setup_opengl_proj(w = self.opt.render_w, h = self.opt.render_h, k = K[j], c2w = C2W[j],near=self.opt.min_depth, far=1000)
            # viewpoint_camera_list.append(viewpoint_camera)
            if self.opt.render_novel_view:
                pc_j = pc
            else:
                pc_j = {key: value[j] for key, value in pc.items()}
            
                if j != 5:
                    mask = self.create_overlap_mask(int(viewpoint_camera['image_width']), int(viewpoint_camera['image_height']), viewpoint_camera['camera_center'], 100, 25, C2W[j], K[j][:3,:3], np.linalg.inv(C2W[j+1]), K[j+1][:3,:3])
                    # mask = self.create_overlap_mask(pc_j['get_xyz'], K[j][:3,:3], C2W[j], int(viewpoint_camera['image_height']), int(viewpoint_camera['image_width']))
                else:
                    mask = self.create_overlap_mask(int(viewpoint_camera['image_width']), int(viewpoint_camera['image_height']), viewpoint_camera['camera_center'], 100, 25, C2W[j], K[j][:3,:3], np.linalg.inv(C2W[0]), K[0][:3,:3])
                    # mask = self.create_overlap_mask(pc_j['get_xyz'], K[0][:3,:3], C2W[0], int(viewpoint_camera['image_height']), int(viewpoint_camera['image_width']))
                print(f'test_img/overlap_mask_{j}.png')
                cv2.imwrite(f'test_img/overlap_mask_{j}.png', mask.cpu().numpy() * 255)

            render_pkg = splatting_render(viewpoint_camera, pc_j, opt = self.opt)

            rgb_marched_i = render_pkg['render'].unsqueeze(0)


            rgb_spaltting.append(rgb_marched_i)


        # self.save_camera_info_to_ply(viewpoint_camera_list)
        exit()
        rgb_spaltting = torch.cat(rgb_spaltting, dim=0)

        # return depth, rgb_spaltting
        return rgb_spaltting


    def get_semantic_gt_loss(self, voxel_semantics, pred, mask_camera):

        # pdb.set_trace()

        preds = pred[0, ...].permute(1, 2, 3, 0) # 200, 200, 16, 18

        if mask_camera is not None:
            voxel_semantics=voxel_semantics.reshape(-1)
            preds=preds.reshape(-1,self.opt.semantic_classes)
            mask_camera=mask_camera.reshape(-1)
            num_total_samples=mask_camera.sum()

            loss_occ=self.loss_occ(preds, voxel_semantics, mask_camera, avg_factor = num_total_samples)

        else:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = preds.reshape(-1, self.num_classes)

            # ce loss
            loss_occ = self.loss_occ(preds, voxel_semantics)


        loss_voxel_sem_scal = sem_scal_loss(preds, voxel_semantics)
        loss_voxel_geo_scal = geo_scal_loss(preds, voxel_semantics, non_empty_idx=17)
        loss_voxel_lovasz = lovasz_softmax(torch.softmax(preds, dim=1), voxel_semantics)

        loss_geometry = loss_voxel_sem_scal + loss_voxel_geo_scal + loss_voxel_lovasz

        self.outputs[("loss_geometry", 0)] = loss_geometry

        self.outputs[("loss_gt_occ", 0)] = loss_occ * 100 + loss_geometry

        self.outputs["pred_occ_logits"] = pred

        self.outputs[('disp', 0)] = torch.ones(6, 1, self.opt.render_h, self.opt.render_w).to('cuda')

        return

    def forward(self, features, inputs, rot_map, scale_map, depth_map, opacity_map, point_cloud, epoch = 0, outputs={}, is_train=True, Voxel_feat_list=None, no_depth=False):

        self.outputs = outputs
        Voxel_feat, meta_similarity, meta_feature, nextcam2curego, feature_size = self.get_voxel(features, inputs)
        # if Voxel_feat_list is None:
        #     # 2D to 3D
        #     Voxel_feat, meta_similarity, meta_feature, nextcam2curego, feature_size = self.get_voxel(features, inputs)
        #     # 3D aggregation
        # Voxel_feat_list = self._3DCNN(Voxel_feat)

        # pdb.set_trace()
        # if self.opt.render_type == 'gt':
        #     preds = Voxel_feat_list[0]
        #     voxel_semantics = inputs['semantics_3d']
        #     mask_camera = inputs['mask_camera_3d']
        #     self.get_semantic_gt_loss(voxel_semantics, preds, mask_camera)
        #     return self.outputs


        # # rendering
        rendering_eps_time = time.time()
        cam_num = self.opt.cam_N * 3 if self.opt.auxiliary_frame else self.opt.cam_N

        for scale in self.opt.scales:

            eps_time = time.time()

            # depth, rgb_marched, semantic, reg_loss = self.get_density(is_train, inputs, rot_map, scale_map, depth_map, opacity_map, cam_num)
            depth, rgb_marched = self.get_density(is_train, inputs, rot_map, scale_map, depth_map, opacity_map, point_cloud, cam_num)
            # print("depth shape", depth[0].shape, len(depth))
            # print("semantic shape", semantic[0].shape, len(semantic))

            #------test----------
            # import imageio
            # # rgb_marched_test = semantic[0].permute(0, 2, 3, 1)
            # rgb_marched_test_split = torch.chunk(semantic[0], 6, dim=-1)
            # for i, t in enumerate(rgb_marched_test_split):
            #     # Chuyển tensor thành numpy array
            #     img_array = t.cpu().numpy()   # Bỏ chiều batch size (1) đi
            #     print("img_array shape", img_array.shape)
            #     # Scale giá trị từ [-1, 1] hoặc [0, 1] sang [0, 255] nếu cần thiết
            #     img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255
            #     img_array = img_array.astype(np.uint8)

            #     # Lưu ảnh dưới dạng file .png
            #     imageio.imwrite(f'/home/vinai/Workspace/phat-intern-dev/VinAI/GaussianOcc/test/image_{i}.png', img_array)
            # # print("rgb_marched shape", rgb_marched[0].shape)
            # exit()
            #---------------------------------------------------------------------
            eps_time = time.time() - eps_time

            # print('single rendering {} :(eps time:'.format(self.opt.render_type), eps_time, 'secs)')
            self.outputs["rgb_marched"] = rgb_marched
            # print("rgb_marched", rgb_marched)

            self.outputs[("disp", scale)] = depth



        self.outputs['render_time'] = time.time() - rendering_eps_time

        return self.outputs


def total_variation(v, mask=None):

    # pdb.set_trace()
    tv2 = v.diff(dim=2).abs()
    tv3 = v.diff(dim=3).abs()
    tv4 = v.diff(dim=4).abs()
    # if mask is not None:
    #     tv2 = tv2[mask[:,:,:-1] & mask[:,:,1:]]
    #     tv3 = tv3[mask[:,:,:,:-1] & mask[:,:,:,1:]]
    #     tv4 = tv4[mask[:,:,:,:,:-1] & mask[:,:,:,:,1:]]
    return (tv2.mean() + tv3.mean() + tv4.mean()) / 3

