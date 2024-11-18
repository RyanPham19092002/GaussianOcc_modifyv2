# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import os
import os.path as osp
import json
import time
import imageio

import cv2
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler as _DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import pickle
import datasets
import networks
from options import MonodepthOptions
from utils.loss_metric import *
from utils.layers import *
from nerfstudio.cameras.camera_utils import get_interpolated_poses
from PIL import Image

from configs.config import ConfigStereoHuman as config
from UniDepth.unidepth.models import UniDepthV1
import utils.basic as basic
import datetime, pytz

# local_rank = int(os.environ.get('LOCAL_RANK', 0))
def get_dist_info(return_gpu_per_machine=False):
    if torch.__version__ < '1.0':
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    if return_gpu_per_machine:
        gpu_per_machine = torch.cuda.device_count()
        return rank, world_size, gpu_per_machine

    return rank, world_size


class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices = indices[self.rank:self.total_size:self.num_replicas]

        return iter(indices)


class Runer:

    def __init__(self, options):
        self.cfg = config()
        self.cfg.load("configs/stage1.yaml")
        self.cfg = self.cfg.get_cfg()

        self.cfg.defrost()
        # dt = datetime.today()
        self.cfg.exp_name = 'exp_{}'.format(datetime.datetime.now(tz=pytz.timezone('Asia/Tokyo')).strftime("%Y_%m_%d_%H_%M_%S"))
        self.cfg.record.ckpt_path = "experiments/%s/ckpt" % self.cfg.exp_name
        self.cfg.record.show_path = "experiments/%s/show" % self.cfg.exp_name
        self.cfg.record.logs_path = "experiments/%s/logs" % self.cfg.exp_name
        self.cfg.record.file_path = "experiments/%s/file" % self.cfg.exp_name
        self.cfg.freeze()

        self.opt = options
        # print("self.opt\n", self.opt)
        self.opt.B = self.opt.batch_size // 6
        if self.opt.debug:
            self.opt.voxels_size = [8, 128, 128]
            self.opt.render_h = 45
            self.opt.render_w = 80
            self.opt.num_workers = 1
            self.opt.model_name = "debug/"
        

        from pathlib import Path
        path = Path(self.opt.load_weights_folder)
        new_path = path.parents[1]

        self.log_path =  new_path

        # print('log path:', self.log_path)
        os.makedirs(osp.join(self.log_path, 'eval'), exist_ok=True)
        os.makedirs(osp.join(self.log_path, 'models'), exist_ok=True)
        os.makedirs(osp.join(self.log_path, 'visual_rgb_depth'), exist_ok=True)
        os.makedirs(osp.join(self.log_path, 'visual_feature'), exist_ok=True)
        os.makedirs(osp.join(self.log_path, 'scene_video'), exist_ok=True)

        self.models = {}
        
        self.local_rank = self.opt.local_rank
        torch.cuda.set_device(self.local_rank)
        dist.init_process_group(backend='nccl')
        self.device = torch.device("cuda", self.local_rank)

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        if self.opt.render_novel_view:
            print("Mode:=========================Render Novel view:=========================")
        else:
            print("Mode:=========================Render Input view:=========================")

        self.models["encoder"] = networks.Encoder_res101(self.opt.input_channel, path=None, network_type=self.opt.encoder)
        self.models["render_img"] = networks.VolumeDecoder(self.opt)
        self.models["depth_2d"] = UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-vitl14") 
        self.models["decoder"] = networks.GSRegresser(self.cfg, rgb_dim=3, depth_dim=1)
        
        self.opt.models_to_load = ['encoder', 'render_img', 'depth_2d', 'decoder']
        # self.models["depth"] = networks.VolumeDecoder(self.opt)
        # self.models["rgb"] = networks.VolumeDecoder(self.opt)
        # print("networks.VolumeDecoder(self.opt)", networks.VolumeDecoder(self.opt))
        # exit()
        # bổ sung ------------------------------------------------
        # self.models["pose_encoder"] = networks.ResnetEncoder(
        # 34, True, num_input_images=self.num_pose_frames)

        # self.models["pose_encoder"] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.models["pose_encoder"])
        # self.models["pose_encoder"] = self.models["pose_encoder"].to(self.device)
        # # self.parameters_to_train += [{'params': self.models["pose_encoder"].parameters(), 'lr': self.opt.learning_rate}]

        # self.models["pose"] = networks.PoseDecoder(
        #     self.models["pose_encoder"].num_ch_enc,
        #     num_input_features=1, num_frames_to_predict_for=2)

        # self.models["pose"] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.models["pose"])
        # self.models["pose"] = (self.models["pose"]).to(self.device)
        # # self.parameters_to_train += [{'params': self.models["pose"].parameters(), 'lr': self.opt.learning_rate}]
        #------------------------------------------------------------------------
        # self.log_print('N_samples: {}'.format(self.models["depth"].N_samples))
        # self.log_print('Voxel size: {}'.format(self.models["depth"].voxel_size))

        self.models["encoder"] = self.models["encoder"].to(self.device)
        self.models["render_img"] = self.models["render_img"].to(self.device)
        self.models["depth_2d"] = self.models["depth_2d"].to(self.device)
        self.models["decoder"] = self.models["decoder"].to(self.device)
        # self.models["depth"] = self.models["depth"].to(self.device)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        for key in self.models.keys():
            # self.models[key] = DDP(self.models[key], device_ids=[self.local_rank], output_device=self.local_rank,
            #                        find_unused_parameters=True, broadcast_buffers=False)
            if key != "depth_2d":  # Bỏ qua DDP cho "depth_2d"
                self.models[key] = DDP(self.models[key], device_ids=[self.local_rank], output_device=self.local_rank,
                                    find_unused_parameters=True, broadcast_buffers=False)
            else:
                self.models[key].infer = self.models["depth_2d"].infer
        
        if self.local_rank == 0:
            self.log_print("Training model named: {}".format(self.opt.model_name))

        datasets_dict = {
            "ddad": datasets.DDADDataset,
            "nusc": datasets.NuscDatasetVis}

        self.dataset = datasets_dict[self.opt.dataset]

        self.opt.batch_size = self.opt.batch_size // 6

        val_dataset = self.dataset(self.opt,
                                   self.opt.height, self.opt.width,
                                   [0], num_scales=1, is_train=False,  # the first is frame_ids
                                   volume_depth=self.opt.volume_depth)
        rank, world_size = get_dist_info()
        self.world_size = world_size
        val_sampler = DistributedSampler(val_dataset, world_size, rank, shuffle=False)

        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, collate_fn=self.my_collate,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=False, sampler=val_sampler)

        self.num_val = len(val_dataset)

        self.opt.batch_size = self.opt.batch_size * 6
        self.num_val = self.num_val * 6

        self.save_opts()
        #--------------------------------------test----------------------------------------------------------------------
        # train_dataset = self.dataset(self.opt,
        #                              self.opt.height, self.opt.width,
        #                              self.opt.frame_ids, num_scales=self.num_scales, is_train=True,
        #                              volume_depth=self.opt.volume_depth)

        # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        # self.train_loader = DataLoader(
        #     train_dataset, self.opt.batch_size, collate_fn=self.my_collate,
        #     num_workers=self.opt.num_workers, pin_memory=True, drop_last=True, sampler=train_sampler)

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                if scale == 0:
                    outputs[("disp", scale)] = disp
                source_scale = 0

            if self.opt.volume_depth:
                depth = disp
            else:
                depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth, abs=False)
            
            outputs[("depth", 0, scale)] = depth


            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if self.opt.gt_pose != 'No':
                    T =  outputs[("cam_T_cam", 0, frame_id)]
                    # print("T\n", T)
                
                else:
                    T = inputs[("cam_T_cam", frame_id)]

                
                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", 0, source_scale)])
                
                # print("cam_points\n", cam_points)
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", frame_id, source_scale)], T)

                # print("pix_coords\n", pix_coords)
                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border", align_corners=True)

                
                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]
                # print("output after \n", outputs)
                # exit()
    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}

        pose_feats = {f_i: inputs["color_aug", 0, f_i] for f_i in [0, -1]}

        for f_i in self.opt.frame_ids[1:]:

            # To maintain ordering we always pass frames in temporal order
            # if f_i < 0:
            pose_inputs = [pose_feats[-1], pose_feats[0]]
            # else:
                # pose_inputs = [pose_feats[0], pose_feats[f_i]]
            # print("pose_inputs", pose_inputs)
            pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]

            coarse_RT = None

            axisangle, translation = self.models["pose"](pose_inputs, joint_pose=True, coarse_RT = coarse_RT)
            
            if self.opt.detach_pose != 'No':
                axisangle = axisangle.detach()
                translation = translation.detach()

            outputs[("axisangle", 0, f_i)] = axisangle
            outputs[("translation", 0, f_i)] = translation

            # Invert the matrix if the frame id is negative
            trans = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
            trans = trans.unsqueeze(1).repeat(1, 6, 1, 1).reshape(-1, 4, 4) # torch.Size([6, 4, 4]

            # pdb.set_trace()
            outputs[("cam_T_cam", 0, f_i)] = torch.linalg.inv(inputs["pose_spatial"]) @ trans @ inputs["pose_spatial"]

        return outputs
    def my_collate(self, batch):
        batch_new = {}
        keys_list = list(batch[0].keys())
        special_key_list = ['id', 'scene_name', 'frame_idx']

        for key in keys_list:
            if key not in special_key_list:
                # print('key:', key)
                batch_new[key] = [item[key] for item in batch]
                try:
                    batch_new[key] = torch.cat(batch_new[key], axis=0)
                except:
                    print('key', key)

            else:
                batch_new[key] = []
                for item in batch:
                    for value in item[key]:
                        # print(value.shape)
                        batch_new[key].append(value)

        return batch_new

    def to_device(self, inputs):

        special_key_list = ['id', ('K_ori', -1), ('K_ori', 1), 'scene_name', 'frame_idx']

        for key, ipt in inputs.items():

            if key in special_key_list:
                inputs[key] = ipt

            else:
                inputs[key] = ipt.to(self.device)

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()
    def depth_to_point_cloud(self, depth_map, K, cam2world, rgb_image=None):
        """
        Tạo point cloud từ depth map và ma trận intrinsic của camera.
        
        depth_map: depth map của ảnh (HxW)
        K: ma trận intrinsic của camera (3x3)
        cam2world: ma trận cam2world (4x4)
        rgb_image: ảnh RGB tương ứng (HxWx3), tùy chọn nếu muốn lưu màu
        
        Returns:
            points_world: point cloud (N x 3)
            colors: (N x 3) nếu có rgb_image, ngược lại trả về None
        """
        # print("depth_map", depth_map)
        # print("K", K)
        # print("cam2w", cam2world)
        # print("rgb", rgb_image)
        h, w = depth_map.shape
        i, j = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')

        # Chuyển từ pixel tọa độ sang không gian camera
        z = depth_map.flatten()
        x = (i.flatten() - K[0, 2]) * z / K[0, 0]
        y = (j.flatten() - K[1, 2]) * z / K[1, 1]
        
        points_camera = np.vstack((x, y, z)).T  # (N x 3)

        # Thêm tọa độ 1 để chuyển thành tọa độ đồng nhất
        points_camera_h = np.hstack([points_camera, np.ones((points_camera.shape[0], 1))])

        # Chuyển về không gian thế giới
        points_world_h = points_camera_h @ cam2world.T
        points_world = points_world_h[:, :3]

        # Lấy màu nếu có RGB image
        if rgb_image is not None:
            colors = rgb_image.reshape(-1, 3) / 255.0
            return points_world, colors
        else:
            return points_world, None

    def warp_point_cloud(self, points_src, T_src, T_tgt):
        """
        Warp các điểm từ không gian camera nguồn về không gian camera đích.
        
        points_src: Điểm 3D trong không gian thế giới (N x 3)
        T_src: Ma trận cam2world của camera nguồn (4x4)
        T_tgt: Ma trận cam2world của camera đích (4x4)
        
        Returns:
            points_tgt: Điểm đã warp trong không gian thế giới (N x 3)
        """
        # Chuyển về không gian camera nguồn
        points_src_h = np.hstack([points_src, np.ones((points_src.shape[0], 1))])
        
        # Chuyển từ thế giới sang không gian camera đích
        points_tgt_h = points_src_h @ np.linalg.inv(T_src).T
        points_tgt_h = points_tgt_h @ T_tgt.T
        
        points_tgt = points_tgt_h[:, :3]
        return points_tgt
    def render_from_point_cloud(self, points_world, colors, K_new, cam2world_new, img_width, img_height):
        """
        Render ảnh từ point cloud với ma trận cam2world mới.

        points_world: point cloud (N x 3)
        colors: Màu tương ứng với các điểm (N x 3)
        K_new: Ma trận intrinsic của view mới (3x3)
        cam2world_new: Ma trận cam2world của view mới (4x4)
        img_width: Chiều rộng ảnh đầu ra
        img_height: Chiều cao ảnh đầu ra

        Returns:
            img_rendered: Ảnh đã được render (HxWx3)
        """
        img_rendered = np.zeros((img_height, img_width, 3), dtype=np.float32)
        
        # Chuyển các điểm từ không gian thế giới về không gian camera mới
        points_world_h = np.hstack([points_world, np.ones((points_world.shape[0], 1))])
        points_camera_new_h = points_world_h @ np.linalg.inv(cam2world_new).T
        points_camera_new = points_camera_new_h[:, :3]

        # Chuyển các điểm từ không gian camera về không gian ảnh (pixel)
        points_2d_h = points_camera_new @ K_new.T
        points_2d = points_2d_h[:, :2] / points_2d_h[:, 2:]

        # Tạo danh sách lưu trữ độ sâu và màu sắc tại mỗi pixel
        depth_list = [[[] for _ in range(img_width)] for _ in range(img_height)]
        color_list = [[[] for _ in range(img_width)] for _ in range(img_height)]

        # Lưu độ sâu và màu sắc cho từng điểm
        for i, (x, y) in enumerate(points_2d):
            x = int(np.round(x))
            y = int(np.round(y))
            if 0 <= x < img_width and 0 <= y < img_height:
                depth_value = points_camera_new[i, 2]  # Độ sâu
                depth_list[y][x].append(depth_value)
                color_list[y][x].append(colors[i] * 255)  # Màu sắc

        # Tạo ảnh render từ danh sách độ sâu và màu sắc
        for y in range(img_height):
            for x in range(img_width):
                if depth_list[y][x]:
                    # Chọn điểm có độ sâu lớn nhất (xa nhất)
                    max_depth_idx = np.argmax(depth_list[y][x])
                    img_rendered[y, x] = color_list[y][x][max_depth_idx] / 255.0  # Chuyển màu sắc về [0, 1]

        return img_rendered
    
    def c2m_NVS(self, pose_a, pose_b, steps):
        row = np.array([0, 0, 0, 1])
        pose_ab = get_interpolated_poses(pose_a, pose_b, steps=steps)[1]
        pose_ab = np.vstack([pose_ab, row])
        pose_ab = torch.tensor(pose_ab, dtype=torch.float32).to('cuda')
        return pose_ab

    def val(self, save_image=True):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        print('begin eval!')
        total_time = []
        total_evl_time = time.time()

        with torch.no_grad():
            loader = self.val_loader
            # loader = self.train_loader      #test
            for idx, data in enumerate(loader):
                # print("data before \n", data.keys())
                # print("dataa K\n", data[('K', 0, 0)])
                # print("data inv_K\n", data[('inv_K', 0, 0)])
                K_render = data[('K', 0, 0)]
                pose_spatial = data['pose_spatial']
   
                
                
                scene_name = data['scene_name'][0]
                frame_idx = data['frame_idx'][0]
                
                if 'ddad' in self.opt.config:
                    # ddad
                    scene_list = ['000159', '000180', '000188', '000194', '000195']
                else:
                    # nuscenes
                    # scene_list = ['scene-0103', 'scene-0553', 'scene-0796','scene-0916']
                    scene_list = ['scene-0553']
                # if scene_name in scene_list and frame_idx < 100:
                if scene_name in scene_list and frame_idx < 20:
                    
                    print("frame_idx: ",frame_idx)
                    eps_time = time.time()

                    input_color = data[("color", 0, 0)].cuda()

                    camera_ids = data["id"]

                    features = self.models["encoder"](input_color)

                    device = torch.device('cuda:0')
                    features = [feat.float().to(device) for feat in features]
                    features_tensor = features

                    predictions_depth = self.models["depth_2d"].infer(input_color)
                    depth_2d = predictions_depth["depth"]
                    point_cloud = predictions_depth["points"]

                    rot_maps, scale_maps, opacity_maps = self.models["decoder"](input_color, depth_2d, features_tensor)
                    
                    output = self.models["render_img"](features, data, rot_maps, scale_maps, depth_2d, opacity_maps, point_cloud, epoch = 0)
                    # output = self.models["depth"](features, data, is_train=False, no_depth=self.opt.use_semantic)
                    
                    # print("output before \n", output.keys())
                    rgb_marched = output['rgb_marched']
                    # print("rgb_marched shape", rgb_marched[0].shape, len(rgb_marched), rgb_marched)
                    # exit()
                    # semantic = output[('semantic', 0)]
                    # print("semantic", semantic[0].shape, len(semantic)) 
                    # print("depth", output[('disp', 0)].shape)
                    
                    # print("data before \n", data.keys())
                    # print("model", self.models.keys())
                    # exit()
                    # output_1 = self.predict_poses(data, features)
                    # print("output_1", output_1)
                    # exit()
                    # print("K_render", data[('K_render', 0, 0)], data[('K_render', 0, 0)].shape)
                    # print("\n")
                    # print("K", data[('K', 0, 0)], data[('K', 0, 0)].shape)
                    # print("\n")
                    # print("inv_K", data[('inv_K', 0, 0)], data[('inv_K', 0, 0)].shape)
                    
                    # print("\n")
                    # print("pose_spatial", data['pose_spatial'], data["pose_spatial"].shape)
                    #render NVS ---------------------------------------
                    # list_pose_spatial = data['pose_spatial']
                    # input_NVS = data
                    # # self.generate_images_pred(data, output)
                    # list_pose_NVS = []
                    
                    # for i in range (len(list_pose_spatial)):
                    #     pose_NVS = np.eye(4)
                    #     if i == len(list_pose_spatial) - 1:
                    #         pose_B = list_pose_spatial[0]
                    #     else:
                    #         pose_B = list_pose_spatial[i+1]
                    #     pose_A = list_pose_spatial[i]
                    #     interpolated_poses = get_interpolated_poses(pose_A, pose_B, 15)
                    #     # print("interpolated_poses", interpolated_poses)
                    #     pose_NVS[:3, :] = torch.tensor(interpolated_poses[11]).double()
                    #     list_pose_NVS.append(pose_NVS)
                    # tensor_pose_NVS = torch.tensor(np.array(list_pose_NVS))
                    # # print("tensor_pose_NVS", tensor_pose_NVS, tensor_pose_NVS.shape)
                    # # exit()
                    # input_NVS['pose_spatial'] = torch.tensor(np.array(list_pose_NVS)).to(torch.float32)
                    # # print("input_NVS", input_NVS['pose_spatial'], input_NVS['pose_spatial'].shape)
                    # # print("data", data['pose_spatial'])
                    # print("models key()", self.models.keys())
                    # output = self.models["depth"](features, input_NVS, is_train=False, no_depth=self.opt.use_semantic)
                    #------------------------------------------------------------------------------------------------
                    eps_time = time.time() - eps_time
                    total_time.append(eps_time)

                    if self.local_rank == 0 and idx % 100 == 0:
                        print('single inference:(eps time:', eps_time, 'secs)')

                    # if not self.opt.use_semantic:
                    pred_depths = output[("disp", 0)].cpu()[:, 0].numpy()
                    # pred_depths = output["depth_predict"].cpu()[:, 0].numpy()
                    # print("rgb_marched shape", rgb_marched.shape)
                    # print("pred_depths shape", pred_depths.shape)
                    # exit()
                    # pred_semantic = output[('semantic', 0)].cpu().numpy()
                    # print("pred depths shape",pred_depths.shape)
                    # print("pred_semantic shape", pred_semantic.shape)
                    # exit()
                    concated_image_list = []
                    concated_depth_list = []
                    concated_pred_list = []
                    points_world_list = []
                    colors_list = []

                    for i in range(input_color.shape[0]):
                        # # semantic_part = semantic[i].view(180, 320, 3, 6).sum(dim=-1).cpu().numpy() / 6
                        # # semantic_part = torch.sum(semantic[i], dim=-1, keepdim=True).cpu().numpy()
                        # rgb_pred = (pred_semantic[i])
                        # # print("semantic_part shape", rgb_pred)
                        # # exit()
                        # semantic_part = rgb_pred[..., [2,1,0]]
                        
                        # min_val = semantic_part.min()
                        # max_val = semantic_part.max()
                        # normalized = (semantic_part - min_val) / (max_val - min_val)
                        # semantic_part = (normalized * 255).astype('uint8')
                        # print("semantic_part shape", semantic_part, semantic_part.shape)
                        # concated_semantic_list.append(semantic_part)
                        color_pred = (rgb_marched[i].cpu().permute(1, 2, 0).numpy())
                        # print("color_pred", color_pred)
                        color_pred = 255 * (color_pred - color_pred.min()) / (color_pred.max() - color_pred.min())
                        color_pred = np.clip(color_pred, 0, 255).astype(np.uint8)
                        color_pred_bgr = cv2.cvtColor(color_pred, cv2.COLOR_RGB2BGR)
                        concated_pred_list.append(cv2.resize(color_pred_bgr.copy(), (320, 180)))
                        # semantic_part = color_pred[..., [2, 1, 0]]
                        # # semantic_part = color_pred
                        # concated_semantic_list.append(semantic_part)
                        # print("semantic_part shape", semantic_part, semantic_part.shape)
                        camera_id = camera_ids[i]

                        color = (input_color[i].cpu().permute(1, 2, 0).numpy()) * 255
                        # print("color shape", color.shape)
                        color = color[..., [2, 1, 0]]
                        # print("color shape af", color, color.shape)
                        # exit()
                        concated_image_list.append(cv2.resize(color.copy(), (320, 180)))
                        
                        # if not self.opt.use_semantic:
                        pred_depth = pred_depths[i]
                        # print("pred_depth shape", pred_depth.shape)
                        # print("pred_depth", pred_depth.shape, pred_depth)
                        
                        pred_depth_color = visualize_depth(pred_depth.copy())
                        concated_depth_list.append(cv2.resize(pred_depth_color.copy(), (320, 180)))
                        #-------------------------------creat Pcloud ---------------------------------------
                        # points, colors = self.depth_to_point_cloud(pred_depth, K_render[i].cpu().numpy(), \
                        #                         pose_spatial[i].cpu().numpy(), cv2.resize(color.copy(), (320, 180)))
                        # # if i != 0:  # 1 là camera "front"
                        # #     points = self.warp_point_cloud(points, pose_spatial[i].cpu().numpy(), pose_spatial[0].cpu().numpy())  # Warp về không gian "front"
                        # points_world_list.append(points)
                        # colors_list.append(colors)

                        


                    # # Hợp nhất tất cả các point cloud
                    # points_world = np.vstack(points_world_list)
                    # colors = np.vstack(colors_list)
                    # pose_lf = self.c2m_NVS(pose_spatial[1], pose_spatial[0], steps=3)
                    # print("pose_lf shape", pose_lf.shape, pose_lf)
                    # img_rendered = self.render_from_point_cloud(points_world, colors, K_render[0][:3, :3].cpu().numpy(), pose_lf.cpu().numpy(), 320, 180)
                    # print("img_rendered", img_rendered, img_rendered.shape)
                    # print("Kiểu dữ liệu của ảnh:", img_rendered.dtype)

                    # # Chuyển đổi kiểu dữ liệu nếu cần
                    # if img_rendered.dtype == np.float32 or img_rendered.dtype == np.float64:
                    #     # Giả sử giá trị pixel nằm trong phạm vi [0, 1], chuyển đổi sang [0, 255]
                    #     img_rendered = (img_rendered * 255).astype(np.uint8)

                    # # Chuyển đổi NumPy array thành image PIL
                    # # img_pil = Image.fromarray(img_rendered, 'BGR')
                    # # imageio.imwrite('rendered_image.png', img_rendered)
                    # # # Lưu ảnh ra file
                    # # img_pil.save('test10-2-RGB.png')
                    # cv2.imwrite('test10-2-RGB.png', img_rendered)
                    # print("Lưu thành côn")
                    # exit()
                    #"-------------------------semantic--------------------------"
                    # semantic_left_front_right = np.concatenate(
                    #     (concated_semantic_list[1], concated_semantic_list[0], concated_semantic_list[5]), axis=1)
                    # semantic_left_rear_right = np.concatenate(
                    #     (concated_semantic_list[2], concated_semantic_list[3], concated_semantic_list[4]), axis=1)

                    image_left_front_right_NVS = np.concatenate(
                        (concated_pred_list[1], concated_pred_list[0], concated_pred_list[5]), axis=1)
                    image_left_rear_right_NVS = np.concatenate(
                        (concated_pred_list[2], concated_pred_list[3], concated_pred_list[4]), axis=1)

                    image_left_front_right = np.concatenate(
                        (concated_image_list[1], concated_image_list[0], concated_image_list[5]), axis=1)
                    image_left_rear_right = np.concatenate(
                        (concated_image_list[2], concated_image_list[3], concated_image_list[4]), axis=1)
                    # image_surround_view = np.concatenate((image_left_front_right, image_left_rear_right), axis=0)


                    surround_depth_view_up = np.concatenate(
                            (concated_depth_list[1], concated_depth_list[0], concated_depth_list[5]), axis=1)
                    surround_depth_view_down = np.concatenate(
                            (concated_depth_list[2], concated_depth_list[3], concated_depth_list[4]), axis=1)

                    if not self.opt.use_semantic:
                        surround_view_up = np.concatenate((image_left_front_right, image_left_front_right_NVS, surround_depth_view_up), axis=0)
                        surround_view_down = np.concatenate((image_left_rear_right, image_left_rear_right_NVS, surround_depth_view_down), axis=0)
                    
                    else:
                        surround_view_up = image_left_front_right
                        surround_view_down = image_left_rear_right
                        #-----------------------------semantic--------------------------------------
                    # surround_view_up_semantic = semantic_left_front_right
                    # surround_view_down_semantic = semantic_left_rear_right
                    
                    # pdb.set_trace()
                    scene_name = data['scene_name'][0]
                    frame_idx = data['frame_idx'][0]
                    # print("create folder scene_video")
                    os.makedirs('{}/scene_video/{}'.format(self.log_path, scene_name), exist_ok=True)
                    cv2.imwrite('{}/scene_video/{}/{:03d}-up.jpg'.format(self.log_path, scene_name, frame_idx), surround_view_up)
                    cv2.imwrite('{}/scene_video/{}/{:03d}-down.jpg'.format(self.log_path, scene_name, frame_idx), surround_view_down)
                    #-------------------------semantic--------------------------
                    # cv2.imwrite('{}/scene_video/{}/{:03d}-up-semantic.jpg'.format(self.log_path, scene_name, frame_idx), surround_view_up_semantic)
                    # cv2.imwrite('{}/scene_video/{}/{:03d}-down-semantic.jpg'.format(self.log_path, scene_name, frame_idx), surround_view_down_semantic)
                    #-----------------------------------------------------------
                    cv2.imwrite('{}/scene_video/{}/{:03d}-depth-up.jpg'.format(self.log_path, scene_name, frame_idx), surround_depth_view_up)
                    cv2.imwrite('{}/scene_video/{}/{:03d}-depth-down.jpg'.format(self.log_path, scene_name, frame_idx), surround_depth_view_down)
                    #NVS-----------------------------------------------------------
                    cv2.imwrite('{}/scene_video/{}/{:03d}-NVS-up.jpg'.format(self.log_path, scene_name, frame_idx), image_left_front_right_NVS)
                    cv2.imwrite('{}/scene_video/{}/{:03d}-NVS-down.jpg'.format(self.log_path, scene_name, frame_idx), image_left_rear_right_NVS)
                    vis_dic = {}
                    # vis_dic['opt'] = self.opt
                    # vis_dic['depth_color'] = concated_depth_list
                    # vis_dic['rgb'] = concated_image_list
                    # vis_dic['pose_spatial'] = data['pose_spatial'].detach().cpu().numpy()   #origin
                    # # vis_dic['pose_spatial'] = input_NVS['pose_spatial'].detach().cpu().numpy() 
                    # # vis_dic['probability'] = output['pred_occ_logits'].detach().cpu().numpy()
                    # np.save('{}/scene_video/{}/{:03d}-out.npy'.format(self.log_path, scene_name, frame_idx), vis_dic)
                    # exit()
                    # print("write finished")
        eps_time = time.time() - total_evl_time

        print('finish eval!')

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = osp.join(self.log_path, "models")
        if not osp.exists(models_dir):
            os.makedirs(models_dir)
        os.makedirs(osp.join(self.log_path, "eval"), exist_ok=True)
        to_save = self.opt.__dict__.copy()

        with open(osp.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = osp.expanduser(self.opt.load_weights_folder)

        if self.local_rank == 0:
            assert osp.isdir(self.opt.load_weights_folder), \
                "Cannot find folder {}".format(self.opt.load_weights_folder)
            self.log_print("loading model from folder {}".format(self.opt.load_weights_folder))
        
        
        for n in self.opt.models_to_load:

            if self.local_rank == 0:
                self.log_print("Loading {} weights...".format(n))
            path = osp.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path, map_location=torch.device('cpu'))
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

    def log_print(self, str):
        print(str)
        with open(osp.join(self.log_path, 'log.txt'), 'a') as f:
            f.writelines(str + '\n')

    def log_print_train(self, str):
        print(str)
        with open(osp.join(self.log_path, 'log_train.txt'), 'a') as f:
            f.writelines(str + '\n')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    options = MonodepthOptions()
    opts = options.parse()
    setup_seed(42)
    runner = Runer(opts)
    runner.val()
