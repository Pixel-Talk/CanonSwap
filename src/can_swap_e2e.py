import numpy as np
import torch
import torch.nn.functional as F
import imageio

import os
import os.path as osp
from skimage.draw import disk

import matplotlib.pyplot as plt
import collections
# from facevid2vid.distributed import master_only, master_only_print, get_rank, is_master
# from facevid2vid.models import Discriminator
# from facevid2vid.trainer import GeneratorFull3 as GeneratorFull
# from facevid2vid.trainer import DiscriminatorFull2 as DiscriminatorFull
# from facevid2vid.models import MultiScaleDiscriminator_wokp as Discriminator
# from facevid2vid.trainer import DiscriminatorFull_wokp_multi as DiscriminatorFull
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

import yaml

from src.modules.spade_generator import SPADEDecoder
from src.modules.warping_network import WarpingNetwork
from src.modules.motion_extractor import MotionExtractor
from src.modules.appearance_feature_extractor import AppearanceFeatureExtractor
from src.modules.stitching_retargeting_network import StitchingRetargetingNetwork
from src.utils.helper import remove_ddp_dumplicate_key, add_ddp_prefix

# from src.modules.modulate import transfer_model
from src.modules.adaptive_modulate import transfer_model
from src.modules.adaptive_modulate import transfer_model2 as transfer_model_big
from src.modules.adaptive_modulate import G3d
# from src.modules.resnet_adain import transfer_model
# from src.modules.distangle import AttributeEncoder, IdentityEncoder, Decoder

from .utils.timer import Timer
from .utils.helper import load_model, concat_feat
from .utils.camera import headpose_pred_to_degree, get_rotation_matrix
from .utils.retargeting_utils import calc_eye_close_ratio, calc_lip_close_ratio
from .config.inference_config import InferenceConfig
from .utils.rprint import rlog as log
import contextlib
import cv2
def to_cpu(losses):
    return {key: value.detach().data.cpu().numpy() for key, value in losses.items()}


class can_swapper(object):
    """
    Wrapper for Human
    """

    def __init__(self, inference_cfg: InferenceConfig):

        self.inference_cfg = inference_cfg
        self.device_id = inference_cfg.device_id
        self.compile = inference_cfg.flag_do_torch_compile
        if inference_cfg.flag_force_cpu:
            self.device = 'cpu'
        else:
            try:
                if torch.backends.mps.is_available():
                    self.device = 'mps'
                else:
                    self.device = 'cuda:' + str(self.device_id)
            except:
                self.device = 'cuda:' + str(self.device_id)

        model_config = yaml.load(open(inference_cfg.models_config, 'r'), Loader=yaml.SafeLoader)['model_params']

        model_config['spade_generator_params']['upscale'] = 2
        self.appearance_feature_extractor = AppearanceFeatureExtractor(**model_config['appearance_feature_extractor_params']).to(self.device).eval()
        self.motion_extractor = MotionExtractor(**model_config['motion_extractor_params']).to(self.device).eval()
        self.warping_module = WarpingNetwork(**model_config['warping_module_params']).to(self.device).eval()
        self.spade_generator = SPADEDecoder(**model_config['spade_generator_params']).to(self.device).eval()
        self.swap_module = transfer_model_big().to(self.device).eval()
        self.refine_module = G3d().to(self.device).eval()
        # self.swap_module = transfer_model().to(self.device).eval()
        # self.load_init_models()
        self.load_cpk()

        # Optimize for inference
        if self.compile:
            torch._dynamo.config.suppress_errors = True  # Suppress errors and fall back to eager execution
            self.warping_module = torch.compile(self.warping_module, mode='max-autotune')
            self.spade_generator = torch.compile(self.spade_generator, mode='max-autotune')

        self.timer = Timer()

        #加载ID extractor
        netArc = "pretrained_weights/arcface_checkpoint.tar"
        self.netArc = torch.load(netArc, map_location=torch.device("cpu"))
        self.netArc.cuda()
        self.netArc.eval()

    def load_cpk(self):
        # 优先尝试加载合并权重文件
        combined_weights_path = "pretrained_weights/combined_weights.pth"

        if os.path.exists(combined_weights_path):
            print(f"loading weights  from path: {combined_weights_path}")
            combined_weights = torch.load(combined_weights_path, map_location=torch.device("cpu"))
            self.appearance_feature_extractor.load_state_dict(combined_weights['appearance_feature_extractor'])
            self.motion_extractor.load_state_dict(combined_weights['motion_extractor'])
            self.warping_module.load_state_dict(combined_weights['warping_module'])
            self.spade_generator.load_state_dict(combined_weights['spade_generator'])
            self.swap_module.load_state_dict(combined_weights['transfer'])
            self.refine_module.load_state_dict(combined_weights['refine'])

            print("model load success！")

    def getid(self, img):
        # img = self.transform(img)
        img = F.interpolate(img, size=(112,112))
        id, _ = self.netArc(img)
        id = F.normalize(id, p=2, dim=1)
        return id

    def swap(self, feature_3d, source_id):
        swap_feature = self.swap_module(feature_3d, source_id)
        return swap_feature

    def inference_ctx(self):
        if self.device == "mps":
            ctx = contextlib.nullcontext()
        else:
            ctx = torch.autocast(device_type=self.device[:4], dtype=torch.float16,
                                 enabled=self.inference_cfg.flag_use_half_precision)
        return ctx

    def update_config(self, user_args):
        for k, v in user_args.items():
            if hasattr(self.inference_cfg, k):
                setattr(self.inference_cfg, k, v)

    def prepare_source(self, img: np.ndarray) -> torch.Tensor:
        """ construct the input as standard
        img: HxWx3, uint8, 256x256
        """
        h, w = img.shape[:2]
        if h != self.inference_cfg.input_shape[0] or w != self.inference_cfg.input_shape[1]:
            x = cv2.resize(img, (self.inference_cfg.input_shape[0], self.inference_cfg.input_shape[1]))
        else:
            x = img.copy()

        if x.ndim == 3:
            x = x[np.newaxis].astype(np.float32) / 255.  # HxWx3 -> 1xHxWx3, normalized to 0~1
        elif x.ndim == 4:
            x = x.astype(np.float32) / 255.  # BxHxWx3, normalized to 0~1
        else:
            raise ValueError(f'img ndim should be 3 or 4: {x.ndim}')
        x = np.clip(x, 0, 1)  # clip to 0~1
        x = torch.from_numpy(x).permute(0, 3, 1, 2)  # 1xHxWx3 -> 1x3xHxW
        x = x.to(self.device)
        return x

    def prepare_videos(self, imgs) -> torch.Tensor:
        """ construct the input as standard
        imgs: NxBxHxWx3, uint8
        """
        if isinstance(imgs, list):
            _imgs = np.array(imgs)[..., np.newaxis]  # TxHxWx3x1
        elif isinstance(imgs, np.ndarray):
            _imgs = imgs
        else:
            raise ValueError(f'imgs type error: {type(imgs)}')

        y = _imgs.astype(np.float32) / 255.
        y = np.clip(y, 0, 1)  # clip to 0~1
        y = torch.from_numpy(y).permute(0, 4, 3, 1, 2)  # TxHxWx3x1 -> Tx1x3xHxW
        y = y.to(self.device)

        return y

    def extract_feature_3d(self, x: torch.Tensor) -> torch.Tensor:
        """ get the appearance feature of the image by F
        x: Bx3xHxW, normalized to 0~1
        """
        with torch.no_grad() , self.inference_ctx():
            feature_3d = self.appearance_feature_extractor(x)

        return feature_3d.float()

    def get_kp_info(self, x: torch.Tensor, **kwargs) -> dict:
        """ get the implicit keypoint information
        x: Bx3xHxW, normalized to 0~1
        flag_refine_info: whether to trandform the pose to degrees and the dimention of the reshape
        return: A dict contains keys: 'pitch', 'yaw', 'roll', 't', 'exp', 'scale', 'kp'
        """

        with torch.no_grad(), self.inference_ctx():
            kp_info = self.motion_extractor(x)

            if self.inference_cfg.flag_use_half_precision:
                # float the dict
                for k, v in kp_info.items():
                    if isinstance(v, torch.Tensor):
                        kp_info[k] = v.float()

        flag_refine_info: bool = kwargs.get('flag_refine_info', True)
        if flag_refine_info:
            bs = kp_info['kp'].shape[0]
            kp_info['pitch'] = headpose_pred_to_degree(kp_info['pitch'])[:, None]  # Bx1
            kp_info['yaw'] = headpose_pred_to_degree(kp_info['yaw'])[:, None]  # Bx1
            kp_info['roll'] = headpose_pred_to_degree(kp_info['roll'])[:, None]  # Bx1
            kp_info['kp'] = kp_info['kp'].reshape(bs, -1, 3)  # BxNx3
            kp_info['exp'] = kp_info['exp'].reshape(bs, -1, 3)  # BxNx3

        return kp_info

    def get_pose_dct(self, kp_info: dict) -> dict:
        pose_dct = dict(
            pitch=headpose_pred_to_degree(kp_info['pitch']).item(),
            yaw=headpose_pred_to_degree(kp_info['yaw']).item(),
            roll=headpose_pred_to_degree(kp_info['roll']).item(),
        )
        return pose_dct

    def get_fs_and_kp_info(self, source_prepared, driving_first_frame):

        # get the canonical keypoints of source image by M
        source_kp_info = self.get_kp_info(source_prepared, flag_refine_info=True)
        source_rotation = get_rotation_matrix(source_kp_info['pitch'], source_kp_info['yaw'], source_kp_info['roll'])

        # get the canonical keypoints of first driving frame by M
        driving_first_frame_kp_info = self.get_kp_info(driving_first_frame, flag_refine_info=True)
        driving_first_frame_rotation = get_rotation_matrix(
            driving_first_frame_kp_info['pitch'],
            driving_first_frame_kp_info['yaw'],
            driving_first_frame_kp_info['roll']
        )

        # get feature volume by F
        source_feature_3d = self.extract_feature_3d(source_prepared)

        return source_kp_info, source_rotation, source_feature_3d, driving_first_frame_kp_info, driving_first_frame_rotation

    def transform_keypoint(self, kp_info: dict):
        """
        transform the implicit keypoints with the pose, shift, and expression deformation
        kp: BxNx3
        """
        kp = kp_info['kp']    # (bs, k, 3)
        pitch, yaw, roll = kp_info['pitch'], kp_info['yaw'], kp_info['roll']

        t, exp = kp_info['t'], kp_info['exp']
        scale = kp_info['scale']

        pitch = headpose_pred_to_degree(pitch)
        yaw = headpose_pred_to_degree(yaw)
        roll = headpose_pred_to_degree(roll)

        bs = kp.shape[0]
        if kp.ndim == 2:
            num_kp = kp.shape[1] // 3  # Bx(num_kpx3)
        else:
            num_kp = kp.shape[1]  # Bxnum_kpx3

        rot_mat = get_rotation_matrix(pitch, yaw, roll)    # (bs, 3, 3)

        # Eqn.2: s * (R * x_c,s + exp) + t
        kp_transformed = kp.view(bs, num_kp, 3) @ rot_mat + exp.view(bs, num_kp, 3)
        kp_transformed *= scale[..., None]  # (bs, k, 3) * (bs, 1, 1) = (bs, k, 3)
        kp_transformed[:, :, 0:2] += t[:, None, 0:2]  # remove z, only apply tx ty

        return kp_transformed

    def retarget_eye(self, kp_source: torch.Tensor, eye_close_ratio: torch.Tensor) -> torch.Tensor:
        """
        kp_source: BxNx3
        eye_close_ratio: Bx3
        Return: Bx(3*num_kp)
        """
        feat_eye = concat_feat(kp_source, eye_close_ratio)

        with torch.no_grad():
            delta = self.stitching_retargeting_module['eye'](feat_eye)

        return delta.reshape(-1, kp_source.shape[1], 3)

    def retarget_lip(self, kp_source: torch.Tensor, lip_close_ratio: torch.Tensor) -> torch.Tensor:
        """
        kp_source: BxNx3
        lip_close_ratio: Bx2
        Return: Bx(3*num_kp)
        """
        feat_lip = concat_feat(kp_source, lip_close_ratio)

        with torch.no_grad():
            delta = self.stitching_retargeting_module['lip'](feat_lip)

        return delta.reshape(-1, kp_source.shape[1], 3)

    def stitch(self, kp_source: torch.Tensor, kp_driving: torch.Tensor) -> torch.Tensor:
        """
        kp_source: BxNx3
        kp_driving: BxNx3
        Return: Bx(3*num_kp+2)
        """
        feat_stiching = concat_feat(kp_source, kp_driving)

        with torch.no_grad():
            delta = self.stitching_retargeting_module['stitching'](feat_stiching)

        return delta

    def stitching(self, kp_source: torch.Tensor, kp_driving: torch.Tensor) -> torch.Tensor:
        """ conduct the stitching
        kp_source: Bxnum_kpx3
        kp_driving: Bxnum_kpx3
        """

        if self.stitching_retargeting_module is not None:

            bs, num_kp = kp_source.shape[:2]

            kp_driving_new = kp_driving.clone()
            delta = self.stitch(kp_source, kp_driving_new)

            delta_exp = delta[..., :3*num_kp].reshape(bs, num_kp, 3)  # 1x20x3
            delta_tx_ty = delta[..., 3*num_kp:3*num_kp+2].reshape(bs, 1, 2)  # 1x1x2

            kp_driving_new += delta_exp
            kp_driving_new[..., :2] += delta_tx_ty

            return kp_driving_new

        return kp_driving

    def warp_decode(self, feature_3d: torch.Tensor, kp_source: torch.Tensor, kp_driving: torch.Tensor) -> torch.Tensor:
        """ get the image after the warping of the implicit keypoints
        feature_3d: Bx32x16x64x64, feature volume
        kp_source: BxNx3
        kp_driving: BxNx3
        """
        # The line 18 in Algorithm 1: D(W(f_s; x_s, x′_d,i)）
        with torch.no_grad(), self.inference_ctx():
            if self.compile:
                # Mark the beginning of a new CUDA Graph step
                torch.compiler.cudagraph_mark_step_begin()
            # get decoder input
            ret_dct = self.warping_module(feature_3d, kp_source=kp_source, kp_driving=kp_driving)
            # decode
            ret_dct['out'] = self.spade_generator(feature=ret_dct['out'])

            # float the dict
            if self.inference_cfg.flag_use_half_precision:
                for k, v in ret_dct.items():
                    if isinstance(v, torch.Tensor):
                        ret_dct[k] = v.float()

        return ret_dct
    def conv_decode(self, out, occlusion_map=None) -> torch.Tensor:
        out = self.warping_module.warp_out(out, occlusion_map)
        out = self.spade_generator(out)
        return out

    def parse_output(self, out: torch.Tensor) -> np.ndarray:
        """ construct the output as standard
        return: 1xHxWx3, uint8
        """
        out = np.transpose(out.data.cpu().numpy(), [0, 2, 3, 1])  # 1x3xHxW -> 1xHxWx3
        out = np.clip(out, 0, 1)  # clip to 0~1
        out = np.clip(out * 255, 0, 255).astype(np.uint8)  # 0~1 -> 0~255

        return out

    def calc_ratio(self, lmk_lst):
        input_eye_ratio_lst = []
        input_lip_ratio_lst = []
        for lmk in lmk_lst:
            # for eyes retargeting
            input_eye_ratio_lst.append(calc_eye_close_ratio(lmk[None]))
            # for lip retargeting
            input_lip_ratio_lst.append(calc_lip_close_ratio(lmk[None]))
        return input_eye_ratio_lst, input_lip_ratio_lst

    def calc_combined_eye_ratio(self, c_d_eyes_i, source_lmk):
        c_s_eyes = calc_eye_close_ratio(source_lmk[None])
        c_s_eyes_tensor = torch.from_numpy(c_s_eyes).float().to(self.device)
        c_d_eyes_i_tensor = torch.Tensor([c_d_eyes_i[0][0]]).reshape(1, 1).to(self.device)
        # [c_s,eyes, c_d,eyes,i]
        combined_eye_ratio_tensor = torch.cat([c_s_eyes_tensor, c_d_eyes_i_tensor], dim=1)
        return combined_eye_ratio_tensor

    def calc_combined_lip_ratio(self, c_d_lip_i, source_lmk):
        c_s_lip = calc_lip_close_ratio(source_lmk[None])
        c_s_lip_tensor = torch.from_numpy(c_s_lip).float().to(self.device)
        c_d_lip_i_tensor = torch.Tensor([c_d_lip_i[0]]).to(self.device).reshape(1, 1) # 1x1
        # [c_s,lip, c_d,lip,i]
        combined_lip_ratio_tensor = torch.cat([c_s_lip_tensor, c_d_lip_i_tensor], dim=1) # 1x2
        return combined_lip_ratio_tensor

# class can_swapper:
#     def __init__(
#         self,
#         visualizer_params={"kp_size": 5, "draw_border": True, "colormap": "gist_rainbow"},
#     ):

#         self.visualizer = Visualizer(**visualizer_params)
#         self.epoch = 0
#         self.steps = 0
#         self.best_loss = float("inf")
#         model_config = yaml.load(open('src/config/models.yaml', 'r'), Loader=yaml.SafeLoader)['model_params']

#         self.g_models = {"afe": AppearanceFeatureExtractor(**model_config['appearance_feature_extractor_params']), "me": MotionExtractor(**model_config['motion_extractor_params']),
#                          "wm": WarpingNetwork(**model_config['warping_module_params']), "generator": SPADEDecoder(**model_config['spade_generator_params']),
#                          "sm": StitchingRetargetingNetwork(**model_config['stitching_retargeting_module_params']["stitching"])}
#         # self.load_init_models()
#         # self.g_models = {"me": MotionExtractor(**model_config['motion_extractor_params']),
#         #                  "wm": WarpingNetwork(**model_config['warping_module_params']), "generator": SPADEDecoder(**model_config['spade_generator_params'])}

#         self.dis_models = {"transfer": transfer_model()}

#         # 遍历每个模型并统计总参数量
#         for name, model in self.g_models.items():
#             total_params = sum(p.numel() for p in model.parameters())
#             print(f"Model {name} has {total_params:,} total parameters.")
#         for name, model in self.dis_models.items():
#             total_params = sum(p.numel() for p in model.parameters())
#             print(f"Model {name} has {total_params:,} total parameters.")

#         self.g_full = GeneratorFull(**self.g_models, **self.dis_models, **self.d_models)
#         self.d_full = DiscriminatorFull(**self.d_models)
#         self.g_loss_names, self.d_loss_names = None, None
#         self.dataloader = dataloader
#         ##锁住g_models
#         for name, model in self.g_models.items():
#             model.eval()
#             for param in model.parameters():
#                 param.requires_grad = False
#         #设置dis_models的优化器
#         for name, model in self.dis_models.items():
#             model.train()
#             for param in model.parameters():
#                 param.requires_grad = True
#         #设置d_models的优化器
#         for name, model in self.d_models.items():
#             model.train()
#             for param in model.parameters():
#                 param.requires_grad = True

#     def __del__(self):
#         self.save_cpk()
#         if is_master():
#             self.log_file.close()

#     @master_only
#     def log_scores(self):
#         loss_mean = np.array(self.g_losses).mean(axis=0)
#         loss_string = "; ".join(["%s - %.5f" % (name, value) for name, value in zip(self.g_loss_names, loss_mean)])
#         loss_string = "G" + str(self.steps).zfill(self.zfill_num) + ") " + loss_string
#         print(loss_string, file=self.log_file)
#         self.g_losses = []
#         loss_mean = np.array(self.d_losses).mean(axis=0)
#         loss_string = "; ".join(["%s - %.5f" % (name, value) for name, value in zip(self.d_loss_names, loss_mean)])
#         loss_string = "D" + str(self.steps).zfill(self.zfill_num) + ") " + loss_string
#         print(loss_string, file=self.log_file)
#         self.d_losses = []
#         # 记录生成器学习率
#         for name, optimizer in self.g_optimizers.items():
#             lr = optimizer.param_groups[0]['lr']
#             lr_string = f"LR (G-{name}): {lr:.6f}"
#             print(lr_string, file=self.log_file)
#         # 记录判别器学习率
#         for name, optimizer in self.d_optimizers.items():
#             lr = optimizer.param_groups[0]['lr']
#             lr_string = f"LR (D-{name}): {lr:.6f}"
#             print(lr_string, file=self.log_file)
#         self.log_file.flush()

#     @master_only
#     def visualize_rec(self, s, d, generated_d, transformed_d, kp_s, kp_d, transformed_kp, occlusion):
#         image = self.visualizer.visualize(s, d, generated_d, transformed_d, kp_s, kp_d, transformed_kp, occlusion)
#         imageio.imsave(os.path.join(self.vis_dir, "%s-rec.png" % str(self.epoch).zfill(self.zfill_num)), image)

#     @master_only
#     def visualize_swap(self, result, mask=None):
#         image = self.visualizer.visualize_swap(result, mask)
#         imageio.imsave(os.path.join(self.vis_dir, "%s-swap.png" % str(self.steps).zfill(self.zfill_num)), image)




#     def load_cpk(self, epoch):
#         ckp_path = os.path.join(self.ckp_dir, "%s-checkpoint.pth.tar" % str(epoch).zfill(self.zfill_num))
#         checkpoint = torch.load(ckp_path, map_location=torch.device("cpu"))
#         for k, v in self.g_models.items():
#             v.module.load_state_dict(checkpoint[k])
#         for k, v in self.dis_models.items():
#             v.module.load_state_dict(checkpoint[k])

#     @master_only
#     def log_iter(self, g_losses, d_losses):
#         g_losses = collections.OrderedDict(g_losses.items())
#         d_losses = collections.OrderedDict(d_losses.items())
#         if self.g_loss_names is None:
#             self.g_loss_names = list(g_losses.keys())
#         if self.d_loss_names is None:
#             self.d_loss_names = list(d_losses.keys())
#         self.g_losses.append(list(g_losses.values()))
#         self.d_losses.append(list(d_losses.values()))

#     @master_only
#     def log_times(self, result, mask=None):
#         self.log_scores()
#         self.visualize_swap(result, mask)

#     @master_only
#     def save_check(self):
#         if (self.epoch + 1) % self.checkpoint_freq == 0 and self.epoch != 0:
#             self.save_cpk()

#     def step(self):
#         master_only_print("Epoch", self.epoch)
#         with tqdm(total=len(self.dataloader.dataset)) as progress_bar:
#             # print('1')
#             for d, M, d_c, M_c, d_id, M_id, flag, swap_res, mask_gt in self.dataloader:
#                 # print('2')
#                 d = d.cuda(non_blocking=True)
#                 M = M.cuda(non_blocking=True)
#                 d_c = d_c.cuda(non_blocking=True)
#                 M_c = M_c.cuda(non_blocking=True)
#                 d_id = d_id.cuda(non_blocking=True)
#                 M_id = M_id.cuda(non_blocking=True)
#                 swap_res = swap_res.cuda(non_blocking=True)
#                 mask_gt = mask_gt.cuda(non_blocking=True)
#                 for optimizer in self.g_optimizers.values():
#                     optimizer.zero_grad()
#                 losses_g, swap_canonical, swap_origin, x_can, x_ori, swap_origin_re, mask = self.g_full(d, M, d_c, M_c, d_id, M_id, flag, swap_res, mask_gt)
#                 loss_g = sum(losses_g.values())
#                 loss_g.backward()
#                 # loss_g.backward()
#                 # for name, param in self.dis_models['transfer'].named_parameters():
#                 #     if param.grad is not None:
#                 #         print(f"{name}: {param.grad.norm()}")
#                 for optimizer in self.g_optimizers.values():
#                     #加入梯度裁剪
#                     torch.nn.utils.clip_grad_norm_(self.dis_models['transfer'].module.parameters(), 1.0)
#                     optimizer.step()
#                     optimizer.zero_grad()
#                 for optimizer in self.d_optimizers.values():
#                     optimizer.zero_grad()
#                 losses_d = self.d_full(d, d_c, swap_canonical, swap_origin, x_can, x_ori)#真实结果， 换脸结果， 换脸的kp
#                 loss_d = sum(losses_d.values())
#                 loss_d.backward()
#                 for optimizer in self.d_optimizers.values():
#                     optimizer.step()
#                     optimizer.zero_grad()
#                 self.log_iter(to_cpu(losses_g), to_cpu(losses_d))
#                 if is_master():
#                     progress_bar.update(len(d))
#                 self.steps += 1
#                 if self.steps % 200 == 0:
#                     self.log_times([d_id, d, d_c, swap_canonical, swap_origin, swap_origin_re], mask)

#         # 调用学习率调度器
#         for scheduler in self.g_schedulers.values():
#             scheduler.step()
#         for scheduler in self.d_schedulers.values():
#             scheduler.step()

#         self.save_check()
#         self.epoch += 1

# class Visualizer:
#     def __init__(self, kp_size=5, draw_border=False, colormap="gist_rainbow"):
#         self.kp_size = kp_size
#         self.draw_border = draw_border
#         self.colormap = plt.get_cmap(colormap)

#     def draw_image_with_kp(self, image, kp_array):
#         image = np.copy(image)
#         spatial_size = np.array(image.shape[:2][::-1])[np.newaxis]
#         kp_array = spatial_size * (kp_array + 1) / 2
#         num_kp = kp_array.shape[0]
#         for kp_ind, kp in enumerate(kp_array):
#             # rr, cc = circle(kp[1], kp[0], self.kp_size, shape=image.shape[:2])
#             rr, cc = disk((kp[0], kp[1]), self.kp_size, shape=image.shape[:2])
#             image[rr, cc] = np.array(self.colormap(kp_ind / num_kp))[:3]
#         return image

#     def create_image_column_with_kp(self, images, kp):
#         image_array = np.array([self.draw_image_with_kp(v, k) for v, k in zip(images, kp)])
#         return self.create_image_column(image_array)

#     def create_image_column(self, images):
#         if self.draw_border:
#             images = np.copy(images)
#             images[:, :, [0, -1]] = (1, 1, 1)
#             images[:, :, [0, -1]] = (1, 1, 1)
#         return np.concatenate(list(images), axis=0)

#     def create_image_grid(self, *args):
#         out = []
#         for arg in args:
#             if type(arg) == tuple:
#                 out.append(self.create_image_column_with_kp(arg[0], arg[1]))
#             else:
#                 out.append(self.create_image_column(arg))
#         return np.concatenate(out, axis=1)

#     def visualize(self, s, d, generated_d, transformed_d, kp_s, kp_d, transformed_kp, occlusion):
#         images = []
#         # Source image with keypoints
#         source = s.data.cpu()
#         kp_source = kp_s.data.cpu().numpy()[:, :, :2]
#         source = np.transpose(source, [0, 2, 3, 1])
#         images.append((source, kp_source))

#         # Equivariance visualization
#         transformed = transformed_d.data.cpu().numpy()
#         transformed = np.transpose(transformed, [0, 2, 3, 1])
#         transformed_kp = transformed_kp.data.cpu().numpy()[:, :, :2]
#         images.append((transformed, transformed_kp))

#         # Driving image with keypoints
#         kp_driving = kp_d.data.cpu().numpy()[:, :, :2]
#         driving = d.data.cpu().numpy()
#         driving = np.transpose(driving, [0, 2, 3, 1])
#         images.append((driving, kp_driving))

#         # Result with and without keypoints
#         prediction = generated_d.data.cpu().numpy()
#         prediction = np.transpose(prediction, [0, 2, 3, 1])
#         images.append(prediction)

#         # Occlusion map
#         occlusion_map = occlusion.data.cpu().repeat(1, 3, 1, 1)
#         occlusion_map = F.interpolate(occlusion_map, size=source.shape[1:3]).numpy()
#         occlusion_map = np.transpose(occlusion_map, [0, 2, 3, 1])
#         images.append(occlusion_map)

#         image = self.create_image_grid(*images)
#         image = image.clip(0, 1)
#         image = (255 * image).astype(np.uint8)
#         return image

#     def visualize_swap(self, result, mask=None):
#         images = []
#         for image_g in result:
#             image_g = image_g.data.cpu().numpy()
#             image_g = np.transpose(image_g, [0, 2, 3, 1])
#             images.append(image_g)

#         if mask is not None:
#             with torch.no_grad():
#                 for mask_g in mask:
#                     mask_g = F.interpolate(mask_g, size=(256, 256))
#                     mask_g = mask_g.repeat(1, 3, 1, 1)
#                     mask_g = mask_g.data.cpu().numpy()
#                     mask_g = np.transpose(mask_g, [0, 2, 3, 1])
#                     images.append(mask_g)
#         # driving = d_id.data.cpu().numpy()
#         # driving = np.transpose(driving, [0, 2, 3, 1])
#         # images.append(driving)

#         # driving = d.data.cpu().numpy()
#         # driving = np.transpose(driving, [0, 2, 3, 1])
#         # images.append(driving)

#         # driving = d_c.data.cpu().numpy()
#         # driving = np.transpose(driving, [0, 2, 3, 1])
#         # images.append(driving)

#         # driving = recon_canonical.data.cpu().numpy()
#         # driving = np.transpose(driving, [0, 2, 3, 1])
#         # images.append(driving)
#         # driving = swap_res.data.cpu().numpy()
#         # driving = np.transpose(driving, [0, 2, 3, 1])
#         # images.append(driving)

#         # prediction = swap_canonical.data.cpu().numpy()
#         # prediction = np.transpose(prediction, [0, 2, 3, 1])
#         # images.append(prediction)

#         # prediction = swap_origin.data.cpu().numpy()
#         # prediction = np.transpose(prediction, [0, 2, 3, 1])
#         # images.append(prediction)

#         image = self.create_image_grid(*images)
#         image = image.clip(0, 1)
#         image = (255 * image).astype(np.uint8)
#         return image
