import numpy as np
import torch
import torch.nn.functional as F
import imageio

import os
import os.path as osp
from skimage.draw import disk

import matplotlib.pyplot as plt
import collections
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

import yaml

from src.modules.spade_generator import SPADEDecoder
from src.modules.warping_network import WarpingNetwork
from src.modules.motion_extractor import MotionExtractor
from src.modules.appearance_feature_extractor import AppearanceFeatureExtractor


from src.modules.adaptive_modulate import transfer_model
from src.modules.adaptive_modulate import transfer_model2 as transfer_model_big
from src.modules.adaptive_modulate import G3d

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
