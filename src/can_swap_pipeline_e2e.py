import torch
torch.backends.cudnn.benchmark = True # disable CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR warning
import imageio
from skimage.draw import disk

import matplotlib.pyplot as plt

import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)
import numpy as np
import os
import os.path as osp
from rich.progress import track

from .config.argument_config import ArgumentConfig
from .config.inference_config import InferenceConfig
from .config.crop_config import CropConfig
from .utils.cropper import Cropper
from .utils.camera import get_rotation_matrix
from .utils.video import images2video, concat_frames, get_fps, add_audio_to_video, has_audio_stream, to_frames
from .utils.crop import prepare_paste_back, paste_back
from .utils.crop import dilation_mask, erode_mask, smooth_mask, blend_images, SoftErosion
from .utils.io import load_image_rgb, load_video, resize_to_limit, dump, load
from .utils.helper import mkdir, basename, dct2device, is_video, is_template, remove_suffix, is_image, is_square_video, calc_motion_multiplier
from .utils.filter import smooth
from .utils.rprint import rlog as log
from .can_swap_e2e import can_swapper
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
def make_abs_path(fn):
    return osp.join(osp.dirname(osp.realpath(__file__)), fn)

from insightface_func.face_detect_crop_single import Face_detect_crop

class CanSwapPipeline(object):

    def __init__(self, inference_cfg: InferenceConfig, crop_cfg: CropConfig):
        self.can_swapper: can_swapper = can_swapper(inference_cfg=inference_cfg)
        self.cropper: Cropper = Cropper(crop_cfg=crop_cfg)
        self.soft_mask = SoftErosion(kernel_size=21, threshold=0.9, iterations=3).cuda()
        self.ID_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

        # visualizer_params={"kp_size": 5, "draw_border": True, "colormap": "gist_rainbow"}
        # self.visualizer = Visualizer(**visualizer_params)

        self.image_processor = SegformerImageProcessor.from_pretrained("pretrained_weights/parsing", local_files_only=True)
        self.model = SegformerForSemanticSegmentation.from_pretrained("pretrained_weights/parsing",local_files_only=True).to(self.can_swapper.device)
        self.valid_list = [1, 2, 4, 5, 6, 7, 10, 11, 12]
        self.valid_list = torch.tensor(self.valid_list, device=self.can_swapper.device)

        self.cropper_insightface = Face_detect_crop(name='antelope', root='pretrained_weights/insightface/models')
        self.cropper_insightface.prepare(ctx_id=0, det_thresh=0.5, det_size=(640,640), mode='None')

    def execute_face_canonical(self, args: ArgumentConfig):
        # for convenience
        inf_cfg = self.can_swapper.inference_cfg
        crop_cfg = self.cropper.crop_cfg
        ######## load source input ########
        img_rgb = load_image_rgb(args.source)  # Assuming args.source is already a numpy array
        img_rgb = resize_to_limit(img_rgb, inf_cfg.source_max_dim, inf_cfg.source_division)
        source_rgb_lst = [img_rgb]

        crop_info = self.cropper.crop_source_image(source_rgb_lst[0], crop_cfg)
        img_crop_256x256 = crop_info['img_crop_256x256']

        I_s = self.can_swapper.prepare_source(img_crop_256x256)
        x_s_info = self.can_swapper.get_kp_info(I_s)
        x_c_s = x_s_info['kp']
        f_s = self.can_swapper.extract_feature_3d(I_s)
        x_s = self.can_swapper.transform_keypoint(x_s_info)

        scale_new = x_s_info['scale']

        x_d_i_new = scale_new * x_c_s
        out = self.can_swapper.warp_decode(f_s, x_s, x_d_i_new)
        I_p = self.can_swapper.parse_output(out['out'])[0]

        # Convert RGB to BGR
        I_p_rgb = cv2.cvtColor(I_p, cv2.COLOR_BGR2RGB)
        # wfp = args.outpath
        cv2.imwrite('source_can.jpg', I_p_rgb)
        # log(f'Animated image: {wfp}')
        return I_p_rgb, x_s_info

    def get_source_id(self, device ,args):
        # source_can, x_s_info = self.execute_face_canonical(args)
        x_s_info = None
        source_can = cv2.imread(args.source)
        source_can_crop = self.cropper_insightface.get(source_can, crop_size=112, max_num=1)
        cv2.imwrite('source_can_112.jpg', source_can_crop[0][0])
        source_can_crop = cv2.cvtColor(source_can_crop[0][0], cv2.COLOR_BGR2RGB)
        source_tensor = self.ID_transform(source_can_crop).unsqueeze(0).to(device)
        source_id = self.can_swapper.getid(source_tensor)
        return source_id, x_s_info

    def make_motion_template(self, I_lst, c_eyes_lst, c_lip_lst, **kwargs):
        n_frames = I_lst.shape[0]
        template_dct = {
            'n_frames': n_frames,
            'output_fps': kwargs.get('output_fps', 25),
            'motion': [],
            'c_eyes_lst': [],
            'c_lip_lst': [],
        }

        for i in track(range(n_frames), description='Making motion templates...', total=n_frames):
            # collect s, R, δ and t for inference
            I_i = I_lst[i]
            x_i_info = self.can_swapper.get_kp_info(I_i)
            x_s = self.can_swapper.transform_keypoint(x_i_info)
            R_i = get_rotation_matrix(x_i_info['pitch'], x_i_info['yaw'], x_i_info['roll'])

            item_dct = {
                'scale': x_i_info['scale'].cpu().numpy().astype(np.float32),
                'R': R_i.cpu().numpy().astype(np.float32),
                'exp': x_i_info['exp'].cpu().numpy().astype(np.float32),
                't': x_i_info['t'].cpu().numpy().astype(np.float32),
                'kp': x_i_info['kp'].cpu().numpy().astype(np.float32),
                'x_s': x_s.cpu().numpy().astype(np.float32),
            }

            template_dct['motion'].append(item_dct)

            c_eyes = c_eyes_lst[i].astype(np.float32)
            template_dct['c_eyes_lst'].append(c_eyes)

            c_lip = c_lip_lst[i].astype(np.float32)
            template_dct['c_lip_lst'].append(c_lip)

        return template_dct

    def execute(self, args: ArgumentConfig):
        # for convenience
        inf_cfg = self.can_swapper.inference_cfg
        device = self.can_swapper.device
        crop_cfg = self.cropper.crop_cfg

        for i in track(range(1), description='🚀Get Source ID...', total=1):
            source_id, x_s_info = self.get_source_id(device, args)

        ######## process target info ########
        driving_rgb_crop_256x256_lst = None
        wfp_template = None

        if is_video(args.driving):
            flag_is_driving_video = True
            # load from video file, AND make motion template
            output_fps = int(get_fps(args.driving))
            log(f"Load driving video from: {args.driving}, FPS is {output_fps}")
            driving_rgb_lst = load_video(args.driving)
        elif is_image(args.driving):
            flag_is_driving_video = False
            driving_img_rgb = load_image_rgb(args.driving)
            output_fps = 25
            log(f"Load driving image from {args.driving}")
            driving_rgb_lst = [driving_img_rgb]
        else:
            raise Exception(f"{args.driving} is not a supported type!")
        n_frames = len(driving_rgb_lst)
        if inf_cfg.flag_crop_driving_video or (not is_square_video(args.driving)):
            ret_d = self.cropper.crop_source_video(driving_rgb_lst, crop_cfg)
            log(f'Target video is cropped, {len(ret_d["frame_crop_lst"])} frames are processed.')
            if len(ret_d["frame_crop_lst"]) is not n_frames and flag_is_driving_video:
                n_frames = min(n_frames, len(ret_d["frame_crop_lst"]))
            driving_rgb_crop_lst, driving_lmk_crop_lst, target_M_c2o_lst = ret_d['frame_crop_lst'], ret_d['lmk_crop_lst'], ret_d['M_c2o_lst']
            driving_rgb_crop_256x256_lst = [cv2.resize(_, (256, 256)) for _ in driving_rgb_crop_lst]
        else:
            driving_lmk_crop_lst = self.cropper.calc_lmks_from_cropped_video(driving_rgb_lst)
            driving_rgb_crop_256x256_lst = [cv2.resize(_, (256, 256)) for _ in driving_rgb_lst]  # force to resize to 256x256
        #######################################
        #获取mask
        masks = []
        for i in track(range(n_frames), description='🚀Parsing...', total=n_frames):
            frame = driving_rgb_crop_256x256_lst[i]
            inputs = self.image_processor(frame, return_tensors="pt").to(device)
            outputs = self.model(**inputs)
            logits = outputs.logits # tensor of shape (N, 19, H, W)
            upsampled_logits = torch.nn.functional.interpolate( #N, 19, 256, 256
            logits,
            size= (512, 512),  # H x W
            mode='bilinear',
            align_corners=False
            )
            labels = upsampled_logits.argmax(dim=1)[0] #N, 256, 256
            mask = torch.isin(labels, self.valid_list).to(dtype=torch.int)
            masks.append(mask)

        ######################################
        c_d_eyes_lst, c_d_lip_lst = self.can_swapper.calc_ratio(driving_lmk_crop_lst)
        # save the motion template
        I_d_lst = self.can_swapper.prepare_videos(driving_rgb_crop_256x256_lst)
        driving_template_dct = self.make_motion_template(I_d_lst, c_d_eyes_lst, c_d_lip_lst, output_fps=output_fps)

        # wfp_template = remove_suffix(args.driving) + '.pkl'
        # dump(wfp_template, driving_template_dct)
        # log(f"Dump motion template to {wfp_template}")

        ######## prepare for pasteback ########
        I_p_pstbk_lst = None
        if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop:
            I_p_pstbk_lst = []
            log("Prepared pasteback mask done.")

        I_p_lst = []
        I_can_lst = []
        rec_can_lst = []
        # R_d_0, x_d_0_info = None, None
        # flag_normalize_lip = inf_cfg.flag_normalize_lip  # not overwrite
        # flag_source_video_eye_retargeting = inf_cfg.flag_source_video_eye_retargeting  # not overwrite
        # lip_delta_before_animation, eye_delta_before_animation = None, None

        ######## animate ########
        if flag_is_driving_video:
            log(f"The animated video consists of {n_frames} frames.")
        else:
            log(f"The output of image-driven portrait animation is an image.")

        for i in track(range(n_frames), description='🚀Swapping...', total=n_frames):
            # if flag_is_driving_video:  # source video
            x_t_info = driving_template_dct['motion'][i]
            x_t_info = dct2device(x_t_info, device)

            # source_lmk = driving_lmk_crop_lst[i]
            # img_crop_256x256 = driving_rgb_crop_256x256_lst[i]
            I_s = I_d_lst[i]

            # f_s = self.can_swapper.appearance_feature_extractor(I_s)

            x_c_t = x_t_info['kp']
            R_t = x_t_info['R']
            x_t =x_t_info['x_s']
            delta_t = x_t_info['exp']
            t_new = x_t_info['t']
            scale_new = x_t_info['scale']
            t_new[..., 2].fill_(0)  # zero tz

            f_s = self.can_swapper.extract_feature_3d(I_s)
            x_can = scale_new * x_c_t
            f_can, occ_map = self.can_swapper.warping_module.warp(f_s, x_t, x_can)


            #for debug
            rec_can = self.can_swapper.conv_decode(f_can, occ_map)
            rec_can_i = self.can_swapper.parse_output(rec_can)[0]
            rec_can_lst.append(rec_can_i)
            #######
            # if i == 0:
            f_can_swap = self.can_swapper.swap_module(f_can, source_id)
            # x_can1 = x_can

            #for debug
            swap_can = self.can_swapper.conv_decode(f_can_swap, occ_map)
            I_can_i = self.can_swapper.parse_output(swap_can)[0]
            I_can_lst.append(I_can_i)

            #for wo refine
            f_can_swap = self.can_swapper.refine_module(f_can_swap)
            out = self.can_swapper.warp_decode(f_can_swap, x_can, x_t) #(N,3,512,512)


            if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop:
                I_p_i = self.can_swapper.parse_output(out['out'])[0]
            else:
                res = blend_images(out['out'], I_s, mask)
                I_p_i = self.can_swapper.parse_output(res)[0] #512,512,3

            I_p_lst.append(I_p_i)

            if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop:  # prepare for paste back
                mask , _ = self.soft_mask(masks[i].unsqueeze(0).unsqueeze(0))
                # print(mask.shape)
                mask = mask.squeeze().data.cpu().numpy()
                mask = np.stack([mask, mask, mask], axis=-1)
                mask_ori_float = prepare_paste_back(mask, target_M_c2o_lst[i], dsize=(driving_rgb_lst[i].shape[1], driving_rgb_lst[i].shape[0]), if_float=True)

            if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop:
                I_p_pstbk = paste_back(I_p_i, target_M_c2o_lst[i], driving_rgb_lst[i], mask_ori_float)
                I_p_pstbk_lst.append(I_p_pstbk)



        mkdir(args.output_dir)
        wfp_concat = None
        ######### build the final concatenation result #########
        frames_concatenated = concat_frames(driving_rgb_crop_256x256_lst, I_can_lst , I_p_lst, rec_can_lst)
        # frames_concatenated = to_frames(I_can_lst)
        # image = self.visualizer.visualize_swap([swap_can, out['out'], res], [mask])
        # imageio.imsave('test.png',image)

        if flag_is_driving_video:
            # flag_source_has_audio = has_audio_stream(args.driving)
            flag_driving_has_audio = has_audio_stream(args.driving)

            wfp_concat = osp.join(args.output_dir, f'{basename(args.source)}--{basename(args.driving)}_concat.mp4')

            # NOTE: update output fps
            output_fps = output_fps
            images2video(frames_concatenated, wfp=wfp_concat, fps=output_fps)

            if flag_driving_has_audio:
            # if False:
                # final result with concatenation
                wfp_concat_with_audio = osp.join(args.output_dir, f'{basename(args.source)}--{basename(args.driving)}_concat_with_audio.mp4')
                audio_from_which_video = args.driving
                log(f"Audio is selected from {audio_from_which_video}, concat mode")
                add_audio_to_video(wfp_concat, audio_from_which_video, wfp_concat_with_audio)
                os.replace(wfp_concat_with_audio, wfp_concat)
                log(f"Replace {wfp_concat_with_audio} with {wfp_concat}")

            # save the animated result
            wfp = osp.join(args.output_dir, f'{basename(args.source)}--{basename(args.driving)}.mp4')
            # wfp = args.outpath
            if I_p_pstbk_lst is not None and len(I_p_pstbk_lst) > 0:
                images2video(I_p_pstbk_lst, wfp=wfp, fps=output_fps)
            else:
                images2video(I_p_lst, wfp=wfp, fps=output_fps)

            ######### build the final result #########
            if flag_driving_has_audio:
                wfp_with_audio = osp.join(args.output_dir, f'{basename(args.source)}--{basename(args.driving)}_with_audio.mp4')
                audio_from_which_video = args.driving
                log(f"Audio is selected from {audio_from_which_video}")
                add_audio_to_video(wfp, audio_from_which_video, wfp_with_audio)
                os.replace(wfp_with_audio, wfp)
                log(f"Replace {wfp_with_audio} with {wfp}")

            # final log
            if wfp_template not in (None, ''):
                log(f'Animated template: {wfp_template}, you can specify `-d` argument with this template path next time to avoid cropping video, motion making and protecting privacy.', style='bold green')
            log(f'Animated video: {wfp}')
            log(f'Animated video with concat: {wfp_concat}')
        else:
            wfp_concat = osp.join(args.output_dir, f'{basename(args.source)}--{basename(args.driving)}_concat.jpg')
            cv2.imwrite(wfp_concat, frames_concatenated[0][..., ::-1])
            wfp = osp.join(args.output_dir, f'{basename(args.source)}--{basename(args.driving)}.jpg')
            # wfp = args.outpath
            if I_p_pstbk_lst is not None and len(I_p_pstbk_lst) > 0:
                cv2.imwrite(wfp, I_p_pstbk_lst[0][..., ::-1])
            else:
                cv2.imwrite(wfp, frames_concatenated[0][..., ::-1])
            # final log
            log(f'Animated image: {wfp}')
            log(f'Animated image with concat: {wfp_concat}')

        return wfp, wfp_concat
