from Midas.run import run_depth
from Midas.monodepth_net import MonoDepthNet
import Midas.midas_utils as MiDaS_utils
from utils import get_MiDaS_samples
config = dict()

config['depth_edge_model_ckpt'] = 'checkpoints/edge-model.pth'
config['depth_feat_model_ckpt'] = 'checkpoints/depth-model.pth'
config['rgb_feat_model_ckpt'] = 'checkpoints/color-model.pth'
config['MiDaS_model_ckpt'] = 'Midas/model.pt'
config['fps'] = 40
config['num_frames'] = 240
config['x_shift_range'] = [0.00, 0.00, -0.02, -0.02]
config['y_shift_range'] = [0.00, 0.00, -0.02, -0.00]
config['z_shift_range'] = [-0.05, -0.05, -0.07, -0.07]
config['traj_types'] = ['double-straight-line', 'double-straight-line', 'circle', 'circle']
config['video_postfix'] = ['dolly-zoom-in', 'zoom-in', 'circle', 'swing']
config['specific'] = ''
config['longer_side_len'] = 960
config['src_folder'] = 'image'
config['depth_folder'] = 'depth'
config['mesh_folder'] = 'mesh'
config['video_folder'] = 'video'
config['load_ply'] = False
config['save_ply'] = False
config['inference_video'] = True
config['gpu_ids'] = 0
config['offscreen_rendering'] = False
config['img_format'] = '.jpg'
config['depth_format'] = '.npy'
config['require_midas'] = True
config['depth_threshold'] = 0.04
config['ext_edge_threshold'] = 0.002
config['sparse_iter'] = 5
config['filter_size'] = [7, 7, 5, 5, 5]
config['sigma_s'] = 4.0
config['sigma_r'] = 0.5
config['redundant_number'] = 12
config['background_thickness'] = 70
config['context_thickness'] = 140
config['background_thickness_2'] = 70
config['context_thickness_2'] = 70
config['discount_factor'] = 1.00
config['log_depth'] = True
config['largest_size'] = 512
config['depth_edge_dilate'] = 10
config['depth_edge_dilate_2'] = 5
config['extrapolate_border'] = True
config['extrapolation_thickness'] = 60
config['repeat_inpaint_edge'] = True
config['crop_border'] = [0.03, 0.03, 0.05, 0.03]
config['anti_flickering'] = True

sample_list = get_MiDaS_samples(config['src_folder'], config['depth_folder'], config, config['specific'])

run_depth([sample_list[0]['ref_img_fi']], config['src_folder'], config['depth_folder'],
                  config['MiDaS_model_ckpt'], MonoDepthNet, MiDaS_utils, target_w=640)
