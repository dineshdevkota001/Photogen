import os
import glob
import cv2
import scipy.misc as misc
from skimage.transform import resize
import numpy as np
from functools import reduce
from operator import mul
import torch
from torch import nn
import matplotlib.pyplot as plt
import re
try:
    import cynetworkx as netx
except ImportError:
    import networkx as netx
from scipy.ndimage import gaussian_filter
from skimage.feature import canny
import collections
import shutil
import imageio
import copy
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from scipy.interpolate import interp1d
from collections import namedtuple
from Mesh.utils import path_planning

def plan_path_e2e(mesh, cc, end_pts, global_mesh, input_edge, mask, valid_map, inpaint_id, npath_map=None, fpath_map=None):
    my_npath_map = np.zeros_like(input_edge) - 1
    my_fpath_map = np.zeros_like(input_edge) - 1
    sub_mesh = mesh.subgraph(list(cc)).copy()
    ends_1, ends_2 = end_pts[0], end_pts[1]
    edge_id = global_mesh.nodes[ends_1]['edge_id']
    npath = [*netx.shortest_path(sub_mesh, (ends_1[0], ends_1[1]), (ends_2[0], ends_2[1]), weight='length')]
    for np_node in npath:
        my_npath_map[np_node[0], np_node[1]] = edge_id
    fpath = []
    if global_mesh.nodes[ends_1].get('far') is None:
        print("None far")
    else:
        fnodes = global_mesh.nodes[ends_1].get('far')
        dmask = mask + 0
        while True:
            dmask = cv2.dilate(dmask, np.ones((3, 3)), iterations=1)
            ffnode = [fnode for fnode in fnodes if (dmask[fnode[0], fnode[1]] > 0 and mask[fnode[0], fnode[1]] == 0 and\
                                                            global_mesh.nodes[fnode].get('inpaint_id') != inpaint_id + 1)]
            if len(ffnode) > 0:
                fnode = ffnode[0]
                break
        e_fnodes = global_mesh.nodes[ends_2].get('far')
        dmask = mask + 0
        while True:
            dmask = cv2.dilate(dmask, np.ones((3, 3)), iterations=1)
            e_ffnode = [e_fnode for e_fnode in e_fnodes if (dmask[e_fnode[0], e_fnode[1]] > 0 and mask[e_fnode[0], e_fnode[1]] == 0 and\
                                                            global_mesh.nodes[e_fnode].get('inpaint_id') != inpaint_id + 1)]
            if len(e_ffnode) > 0:
                e_fnode = e_ffnode[0]
                break            
        fpath.append((fnode[0], fnode[1]))
        if len(e_ffnode) == 0 or len(ffnode) == 0:
            return my_npath_map, my_fpath_map, [], []
        barrel_dir = np.array([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]])
        n2f_dir = (int(fnode[0] - npath[0][0]), int(fnode[1] - npath[0][1]))
        while True:
            if barrel_dir[0, 0] == n2f_dir[0] and barrel_dir[0, 1] == n2f_dir[1]:
                n2f_barrel = barrel_dir.copy()
                break
            barrel_dir = np.roll(barrel_dir, 1, axis=0)
        for step in range(0, len(npath)):
            if step == 0:
                continue
            elif step == 1:
                next_dir = (npath[step][0] - npath[step - 1][0], npath[step][1] - npath[step - 1][1])
                while True:
                    if barrel_dir[0, 0] == next_dir[0] and barrel_dir[0, 1] == next_dir[1]:
                        next_barrel = barrel_dir.copy()
                        break
                    barrel_dir = np.roll(barrel_dir, 1, axis=0)
                barrel_pair = np.stack((n2f_barrel, next_barrel), axis=0)
                n2f_dir = (barrel_pair[0, 0, 0], barrel_pair[0, 0, 1])
            elif step > 1:
                next_dir = (npath[step][0] - npath[step - 1][0], npath[step][1] - npath[step - 1][1])
                while True:
                    if barrel_pair[1, 0, 0] == next_dir[0] and barrel_pair[1, 0, 1] == next_dir[1]:
                        next_barrel = barrel_pair.copy()
                        break
                    barrel_pair = np.roll(barrel_pair, 1, axis=1)
                n2f_dir = (barrel_pair[0, 0, 0], barrel_pair[0, 0, 1])
            new_locs = []
            if abs(n2f_dir[0]) == 1:
                new_locs.append((npath[step][0] + n2f_dir[0], npath[step][1]))
            if abs(n2f_dir[1]) == 1:
                new_locs.append((npath[step][0], npath[step][1] + n2f_dir[1]))
            if len(new_locs) > 1:
                new_locs = sorted(new_locs, key=lambda xx: np.hypot((xx[0] - fpath[-1][0]), (xx[1] - fpath[-1][1])))
            break_flag = False
            for new_loc in new_locs:
                new_loc_nes = [xx for xx in [(new_loc[0] + 1, new_loc[1]), (new_loc[0] - 1, new_loc[1]),
                                            (new_loc[0], new_loc[1] + 1), (new_loc[0], new_loc[1] - 1)]\
                                    if xx[0] >= 0 and xx[0] < my_fpath_map.shape[0] and xx[1] >= 0 and xx[1] < my_fpath_map.shape[1]]
                if fpath_map is not None and np.sum([fpath_map[nlne[0], nlne[1]] for nlne in new_loc_nes]) != 0:
                    break_flag = True
                    break
                if my_npath_map[new_loc[0], new_loc[1]] != -1:
                    continue
                if npath_map is not None and npath_map[new_loc[0], new_loc[1]] != edge_id:
                    break_flag = True
                    break
                fpath.append(new_loc)
            if break_flag is True:
                break
        if (e_fnode[0], e_fnode[1]) not in fpath:
            fpath.append((e_fnode[0], e_fnode[1]))
        if step != len(npath) - 1:
            for xx in npath[step:]:
                if my_npath_map[xx[0], xx[1]] == edge_id:
                    my_npath_map[xx[0], xx[1]] = -1
            npath = npath[:step]
        if len(fpath) > 0:
            for fp_node in fpath:
                my_fpath_map[fp_node[0], fp_node[1]] = edge_id
    
    return my_fpath_map, my_npath_map, npath, fpath

def plan_path(mesh, info_on_pix, cc, end_pt, global_mesh, input_edge, mask, valid_map, inpaint_id, npath_map=None, fpath_map=None, npath=None):
    my_npath_map = np.zeros_like(input_edge) - 1
    my_fpath_map = np.zeros_like(input_edge) - 1
    sub_mesh = mesh.subgraph(list(cc)).copy()
    pnodes = netx.periphery(sub_mesh)
    ends = [*end_pt]
    edge_id = global_mesh.nodes[ends[0]]['edge_id']
    pnodes = sorted(pnodes, 
                    key=lambda x: np.hypot((x[0] - ends[0][0]), (x[1] - ends[0][1])),
                    reverse=True)[0]
    if npath is None:
        npath = [*netx.shortest_path(sub_mesh, (ends[0][0], ends[0][1]), pnodes, weight='length')]
    else:
        if (ends[0][0], ends[0][1]) == npath[0]:
            npath = npath
        elif (ends[0][0], ends[0][1]) == npath[-1]:
            npath = npath[::-1]
        else:
            import pdb; pdb.set_trace()
    for np_node in npath:
        my_npath_map[np_node[0], np_node[1]] = edge_id
    fpath = []
    if global_mesh.nodes[ends[0]].get('far') is None:
        print("None far")
    else:
        fnodes = global_mesh.nodes[ends[0]].get('far')
        dmask = mask + 0
        did = 0
        while True:
            did += 1
            if did > 3:
                return my_fpath_map, my_npath_map, -1
            dmask = cv2.dilate(dmask, np.ones((3, 3)), iterations=1)
            ffnode = [fnode for fnode in fnodes if (dmask[fnode[0], fnode[1]] > 0 and mask[fnode[0], fnode[1]] == 0 and\
                                                            global_mesh.nodes[fnode].get('inpaint_id') != inpaint_id + 1)]
            if len(ffnode) > 0:
                fnode = ffnode[0]
                break
        
        fpath.append((fnode[0], fnode[1]))
        disp_diff = 0.
        for n_loc in npath:
            if mask[n_loc[0], n_loc[1]] != 0:
                disp_diff = abs(abs(1. / info_on_pix[(n_loc[0], n_loc[1])][0]['depth']) - abs(1. / ends[0][2]))
                break
        barrel_dir = np.array([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]])
        n2f_dir = (int(fnode[0] - npath[0][0]), int(fnode[1] - npath[0][1]))
        while True:
            if barrel_dir[0, 0] == n2f_dir[0] and barrel_dir[0, 1] == n2f_dir[1]:
                n2f_barrel = barrel_dir.copy()
                break
            barrel_dir = np.roll(barrel_dir, 1, axis=0)
        for step in range(0, len(npath)):
            if step == 0:
                continue
            elif step == 1:
                next_dir = (npath[step][0] - npath[step - 1][0], npath[step][1] - npath[step - 1][1])
                while True:
                    if barrel_dir[0, 0] == next_dir[0] and barrel_dir[0, 1] == next_dir[1]:
                        next_barrel = barrel_dir.copy()
                        break
                    barrel_dir = np.roll(barrel_dir, 1, axis=0)
                barrel_pair = np.stack((n2f_barrel, next_barrel), axis=0)
                n2f_dir = (barrel_pair[0, 0, 0], barrel_pair[0, 0, 1])
            elif step > 1:
                next_dir = (npath[step][0] - npath[step - 1][0], npath[step][1] - npath[step - 1][1])
                while True:
                    if barrel_pair[1, 0, 0] == next_dir[0] and barrel_pair[1, 0, 1] == next_dir[1]:
                        next_barrel = barrel_pair.copy()
                        break
                    barrel_pair = np.roll(barrel_pair, 1, axis=1)
                n2f_dir = (barrel_pair[0, 0, 0], barrel_pair[0, 0, 1])
            new_locs = []
            if abs(n2f_dir[0]) == 1:
                new_locs.append((npath[step][0] + n2f_dir[0], npath[step][1]))
            if abs(n2f_dir[1]) == 1:
                new_locs.append((npath[step][0], npath[step][1] + n2f_dir[1]))
            if len(new_locs) > 1:
                new_locs = sorted(new_locs, key=lambda xx: np.hypot((xx[0] - fpath[-1][0]), (xx[1] - fpath[-1][1])))
            break_flag = False
            for new_loc in new_locs:
                new_loc_nes = [xx for xx in [(new_loc[0] + 1, new_loc[1]), (new_loc[0] - 1, new_loc[1]),
                                        (new_loc[0], new_loc[1] + 1), (new_loc[0], new_loc[1] - 1)]\
                                if xx[0] >= 0 and xx[0] < my_fpath_map.shape[0] and xx[1] >= 0 and xx[1] < my_fpath_map.shape[1]]
                if fpath_map is not None and np.all([(fpath_map[nlne[0], nlne[1]] == -1) for nlne in new_loc_nes]) != True:
                    break_flag = True
                    break
                if np.all([(my_fpath_map[nlne[0], nlne[1]] == -1) for nlne in new_loc_nes]) != True:
                    break_flag = True
                    break 
                if my_npath_map[new_loc[0], new_loc[1]] != -1:
                    continue
                if npath_map is not None and npath_map[new_loc[0], new_loc[1]] != edge_id:
                    break_flag = True
                    break
                if valid_map[new_loc[0], new_loc[1]] == 0:
                    break_flag = True
                    break
                fpath.append(new_loc)
            if break_flag is True:
                break
        if step != len(npath) - 1:
            for xx in npath[step:]:
                if my_npath_map[xx[0], xx[1]] == edge_id:
                    my_npath_map[xx[0], xx[1]] = -1
            npath = npath[:step]
        if len(fpath) > 0:
            for fp_node in fpath:
                my_fpath_map[fp_node[0], fp_node[1]] = edge_id

    return my_fpath_map, my_npath_map, disp_diff

def get_MiDaS_samples(image_folder, depth_folder, config, specific=None, aft_certain=None):
    lines = [os.path.splitext(os.path.basename(xx))[0] for xx in glob.glob(os.path.join(image_folder, '*' + config['img_format']))]
    samples = []
    generic_pose = np.eye(4)
    assert len(config['traj_types']) == len(config['x_shift_range']) ==\
           len(config['y_shift_range']) == len(config['z_shift_range']) == len(config['video_postfix']), \
           "The number of elements in 'traj_types', 'x_shift_range', 'y_shift_range', 'z_shift_range' and \
               'video_postfix' should be equal."
    tgt_pose = [[generic_pose * 1]]
    tgts_poses = []
    for traj_idx in range(len(config['traj_types'])):
        tgt_poses = []
        sx, sy, sz = path_planning(config['num_frames'], config['x_shift_range'][traj_idx], config['y_shift_range'][traj_idx],
                                   config['z_shift_range'][traj_idx], path_type=config['traj_types'][traj_idx])
        for xx, yy, zz in zip(sx, sy, sz):
            tgt_poses.append(generic_pose * 1.)
            tgt_poses[-1][:3, -1] = np.array([xx, yy, zz])
        tgts_poses += [tgt_poses]    
    tgt_pose = generic_pose * 1
    
    aft_flag = True
    if aft_certain is not None and len(aft_certain) > 0:
        aft_flag = False
    for seq_dir in lines:
        if specific is not None and len(specific) > 0:
            if specific != seq_dir:
                continue
        if aft_certain is not None and len(aft_certain) > 0:
            if aft_certain == seq_dir:
                aft_flag = True
            if aft_flag is False:
                continue
        samples.append({})
        sdict = samples[-1]            
        sdict['depth_fi'] = os.path.join(depth_folder, seq_dir + config['depth_format'])
        sdict['ref_img_fi'] = os.path.join(image_folder, seq_dir + config['img_format'])
        H, W = imageio.imread(sdict['ref_img_fi']).shape[:2]
        sdict['int_mtx'] = np.array([[max(H, W), 0, W//2], [0, max(H, W), H//2], [0, 0, 1]]).astype(np.float32)
        if sdict['int_mtx'].max() > 1:
            sdict['int_mtx'][0, :] = sdict['int_mtx'][0, :] / float(W)
            sdict['int_mtx'][1, :] = sdict['int_mtx'][1, :] / float(H)
        sdict['ref_pose'] = np.eye(4)
        sdict['tgt_pose'] = tgt_pose
        sdict['tgts_poses'] = tgts_poses
        sdict['video_postfix'] = config['video_postfix']
        sdict['tgt_name'] = [os.path.splitext(os.path.basename(sdict['depth_fi']))[0]]
        sdict['src_pair_name'] = sdict['tgt_name'][0]

    return samples

def get_valid_size(imap):
    x_max = np.where(imap.sum(1).squeeze() > 0)[0].max() + 1
    x_min = np.where(imap.sum(1).squeeze() > 0)[0].min()
    y_max = np.where(imap.sum(0).squeeze() > 0)[0].max() + 1
    y_min = np.where(imap.sum(0).squeeze() > 0)[0].min()
    size_dict = {'x_max':x_max, 'y_max':y_max, 'x_min':x_min, 'y_min':y_min}
    
    return size_dict

def dilate_valid_size(isize_dict, imap, dilate=[0, 0]):
    osize_dict = copy.deepcopy(isize_dict)
    osize_dict['x_min'] = max(0, osize_dict['x_min'] - dilate[0])
    osize_dict['x_max'] = min(imap.shape[0], osize_dict['x_max'] + dilate[0])
    osize_dict['y_min'] = max(0, osize_dict['y_min'] - dilate[0])
    osize_dict['y_max'] = min(imap.shape[1], osize_dict['y_max'] + dilate[1])

    return osize_dict

def crop_maps_by_size(size, *imaps):
    omaps = []
    for imap in imaps:
        omaps.append(imap[size['x_min']:size['x_max'], size['y_min']:size['y_max']].copy())
    
    return omaps

def read_MiDaS_depth(disp_fi, disp_rescale=10., h=None, w=None):
    if 'npy' in os.path.splitext(disp_fi)[-1]:
        disp = np.load(disp_fi)
    else:
        disp = imageio.imread(disp_fi, as_gray=True).astype(np.float32)
    disp = disp - disp.min()
    disp = cv2.blur(disp / disp.max(), ksize=(3, 3)) * disp.max()
    disp = (disp / disp.max()) * disp_rescale
    if h is not None and w is not None:
        disp = resize(disp / disp.max(), (h, w), order=1) * disp.max()
    depth = 1. / np.maximum(disp, 0.05)

    return depth

def follow_image_aspect_ratio(depth, image):
    H, W = image.shape[:2]
    image_aspect_ratio = H / W
    dH, dW = depth.shape[:2]
    depth_aspect_ratio = dH / dW
    if depth_aspect_ratio > image_aspect_ratio:
        resize_H = dH
        resize_W = dH / image_aspect_ratio
    else:
        resize_W = dW
        resize_H = dW * image_aspect_ratio
    depth = resize(depth / depth.max(), 
                    (int(resize_H), 
                    int(resize_W)), 
                    order=0) * depth.max()
    
    return depth

def depth_resize(depth, origin_size, image_size):
    if origin_size[0] is not 0:
        max_depth = depth.max()
        depth = depth / max_depth
        depth = resize(depth, origin_size, order=1, mode='edge')
        depth = depth * max_depth
    else:
        max_depth = depth.max()
        depth = depth / max_depth
        depth = resize(depth, image_size, order=1, mode='edge')
        depth = depth * max_depth

    return depth

def vis_depth_edge_connectivity(depth, config):
    disp = 1./depth
    u_diff = (disp[1:, :] - disp[:-1, :])[:-1, 1:-1]
    b_diff = (disp[:-1, :] - disp[1:, :])[1:, 1:-1]
    l_diff = (disp[:, 1:] - disp[:, :-1])[1:-1, :-1]
    r_diff = (disp[:, :-1] - disp[:, 1:])[1:-1, 1:]
    u_over = (np.abs(u_diff) > config['depth_threshold']).astype(np.float32)
    b_over = (np.abs(b_diff) > config['depth_threshold']).astype(np.float32)
    l_over = (np.abs(l_diff) > config['depth_threshold']).astype(np.float32)
    r_over = (np.abs(r_diff) > config['depth_threshold']).astype(np.float32)
    concat_diff = np.stack([u_diff, b_diff, r_diff, l_diff], axis=-1)
    concat_over = np.stack([u_over, b_over, r_over, l_over], axis=-1)
    over_diff = concat_diff * concat_over
    pos_over = (over_diff > 0).astype(np.float32).sum(-1).clip(0, 1)
    neg_over = (over_diff < 0).astype(np.float32).sum(-1).clip(0, 1)
    neg_over[(over_diff > 0).astype(np.float32).sum(-1) > 0] = 0
    _, edge_label = cv2.connectedComponents(pos_over.astype(np.uint8), connectivity=8)
    T_junction_maps = np.zeros_like(pos_over)
    for edge_id in range(1, edge_label.max() + 1):
        edge_map = (edge_label == edge_id).astype(np.uint8)
        edge_map = np.pad(edge_map, pad_width=((1,1),(1,1)), mode='constant')
        four_direc = np.roll(edge_map, 1, 1) + np.roll(edge_map, -1, 1) + np.roll(edge_map, 1, 0) + np.roll(edge_map, -1, 0)
        eight_direc = np.roll(np.roll(edge_map, 1, 1), 1, 0) + np.roll(np.roll(edge_map, 1, 1), -1, 0) + \
                      np.roll(np.roll(edge_map, -1, 1), 1, 0) + np.roll(np.roll(edge_map, -1, 1), -1, 0)
        eight_direc = (eight_direc + four_direc)[1:-1,1:-1]
        pos_over[eight_direc > 2] = 0
        T_junction_maps[eight_direc > 2] = 1
    _, edge_label = cv2.connectedComponents(pos_over.astype(np.uint8), connectivity=8)
    edge_label = np.pad(edge_label, 1, mode='constant')

    return edge_label



def max_size(mat, value=0):
    if not (mat and mat[0]): return (0, 0)
    it = iter(mat)
    prev = [(el==value) for el in next(it)]
    max_size = max_rectangle_size(prev)
    for row in it:
        hist = [(1+h) if el == value else 0 for h, el in zip(prev, row)]
        max_size = max(max_size, max_rectangle_size(hist), key=get_area)
        prev = hist                                               
    return max_size

def max_rectangle_size(histogram):
    Info = namedtuple('Info', 'start height')
    stack = []
    top = lambda: stack[-1]
    max_size = (0, 0) # height, width of the largest rectangle
    pos = 0 # current position in the histogram
    for pos, height in enumerate(histogram):
        start = pos # position where rectangle starts
        while True:
            if not stack or height > top().height:
                stack.append(Info(start, height)) # push
            if stack and height < top().height:
                max_size = max(max_size, (top().height, (pos-top().start)),
                               key=get_area)
                start, _ = stack.pop()
                continue
            break # height == top().height goes here
                
    pos += 1
    for start, height in stack:
        max_size = max(max_size, (height, (pos-start)),
                       key=get_area)

    return max_size

def get_area(size):
    return reduce(mul, size)

def find_anchors(matrix):
    matrix = [[*x] for x in matrix]
    mh, mw = max_size(matrix)
    matrix = np.array(matrix)
    # element = np.zeros((mh, mw))
    for i in range(matrix.shape[0] + 1 - mh):
        for j in range(matrix.shape[1] + 1 - mw):
            if matrix[i:i + mh, j:j + mw].max() == 0:
                return i, i + mh, j, j + mw
