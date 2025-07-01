import matplotlib
matplotlib.use('Agg')
import os, sys
import yaml
from argparse import ArgumentParser
from tqdm import tqdm
import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch

from scipy.spatial import ConvexHull
import scipy.io as io
import json
import cv2
import torch.nn.functional as F
import struct, time
from pathlib import Path

from GFVC.DS.sync_batchnorm import DataParallelWithCallback
from GFVC.DS.modules.util import *
from GFVC.DS.modules.generator import OcclusionAwareGenerator ###
from GFVC.DS.modules.keypoint_detector import Encoder as Kpdetector ###
from GFVC.DS.animate import normalize_kp
from GFVC.DS.modules.CFF import cff4, Depth_Net
from GFVC.DS.modules.RDloss import VideoCompressor
from GFVC.DS.modules.generator_refine import generator_refine


def load_checkpoints(config_path, checkpoint_path, cpu=False):
    with open(config_path) as f:
        # config = yaml.load(f)
        config = yaml.load(f.read(), Loader=yaml.FullLoader)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    generator_Refine = generator_refine()
    if not cpu:
        generator.cuda()
        generator_Refine.cuda()
    videocompressor = VideoCompressor(**config['model_params']['videocompressor_params']).to('cuda')
    # kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
    #                          **config['model_params']['common_params'])
    kp_detector = Kpdetector()
    cross_scale_m_fusion = cff4(10).to('cuda')
    depth_net = Depth_Net()
    if not cpu:
        kp_detector.cuda()
        depth_net.cuda()
    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)
    generator.load_state_dict(checkpoint['generator'], strict=False)
    kp_detector.load_state_dict(checkpoint['kp_detector'], strict=False)  ####
    cross_scale_m_fusion.load_state_dict(checkpoint['cross_scale_m_fusion'], strict=False)
    videocompressor.load_state_dict(checkpoint['videocompressor'], strict=False)
    generator_Refine.load_state_dict(checkpoint['Generator_Refine'], strict=False)

    if not cpu:
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()
    cross_scale_m_fusion.eval()
    videocompressor.eval()
    return kp_detector, generator, cross_scale_m_fusion, depth_net, videocompressor


def make_prediction(reference_frame, kp_reference, kp_current, generator, M_i, generator_Refine, relative=False, adapt_movement_scale=False, cpu=False):
        
    kp_norm = normalize_kp(kp_source=kp_reference, kp_driving=kp_current,
                           kp_driving_initial=kp_reference, use_relative_movement=relative,
                           use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
    
    out = generator(reference_frame, kp_reference, kp_current, M_i)

    prediction=np.transpose(out.data.cpu().numpy(), [0, 1, 2, 3])[0]

    return prediction



