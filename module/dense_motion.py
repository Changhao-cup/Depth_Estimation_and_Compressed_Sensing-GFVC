from torch import nn
import torch.nn.functional as F
import torch
from modules.util import *
import cv2
import numpy as np
from torch.autograd import Variable
import torchvision.transforms as transforms
from modules.vggloss import *
from modules.flowwarp import *
from einops.layers.torch import Rearrange


# USE_CUDA = torch.cuda.is_available()
# device = torch.device("cuda:0" if USE_CUDA else "cpu")
# -
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")
#

class DenseMotionNetwork(nn.Module):
    """
    Module that predicting a dense motion from sparse motion representation given by kp_source and kp_driving
    """

    def __init__(self, block_expansion, num_blocks, num_down_blocks, max_features, num_kp, num_channels, num_bottleneck_blocks,
                 estimate_occlusion_map=False, scale_factor=1, kp_variance=0.01):

        super(DenseMotionNetwork, self).__init__()
        self.hourglass = Hourglass(block_expansion=block_expansion, in_features=num_channels + 1,
                                   max_features=max_features, num_blocks=num_blocks)
        A = torch.from_numpy(self.load_sampling_matrix()).float()
        self.A = nn.Parameter(
            Rearrange("m (1 b1 b2) -> m 1 b1 b2", b1=32)(A),
            requires_grad=True,
        )
        self.flow = nn.Conv2d(self.hourglass.out_filters, 2, kernel_size=(7, 7), padding=(3, 3))
        self.flow2 = nn.Conv2d(4, 2, kernel_size=(7, 7), padding=(3, 3))

        # if estimate_occlusion_map:
        #     self.occlusion = nn.Conv2d(self.hourglass.out_filters, 1, kernel_size=(7, 7), padding=(3, 3))
        # else:
        #     self.occlusion = None

        self.num_kp = num_kp
        self.scale_factor = scale_factor
        self.kp_variance = kp_variance
        self.num_down_blocks = num_down_blocks

        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)
            # self.down_128 = AntiAliasInterpolation2d(num_channels, self.scale_factor * 2)

        ###heatmap_difference upscale    
        up_blocks = []
        for i in range(num_down_blocks + 1):
            up_blocks.append(DHFE_UpBlock2d(num_kp, num_kp))
        self.up_blocks = nn.ModuleList(up_blocks)

        ####sparse motion warp---downscale-->upscale

        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))

        # motiondown_blocks = []
        depth_motion_down = []
        spatial_motion_down = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            spatial_motion_down.append(DHFE_DownBlock2d(in_features, out_features))
            depth_motion_down.append(DHFE(in_features, out_features))
        spatial_motion_down.append(DHFE_DownBlock2d(in_features, out_features))
        self.spatial_motion_down = nn.ModuleList(spatial_motion_down)
        self.depth_motion_down = nn.ModuleList(depth_motion_down)

        spatial_motion_up = []
        depth_motion_up = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
            spatial_motion_up.append(DHFE_UpBlock2d(in_features, out_features))
            depth_motion_up.append(DHFE(in_features, out_features))
        self.spatial_motion_up = nn.ModuleList(spatial_motion_up)
        self.depth_motion_up = nn.ModuleList(depth_motion_up)

        self.bottleneck = torch.nn.Sequential()
        in_features = min(max_features, block_expansion * (2 ** num_down_blocks))
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))

        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        self.conv4 = DHFE(num_kp * 64, num_kp * 102)
        self.conv3 = DHFE(num_kp * 32, num_kp*64)
        self.conv2 = DHFE(num_kp * 8, num_kp * 32)
        self.conv1 = DHFE(num_kp, num_kp * 8)

        self.dh  = DHFE(1, 1)


    def create_heatmap_representations(self, source_image, heatmap_driving, heatmap_source):
        """
        Eq 6. in the paper H_k(z)
        8*8 Feature-->upscale-->64*64 Feature---->Feature Difference ####torch.Size([40, 1, 64, 64])
        """

        # heatmap = heatmap_driving['value']  - heatmap_source['value']
        bs, _, h, w = heatmap_driving.shape

        heatmap = heatmap_driving - heatmap_source
        return heatmap

    ###Gunnar Farneback算法计算稠密光流
    def create_sparse_motions(self, source_image, heatmap_driving, heatmap_source):
        """
        Eq 4. in the paper T_{s<-d}(z)
        """

        # feature map-->img-->sparse motion detecion point p0 and p1
        heatmap_source_lt = heatmap_source
        heatmap_source_lt = heatmap_source_lt.cuda().data.cpu().numpy()
        heatmap_source_lt = (heatmap_source_lt - np.min(heatmap_source_lt)) / (
                    np.max(heatmap_source_lt) - np.min(heatmap_source_lt)) * 255.0  # 转为0-255
        heatmap_source_lt = np.round(heatmap_source_lt)  # 转换数据类型
        heatmap_source_lt = heatmap_source_lt.astype(np.uint8)

        heatmap_driving_lt = heatmap_driving
        heatmap_driving_lt = heatmap_driving_lt.cuda().data.cpu().numpy()
        heatmap_driving_lt = (heatmap_driving_lt - np.min(heatmap_driving_lt)) / (
                    np.max(heatmap_driving_lt) - np.min(heatmap_driving_lt)) * 255.0  # 转为0-255
        heatmap_driving_lt = np.round(heatmap_driving_lt)  # 转换数据类型
        heatmap_driving_lt = heatmap_driving_lt.astype(np.uint8)

        bs, _, h, w = source_image.shape  ##bs=40

        GFflow = []
        for tensorchannel in range(0, bs):
            heatmap_source_lt11 = heatmap_source_lt[tensorchannel].transpose([1, 2, 0])  # 取出其中一张并转换维度
            heatmap_driving_lt11 = heatmap_driving_lt[tensorchannel].transpose([1, 2, 0])  # 取出其中一张并转换维度
            # flow = cv2.calcOpticalFlowFarneback(heatmap_driving_lt11,heatmap_source_lt11,None, 0.5, 2, 15, 3, 5, 1.2, 0)  ####
            flow = cv2.calcOpticalFlowFarneback(heatmap_source_lt11, heatmap_driving_lt11, None, 0.5, 2, 15, 3, 5, 1.2, 0)  ####
            GFflow.append(flow)

        tmp_flow = torch.Tensor(np.array(GFflow)).cuda()  # .to(device)
        return tmp_flow


    def create_deformed_source_image(self, source_image, sparse_motion):
        ''' [bs, 3, 64, 64])-->[bs, 64, 64, 64]-->[bs, 128, 32, 32]-->[bs, 256, 16, 16]-->[bs, 512, 8, 8]
        '''
        # Encoding (downsampling) part
        bs, h, w, _ = sparse_motion.shape

        out = self.first(source_image)
        for i in range(self.num_down_blocks):
            out = self.depth_motion_down[i](out)
            ########
        # warping
        out = warp(out, sparse_motion)
        out = self.DH(out)
        out = warp(out, sparse_motion)

        # Decoding part
        out = self.bottleneck(out)
        for i in range(self.num_down_blocks):
            out = self.depth_motion_up[i](out)
        ##deformed image
        out = self.final(out)
        return out

    def forward(self, source_image, heatmap_source, heatmap_driving, depth_source, depth_driving):
        if self.scale_factor != 1:
            source_image = self.down(source_image)
        # bs, c, h, w = source_image_.shape
        out_dict = dict()

        bs, c, h, w = source_image.shape
        heatmap_representation = self.create_heatmap_representations(source_image, heatmap_source, heatmap_driving)
        depth_motion = self.create_sparse_motions(source_image, depth_source, depth_driving)
        deformed_source = self.create_deformed_source_image(source_image, depth_motion)
        out_dict['sparse_deformed'] = deformed_source
        heatmap_representation = heatmap_representation.unsqueeze(1).view(bs, 1, -1, h, w)
        deformed_source = deformed_source.unsqueeze(1).view(bs, 1, -1, h, w)
        input = torch.cat([heatmap_representation, deformed_source], dim=2)
        input = input.view(bs, -1, h, w)
        prediction = self.hourglass(input)
        deformation = self.flow(prediction)  ##([40, 2, 64, 64])
        deformation = deformation.permute(0, 2, 3, 1)
        out_dict['deformation_depth'] = deformation
        # occulusion map
        if self.occlusion:
            occlusion_map = torch.sigmoid(self.occlusion(prediction))  ##([40, 1, 64, 64])
            # print(occlusion_map.shape)
            out_dict['occlusion_map'] = occlusion_map
        return out_dict