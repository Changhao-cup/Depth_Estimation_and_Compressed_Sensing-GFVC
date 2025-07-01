import torch
from torch import nn
import torch.nn.functional as F
from modules.util import *
from modules.dense_motion import DenseMotionNetwork
from modules.util import AntiAliasInterpolation2d, make_coordinate_grid
from .GDN import GDN
import math
from modules.flowwarp import *

import depth


class DepthAwareAttention(nn.Module):
    """ depth-aware attention Layer"""

    def __init__(self, in_dim, activation):
        super(DepthAwareAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, source, feat):
        """
            inputs :
                source : input feature maps( B X C X W X H) 256,64,64
                driving : input feature maps( B X C X W X H) 256,64,64
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = source.size()
        proj_query = self.activation(self.query_conv(source)).view(m_batchsize, -1, width * height).permute(0, 2,
                                                                                                            1)  # B X CX(N) [bz,32,64,64]
        proj_key = self.activation(self.key_conv(feat)).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.activation(self.value_conv(feat)).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        out = self.gamma * out + feat

        return out, attention
class OcclusionAwareGenerator(nn.Module):
    """
    Generator that given source image and and keypoints try to transform image according to movement trajectories
    induced by keypoints. Generator follows Johnson architecture.
    """

    def __init__(self, num_channels, num_kp, block_expansion, max_features, num_down_blocks,
                 num_bottleneck_blocks, estimate_occlusion_map=False, dense_motion_params=None, estimate_jacobian=False,
                 dense_motion_params_depth=None):
        super(OcclusionAwareGenerator, self).__init__()


        self.temperature =0.1

        if dense_motion_params is not None:
            self.dense_motion_network = DenseMotionNetwork(num_kp=num_kp, num_channels=num_channels,
                                                           estimate_occlusion_map=estimate_occlusion_map,
                                                           **dense_motion_params)
        else:
            self.dense_motion_network = None


        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DHFE_DownBlock2d(in_features, out_features))
        self.down_blocks = nn.ModuleList(down_blocks)

        up_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
            up_blocks.append(DHFE_UpBlock2d(in_features, out_features))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.bottleneck = torch.nn.Sequential()
        in_features = min(max_features, block_expansion * (2 ** num_down_blocks))
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))
        bottleneck = []
        # self.bottleneck = torch.nn.Sequential()
        in_features = min(max_features, block_expansion * (2 ** num_down_blocks))
        for i in range(num_bottleneck_blocks):
            bottleneck.append(ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))
        self.bottleneck = nn.ModuleList(bottleneck)

        # self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        self.final = DHFE(block_expansion, num_channels)
        self.estimate_occlusion_map = estimate_occlusion_map
        self.num_channels = num_channels
        self.num_kp = num_kp
        self.block_size = 32

        # self.CompressFeatureFusion = cff3(1)

        self.src_first = SameBlock2d(1, block_expansion, kernel_size=(7, 7), padding=(3, 3))
        src_down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            src_down_blocks.append(DHFE(in_features, out_features))
        self.src_down_blocks = nn.ModuleList(src_down_blocks)

        self.depth_encoder = depth.ResnetEncoder(50, False).cuda()
        self.depth_decoder = depth.DepthDecoder(num_ch_enc=self.depth_encoder.num_ch_enc, scales=range(4)).cuda()
        loaded_dict_enc = torch.load('/media/lab/CH/depth/models/depth_face_model_Voxceleb2_10w/encoder.pth', map_location='cpu')
        loaded_dict_dec = torch.load('/media/lab/CH/depth/models/depth_face_model_Voxceleb2_10w/depth.pth', map_location='cpu')
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.depth_encoder.state_dict()}
        self.depth_encoder.load_state_dict(filtered_dict_enc)
        self.depth_decoder.load_state_dict(loaded_dict_dec)
        self.set_requires_grad(self.depth_encoder, False)
        self.set_requires_grad(self.depth_decoder, False)
        self.depth_decoder.eval()
        self.depth_encoder.eval()
        self.src_first = SameBlock2d(1, block_expansion, kernel_size=(7, 7), padding=(3, 3))
        self.AttnModule = DepthAwareAttention(256, nn.ReLU())
        self.decoder = SPADEGenerator(512, 64, 2, 6)
    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode='bilinear')
            deformation = deformation.permute(0, 2, 3, 1)
        return warp(inp, deformation) #F.grid_sample(inp, deformation)  #########

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def forward(self, source_image, heatmap_source, heatmap_driving, M_i):
        
        # Encoding (downsampling) part
        out = self.first(source_image)
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
        # Transforming feature representation according to deformation and occlusion

        outputs_source_depth = self.depth_decoder(self.depth_encoder(source_image))

        driving_depth = self.src_first(M_i)
        for i in range(len(self.src_down_blocks)):
            driving_depth = self.src_down_blocks[i](driving_depth)
        output_dict = {}
        dense_motion = self.dense_motion_network(source_image=source_image,heatmap_source=heatmap_source,
                                         heatmap_driving=heatmap_driving, depth_source=outputs_source_depth['disp', 2],
                                        depth_driving=M_i)

        occlusion_map = dense_motion['occlusion_map']  #64*64*1
        output_dict['occlusion_map'] = occlusion_map

        deformation = dense_motion['deformation'] #64*64*2
        output_dict['deformation'] = deformation

        deformed_sparse_source = dense_motion['sparse_deformed'] #64*64*3
        output_dict['sparse_deformed'] = deformed_sparse_source

        output_dict["deformed"] = self.deform_input(source_image, deformation)


        deformation_depth = dense_motion['deformation_depth']
        output_dict["deformation_depth"] = deformation_depth


        output_dict["deformed"] = self.deform_input(source_image, deformation)   #### deformation 64*64 interpolate 256*256


        out = self.deform_input(out, deformation)
        out = out * occlusion_map
        out, _ = self.AttnModule(driving_depth, out)

        prediction_deformation = self.deform_input(source_image, deformation_depth)
        output_dict['prediction_deformation'] = self.deform_input(source_image, deformation_depth)
        out = self.decoder(out, prediction_deformation)

        return out


class SPADEGenerator(nn.Module):
    def __init__(self, max_features, block_expansion, num_down_blocks, num_bottleneck_blocks):
        super().__init__()
        ic = 256
        cc = 4
        oc = 64
        norm_G = 'spadespectralinstance'
        label_nc = 3 + cc

        self.compress = nn.Conv2d(ic, cc, 3, padding=1)
        self.fc = nn.Conv2d(ic, 2 * ic, 3, padding=1)

        self.G_middle_0 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_1 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_2 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        # self.G_middle_3 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        # self.G_middle_4 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        # self.G_middle_5 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.GD = DHFE(max_features, max_features//2)
        self.bottleneck = torch.nn.Sequential()
        in_features = min(max_features, block_expansion * (2 ** num_down_blocks))
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))
        up_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
            up_blocks.append(DHFE_UpBlock2d(in_features, out_features))
        self.up_blocks = nn.ModuleList(up_blocks)
        # self.up = nn.Upsample(scale_factor=2)

    def forward(self, feature, image):
        cp = self.compress(feature)
        seg = torch.cat((F.interpolate(cp, size=(image.shape[2], image.shape[3])), image), dim=1)  # 7, 256, 256

        x = feature  # 256, 64, 64
        x = self.fc(x)  # 512, 64, 64
        x = self.G_middle_0(x, seg)
        x = self.G_middle_1(x, seg)
        x = self.G_middle_2(x, seg)
        # x = self.G_middle_3(x, seg)
        # x = self.G_middle_4(x, seg)
        # x = self.G_middle_5(x, seg)
        x = self.GD(x)
        x = self.bottleneck(x)
        for i in range(len(self.up_blocks)):
            x = self.up_blocks[i](x)
        x = self.conv_img(x)
        x = F.sigmoid(x)
        return x