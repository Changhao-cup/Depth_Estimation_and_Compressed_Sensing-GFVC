import numpy as np
import torch
import torch.nn.functional as F
import imageio
from torch import nn

import os
from skimage.draw import circle

import matplotlib.pyplot as plt
import collections
from modules.util import make_coordinate_grid
from flowvisual import *
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir, checkpoint_freq=100, visualizer_params=None, zfill_num=8, log_file_name='log.txt'):

        self.loss_list = []
        self.cpk_dir = log_dir
        self.visualizations_dir = os.path.join(log_dir, 'train-vis')
        if not os.path.exists(self.visualizations_dir):
            os.makedirs(self.visualizations_dir)
        self.log_file = open(os.path.join(log_dir, log_file_name), 'a')
        self.zfill_num = zfill_num
        self.visualizer = Visualizer(**visualizer_params)
        self.checkpoint_freq = checkpoint_freq
        self.epoch = 0
        self.best_loss = float('inf')
        self.names = None
        self.writer = SummaryWriter("runs/output_picture")

    def log_scores(self, loss_names):
        loss_mean = np.array(self.loss_list).mean(axis=0)

        loss_string = "; ".join(["%s - %.5f" % (name, value) for name, value in zip(loss_names, loss_mean)])
        loss_string = str(self.epoch).zfill(self.zfill_num) + ") " + loss_string

        print(loss_string, file=self.log_file)
        print(loss_string)
        self.loss_list = []
        self.log_file.flush()

    def visualize_rec(self, inp, out):
        image = self.visualizer.visualize(inp['driving'], inp['source'], out)
        imageio.imsave(os.path.join(self.visualizations_dir, "%s-rec.png" % str(self.epoch).zfill(self.zfill_num)), image)

    def visualize_rec_training(self, inp, out, batch_idx, tag):
        image = self.visualizer.visualize(inp['driving'], inp['source'], out)
        image = torch.from_numpy(image/255)
        self.writer.add_images(tag, image.permute(2, 0, 1).unsqueeze(0), global_step=batch_idx)
        self.writer.flush()


    def save_cpk(self, emergent=False):
        cpk = {k: v.state_dict() for k, v in self.models.items()}
        cpk['epoch'] = self.epoch
        cpk_path = os.path.join(self.cpk_dir, '%s-checkpoint.pth.tar' % str(self.epoch).zfill(self.zfill_num)) 
        if not (os.path.exists(cpk_path) and emergent):
            torch.save(cpk, cpk_path)

    @staticmethod
    def load_cpk(checkpoint_path, generator=None, Generator_Refine=None, discriminator=None, kp_detector=None, videocompressor=None, cross_scale_m_fusion=None,
                 optimizer_generator=None, optimizer_Generator_Refine=None, optimizer_discriminator=None, optimizer_kp_detector=None, optimizer_cross_scale_m_fusion=None,
                 optimizer_videocompressor=None):
        checkpoint = torch.load(checkpoint_path)
        if generator is not None:
            generator.load_state_dict(checkpoint['generator'])
        if Generator_Refine is not None:
            Generator_Refine.load_state_dict(checkpoint['Generator_Refine'])
        if kp_detector is not None:
            kp_detector.load_state_dict(checkpoint['kp_detector'])
        if discriminator is not None:
            try:
               discriminator.load_state_dict(checkpoint['discriminator'])
            except:
               print ('No discriminator in the state-dict. Dicriminator will be randomly initialized')
        if videocompressor is not None:
            videocompressor.load_state_dict(checkpoint['videocompressor'])
        if cross_scale_m_fusion is not None:
            cross_scale_m_fusion.load_state_dict(checkpoint['cross_scale_m_fusion'])
        if optimizer_generator is not None:
            optimizer_generator.load_state_dict(checkpoint['optimizer_generator'])
        if optimizer_Generator_Refine is not None:
            optimizer_Generator_Refine.load_state_dict(checkpoint['optimizer_Generator_Refine'])
        if optimizer_discriminator is not None:
            try:
                optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator'])
            except RuntimeError as e:
                print ('No discriminator optimizer in the state-dict. Optimizer will be not initialized')
        if optimizer_kp_detector is not None:
            optimizer_kp_detector.load_state_dict(checkpoint['optimizer_kp_detector'])
        if optimizer_cross_scale_m_fusion is not None:
            optimizer_cross_scale_m_fusion.load_state_dict(checkpoint['optimizer_cross_scale_m_fusion'])
        if optimizer_videocompressor is not None:
            optimizer_videocompressor.load_state_dict(checkpoint['optimizer_videocompressor'])

        return 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if 'models' in self.__dict__:
            self.save_cpk()
        self.log_file.close()

    def log_iter(self, losses):
        losses = collections.OrderedDict(losses.items())
        if self.names is None:
            self.names = list(losses.keys())
        self.loss_list.append(list(losses.values()))

    def log_epoch(self, epoch, models, inp, out):
        self.epoch = epoch
        self.models = models
        if (self.epoch + 1) % self.checkpoint_freq == 0:
            self.save_cpk()
        self.log_scores(self.names)
        self.visualize_rec(inp, out)

    def loss_print(self):
        self.log_scores(self.names)


class Visualizer:
    def __init__(self, kp_size=5, draw_border=False, colormap='gist_rainbow'):
        self.kp_size = kp_size
        self.draw_border = draw_border
        self.colormap = plt.get_cmap(colormap)
            
    def draw_image_with_kp(self, image, kp_array):
        image = np.copy(image)
        spatial_size = np.array(image.shape[:2][::-1])[np.newaxis] #[[256 256]]
        kp_array = spatial_size * (kp_array + 1) / 2 #(10, 64, 64, 2)
        num_kp = kp_array.shape[0] #10

        for kp_ind, kp in enumerate(kp_array):
            print(kp_ind)
            print(kp)
            
            rr, cc = circle(kp[1], kp[0], self.kp_size, shape=image.shape[:2])
            image[rr, cc] = np.array(self.colormap(kp_ind / num_kp))[:3]
        return image

    def create_image_column_with_kp(self, images, kp):
        image_array = np.array([self.draw_image_with_kp(v, k) for v, k in zip(images, kp)])
        return self.create_image_column(image_array)

    def create_image_column(self, images):
        if self.draw_border:
            images = np.copy(images)
            images[:, :, [0, -1]] = (1, 1, 1)
            images[:, :, [0, -1]] = (1, 1, 1)
        return np.concatenate(list(images), axis=0)

    def create_image_grid(self, *args):
        out = []
        for arg in args:
            if type(arg) == tuple:
                out.append(self.create_image_column_with_kp(arg[0], arg[1]))
            else:
                out.append(self.create_image_column(arg))
        return np.concatenate(out, axis=1)

    def visualize(self, driving, source, out):
        images = []

        # Source image 
        source = source.data.cpu()
        source = np.transpose(source, [0, 2, 3, 1])
        images.append((source))
        
        # Driving image 
        driving = driving.data.cpu().numpy()
        driving = np.transpose(driving, [0, 2, 3, 1])
        images.append((driving))        

        if 'depth_source' in out:
            depth_driving = out['depth_source'].data.cpu().numpy()
            bs, h, w, c = depth_driving.shape
            depth = []
            for batch in range(0, bs):
                de = depth_to_image(depth_driving[batch:batch+1,:,:,:].reshape(h, w, c))
                depth.append(de)
            depth_image = np.array(depth)
            depth_image = np.transpose(depth_image, [0,3,1,2])
            depth_image = torch.from_numpy(depth_image).type(source.type())

            depth_image = F.interpolate(depth_image, size=source.shape[1:3]).numpy()
            depth_image = np.transpose(depth_image, [0, 2, 3, 1])
            images.append(depth_image)

        if 'depth_driving_prediction' in out:
            depth_driving = out['depth_driving_prediction'].data.cpu().numpy()
            bs, h, w, c = depth_driving.shape
            depth = []
            for batch in range(0, bs):
                de = depth_to_image(depth_driving[batch:batch+1,:,:,:].reshape(h, w, c))
                depth.append(de)
            depth_image = np.array(depth)
            depth_image = np.transpose(depth_image, [0,3,1,2])
            depth_image = torch.from_numpy(depth_image).type(source.type())

            depth_image = F.interpolate(depth_image, size=source.shape[1:3]).numpy()
            depth_image = np.transpose(depth_image, [0, 2, 3, 1])
            images.append(depth_image)
        
        ### sparse motion deformed image
        if 'sparse_deformed' in out:        
            sparse_deformed = out['sparse_deformed'].data.cpu().repeat(1, 1, 1, 1)
            sparse_deformed = F.interpolate(sparse_deformed, size=source.shape[1:3]).numpy()
            sparse_deformed = np.transpose(sparse_deformed, [0, 2, 3, 1])
            images.append(sparse_deformed)

        if 'spatial_deformed' in out:
            sparse_deformed = out['spatial_deformed'].data.cpu().repeat(1, 1, 1, 1)
            sparse_deformed = F.interpolate(sparse_deformed, size=source.shape[1:3]).numpy()
            sparse_deformed = np.transpose(sparse_deformed, [0, 2, 3, 1])
            images.append(sparse_deformed)

        # if 'depth_motion' in out:
        #     depth_motion = out['depth_motion'].data.cpu().numpy()
        #
        #     bs, h, w, c = depth_motion.shape
        #     flow=[]
        #     for batch in range(0,bs):
        #         sf =flow_to_image(depth_motion[batch:batch+1,:,:,:].reshape(h, w, c))
        #         flow.append(sf)
        #
        #     depth_motion= np.array(flow)
        #     depth_motion = np.transpose(depth_motion, [0, 3, 1, 2])
        #     depth_motion = torch.from_numpy(depth_motion).type(source.type())  ###.type(dtype=torch.float64)
        #     depth_motion = F.interpolate(depth_motion, size=source.shape[1:3]).numpy()
        #     depth_motion = np.transpose(depth_motion, [0, 2, 3, 1])
        #     images.append(depth_motion)


        #Dense Flow
        if 'deformation_depth' in out:
            denseflow = out['deformation_depth'].data.cpu().numpy()

            bs, h, w, c = denseflow.shape
            flow = []
            for batch in range(0, bs):
                df = flow_to_image(denseflow[batch:batch + 1, :, :, :].reshape(h, w, c))
                flow.append(df)

            dense_flow = np.array(flow)
            dense_flow = np.transpose(dense_flow, [0, 3, 1, 2])
            dense_flow = torch.from_numpy(dense_flow).type(source.type())
            dense_flow = F.interpolate(dense_flow, size=source.shape[1:3]).numpy()
            dense_flow = np.transpose(dense_flow, [0, 2, 3, 1])
            images.append(dense_flow)

                # denseflow Deformed image
        if 'prediction_deformation' in out:
            deformed = out['prediction_deformation'].data.cpu().numpy()
            deformed = np.transpose(deformed, [0, 2, 3, 1])
            images.append(deformed)

                ## Occlusion map

        if 'occlusion_map_depth' in out:
            occlusion_map = out['occlusion_map_depth'].data.cpu().repeat(1, 3, 1, 1)
            occlusion_map = F.interpolate(occlusion_map, size=source.shape[1:3]).numpy()
            occlusion_map = np.transpose(occlusion_map, [0, 2, 3, 1])
            images.append(occlusion_map)

        # Driving Result 
        prediction = out['prediction'].data.cpu().numpy()
        prediction = np.transpose(prediction, [0, 2, 3, 1])
        images.append(prediction)

        image = self.create_image_grid(*images)
        image = (255 * image).astype(np.uint8)
        return image
