# -*- coding: utf-8 -*-
from tqdm import trange, tqdm
import torch

from torch.utils.data import DataLoader

from logger import Logger
from modules.model import GeneratorFullModel, DiscriminatorFullModel

from torch.optim.lr_scheduler import MultiStepLR

from sync_batchnorm import DataParallelWithCallback

from frames_dataset import DatasetRepeater
from modules.util import get_parameter_groups


def train(config, generator, cross_scale_m_fusion, discriminator, kp_detector, videocompressor, Generator_Refine,
          checkpoint, log_dir, dataset, dataset_test, device_ids):
    train_params = config['train_params']
    
    rdlambdas = config['train_params']['loss_weights']['rdlambda']
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=train_params['lr_generator'], betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=train_params['lr_discriminator'], betas=(0.5, 0.999))
    optimizer_videocompressor = torch.optim.Adam(videocompressor.parameters(), lr=train_params['lr_videocompressor'], betas=(0.5, 0.999))
    optimizer_aux = torch.optim.Adam(videocompressor.parameters(), lr=train_params['lr_videocompressor'], betas=(0.5, 0.999))   ###
    optimizer_kp_detector = torch.optim.Adam(kp_detector.parameters(), lr=train_params['lr_kp_detector'], betas=(0.5, 0.999))
    optimizer_cross_scale_m_fusion = torch.optim.Adam(cross_scale_m_fusion.parameters(), lr=train_params['lr_kp_detector'], betas=(0.5, 0.999))
    optimizer_Generator_Refine = torch.optim.Adam(Generator_Refine.parameters(), lr=train_params['lr_kp_detector'], betas=(0.5, 0.999))

    if checkpoint is not None:
        start_epoch = Logger.load_cpk(checkpoint, generator, Generator_Refine, discriminator, kp_detector,videocompressor,cross_scale_m_fusion,
                                      optimizer_generator, optimizer_Generator_Refine, optimizer_discriminator,
                                      None if train_params['lr_kp_detector'] == 0 else optimizer_kp_detector, optimizer_cross_scale_m_fusion,
                                      optimizer_videocompressor)
    else:
        start_epoch = 0
    scheduler_generator = MultiStepLR(optimizer_generator, train_params['epoch_milestones'], gamma=0.1,
                                      last_epoch=start_epoch - 1)
    scheduler_Generator_Refine = MultiStepLR(optimizer_Generator_Refine, train_params['epoch_milestones'], gamma=0.1,
                                        last_epoch=-1 + start_epoch)
    scheduler_discriminator = MultiStepLR(optimizer_discriminator, train_params['epoch_milestones'], gamma=0.1,
                                          last_epoch=start_epoch - 1)
    scheduler_kp_detector = MultiStepLR(optimizer_kp_detector, train_params['epoch_milestones'], gamma=0.1,
                                        last_epoch=-1 + start_epoch)
    scheduler_videocompressor = MultiStepLR(optimizer_videocompressor, train_params['epoch_milestones'], gamma=0.1,
                                      last_epoch=start_epoch - 1)
    scheduler_cross_scale_m_fusion = MultiStepLR(optimizer_cross_scale_m_fusion, train_params['epoch_milestones'], gamma=0.1,
                                      last_epoch=start_epoch - 1)
    scheduler_aux = MultiStepLR(optimizer_aux, train_params['epoch_milestones'], gamma=0.1,last_epoch=start_epoch - 1)   #####
    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])
    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=48, drop_last=True)
    dataloader_test = DataLoader(dataset_test, batch_size=train_params['batch_size'], num_workers=48, shuffle=True)
    generator_full = GeneratorFullModel(kp_detector, cross_scale_m_fusion, generator, discriminator, videocompressor,
                                        Generator_Refine, train_params) #####
    discriminator_full = DiscriminatorFullModel(kp_detector, generator, discriminator, videocompressor, train_params) #####



    if torch.cuda.is_available():
        generator_full = torch.nn.DataParallel(generator_full, device_ids=device_ids).to(device_ids[0])
        discriminator_full = torch.nn.DataParallel(discriminator_full, device_ids=device_ids).to(device_ids[0])  

    final_epoch = train_params['num_epochs']
    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, final_epoch):
            loop = tqdm(dataloader, total=len(dataloader), position=0)
            generator_full.train()
            discriminator_full.train()
            for batch_idx, x in enumerate(loop):

                optimizer_generator.step()
                optimizer_generator.zero_grad()
                optimizer_Generator_Refine.step()
                optimizer_Generator_Refine.zero_grad()
                optimizer_kp_detector.step()
                optimizer_kp_detector.zero_grad()
                optimizer_videocompressor.step()
                optimizer_videocompressor.zero_grad()
                optimizer_cross_scale_m_fusion.step()
                optimizer_cross_scale_m_fusion.zero_grad()

                optimizer_aux.step() ###
                optimizer_aux.zero_grad() ###


                #lambda_var = abs(rdlambdas)
                lambda_var = rdlambdas                      


                losses_generator, generated = generator_full(x,lambda_var, epoch) #####
                if batch_idx % 600 == 0:
                    logger.visualize_rec_training(x, generated, batch_idx, 'Image_Training')

                loss_values = [val.mean() for val in losses_generator.values()]
                loss = sum(loss_values)
                loss.backward(retain_graph=True)  ####

                if train_params['loss_weights']['generator_gan'] != 0:
                    optimizer_discriminator.zero_grad()
                    losses_discriminator = discriminator_full(x, generated)
                    loss_values = [val.mean() for val in losses_discriminator.values()]
                    loss = sum(loss_values)

                    loss.backward(retain_graph=True)
                    optimizer_discriminator.step()
                    optimizer_discriminator.zero_grad()
                else:
                    losses_discriminator = {}

                losses_generator.update(losses_discriminator)
                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                logger.log_iter(losses=losses)
                # print(losses)

                aux_loss = videocompressor.entropy_bottleneck.loss() ###
                # print(aux_loss)
                aux_loss.backward(retain_graph=True) ###
                
                loop.set_description(f'Train_Epoch[{epoch}/{final_epoch}]')
                loop.set_postfix(lambda_var=lambda_var, aux_loss=f"{aux_loss}")

            scheduler_generator.step()
            scheduler_Generator_Refine.step()
            scheduler_discriminator.step()
            scheduler_kp_detector.step()
            scheduler_videocompressor.step()
            scheduler_cross_scale_m_fusion.step()
            scheduler_aux.step()   ####
            logger.log_epoch(epoch, {'generator': generator,
                                     'discriminator': discriminator,
                                     'kp_detector': kp_detector,
                                     'videocompressor':videocompressor,
                                     'cross_scale_m_fusion': cross_scale_m_fusion,
                                     'Generator_Refine': Generator_Refine,
                                     'optimizer_generator': optimizer_generator,
                                     'optimizer_Generator_Refine': optimizer_Generator_Refine,
                                     'optimizer_discriminator': optimizer_discriminator,
                                     'optimizer_videocompressor': optimizer_videocompressor,
                                     'optimizer_kp_detector': optimizer_kp_detector,
                                     'optimizer_cross_scale_m_fusion': optimizer_cross_scale_m_fusion}, inp=x, out=generated)

            loop = tqdm(dataloader_test, total=len(dataloader_test), position=0)
            generator_full.eval()
            discriminator_full.eval()
            for batch_idx, x in enumerate(loop):
                losses_generator, generated = generator_full(x, lambda_var, epoch)

                if batch_idx % 50 == 0:
                    logger.visualize_rec_training(x, generated, batch_idx, 'Image_Testing')
                loss_values = [val.mean() for val in losses_generator.values()]
                loss = sum(loss_values)
                # loss.backward(retain_graph=True)  ####

                if train_params['loss_weights']['generator_gan'] != 0:

                    losses_discriminator = discriminator_full(x, generated)
                    loss_values = [val.mean() for val in losses_discriminator.values()]
                    loss = sum(loss_values)

                    # loss.backward(retain_graph=True)

                else:
                    losses_discriminator = {}

                losses_generator.update(losses_discriminator)
                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                logger.log_iter(losses=losses)
                # print(losses)

                aux_loss = videocompressor.entropy_bottleneck.loss()  ###
                # print(aux_loss)###

                loop.set_description(f'Test_Epoch[{epoch}/{final_epoch}]')
                loop.set_postfix(lambda_var=lambda_var, aux_loss=f"{aux_loss}")

            logger.loss_print()
