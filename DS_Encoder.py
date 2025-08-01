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

from arithmetic.value_encoder import *
from arithmetic.value_decoder import *
from GFVC.utils import *
from GFVC.DS_utils import *



    
    
if __name__ == "__main__":
   
    parser = ArgumentParser()
    parser.add_argument("--original_seq", default='/home/media/CH/GFVC_SOftware/GFVC_Software-main-318/output.rgb', type=str, help="path to the input testing sequence")
    parser.add_argument("--encoding_frames", default=125, help="the number of encoding frames")
    parser.add_argument("--seq_width", default=256, help="the width of encoding frames")
    parser.add_argument("--seq_height", default=256, help="the height of encoding frames")
    parser.add_argument("--quantization_factor", default=2, type=int, help="the quantization factor for the residual conversion from float-type to int-type")
    parser.add_argument("--Iframe_QP", default=22, help="the quantization parameters for encoding the Intra frame")
    parser.add_argument("--Iframe_format", default='YUV420', type=str,help="the quantization parameters for encoding the Intra frame")
    
    
    opt = parser.parse_args()
    
    
    frames=int(opt.encoding_frames)
    width=opt.seq_width
    height=opt.seq_width
    Qstep=opt.quantization_factor
    QP=opt.Iframe_QP
    Iframe_format=opt.Iframe_format
    original_seq=opt.original_seq
    seq = os.path.splitext(os.path.split(opt.original_seq)[-1])[0]
    

    ##DS
    checkpoint_path='../checkpoint.pth.tar'
    config_path='../vox-256.yaml'
    Analysis_Model, Synthesis_Model, cross_scale_m_fusion, Depth_Net, videocompressor, generator_Refine = load_checkpoints(
        config_path, checkpoint_path, cpu=False)

    modeldir = 'MODEL'
    model_dirname='./experiment/'+modeldir+"/"+'Iframe_'+str(Iframe_format)   
    
    
    #############################
    listR,listG,listB=RawReader_planar(original_seq, width, height,frames)

    driving_kp =model_dirname+'/kp/'+seq+'_QP'+str(QP)+'/'    
    os.makedirs(driving_kp,exist_ok=True)     # the frames to be compressed by vtm                 

    dir_enc =model_dirname+'/enc/'+seq+'_QP'+str(QP)+'/'
    os.makedirs(dir_enc,exist_ok=True)     # the frames to be compressed by vtm                 

    f_org=open(original_seq,'rb')

    seq_kp_integer=[]
    start=time.time() 

    sum_bits = 0
    for frame_idx in range(0, frames):            

        frame_idx_str = str(frame_idx).zfill(4)   
        
        if frame_idx in [0]:      # I-frame      
            
            if Iframe_format=='YUV420':
                
                # wtite ref and cur (rgb444) to file (yuv420)
                f_temp=open(dir_enc+'frame'+frame_idx_str+'_org.yuv','w')
                img_input_rgb = cv2.merge([listR[frame_idx],listG[frame_idx],listB[frame_idx]])
                img_input_yuv = cv2.cvtColor(img_input_rgb, cv2.COLOR_RGB2YUV_I420)  #COLOR_RGB2YUV
                img_input_yuv.tofile(f_temp)
                f_temp.close()            

                os.system("./vtm/encode.sh "+dir_enc+'frame'+frame_idx_str+" "+str(QP)+" "+str(width)+" "+str(height))   ########################

                bin_file=dir_enc+'frame'+frame_idx_str+'.bin'
                bits=os.path.getsize(bin_file)*8
                sum_bits += bits
                
                #  read the rec frame (yuv420) and convert to rgb444
                rec_ref_yuv=yuv420_to_rgb444(dir_enc+'frame'+frame_idx_str+'_rec.yuv', width, height, 0, 1, False, False) 
                img_rec = rec_ref_yuv[frame_idx]
                img_rec = img_rec[:,:,::-1].transpose(2, 0, 1)    # HxWx3
                img_rec = resize(img_rec, (3, height, width))    # normlize to 0-1                 
            
            elif Iframe_format=='RGB444':
                # wtite ref and cur (rgb444) 
                f_temp=open(dir_enc+'frame'+frame_idx_str+'_org.rgb','w')
                img_input_rgb = cv2.merge([listR[frame_idx],listG[frame_idx],listB[frame_idx]])
                img_input_rgb = img_input_rgb.transpose(2, 0, 1)   # 3xHxW
                img_input_rgb.tofile(f_temp)
                f_temp.close()

                os.system("./vtm/encode_rgb444.sh "+dir_enc+'frame'+frame_idx_str+" "+QP+" "+str(width)+" "+str(height))   ########################
                
                bin_file=dir_enc+'frame'+frame_idx_str+'.bin'
                bits=os.path.getsize(bin_file)*8
                sum_bits += bits
                
                f_temp=open(dir_enc+'frame'+frame_idx_str+'_rec.rgb','rb')
                img_rec=np.fromfile(f_temp,np.uint8,3*height*width).reshape((3,height,width))   # 3xHxW RGB         
                img_rec = resize(img_rec, (3, height, width))    # normlize to 0-1                  
            

            with torch.no_grad(): 
                reference = torch.tensor(img_rec[np.newaxis].astype(np.float32))
                reference = reference.cuda()    # require GPU

                #add
                reference_depth = Depth_Net(reference)
                kp_reference = Analysis_Model(reference, reference_depth['disp', 0]) ################
                kp_reference = {'value': kp_reference}
                kp_value = kp_reference['value']
                #add
                kp_value_list = kp_value.tolist()
                kp_value_list = str(kp_value_list)
                kp_value_list = "".join(kp_value_list.split())

                with open(driving_kp+'/frame'+frame_idx_str+'.txt','w')as f:
                    f.write(kp_value_list)  

                kp_value_frame=json.loads(kp_value_list)
                kp_value_frame= eval('[%s]'%repr(kp_value_frame).replace('[', '').replace(']', ''))
                seq_kp_integer.append(kp_value_frame)      


        else:

            interframe = cv2.merge([listR[frame_idx],listG[frame_idx],listB[frame_idx]])
            interframe = resize(interframe, (width, height))[..., :3]

            with torch.no_grad(): 
                interframe = torch.tensor(interframe[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
                interframe = interframe.cuda()    # require GPU                  

                ###extraction

                #add
                interframe_depth = Depth_Net(interframe)
                kp_interframe = Analysis_Model(interframe, interframe_depth['disp', 0]) ################
                kp_interframe = {"value": kp_interframe}
                kp_value = kp_interframe['value']
                #add

                kp_value_list = kp_value.tolist()
                kp_value_list = str(kp_value_list)
                kp_value_list = "".join(kp_value_list.split())

                with open(driving_kp+'/frame'+frame_idx_str+'.txt','w')as f:
                    f.write(kp_value_list)  

                kp_value_frame=json.loads(kp_value_list)
                kp_value_frame= eval('[%s]'%repr(kp_value_frame).replace('[', '').replace(']', ''))
                seq_kp_integer.append(kp_value_frame)     




    rec_sem=[]
    for frame in range(1,frames):
        frame_idx = str(frame).zfill(4)
        if frame==1:
            rec_sem.append(seq_kp_integer[0])

            ### residual
            kp_difference=(np.array(seq_kp_integer[frame])-np.array(seq_kp_integer[frame-1])).tolist()
            ## quantization
            kp_difference=[i*Qstep for i in kp_difference]
            kp_difference= list(map(round, kp_difference[:]))
            frame_idx = str(frame).zfill(4)
            bin_file=driving_kp+'/frame'+str(frame_idx)+'.bin'
            final_encoder_expgolomb(kp_difference,bin_file)     
            bits=os.path.getsize(bin_file)*8
            sum_bits += bits          
            #### decoding for residual
            res_dec = final_decoder_expgolomb(bin_file)
            res_difference_dec = data_convert_inverse_expgolomb(res_dec)   
            ### (i)_th frame + (i+1-i)_th residual =(i+1)_th frame
            res_difference_dec=[i/Qstep for i in res_difference_dec]
            rec_semantics=(np.array(res_difference_dec)+np.array(rec_sem[frame-1])).tolist()
            rec_sem.append(rec_semantics)

        else:

            ### residual
            kp_difference=(np.array(seq_kp_integer[frame])-np.array(rec_sem[frame-1])).tolist()
            ## quantization
            kp_difference=[i*Qstep for i in kp_difference]
            kp_difference= list(map(round, kp_difference[:]))
            frame_idx = str(frame).zfill(4)
            bin_file=driving_kp+'/frame'+str(frame_idx)+'.bin'
            final_encoder_expgolomb(kp_difference,bin_file)     
            bits=os.path.getsize(bin_file)*8
            sum_bits += bits          

            #### decoding for residual
            res_dec = final_decoder_expgolomb(bin_file)
            res_difference_dec = data_convert_inverse_expgolomb(res_dec)   
            ### (i)_th frame + (i+1-i)_th residual =(i+1)_th frame
            res_difference_dec=[i/Qstep for i in res_difference_dec]
            rec_semantics=(np.array(res_difference_dec)+np.array(rec_sem[frame-1])).tolist()
            rec_sem.append(rec_semantics)

    end=time.time()
    print("Extracting kp success. Time is %.4fs. Key points coding %d bits." %(end-start, sum_bits))   

