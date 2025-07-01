# +
# get file size in python
import os
import numpy as np

Inputformat = 'Iframe_YUV420'  # 'Iframe_RGB444' OR 'Iframe_YUV420'
testingdata_name = 'VOXCELEB'  #'CFVQA' OR 'VOXCELEB'
Model = 'DS'  ## 'FV2V' OR 'FOMM' OR 'CFTE'
txt_path_ssim_YUV420 = '../experiment/' + Model + '/' + Inputformat + '/evaluation-YUV420/' + testingdata_name + '_result_' + 'ssim.txt'
txt_path_psnr_YUV420 = '../experiment/' + Model + '/' + Inputformat + '/evaluation-YUV420/' + testingdata_name + '_result_' + 'psnr.txt'
txt_path_psnr = '../experiment/' + Model + '/' + Inputformat + '/evaluation/' + testingdata_name + '_result_' + 'psnr.txt'
txt_path_ssim = '../experiment/' + Model + '/' + Inputformat + '/evaluation/' + testingdata_name + '_result_' + 'ssim.txt'
txt_path_LPIPS = '../experiment/' + Model + '/' + Inputformat + '/evaluation/' + testingdata_name + '_result_' + 'lpips.txt'
txt_path_DISTS = '../experiment/' + Model + '/' + Inputformat + '/evaluation/' + testingdata_name + '_result_' + 'dists.txt'
txt_path_Kbps = '../experiment/' + Model + '/' + Inputformat + '/resultBit/' + testingdata_name + '_' + 'resultBit.txt'

RGB444 = 1
YUV420 = 1
if RGB444:
    print('-----------------------------ssim-----------------------------')
    print('-----------------------------ssim-----------------------------')
    with open(txt_path_ssim, 'r') as file:
        # content = file.read()
        for line in file:
            words = line.split()
            for num in range(4):
                print(words[num])
    print('-----------------------------psnr-----------------------------')
    print('-----------------------------psnr-----------------------------')
    with open(txt_path_psnr, 'r') as file:
        # content = file.read()
        for line in file:
            words = line.split()
            for num in range(4):
                print(words[num])
    print('-----------------------------LPIPS-----------------------------')
    print('-----------------------------LPIPS-----------------------------')
    with open(txt_path_LPIPS, 'r') as file:
        # content = file.read()
        for line in file:
            words = line.split()
            for num in range(4):
                print(1 - float(words[num]))

    print('-----------------------------DISTS-----------------------------')
    print('-----------------------------DISTS-----------------------------')
    with open(txt_path_DISTS, 'r') as file:
        # content = file.read()
        for line in file:
            words = line.split()
            for num in range(4):
                print(1 - float(words[num]))


    print('-----------------------------Kbps-----------------------------')
    print('-----------------------------Kbps-----------------------------')
    with open(txt_path_Kbps, 'r') as file:
        # content = file.read()
        for line in file:
            words = line.split()
            for num in range(4):
                print(words[num])
if YUV420:
    print('-----------------------------YUV420_psnr-----------------------------')
    print('-----------------------------YUV420_psnr-----------------------------')
    with open(txt_path_psnr_YUV420, 'r') as file:
        # content = file.read()
        for line in file:
            words = line.split()
            for num in range(4):
                print(words[num])
    print('-----------------------------YUV420_ssim-----------------------------')
    print('-----------------------------YUV420_ssim-----------------------------')
    with open(txt_path_ssim_YUV420, 'r') as file:
        # content = file.read()
        for line in file:
            words = line.split()
            for num in range(4):
                print(words[num])



