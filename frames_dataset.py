import os
import random
import cv2
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from imageio import mimread

import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from augmentation import AllAugmentationTransform
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import glob


def read_video(name, frame_shape):
    """
    Read video which can be:
      - an image of concatenated frames
      - '.mp4' and'.gif'
      - folder with videos
    """

    if os.path.isdir(name):
        frames = sorted(os.listdir(name))
        num_frames = len(frames)
        video_array = np.array(
            [img_as_float32(io.imread(os.path.join(name, frames[idx]))) for idx in range(num_frames)])
    elif name.lower().endswith('.png') or name.lower().endswith('.jpg'):
        image = io.imread(name)

        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)

        if image.shape[2] == 4:
            image = image[..., :3]

        image = img_as_float32(image)

        video_array = np.moveaxis(image, 1, 0)

        video_array = video_array.reshape((-1,) + frame_shape)
        video_array = np.moveaxis(video_array, 1, 2)
    elif name.lower().endswith('.gif') or name.lower().endswith('.mp4') or name.lower().endswith('.mov'):
        video = np.array(mimread(name))
        if len(video.shape) == 3:
            video = np.array([gray2rgb(frame) for frame in video])
        if video.shape[-1] == 4:
            video = video[..., :3]
        video_array = img_as_float32(video)
    else:
        raise Exception("Unknown file extensions  %s" % name)

    return video_array

def process_video(video_path):
    with os.scandir(video_path) as frame_entries:
        return [entry.path for entry in frame_entries]

def extract_number(s):
    return int(s.split('_')[-1])

class FramesDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """
    def __init__(self, root_dir, frame_shape=(256, 256, 3), id_sampling=False, is_train=True,
                 random_seed=0, pairs_list=None, augmentation_params=None, train_sample_size=None, test_sample_size=None):
        random_seed = random.randint(1, 2025)
        self.root_dir_Vox2 = root_dir
        self.root_dir_CFVQA = "./dataset/CFVQA/output_frames"
        self.root_dir_test = "./dataset/test_GFVC"
        self.videos = os.listdir(root_dir)
        self.frame_shape = tuple(frame_shape)
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling
        if is_train:
            train_videos = []
            root_list = os.listdir(self.root_dir_CFVQA)
            for cfv in root_list:
                frames_list = os.listdir(os.path.join(self.root_dir_CFVQA, cfv))
                for indx in frames_list:
                    train_videos.append(os.path.join(self.root_dir_CFVQA, cfv, indx))
            root_list = os.listdir(self.root_dir_Vox2)
            for vox in root_list:
                frames_list = os.listdir(os.path.join(self.root_dir_Vox2, vox))
                for indx in frames_list:
                    train_videos.append(os.path.join(self.root_dir_Vox2, vox, indx))
            if train_sample_size is not None and len(train_videos) > train_sample_size:
                train_videos = random.sample(train_videos, train_sample_size)
        else:
            test_videos = []
            test_root = os.listdir(self.root_dir_test)
            test_root = sorted(test_root, key=extract_number)
            for tr in test_root:
                frame_root = os.listdir(self.root_dir_test + '/' + tr)
                for indx in frame_root:
                    test_videos.append(os.path.join(self.root_dir_test, tr, indx))

            if test_sample_size is not None and len(test_videos) > test_sample_size:
                test_videos = random.sample(test_videos, test_sample_size)

        if is_train:
            self.videos = train_videos
        else:
            self.videos = test_videos
        self.is_train = is_train
        if is_train:
            self.transform = None
        else:
            self.transform = None


    def __len__(self):
        return len(self.videos)


    def __getitem__(self, idx):
        if self.is_train and self.id_sampling:
            name = self.videos[idx]
            # path = np.random.choice(glob.glob(os.path.join(self.root_dir, name + '*.mp4')))
            # path = os.path.join(self.root_dir, name)
            path = name
        else:
            name = self.videos[idx]
            # path = os.path.join(self.root_dir_test, name)
            path = name

        video_name = os.path.basename(path)

        if self.is_train and os.path.isdir(path):
            frames = os.listdir(path)
            frames.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            frame_idx = [frames[0], frames[1]]
            video_array = [img_as_float32(io.imread(os.path.join(path, idx))) for idx in frame_idx]
        else:
            frames = os.listdir(path)
            frames.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            frame_idx = np.array([0, 1])
            video_array = [img_as_float32(io.imread(os.path.join(path, frames[idx]))) for idx in frame_idx]

        if self.transform is not None:
            video_array = self.transform(video_array)

        out = {}
        if self.is_train:
            source = np.array(video_array[0], dtype='float32')
            driving = np.array(video_array[1], dtype='float32')

            out['driving'] = driving.transpose((2, 0, 1))
            out['source'] = source.transpose((2, 0, 1))
        else:
            source = np.array(video_array[0], dtype='float32')
            driving = np.array(video_array[1], dtype='float32')

            out['driving'] = driving.transpose((2, 0, 1))
            out['source'] = source.transpose((2, 0, 1))

        out['name'] = video_name

        return out


class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """

    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]


class PairedDataset(Dataset):
    """
    Dataset of pairs for animation.
    """

    def __init__(self, initial_dataset, number_of_pairs, seed=0):
        self.initial_dataset = initial_dataset
        pairs_list = self.initial_dataset.pairs_list

        np.random.seed(seed)

        if pairs_list is None:
            max_idx = min(number_of_pairs, len(initial_dataset))
            nx, ny = max_idx, max_idx
            xy = np.mgrid[:nx, :ny].reshape(2, -1).T
            number_of_pairs = min(xy.shape[0], number_of_pairs)
            self.pairs = xy.take(np.random.choice(xy.shape[0], number_of_pairs, replace=False), axis=0)
        else:
            videos = self.initial_dataset.videos
            name_to_index = {name: index for index, name in enumerate(videos)}
            pairs = pd.read_csv(pairs_list)
            pairs = pairs[np.logical_and(pairs['source'].isin(videos), pairs['driving'].isin(videos))]

            number_of_pairs = min(pairs.shape[0], number_of_pairs)
            self.pairs = []
            self.start_frames = []
            for ind in range(number_of_pairs):
                self.pairs.append(
                    (name_to_index[pairs['driving'].iloc[ind]], name_to_index[pairs['source'].iloc[ind]]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        first = self.initial_dataset[pair[0]]
        second = self.initial_dataset[pair[1]]
        first = {'driving_' + key: value for key, value in first.items()}
        second = {'source_' + key: value for key, value in second.items()}

        return {**first, **second}
