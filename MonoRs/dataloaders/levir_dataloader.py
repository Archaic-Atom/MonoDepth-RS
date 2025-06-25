import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.distributed
from torchvision import transforms

import numpy as np
from PIL import Image, ImageOps
import os
import random
import copy

import cv2

from utils import DistributedSamplerNoEvenlyDivisible


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode):
    return transforms.Compose([
        ToTensor(mode=mode)
    ])


class NewDataLoader(object):
    def __init__(self, args, mode):
        if mode == 'train':
            self.training_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            if args.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.training_samples)
            else:
                self.train_sampler = None

            self.data = DataLoader(self.training_samples, args.batch_size,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=args.num_threads,
                                   pin_memory=True,
                                   sampler=self.train_sampler)

        elif mode == 'online_eval':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            if args.distributed:
                # self.eval_sampler = torch.utils.data.distributed.DistributedSampler(self.testing_samples, shuffle=False)
                self.eval_sampler = DistributedSamplerNoEvenlyDivisible(self.testing_samples, shuffle=False)
            else:
                self.eval_sampler = None

            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=True,
                                   sampler=self.eval_sampler)
            # print("ok123")

        elif mode == 'test':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=1)

        else:
            print('mode should be one of \'train, test\'. Got {}'.format(mode))


class DataLoadPreprocess(Dataset):

    def __init__(self, args, mode, transform=None):
        self.args = args

        if mode == 'online_eval':
            with open(args.filenames_file_eval, 'r') as f:
                self.filenames = f.readlines()
        else:
            with open(args.filenames_file, 'r') as f:
                self.filenames = f.readlines()

        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        focal = 518.8579

        if self.mode == 'train':
            rgb_file = sample_path.split()[0]
            depth_file = sample_path.split()[1]

            image_path = os.path.join(self.args.data_path, rgb_file)
            depth_path = os.path.join(self.args.gt_path, depth_file)

            image = Image.open(image_path)
            depth_gt = Image.open(depth_path)

            image = np.asarray(image, dtype=np.float32) / 255.0

            # cams_path=depth_path.replace("Depths", "Cams")
            # cams_path=cams_path.replace(".tiff", ".txt")
            #
            # words = open(cams_path).read().split()
            # depth=words[9].split(",")
            # min_depth=float(depth[0])
            # max_depth=float(depth[1])



            depth_gt = np.array(depth_gt).astype(np.float32)
            # print("depth_gt",depth_gt.shape)# 获取tiff文件
            depth_gt = np.expand_dims(depth_gt, axis=2)
            # print("depth_gt", depth_gt.shape)

            image, depth_gt = self.train_preprocess(image, depth_gt)
            image, depth_gt = self.Cut_Flip(image, depth_gt)
            sample = {'image': image, 'depth': depth_gt, 'focal': focal}

        else:
            if self.mode == 'online_eval':
                data_path = self.args.data_path_eval
            else:
                data_path = self.args.data_path

            image_path = os.path.join(data_path, "./" + sample_path.split()[0])
            image=Image.open(image_path)


            # padding
            # (left, top, right, bottom)
            # border_width = (0, 0, 20, 20)  # 边框宽度
            # border_color = (0, 0, 0)  # 灰色填充
            # image = ImageOps.expand(image, border=border_width, fill=border_color)

            image = np.asarray(image, dtype=np.float32) / 255.0
            # print("image", image.shape)

            # cams_path = depth_path.replace("Depths", "Cams")
            # cams_path = cams_path.replace(".tiff", ".txt")
            #
            # words = open(cams_path).read().split()
            # depth = words[9].split(",")
            # min_depth = float(depth[0])
            # max_depth = float(depth[1])

            # depth
            if self.mode == 'online_eval':
                gt_path = self.args.gt_path_eval
                depth_path = os.path.join(gt_path, "./" + sample_path.split()[1])
                has_valid_depth = False
                try:
                    depth_gt = Image.open(depth_path)
                    has_valid_depth = True
                except IOError:
                    depth_gt = False

                if has_valid_depth:
                    depth_gt = np.array(depth_gt).astype(np.float32)
                    # print("depth_gt--",depth_gt.shape)
                    depth_gt = np.expand_dims(depth_gt, axis=2)
                    # print("depth_gt", depth_gt.shape)

            if self.mode == 'online_eval':
                sample = {'image': image, 'depth': depth_gt, 'focal': focal}
            else:
                sample = {'image': image, 'focal': focal}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        return img, depth

    def train_preprocess(self, image, depth_gt):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()

        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)

        return image, depth_gt

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        # if self.args.dataset == 'nyu':
        #     brightness = random.uniform(0.75, 1.25)
        # else:

        brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def Cut_Flip(self, image, depth):

        p = random.random()
        if p < 0.5:
            return image, depth
        image_copy = copy.deepcopy(image)
        depth_copy = copy.deepcopy(depth)
        h, w, c = image.shape

        N = 2
        h_list = []
        h_interval_list = []  # hight interval
        for i in range(N - 1):
            h_list.append(random.randint(int(0.2 * h), int(0.8 * h)))
        h_list.append(h)
        h_list.append(0)
        h_list.sort()
        h_list_inv = np.array([h] * (N + 1)) - np.array(h_list)
        for i in range(len(h_list) - 1):
            h_interval_list.append(h_list[i + 1] - h_list[i])
        for i in range(N):
            image[h_list[i]:h_list[i + 1], :, :] = image_copy[h_list_inv[i] - h_interval_list[i]:h_list_inv[i], :, :]
            depth[h_list[i]:h_list[i + 1], :, :] = depth_copy[h_list_inv[i] - h_interval_list[i]:h_list_inv[i], :, :]

        return image, depth

    def __len__(self):
        return len(self.filenames)


class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, sample):

        image, focal = sample['image'], sample['focal']
        image = self.to_tensor(image)
        image = self.normalize(image)



        if self.mode == 'test':
            return {'image': image, 'focal': focal}

        # min_depth= torch.from_numpy(sample['min_depth'])
        # max_depth= torch.from_numpy(sample['max_depth'])
        depth = sample['depth']
        if self.mode == 'train':
            depth = self.to_tensor(depth)
            return {'image': image, 'depth': depth, 'focal': focal}
        else:
            return {'image': image, 'depth': depth, 'focal': focal}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            # print(pic.shape)
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img



