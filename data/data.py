import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transforms

from natsort import natsorted

from PIL import Image, ImageOps, ImageFilter
import os.path as osp
import sys
import random

from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator


class DataLoaderX(DataLoader):
    def _iter_(self):
        return BackgroundGenerator(super()._iter_())


class NUST_SIRST(Data.Dataset):
    def __init__(self, mode='train'):
        #path of dataset
        self.base_dir = '/data1/hrm/Datasets/NUST-SIRST/MDvsFA_cGAN/data'
        self.mean = [0.410, 0.410, 0.410]
        self.std = [0.105, 0.105, 0.105]
        # 都是128*128的图片
        self.base_size = 256
        self.crop_size = 256

        self.mode = mode
        if mode == 'train':
            self.imgs_dir = osp.join(self.base_dir, 'training')
            self.label_dir = osp.join(self.base_dir, 'training')
        elif mode == 'test':
            self.imgs_dir = osp.join(self.base_dir, 'test_org')
            self.label_dir = osp.join(self.base_dir, 'test_gt')
        else:
            raise ValueError("Unkown self.mode")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

    def __getitem__(self, i):

        if self.mode == 'train':
            # has been broken
            if i  in [239, 245, 260, 264, 2543, 2553, 2561, 2808, 2817, 2819, 3503, 3504, 3947, 3949, 3962, 7389, 7395, 8094, 8105, 8112, 8757, 8772]:
                i += 3

            img_path = osp.join(self.imgs_dir, '%06d_1.png' % i)
            label_path = osp.join(self.label_dir, '%06d_2.png' % i)

            img = Image.open(img_path).convert('RGB')
            mask = Image.open(label_path).convert('L')
            
            # transform
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'test':
            img_path = osp.join(self.imgs_dir, '%05d.png' % i)
            label_path = osp.join(self.label_dir, '%05d.png' % i)
            
            img = Image.open(img_path).convert('RGB')
            mask = Image.open(label_path).convert('L')
            
            # transform
            img, mask = self._testval_sync_transform(img, mask)
        else:
            raise ValueError("Unkown self.mode")

        # img = self.transform(img)
        # print(torch.max(img))
        # print(torch.min(img))

        img, mask = self.transform(img), transforms.ToTensor()(mask)
        return img, mask

    def __len__(self):
        if self.mode == 'train':
            return 10000
        elif self.mode == 'test':
            return 100
        else:
            raise ValueError("Unkown self.mode")

    def _sync_transform(self, img, mask):
        
        # 随机翻转旋转
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        if random.random() < 0.5:
            img = img.transpose(Image.ROTATE_90)
            mask = mask.transpose(Image.ROTATE_90)
        
        crop_size = self.crop_size
        
        #训练时，长边先Resize到基于base_size的随机的long_size，然后短边pad到crop_size大小，最后crop到crop_size大小
        
        # long_size = random.randint(256, 281)，然后长边等比例缩放到该大小
        long_size = random.randint(int(self.base_size * 1.0), int(self.base_size * 1.1))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        
        # 短边pad到crop_size的大小，并且是两边都进行pad
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            padh = padh + 1 if padh % 2 != 0 else padh
            padw = padw + 1 if padw % 2 != 0 else padw
            img = ImageOps.expand(img, border=(int(padw/2), int(padh/2), int(padw/2), int(padh/2)), fill=0)
            mask = ImageOps.expand(mask, border=(int(padw/2), int(padh/2), int(padw/2), int(padh/2)), fill=0)
        
        # 裁剪crop_size的大小
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        
        # 不进行高斯模糊
        # gaussian blur as in PSP
        # if random.random() < 0.5:
            # img = img.filter( ImageFilter.GaussianBlur( radius=random.random() ) )

        return img, mask


    # 测试的时候直接resize到base_size的正方形
    def _testval_sync_transform(self, img, mask):
        crop_size = self.crop_size
        img = img.resize((crop_size, crop_size), Image.BILINEAR)
        mask = mask.resize((crop_size, crop_size), Image.NEAREST)

        return img, mask


class NUAA_SIRST(Data.Dataset):
    def __init__(self, mode='train'):
        #path of dataset
        self.base_dir = '/data1/hrm/Datasets/NUAA-SIRST/sirst'
        self.mean = [0.442, 0.442, 0.442]
        self.std = [0.111, 0.111, 0.111]
        # 杂乱，从222*222到367*305都有，官方缩放成了480*480
        self.base_size = 512
        self.crop_size = 512

        # list of images in dataset
        if mode == 'train':
            txtfile = 'trainval.txt'
        elif mode == 'test':
            txtfile = 'test.txt'

        self.list_file = osp.join(self.base_dir, 'idx_427', txtfile)
        self.imgs_dir = osp.join(self.base_dir, 'images')
        self.label_dir = osp.join(self.base_dir, 'masks')

        self.names = []
        with open(self.list_file, 'r') as f:
            self.names += [line.strip() for line in f.readlines()]

        self.mode = mode
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.imgs_dir, name+'.png')
        label_path = osp.join(self.label_dir, name+'_pixels0.png')

        #由于输入的三通道、单通道图像都有，所以统一转成RGB的三通道，这也符合Unet等网络的期待尺寸
        img = Image.open(img_path).convert('RGB')
        #为后续多类型目标检测分割预留扩展空间 
        mask = Image.open(label_path).convert('L')

        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'test':
            img, mask = self._testval_sync_transform(img, mask)
        else:
            raise ValueError("Unkown self.mode")

        # img = self.transform(img)
        # print(torch.max(img))
        # print(torch.min(img))

        img, mask = self.transform(img), transforms.ToTensor()(mask)
        return img, mask

    def __len__(self):
        return len(self.names)

    def _sync_transform(self, img, mask):
        
        # 随机翻转旋转
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random() < 0.5:
            img = img.transpose(Image.ROTATE_90)
            mask = mask.transpose(Image.ROTATE_90)
        
        crop_size = self.crop_size
        
        #训练时，长边先Resize到基于base_size的随机的long_size，然后短边pad到crop_size大小，最后crop到crop_size大小
        
        # long_size = random.randint(480, 528)，然后长边等比例缩放到该大小
        long_size = random.randint(int(self.base_size * 1.0), int(self.base_size * 1.1))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        
        # 短边pad到crop_size的大小，并且是两边都进行pad
        if short_size < crop_size:

            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0

            padh = padh + 1 if padh % 2 != 0 else padh
            padw = padw + 1 if padw % 2 != 0 else padw

            img = ImageOps.expand(img, border=(int(padw/2), int(padh/2), int(padw/2), int(padh/2)), fill=0)
            mask = ImageOps.expand(mask, border=(int(padw/2), int(padh/2), int(padw/2), int(padh/2)), fill=0)
        
        # 裁剪crop_size的大小
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        
        # 不进行高斯模糊
        # gaussian blur as in PSP
        # if random.random() < 0.5:
            # img = img.filter( ImageFilter.GaussianBlur( radius=random.random() ) )

        return img, mask


    # 测试的时候直接resize到base_size的正方形
    def _testval_sync_transform(self, img, mask):
        crop_size = self.crop_size
        img = img.resize((crop_size, crop_size), Image.BILINEAR)
        mask = mask.resize((crop_size, crop_size), Image.NEAREST)

        return img, mask



class NUAA_SIRST_WG_Noise(Data.Dataset):
    def __init__(self, mode='train'):
        #path of dataset
        self.base_dir = '/data1/hrm/Datasets/NUAA-SIRST-WG-Noise/sirst'
        self.mean = [0.442, 0.442, 0.442]
        self.std = [0.111, 0.111, 0.111]
        # 杂乱，从222*222到367*305都有，官方缩放成了480*480
        self.base_size = 512
        self.crop_size = 512

        # list of images in dataset
        if mode == 'train':
            txtfile = 'trainval.txt'
        elif mode == 'test':
            txtfile = 'test.txt'

        self.list_file = osp.join(self.base_dir, 'idx_427', txtfile)
        self.imgs_dir = osp.join(self.base_dir, 'images')
        self.label_dir = osp.join(self.base_dir, 'masks')

        self.names = []
        with open(self.list_file, 'r') as f:
            self.names += [line.strip() for line in f.readlines()]

        self.mode = mode
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.imgs_dir, name+'.png')
        label_path = osp.join(self.label_dir, name+'_pixels0.png')

        #由于输入的三通道、单通道图像都有，所以统一转成RGB的三通道，这也符合Unet等网络的期待尺寸
        img = Image.open(img_path).convert('RGB')
        #为后续多类型目标检测分割预留扩展空间 
        mask = Image.open(label_path).convert('L')

        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'test':
            img, mask = self._testval_sync_transform(img, mask)
        else:
            raise ValueError("Unkown self.mode")

        # img = self.transform(img)
        # print(torch.max(img))
        # print(torch.min(img))

        img, mask = self.transform(img), transforms.ToTensor()(mask)
        return img, mask

    def __len__(self):
        return len(self.names)

    def _sync_transform(self, img, mask):
        
        # 随机翻转旋转
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random() < 0.5:
            img = img.transpose(Image.ROTATE_90)
            mask = mask.transpose(Image.ROTATE_90)
        
        crop_size = self.crop_size
        
        #训练时，长边先Resize到基于base_size的随机的long_size，然后短边pad到crop_size大小，最后crop到crop_size大小
        
        # long_size = random.randint(480, 528)，然后长边等比例缩放到该大小
        long_size = random.randint(int(self.base_size * 1.0), int(self.base_size * 1.1))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        
        # 短边pad到crop_size的大小，并且是两边都进行pad
        if short_size < crop_size:

            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0

            padh = padh + 1 if padh % 2 != 0 else padh
            padw = padw + 1 if padw % 2 != 0 else padw

            img = ImageOps.expand(img, border=(int(padw/2), int(padh/2), int(padw/2), int(padh/2)), fill=0)
            mask = ImageOps.expand(mask, border=(int(padw/2), int(padh/2), int(padw/2), int(padh/2)), fill=0)
        
        # 裁剪crop_size的大小
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        
        # 不进行高斯模糊
        # gaussian blur as in PSP
        # if random.random() < 0.5:
            # img = img.filter( ImageFilter.GaussianBlur( radius=random.random() ) )

        return img, mask


    # 测试的时候直接resize到base_size的正方形
    def _testval_sync_transform(self, img, mask):
        crop_size = self.crop_size
        img = img.resize((crop_size, crop_size), Image.BILINEAR)
        mask = mask.resize((crop_size, crop_size), Image.NEAREST)

        return img, mask


class NUAA_SIRST_WG_DeNoise(Data.Dataset):
    def __init__(self, mode='train'):
        #path of dataset
        self.base_dir = '/data1/hrm/Datasets/NUAA-SIRST-WG-Denoise/sirst'
        self.mean = [0.442, 0.442, 0.442]
        self.std = [0.111, 0.111, 0.111]
        # 杂乱，从222*222到367*305都有，官方缩放成了480*480
        self.base_size = 512
        self.crop_size = 512

        # list of images in dataset
        if mode == 'train':
            txtfile = 'trainval.txt'
        elif mode == 'test':
            txtfile = 'test.txt'

        self.list_file = osp.join(self.base_dir, 'idx_427', txtfile)
        self.imgs_dir = osp.join(self.base_dir, 'images')
        self.label_dir = osp.join(self.base_dir, 'masks')

        self.names = []
        with open(self.list_file, 'r') as f:
            self.names += [line.strip() for line in f.readlines()]

        self.mode = mode
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.imgs_dir, name+'.png')
        label_path = osp.join(self.label_dir, name+'_pixels0.png')

        #由于输入的三通道、单通道图像都有，所以统一转成RGB的三通道，这也符合Unet等网络的期待尺寸
        img = Image.open(img_path).convert('RGB')
        #为后续多类型目标检测分割预留扩展空间 
        mask = Image.open(label_path).convert('L')

        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'test':
            img, mask = self._testval_sync_transform(img, mask)
        else:
            raise ValueError("Unkown self.mode")

        # img = self.transform(img)
        # print(torch.max(img))
        # print(torch.min(img))

        img, mask = self.transform(img), transforms.ToTensor()(mask)
        return img, mask

    def __len__(self):
        return len(self.names)

    def _sync_transform(self, img, mask):
        
        # 随机翻转旋转
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random() < 0.5:
            img = img.transpose(Image.ROTATE_90)
            mask = mask.transpose(Image.ROTATE_90)
        
        crop_size = self.crop_size
        
        #训练时，长边先Resize到基于base_size的随机的long_size，然后短边pad到crop_size大小，最后crop到crop_size大小
        
        # long_size = random.randint(480, 528)，然后长边等比例缩放到该大小
        long_size = random.randint(int(self.base_size * 1.0), int(self.base_size * 1.1))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        
        # 短边pad到crop_size的大小，并且是两边都进行pad
        if short_size < crop_size:

            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0

            padh = padh + 1 if padh % 2 != 0 else padh
            padw = padw + 1 if padw % 2 != 0 else padw

            img = ImageOps.expand(img, border=(int(padw/2), int(padh/2), int(padw/2), int(padh/2)), fill=0)
            mask = ImageOps.expand(mask, border=(int(padw/2), int(padh/2), int(padw/2), int(padh/2)), fill=0)
        
        # 裁剪crop_size的大小
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        
        # 不进行高斯模糊
        # gaussian blur as in PSP
        # if random.random() < 0.5:
            # img = img.filter( ImageFilter.GaussianBlur( radius=random.random() ) )

        return img, mask


    # 测试的时候直接resize到base_size的正方形
    def _testval_sync_transform(self, img, mask):
        crop_size = self.crop_size
        img = img.resize((crop_size, crop_size), Image.BILINEAR)
        mask = mask.resize((crop_size, crop_size), Image.NEAREST)

        return img, mask



class IRSTD_1k(Data.Dataset):

    def __init__(self, mode='train'):

        self.base_dir = '/data1/hrm/Datasets/IRSTD-1k/IRSTD-1k'
        self.mean = [0.343, 0.343, 0.343]
        self.std = [0.157, 0.157, 0.157]
        # 全是512*512的图片
        self.base_size = 512
        self.crop_size = 512

        if mode == 'train':
            txtfile = 'trainval.txt'

        elif mode == 'test':
            txtfile = 'test.txt'

        self.list_dir = osp.join(self.base_dir, txtfile)
        self.imgs_dir = osp.join(self.base_dir, 'IRSTD1k_Img')
        self.label_dir = osp.join(self.base_dir, 'IRSTD1k_Label')
        
        self.names = []
        with open(self.list_dir, 'r') as f:
            self.names += [line.strip() for line in f.readlines()]

        self.mode = mode
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.imgs_dir, name+'.png')
        label_path = osp.join(self.label_dir, name+'.png')

        # #由于输入的三通道、单通道图像都有，所以统一转成RGB的三通道，这也符合Unet等网络的期待尺寸
        img = Image.open(img_path).convert('RGB')
        #为后续多类型目标检测分割预留扩展空间
        mask = Image.open(label_path).convert('L')

        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'test':
            img, mask = self._testval_sync_transform(img, mask)
        else:
            raise ValueError("Unkown self.mode")

        #img = self.transform(img)
        # print(max(mask))
        # print(min(mask))

        img, mask= self.transform(img), transforms.ToTensor()(mask)
        return img, mask

    def __len__(self):
        return len(self.names)

    def _sync_transform(self, img, mask):
        
        # 随机翻转旋转
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        if random.random() < 0.5:
            img = img.transpose(Image.ROTATE_90)
            mask = mask.transpose(Image.ROTATE_90)
        
        crop_size = self.crop_size
        
        #训练时，长边先Resize到基于base_size的随机的long_size，然后短边pad到crop_size大小，最后crop到crop_size大小
        
        # long_size = random.randint(512, 563)，然后长边等比例缩放到该大小
        long_size = random.randint(int(self.base_size * 1.0), int(self.base_size * 1.1))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        
        # 短边pad到crop_size的大小，并且是两边都进行pad
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            padh = padh + 1 if padh % 2 != 0 else padh
            padw = padw + 1 if padw % 2 != 0 else padw
            img = ImageOps.expand(img, border=(int(padw/2), int(padh/2), int(padw/2), int(padh/2)), fill=0)
            mask = ImageOps.expand(mask, border=(int(padw/2), int(padh/2), int(padw/2), int(padh/2)), fill=0)
        
        # 裁剪crop_size的大小
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        
        # 不进行高斯模糊
        # gaussian blur as in PSP
        # if random.random() < 0.5:
            # img = img.filter( ImageFilter.GaussianBlur( radius=random.random() ) )

        return img, mask


    # 测试的时候直接resize到base_size的正方形
    def _testval_sync_transform(self, img, mask):
        crop_size = self.crop_size
        img = img.resize((crop_size, crop_size), Image.BILINEAR)
        mask = mask.resize((crop_size, crop_size), Image.NEAREST)

        return img, mask


class SIRST_Aug(Data.Dataset):
    def __init__(self, mode='train'):
        #path of dataset
        self.base_dir = '/data1/hrm/Datasets/SIRST-Aug/sirst_aug'
        self.mean = [0.290, 0.290, 0.290]
        self.std = [0.194, 0.194, 0.194]

        # 全是256x256的图片
        # 但已经增强过，这里就不再增强了
        self.base_size = 256
        self.crop_size = 256

        self.mode = mode
        if mode == 'train':
            self.data_dir = osp.join(self.base_dir, 'trainval')
        elif mode == 'test':
            self.data_dir = osp.join(self.base_dir, 'test')
        else:
            raise ValueError("Unkown self.mode")

        # for __len__
        self.names = []
        for filename in os.listdir(osp.join(self.data_dir, 'images')):
            if filename.endswith('png'):
                self.names.append(filename)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

    def __getitem__(self, i):
        img_path = osp.join(self.data_dir, 'images', '%06d.png' % i)
        label_path = osp.join(self.data_dir, 'masks', '%06d.png' % i)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(label_path).convert('L')

        img, mask = self.transform(img), transforms.ToTensor()(mask)
        return img, mask

    def __len__(self):
        return len(self.names)


class SIRST_V2(Data.Dataset):
    def __init__(self, mode='train'):
        #path of dataset
        self.base_dir = '/data1/hrm/Datasets/SIRST-V2/open-sirst-v2'
        self.mean = [0.399, 0.399, 0.399]
        self.std = [0.153, 0.153, 0.153]

        # 杂乱，从640*512到1280*1024的都有
        self.base_size = 512
        self.crop_size = 512

        # list of images in dataset
        if mode == 'train':
            txtfile = 'trainval_full.txt'
        elif mode == 'test':
            txtfile = 'test_full.txt'

        self.list_file = osp.join(self.base_dir, 'splits', txtfile)
        self.imgs_dir = osp.join(self.base_dir, 'mixed')
        self.label_dir = osp.join(self.base_dir, 'annotations', 'masks')

        self.names = []
        with open(self.list_file, 'r') as f:
            self.names += [line.strip() for line in f.readlines()]

        self.mode = mode
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.imgs_dir, name+'.png')
        label_path = osp.join(self.label_dir, name+'_pixels0.png')

        #由于输入的三通道、单通道图像都有，所以统一转成RGB的三通道，这也符合Unet等网络的期待尺寸
        img = Image.open(img_path).convert('RGB')
        #为后续多类型目标检测分割预留扩展空间 
        mask = Image.open(label_path).convert('L')

        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'test':
            img, mask = self._testval_sync_transform(img, mask)
        else:
            raise ValueError("Unkown self.mode")

        # img = self.transform(img)
        # print(torch.max(img))
        # print(torch.min(img))

        img, mask = self.transform(img), transforms.ToTensor()(mask)
        return img, mask

    def __len__(self):
        return len(self.names)

    def _sync_transform(self, img, mask):
        
        # 随机翻转旋转
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        if random.random() < 0.5:
            img = img.transpose(Image.ROTATE_90)
            mask = mask.transpose(Image.ROTATE_90)
        
        crop_size = self.crop_size
        
        #训练时，长边先Resize到基于base_size的随机的long_size，然后短边pad到crop_size大小，最后crop到crop_size大小
        
        # long_size = random.randint(512, 563)，然后长边等比例缩放到该大小
        long_size = random.randint(int(self.base_size * 1.0), int(self.base_size * 1.1))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        
        # 短边pad到crop_size的大小，并且是两边都进行pad
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            padh = padh + 1 if padh % 2 != 0 else padh
            padw = padw + 1 if padw % 2 != 0 else padw
            img = ImageOps.expand(img, border=(int(padw/2), int(padh/2), int(padw/2), int(padh/2)), fill=0)
            mask = ImageOps.expand(mask, border=(int(padw/2), int(padh/2), int(padw/2), int(padh/2)), fill=0)
        
        # 裁剪crop_size的大小
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        
        # 不进行高斯模糊
        # gaussian blur as in PSP
        # if random.random() < 0.5:
            # img = img.filter( ImageFilter.GaussianBlur( radius=random.random() ) )

        return img, mask


    # 测试的时候直接resize到base_size的正方形
    def _testval_sync_transform(self, img, mask):
        crop_size = self.crop_size
        img = img.resize((crop_size, crop_size), Image.BILINEAR)
        mask = mask.resize((crop_size, crop_size), Image.NEAREST)

        return img, mask


class NUDT_SIRST(Data.Dataset):
    def __init__(self, mode='train'):   
        #path of dataset
        self.base_dir = '/data1/hrm/Datasets/NUDT-SIRST/NUDT-SIRST'

        self.mean = [0.424, 0.424, 0.424]
        self.std = [0.130, 0.130, 0.130]
        # 全是256*256的
        self.base_size = 256
        self.crop_size = 256

        # list of images in dataset
        if mode == 'train':
            txtfile = 'train_NUDT-SIRST.txt'
        elif mode == 'test':
            # txtfile = 'test.txt'
            txtfile = 'test_NUDT-SIRST.txt'

        self.list_dir = osp.join(self.base_dir,  txtfile)
        self.imgs_dir = osp.join(self.base_dir, 'images')
        self.label_dir = osp.join(self.base_dir, 'masks')

        self.names = []
        with open(self.list_dir, 'r') as f:
            self.names += [line.strip() for line in f.readlines()]

        self.mode = mode
        self.transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(self.mean, self.std) # mean and std
        ])

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.imgs_dir, name+'.png')
        label_path = osp.join(self.label_dir, name+'.png')

        #由于输入的三通道、单通道图像都有，所以统一转成RGB的三通道，这也符合Unet等网络的期待尺寸
        img = Image.open(img_path).convert('RGB')
        #为后续多类型目标检测分割预留扩展空间 
        mask = Image.open(label_path).convert('L')

        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'test':
            img, mask = self._testval_sync_transform(img, mask)
        else:
            raise ValueError("Unkown self.mode")

        # img = self.transform(img)
        # print(torch.max(img))
        # print(torch.min(img))

        img, mask = self.transform(img), transforms.ToTensor()(mask)
        return img, mask

    def __len__(self):
        return len(self.names)

    def _sync_transform(self, img, mask):
        
        # 随机翻转旋转
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        if random.random() < 0.5:
            img = img.transpose(Image.ROTATE_90)
            mask = mask.transpose(Image.ROTATE_90)
        
        crop_size = self.crop_size
        
        #训练时，长边先Resize到基于base_size的随机的long_size，然后短边pad到crop_size大小，最后crop到crop_size大小
        
        # long_size = random.randint(256, 281)，然后长边等比例缩放到该大小
        long_size = random.randint(int(self.base_size * 1.0), int(self.base_size * 1.1))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        
        # 短边pad到crop_size的大小，并且是两边都进行pad
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            padh = padh + 1 if padh % 2 != 0 else padh
            padw = padw + 1 if padw % 2 != 0 else padw
            img = ImageOps.expand(img, border=(int(padw/2), int(padh/2), int(padw/2), int(padh/2)), fill=0)
            mask = ImageOps.expand(mask, border=(int(padw/2), int(padh/2), int(padw/2), int(padh/2)), fill=0)
        
        # 裁剪crop_size的大小
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        
        # 不进行高斯模糊
        # gaussian blur as in PSP
        # if random.random() < 0.5:
            # img = img.filter( ImageFilter.GaussianBlur( radius=random.random() ) )

        return img, mask


    # 测试的时候直接resize到base_size的正方形
    def _testval_sync_transform(self, img, mask):
        crop_size = self.crop_size
        img = img.resize((crop_size, crop_size), Image.BILINEAR)
        mask = mask.resize((crop_size, crop_size), Image.NEAREST)

        return img, mask


class SIRST3(Data.Dataset):
    def __init__(self, mode='train'):
        #path of dataset
        self.base_dir = '/data1/hrm/Datasets/SIRST3/SIRST3'
        self.mean = [0.386, 0.386, 0.386]
        self.std = [0.141, 0.141, 0.141]
        # 256*256, 512*512以及256*256附近的分辨率
        self.base_size = 480
        self.crop_size = 480

        # list of images in dataset
        if mode == 'train':
            txtfile = 'train_SIRST3.txt'
        elif mode == 'test':
            txtfile = 'test_SIRST3.txt'

        self.list_file = osp.join(self.base_dir, 'img_idx', txtfile)
        self.imgs_dir = osp.join(self.base_dir, 'images')
        self.label_dir = osp.join(self.base_dir, 'masks')

        self.names = []
        with open(self.list_file, 'r') as f:
            self.names += [line.strip() for line in f.readlines()]

        self.mode = mode
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.imgs_dir, name+'.png')
        label_path = osp.join(self.label_dir, name+'.png')

        #由于输入的三通道、单通道图像都有，所以统一转成RGB的三通道，这也符合Unet等网络的期待尺寸
        img = Image.open(img_path).convert('RGB')
        #为后续多类型目标检测分割预留扩展空间 
        mask = Image.open(label_path).convert('L')

        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'test':
            img, mask = self._testval_sync_transform(img, mask)
        else:
            raise ValueError("Unkown self.mode")

        # img = self.transform(img)
        # print(torch.max(img))
        # print(torch.min(img))

        img, mask = self.transform(img), transforms.ToTensor()(mask)
        return img, mask

    def __len__(self):
        return len(self.names)

    def _sync_transform(self, img, mask):
        
        # 随机翻转旋转
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        if random.random() < 0.5:
            img = img.transpose(Image.ROTATE_90)
            mask = mask.transpose(Image.ROTATE_90)
        
        crop_size = self.crop_size
        
        #训练时，长边先Resize到基于base_size的随机的long_size，然后短边pad到crop_size大小，最后crop到crop_size大小
        
        # long_size = random.randint(256, 281)，然后长边等比例缩放到该大小
        long_size = random.randint(int(self.base_size * 1.0), int(self.base_size * 1.1))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        
        # 短边pad到crop_size的大小，并且是两边都进行pad
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            padh = padh + 1 if padh % 2 != 0 else padh
            padw = padw + 1 if padw % 2 != 0 else padw
            img = ImageOps.expand(img, border=(int(padw/2), int(padh/2), int(padw/2), int(padh/2)), fill=0)
            mask = ImageOps.expand(mask, border=(int(padw/2), int(padh/2), int(padw/2), int(padh/2)), fill=0)
        
        # 裁剪crop_size的大小
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        
        # 不进行高斯模糊
        # gaussian blur as in PSP
        # if random.random() < 0.5:
            # img = img.filter( ImageFilter.GaussianBlur( radius=random.random() ) )

        return img, mask


    # 测试的时候直接resize到base_size的正方形
    def _testval_sync_transform(self, img, mask):
        crop_size = self.crop_size
        img = img.resize((crop_size, crop_size), Image.BILINEAR)
        mask = mask.resize((crop_size, crop_size), Image.NEAREST)

        return img, mask


class NUDT_SIRST_Sea(Data.Dataset):
    def __init__(self, mode='train'):
        #path of dataset
        self.base_dir = '/data1/hrm/Datasets/NUDT-SIRST-Sea/NUDT-SIRST-Sea'
        self.mean = [0.159, 0.159, 0.159]
        self.std = [0.074, 0.074, 0.074]
        # 1024*1024, 740*1024, 1024*740以及740*740共3种分辨率
        self.base_size = 1024
        self.crop_size = 1024
        # self.base_size = 512 # must 1024*1024
        # self.crop_size = 480 # must 1024*1024

        # list of images in dataset
        if mode == 'train':
            txtfile = 'train.txt'
        elif mode == 'test':
            txtfile = 'test.txt'

        self.list_dir_file = osp.join(self.base_dir, 'idx_4961+847', txtfile)
        self.images_dir = osp.join(self.base_dir, 'images')
        self.labels_dir = osp.join(self.base_dir, 'masks')
        self.target_images_dir = osp.join(self.base_dir, 'Target_image')
        self.target_labels_dir = osp.join(self.base_dir, 'Target_mask')


        self.names = []
        with open(self.list_dir_file, 'r') as f:
            self.names += [line.strip() for line in f.readlines()]

        self.mode = mode
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])


    def __getitem__(self, idx):

        name = self.names[idx]
        img_path = osp.join(self.images_dir, name+'.png')
        label_path = osp.join(self.labels_dir, name+'.png')

        img = Image.open(img_path).convert('RGB')         ##由于输入的三通道、单通道图像都有，所以统一转成RGB的三通道，这也符合Unet等网络的期待尺寸
        mask = Image.open(label_path).convert('L')


        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'test':
            img, mask = self._testval_sync_transform(img, mask)
        else:
            raise ValueError("Unkown self.mode")


        # img = self.transform(img)
        # print(torch.max(img))
        # print(torch.min(img))

        img, mask = self.transform(img), transforms.ToTensor()(mask)
        return img, mask

    def __len__(self):
        return len(self.names)

    def _copy_paste_transform(self, img, label, CP_num):

        w, h = img.size
        target_images_files = os.listdir(self.target_images_dir)
        target_labels_files = os.listdir(self.target_labels_dir)
        range_k = len(target_images_files)
        dice = random.randint(0, 1)

        if dice==0:
            img=img
            label=label
        else:
            for i in range(CP_num):
                k = random.randint(0,range_k-1 )
                x = random.randint(0, w-1)
                y = random.randint(0, h-1)
                T_I_path = osp.join(self.target_images_dir, target_images_files[k])
                T_L_path = osp.join(self.target_labels_dir, target_labels_files[k])
                #由于输入的三通道、单通道图像都有，所以统一转成RGB的三通道，这也符合Unet等网络的期待尺寸
                T_img = Image.open(T_I_path).convert('RGB')
                #只认单通道
                T_label = Image.open(T_L_path).convert('L')
                img.paste(T_img, (x, y))
                label.paste(T_label, (x, y))

        return img, label

    def _sync_transform(self, img, mask):
        
        # 随机翻转旋转
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        if random.random() < 0.5:
            img = img.transpose(Image.ROTATE_90)
            mask = mask.transpose(Image.ROTATE_90)
        
        crop_size = self.crop_size
        
        #训练时，长边先Resize到基于base_size的随机的long_size，然后短边pad到crop_size大小，最后crop到crop_size大小
        
        # long_size = random.randint(1024, 1126)，然后长边等比例缩放到该大小
        long_size = random.randint(int(self.base_size * 1.0), int(self.base_size * 1.1))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        
        # 短边pad到crop_size的大小，并且是两边都进行pad
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            padh = padh + 1 if padh % 2 != 0 else padh
            padw = padw + 1 if padw % 2 != 0 else padw
            img = ImageOps.expand(img, border=(int(padw/2), int(padh/2), int(padw/2), int(padh/2)), fill=0)
            mask = ImageOps.expand(mask, border=(int(padw/2), int(padh/2), int(padw/2), int(padh/2)), fill=0)
        
        # 裁剪crop_size的大小
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        
        # 不进行高斯模糊
        # gaussian blur as in PSP
        # if random.random() < 0.5:
            # img = img.filter( ImageFilter.GaussianBlur( radius=random.random() ) )
        
        # final transform
        img, mask = self._copy_paste_transform(img, mask, random.randint(1,100)) # CP
        return img, mask

    # 测试的时候直接resize到base_size的正方形
    def _testval_sync_transform(self, img, mask):
        crop_size = self.crop_size
        img  = img.resize ((crop_size, crop_size), Image.BILINEAR)
        mask = mask.resize((crop_size, crop_size), Image.NEAREST)
        return img, mask


class IRDST(Data.Dataset):
    def __init__(self, mode='train'):   
        #path of dataset
        self.base_dir = '/data1/hrm/Datasets/IRDST/IRDST/real'

        self.mean = [0.398, 0.398, 0.398]
        self.std = [0.222, 0.222, 0.222]
        # 主要是992*742和720*480这2种分辨率
        self.base_size = 480
        self.crop_size = 480

        # list of images in dataset
        if mode == 'train':
            txtfile = 'train_IRDST-real.txt'
        elif mode == 'test':
            # txtfile = 'test.txt'
            txtfile = 'test_IRDST-real.txt'

        self.list_dir = osp.join(self.base_dir,  txtfile)
        self.imgs_dir = osp.join(self.base_dir, 'images')
        self.mask_dir = osp.join(self.base_dir, 'masks')

        self.names = []
        with open(self.list_dir, 'r') as f:
            self.names += [line.strip() for line in f.readlines()]

        self.mode = mode
        self.transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(self.mean, self.std) # mean and std
        ])

    def __getitem__(self, i):
        name = self.names[i]
        # 这里的文件名带有了斜杠
        img_path = self.imgs_dir+name+'.png'
        mask_path = self.mask_dir+name+'.png'

        #由于输入的三通道、单通道图像都有，所以统一转成RGB的三通道，这也符合Unet等网络的期待尺寸
        img = Image.open(img_path).convert('RGB')  
        #为后续多类型目标检测分割预留扩展空间 
        mask = Image.open(mask_path).convert('L')

        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'test':
            img, mask = self._testval_sync_transform(img, mask)
        else:
            raise ValueError("Unkown self.mode")

        img, mask = self.transform(img), transforms.ToTensor()(mask)
        return img, mask

    def __len__(self):
        return len(self.names)

    def _sync_transform(self, img, mask):
        
        # 随机翻转旋转
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        if random.random() < 0.5:
            img = img.transpose(Image.ROTATE_90)
            mask = mask.transpose(Image.ROTATE_90)
        
        crop_size = self.crop_size
        
        #训练时，长边先Resize到基于base_size的随机的long_size，然后短边pad到crop_size大小，最后crop到crop_size大小
        
        # long_size = random.randint(256, 281)，然后长边等比例缩放到该大小
        long_size = random.randint(int(self.base_size * 1.0), int(self.base_size * 1.1))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        
        # 短边pad到crop_size的大小，并且是两边都进行pad
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            padh = padh + 1 if padh % 2 != 0 else padh
            padw = padw + 1 if padw % 2 != 0 else padw
            img = ImageOps.expand(img, border=(int(padw/2), int(padh/2), int(padw/2), int(padh/2)), fill=0)
            mask = ImageOps.expand(mask, border=(int(padw/2), int(padh/2), int(padw/2), int(padh/2)), fill=0)
        
        # 裁剪crop_size的大小
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        
        # 不进行高斯模糊
        # gaussian blur as in PSP
        # if random.random() < 0.5:
            # img = img.filter( ImageFilter.GaussianBlur( radius=random.random() ) )

        return img, mask


    # 测试的时候直接resize到base_size的正方形
    def _testval_sync_transform(self, img, mask):
        crop_size = self.crop_size
        img = img.resize((crop_size, crop_size), Image.BILINEAR)
        mask = mask.resize((crop_size, crop_size), Image.NEAREST)

        return img, mask



class GrokCV_NUAA_SIRST(Data.Dataset):
    def __init__(self, mode='train'):
        #path of dataset
        self.base_dir = '/data1/hrm/Datasets/GrokCV/NUAA-SIRST/NUAA'

        self.mean = [0.442, 0.442, 0.442]
        self.std = [0.111, 0.111, 0.111]
        # 杂乱，从222*222到367*305都有，官方缩放成了480*480
        # self.base_size = 512 # default
        # self.crop_size = 512 # default
        self.base_size = 512 # for MLP_Net
        self.crop_size = 512 # for MLP_Net

        if mode == 'train':
            self.data_dir = 'trainval'
        elif mode == 'test':
            self.data_dir = 'test'

        self.imgs_dir = osp.join(self.base_dir, self.data_dir, 'images')
        self.label_dir = osp.join(self.base_dir, self.data_dir, 'masks')


        self.img_names = []
        for img in natsorted( os.listdir(self.imgs_dir) ):
            if img.endswith('png'):
                self.img_names.append(img)

        self.mode = mode
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

    def __getitem__(self, i):
        name = self.img_names[i]
        img_path = osp.join(self.imgs_dir, name)
        label_path = osp.join(self.label_dir, name)

        #由于输入的三通道、单通道图像都有，所以统一转成RGB的三通道，这也符合Unet等网络的期待尺寸
        img = Image.open(img_path).convert('RGB')
        #为后续多类型目标检测分割预留扩展空间 
        mask = Image.open(label_path).convert('L')

        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'test':
            img, mask = self._testval_sync_transform(img, mask)
        else:
            raise ValueError("Unkown self.mode")

        # img = self.transform(img)
        # print(torch.max(img))
        # print(torch.min(img))

        img, mask = self.transform(img), transforms.ToTensor()(mask)
        return img, mask

    def __len__(self):
        return len(self.img_names)

    def _sync_transform(self, img, mask):
        
        # 随机翻转旋转
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random() < 0.5:
            img = img.transpose(Image.ROTATE_90)
            mask = mask.transpose(Image.ROTATE_90)
        
        crop_size = self.crop_size
        
        #训练时，长边先Resize到基于base_size的随机的long_size，然后短边pad到crop_size大小，最后crop到crop_size大小
        
        # long_size = random.randint(480, 528)，然后长边等比例缩放到该大小
        long_size = random.randint(int(self.base_size * 1.0), int(self.base_size * 1.1))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        
        # 短边pad到crop_size的大小，并且是两边都进行pad
        if short_size < crop_size:

            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0

            padh = padh + 1 if padh % 2 != 0 else padh
            padw = padw + 1 if padw % 2 != 0 else padw

            img = ImageOps.expand(img, border=(int(padw/2), int(padh/2), int(padw/2), int(padh/2)), fill=0)
            mask = ImageOps.expand(mask, border=(int(padw/2), int(padh/2), int(padw/2), int(padh/2)), fill=0)
        
        # 裁剪crop_size的大小
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        
        # 不进行高斯模糊
        # gaussian blur as in PSP
        # if random.random() < 0.5:
            # img = img.filter( ImageFilter.GaussianBlur( radius=random.random() ) )

        return img, mask


    # 测试的时候直接resize到base_size的正方形
    def _testval_sync_transform(self, img, mask):
        crop_size = self.crop_size
        img = img.resize((crop_size, crop_size), Image.BILINEAR)
        mask = mask.resize((crop_size, crop_size), Image.NEAREST)

        return img, mask


class GrokCV_NUDT_SIRST(Data.Dataset):
    def __init__(self, mode='train'):
        #path of dataset
        self.base_dir = '/data1/hrm/Datasets/GrokCV/NUDT-SIRST/NUDT'

        self.mean = [0.424, 0.424, 0.424]
        self.std = [0.130, 0.130, 0.130]
        # 杂乱，从222*222到367*305都有，官方缩放成了480*480
        self.base_size = 256
        self.crop_size = 256

        if mode == 'train':
            self.data_dir = 'trainval'
        elif mode == 'test':
            self.data_dir = 'test'

        self.imgs_dir = osp.join(self.base_dir, self.data_dir, 'images')
        self.label_dir = osp.join(self.base_dir, self.data_dir, 'masks')


        self.img_names = []
        for img in natsorted( os.listdir(self.imgs_dir) ):
            if img.endswith('png'):
                self.img_names.append(img)

        self.mode = mode
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

    def __getitem__(self, i):
        name = self.img_names[i]
        img_path = osp.join(self.imgs_dir, name)
        label_path = osp.join(self.label_dir, name)

        #由于输入的三通道、单通道图像都有，所以统一转成RGB的三通道，这也符合Unet等网络的期待尺寸
        img = Image.open(img_path).convert('RGB')
        #为后续多类型目标检测分割预留扩展空间 
        mask = Image.open(label_path).convert('L')

        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'test':
            img, mask = self._testval_sync_transform(img, mask)
        else:
            raise ValueError("Unkown self.mode")

        # img = self.transform(img)
        # print(torch.max(img))
        # print(torch.min(img))

        img, mask = self.transform(img), transforms.ToTensor()(mask)
        return img, mask

    def __len__(self):
        return len(self.img_names)

    def _sync_transform(self, img, mask):
        
        # 随机翻转旋转
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random() < 0.5:
            img = img.transpose(Image.ROTATE_90)
            mask = mask.transpose(Image.ROTATE_90)
        
        crop_size = self.crop_size
        
        #训练时，长边先Resize到基于base_size的随机的long_size，然后短边pad到crop_size大小，最后crop到crop_size大小
        
        # long_size = random.randint(480, 528)，然后长边等比例缩放到该大小
        long_size = random.randint(int(self.base_size * 1.0), int(self.base_size * 1.1))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        
        # 短边pad到crop_size的大小，并且是两边都进行pad
        if short_size < crop_size:

            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0

            padh = padh + 1 if padh % 2 != 0 else padh
            padw = padw + 1 if padw % 2 != 0 else padw

            img = ImageOps.expand(img, border=(int(padw/2), int(padh/2), int(padw/2), int(padh/2)), fill=0)
            mask = ImageOps.expand(mask, border=(int(padw/2), int(padh/2), int(padw/2), int(padh/2)), fill=0)
        
        # 裁剪crop_size的大小
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        
        # 不进行高斯模糊
        # gaussian blur as in PSP
        # if random.random() < 0.5:
            # img = img.filter( ImageFilter.GaussianBlur( radius=random.random() ) )

        return img, mask


    # 测试的时候直接resize到base_size的正方形
    def _testval_sync_transform(self, img, mask):
        crop_size = self.crop_size
        img = img.resize((crop_size, crop_size), Image.BILINEAR)
        mask = mask.resize((crop_size, crop_size), Image.NEAREST)

        return img, mask


class GrokCV_IRSTD_1k(Data.Dataset):
    def __init__(self, mode='train'):
        #path of dataset
        self.base_dir = '/data1/hrm/Datasets/GrokCV/IRSTD-1k/IRSTD-1k'

        self.mean = [0.343, 0.343, 0.343]
        self.std = [0.157, 0.157, 0.157]
        # 杂乱，从222*222到367*305都有，官方缩放成了480*480
        self.base_size = 512
        self.crop_size = 512

        if mode == 'train':
            self.data_dir = 'trainval'
        elif mode == 'test':
            self.data_dir = 'test'

        self.imgs_dir = osp.join(self.base_dir, self.data_dir, 'images')
        self.label_dir = osp.join(self.base_dir, self.data_dir, 'masks')


        self.img_names = []
        for img in natsorted( os.listdir(self.imgs_dir) ):
            if img.endswith('png'):
                self.img_names.append(img)

        self.mode = mode
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

    def __getitem__(self, i):
        name = self.img_names[i]
        img_path = osp.join(self.imgs_dir, name)
        label_path = osp.join(self.label_dir, name)

        #由于输入的三通道、单通道图像都有，所以统一转成RGB的三通道，这也符合Unet等网络的期待尺寸
        img = Image.open(img_path).convert('RGB')
        #为后续多类型目标检测分割预留扩展空间 
        mask = Image.open(label_path).convert('L')

        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'test':
            img, mask = self._testval_sync_transform(img, mask)
        else:
            raise ValueError("Unkown self.mode")

        # img = self.transform(img)
        # print(torch.max(img))
        # print(torch.min(img))

        img, mask = self.transform(img), transforms.ToTensor()(mask)
        return img, mask

    def __len__(self):
        return len(self.img_names)

    def _sync_transform(self, img, mask):
        
        # 随机翻转旋转
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random() < 0.5:
            img = img.transpose(Image.ROTATE_90)
            mask = mask.transpose(Image.ROTATE_90)
        
        crop_size = self.crop_size
        
        #训练时，长边先Resize到基于base_size的随机的long_size，然后短边pad到crop_size大小，最后crop到crop_size大小
        
        # long_size = random.randint(480, 528)，然后长边等比例缩放到该大小
        long_size = random.randint(int(self.base_size * 1.0), int(self.base_size * 1.1))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        
        # 短边pad到crop_size的大小，并且是两边都进行pad
        if short_size < crop_size:

            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0

            padh = padh + 1 if padh % 2 != 0 else padh
            padw = padw + 1 if padw % 2 != 0 else padw

            img = ImageOps.expand(img, border=(int(padw/2), int(padh/2), int(padw/2), int(padh/2)), fill=0)
            mask = ImageOps.expand(mask, border=(int(padw/2), int(padh/2), int(padw/2), int(padh/2)), fill=0)
        
        # 裁剪crop_size的大小
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        
        # 不进行高斯模糊
        # gaussian blur as in PSP
        # if random.random() < 0.5:
            # img = img.filter( ImageFilter.GaussianBlur( radius=random.random() ) )

        return img, mask


    # 测试的时候直接resize到base_size的正方形
    def _testval_sync_transform(self, img, mask):
        crop_size = self.crop_size
        img = img.resize((crop_size, crop_size), Image.BILINEAR)
        mask = mask.resize((crop_size, crop_size), Image.NEAREST)

        return img, mask


class GrokCV_SIRST_Aug(Data.Dataset):
    def __init__(self, mode='train'):
        #path of dataset
        self.base_dir = '/data1/hrm/Datasets/GrokCV/SIRST-Aug/SIRST_AUG'

        self.mean = [0.290, 0.290, 0.290]
        self.std = [0.194, 0.194, 0.194]
        # 杂乱，从222*222到367*305都有，官方缩放成了480*480
        self.base_size = 256
        self.crop_size = 256

        if mode == 'train':
            self.data_dir = 'trainval'
        elif mode == 'test':
            self.data_dir = 'test'

        self.imgs_dir = osp.join(self.base_dir, self.data_dir, 'images')
        self.label_dir = osp.join(self.base_dir, self.data_dir, 'masks')


        self.img_names = []
        for img in natsorted( os.listdir(self.imgs_dir) ):
            if img.endswith('png'):
                self.img_names.append(img)

        self.mode = mode
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

    def __getitem__(self, i):
        name = self.img_names[i]
        img_path = osp.join(self.imgs_dir, name)
        label_path = osp.join(self.label_dir, name)

        #由于输入的三通道、单通道图像都有，所以统一转成RGB的三通道，这也符合Unet等网络的期待尺寸
        img = Image.open(img_path).convert('RGB')
        #为后续多类型目标检测分割预留扩展空间 
        mask = Image.open(label_path).convert('L')

        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'test':
            img, mask = self._testval_sync_transform(img, mask)
        else:
            raise ValueError("Unkown self.mode")

        # img = self.transform(img)
        # print(torch.max(img))
        # print(torch.min(img))

        img, mask = self.transform(img), transforms.ToTensor()(mask)
        return img, mask

    def __len__(self):
        return len(self.img_names)

    def _sync_transform(self, img, mask):
        
        # 随机翻转旋转
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random() < 0.5:
            img = img.transpose(Image.ROTATE_90)
            mask = mask.transpose(Image.ROTATE_90)
        
        crop_size = self.crop_size
        
        #训练时，长边先Resize到基于base_size的随机的long_size，然后短边pad到crop_size大小，最后crop到crop_size大小
        
        # long_size = random.randint(480, 528)，然后长边等比例缩放到该大小
        long_size = random.randint(int(self.base_size * 1.0), int(self.base_size * 1.1))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        
        # 短边pad到crop_size的大小，并且是两边都进行pad
        if short_size < crop_size:

            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0

            padh = padh + 1 if padh % 2 != 0 else padh
            padw = padw + 1 if padw % 2 != 0 else padw

            img = ImageOps.expand(img, border=(int(padw/2), int(padh/2), int(padw/2), int(padh/2)), fill=0)
            mask = ImageOps.expand(mask, border=(int(padw/2), int(padh/2), int(padw/2), int(padh/2)), fill=0)
        
        # 裁剪crop_size的大小
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        
        # 不进行高斯模糊
        # gaussian blur as in PSP
        # if random.random() < 0.5:
            # img = img.filter( ImageFilter.GaussianBlur( radius=random.random() ) )

        return img, mask


    # 测试的时候直接resize到base_size的正方形
    def _testval_sync_transform(self, img, mask):
        crop_size = self.crop_size
        img = img.resize((crop_size, crop_size), Image.BILINEAR)
        mask = mask.resize((crop_size, crop_size), Image.NEAREST)

        return img, mask
