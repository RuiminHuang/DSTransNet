import os
from PIL import Image
import numpy as np
import tqdm
import os.path as osp


def cal_NUST_SIRST():
    base_dir = '/data1/hrm/Datasets/NUST-SIRST/MDvsFA_cGAN/data'
    imgs_dir = osp.join(base_dir, 'training')

    img_channels = 3
    cumulative_mean = np.zeros(img_channels)
    cumulative_std = np.zeros(img_channels)

    for i in tqdm.tqdm( range(10000) ):
        
        # has been broken
        if i  in [239, 245, 260, 264, 2543, 2553, 2561, 2808, 2817, 2819, 3503, 3504, 3947, 3949, 3962, 7389, 7395, 8094, 8105, 8112, 8757, 8772]:
            i += 3
        
        img_path = osp.join(imgs_dir, '%06d_1.png' % i)
        img = np.array( Image.open(img_path).convert('RGB') ) / 255.

        for d in range(3):
            cumulative_mean[d] += img[:, :, d].mean()
            cumulative_std[d] += img[:, :, d].std()

    mean = cumulative_mean / 10000
    std = cumulative_std / 10000
    print("cal_NUST_SIRST")
    print(f"mean: {mean}")
    print(f"std: {std}")


def cal_NUAA_SIRST():
    base_dir = '/data1/hrm/Datasets/NUAA-SIRST/sirst'
    txtfile = 'trainval.txt'
    
    list_file = osp.join(base_dir, 'idx_427', txtfile)
    imgs_dir = osp.join(base_dir, 'images')


    names = []
    with open(list_file, 'r') as f:
        names += [line.strip() for line in f.readlines()]

    img_channels = 3
    cumulative_mean = np.zeros(img_channels)
    cumulative_std = np.zeros(img_channels)

    for i in tqdm.tqdm( range( len(names) ) ):
        
        name = names[i]
        img_path = osp.join(imgs_dir, name+'.png')
        img = np.array( Image.open(img_path).convert('RGB') ) / 255.

        for d in range(3):
            cumulative_mean[d] += img[:, :, d].mean()
            cumulative_std[d] += img[:, :, d].std()

    mean = cumulative_mean / len(names)
    std = cumulative_std / len(names)
    print("cal_NUAA_SIRST")
    print(f"mean: {mean}")
    print(f"std: {std}")


def cal_IRSTD_1k():
    base_dir = '/data1/hrm/Datasets/IRSTD-1k/IRSTD-1k'
    txtfile = 'trainval.txt'
    
    list_file = osp.join(base_dir, txtfile)
    imgs_dir = osp.join(base_dir, 'IRSTD1k_Img')


    names = []
    with open(list_file, 'r') as f:
        names += [line.strip() for line in f.readlines()]

    img_channels = 3
    cumulative_mean = np.zeros(img_channels)
    cumulative_std = np.zeros(img_channels)

    for i in tqdm.tqdm( range( len(names) ) ):
        
        name = names[i]
        img_path = osp.join(imgs_dir, name+'.png')
        img = np.array( Image.open(img_path).convert('RGB') ) / 255.

        for d in range(3):
            cumulative_mean[d] += img[:, :, d].mean()
            cumulative_std[d] += img[:, :, d].std()

    mean = cumulative_mean / len(names)
    std = cumulative_std / len(names)
    print("cal_IRSTD_1k")
    print(f"mean: {mean}")
    print(f"std: {std}")


def cal_SIRST_Aug():
    base_dir = '/data1/hrm/Datasets/SIRST-Aug/sirst_aug'
    imgs_dir = osp.join(base_dir, 'trainval')


    # for __len__
    names = []
    for filename in os.listdir(osp.join(imgs_dir, 'images')):
        if filename.endswith('png'):
            names.append(filename)


    img_channels = 3
    cumulative_mean = np.zeros(img_channels)
    cumulative_std = np.zeros(img_channels)

    for i in tqdm.tqdm( range(len(names)) ):
        
        img_path = osp.join(imgs_dir, 'images', '%06d.png' % i)
        img = np.array( Image.open(img_path).convert('RGB') ) / 255.

        for d in range(3):
            cumulative_mean[d] += img[:, :, d].mean()
            cumulative_std[d] += img[:, :, d].std()

    mean = cumulative_mean / len(names)
    std = cumulative_std / len(names)
    print("SIRST_Aug")
    print(f"mean: {mean}")
    print(f"std: {std}")


def cal_SIRST_V2():
    base_dir = '/data1/hrm/Datasets/SIRST-V2/open-sirst-v2'
    txtfile = 'trainval_full.txt'
    
    list_file = osp.join(base_dir, 'splits', txtfile)
    imgs_dir = osp.join(base_dir, 'mixed')


    names = []
    with open(list_file, 'r') as f:
        names += [line.strip() for line in f.readlines()]

    img_channels = 3
    cumulative_mean = np.zeros(img_channels)
    cumulative_std = np.zeros(img_channels)

    for i in tqdm.tqdm( range( len(names) ) ):
        
        name = names[i]
        img_path = osp.join(imgs_dir, name+'.png')
        img = np.array( Image.open(img_path).convert('RGB') ) / 255.

        for d in range(3):
            cumulative_mean[d] += img[:, :, d].mean()
            cumulative_std[d] += img[:, :, d].std()

    mean = cumulative_mean / len(names)
    std = cumulative_std / len(names)
    print("cal_SIRST_V2")
    print(f"mean: {mean}")
    print(f"std: {std}")


def cal_NUDT_SIRST():
    base_dir = '/data1/hrm/Datasets/NUDT-SIRST/NUDT-SIRST'
    txtfile = 'train_NUDT-SIRST.txt'
    
    list_file = osp.join(base_dir, txtfile)
    imgs_dir = osp.join(base_dir, 'images')


    names = []
    with open(list_file, 'r') as f:
        names += [line.strip() for line in f.readlines()]

    img_channels = 3
    cumulative_mean = np.zeros(img_channels)
    cumulative_std = np.zeros(img_channels)

    for i in tqdm.tqdm( range( len(names) ) ):
        
        name = names[i]
        img_path = osp.join(imgs_dir, name+'.png')
        img = np.array( Image.open(img_path).convert('RGB') ) / 255.

        for d in range(3):
            cumulative_mean[d] += img[:, :, d].mean()
            cumulative_std[d] += img[:, :, d].std()

    mean = cumulative_mean / len(names)
    std = cumulative_std / len(names)
    print("cal_NUDT_SIRST")
    print(f"mean: {mean}")
    print(f"std: {std}")


def cal_SIRST3():
    base_dir = '/data1/hrm/Datasets/SIRST3/SIRST3'
    txtfile = 'train_SIRST3.txt'
    
    list_file = osp.join(base_dir, 'img_idx', txtfile)
    imgs_dir = osp.join(base_dir, 'images')


    names = []
    with open(list_file, 'r') as f:
        names += [line.strip() for line in f.readlines()]

    img_channels = 3
    cumulative_mean = np.zeros(img_channels)
    cumulative_std = np.zeros(img_channels)

    for i in tqdm.tqdm( range( len(names) ) ):
        
        name = names[i]
        img_path = osp.join(imgs_dir, name+'.png')
        img = np.array( Image.open(img_path).convert('RGB') ) / 255.

        for d in range(3):
            cumulative_mean[d] += img[:, :, d].mean()
            cumulative_std[d] += img[:, :, d].std()

    mean = cumulative_mean / len(names)
    std = cumulative_std / len(names)
    print("cal_SIRST3")
    print(f"mean: {mean}")
    print(f"std: {std}")


def cal_NUDT_SIRST_Sea():
    base_dir = '/data1/hrm/Datasets/NUDT-SIRST-Sea/NUDT-SIRST-Sea'
    txtfile = 'train.txt'
    
    list_file = osp.join(base_dir, 'idx_4961+847', txtfile)
    imgs_dir = osp.join(base_dir, 'images')

    names = []
    with open(list_file, 'r') as f:
        names += [line.strip() for line in f.readlines()]

    img_channels = 3
    cumulative_mean = np.zeros(img_channels)
    cumulative_std = np.zeros(img_channels)

    for i in tqdm.tqdm( range( len(names) ) ):
        
        name = names[i]
        img_path = osp.join(imgs_dir, name+'.png')
        img = np.array( Image.open(img_path).convert('RGB') ) / 255.

        for d in range(3):
            cumulative_mean[d] += img[:, :, d].mean()
            cumulative_std[d] += img[:, :, d].std()

    mean = cumulative_mean / len(names)
    std = cumulative_std / len(names)
    print("cal_NUDT_SIRST_Sea")
    print(f"mean: {mean}")
    print(f"std: {std}")


def cal_IRDST():

    base_dir = '/data1/hrm/Datasets/IRDST/IRDST/real'
    txtfile = 'train_IRDST-real.txt'
    
    list_file = osp.join(base_dir, txtfile)
    imgs_dir = osp.join(base_dir, 'images')


    names = []
    with open(list_file, 'r') as f:
        names += [line.strip() for line in f.readlines()]

    img_channels = 3
    cumulative_mean = np.zeros(img_channels)
    cumulative_std = np.zeros(img_channels)

    for i in tqdm.tqdm( range( len(names) ) ):
        
        name = names[i]

        img_path = imgs_dir+name+'.png'
        img = np.array( Image.open(img_path).convert('RGB') ) / 255.

        for d in range(3):
            cumulative_mean[d] += img[:, :, d].mean()
            cumulative_std[d] += img[:, :, d].std()

    mean = cumulative_mean / len(names)
    std = cumulative_std / len(names)
    print("cal_IRDST")
    print(f"mean: {mean}")
    print(f"std: {std}")

if __name__ == '__main__':
    cal_NUST_SIRST()
    cal_NUAA_SIRST()
    cal_IRSTD_1k()
    cal_SIRST_Aug()
    cal_SIRST_V2()
    cal_NUDT_SIRST()
    cal_SIRST3()
    cal_NUDT_SIRST_Sea()
    cal_IRDST()
