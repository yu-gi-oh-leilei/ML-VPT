from cProfile import label
import os
import random
import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms as transforms
from dataset.transforms import SLCutoutPIL, CutoutPIL
from dataset.transforms import MultiScaleCrop
from dataset.two_transformer import CustomDataAugmentation
from dataset.randaugment import RandAugment
import torchvision.transforms as transforms
from PIL import ImageDraw
from dataset.handlers import COCO2014_handler, VG256_handler, \
                             VOC2007_handler, VOC2012_handler, \
                             CUB_200_2011_handler, NUS_WIDE_handler, OBJECTS_365_handler

np.set_printoptions(suppress=True)

HANDLER_DICT = {
    'coco': COCO2014_handler,
    'nus': NUS_WIDE_handler,
    'voc2007': VOC2007_handler,
    'voc2012': VOC2012_handler,
    'vg256': VG256_handler,
    'cub': CUB_200_2011_handler,
    'objects365': OBJECTS_365_handler
}


def distributedsampler(cfg, train_dataset, val_dataset):
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    assert cfg.OPTIMIZER.batch_size // dist.get_world_size() == cfg.OPTIMIZER.batch_size / dist.get_world_size(), 'Batch size is not divisible by num of gpus.'
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.OPTIMIZER.batch_size // dist.get_world_size(), 
        # batch_size=cfg.OPTIMIZER.batch_size,
        shuffle=(train_sampler is None),
        num_workers=cfg.DATA.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=cfg.OPTIMIZER.batch_size // dist.get_world_size(), 
        # batch_size=cfg.OPTIMIZER.batch_size, 
        shuffle=False,
        num_workers=cfg.DATA.num_workers, pin_memory=True, sampler=val_sampler, drop_last=False)
    return train_loader, val_loader, train_sampler

def get_datasets(cfg, logger):

    if cfg.DATA.TRANSFORM.crop:
        train_data_transform_list = [transforms.Resize((cfg.DATA.TRANSFORM.img_size+64, cfg.DATA.TRANSFORM.img_size+64)),
                                                MultiScaleCrop(cfg.DATA.TRANSFORM.img_size, scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor()]
    else:
        train_data_transform_list = [transforms.Resize((cfg.DATA.TRANSFORM.img_size, cfg.DATA.TRANSFORM.img_size)),
                                                RandAugment(),
                                                transforms.ToTensor()]

    test_data_transform_list =  [transforms.Resize((cfg.DATA.TRANSFORM.img_size, cfg.DATA.TRANSFORM.img_size)),
                                            transforms.ToTensor()]
    if cfg.DATA.TRANSFORM.cutout and cfg.DATA.TRANSFORM.crop is not True:
        logger.info("Using Cutout!!!")
        assert cfg.DATA.TRANSFORM.length == cfg.DATA.TRANSFORM.img_size // 2, "Cutout length should be half of the image size" 
        train_data_transform_list.insert(1, SLCutoutPIL(n_holes=cfg.DATA.TRANSFORM.n_holes, length=cfg.DATA.TRANSFORM.length))
    
    if cfg.DATA.TRANSFORM.remove_norm is False:
        if cfg.DATA.TRANSFORM.orid_norm:
            normalize = transforms.Normalize(mean=[0, 0, 0],
                                            std=[1, 1, 1])
            logger.info("mean=[0, 0, 0], std=[1, 1, 1]")
        else:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
            logger.info("mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]")
        train_data_transform_list.append(normalize)
        test_data_transform_list.append(normalize)
    else:
        logger.info('remove normalize')

    train_data_transform = transforms.Compose(train_data_transform_list)
    test_data_transform = transforms.Compose(test_data_transform_list)
    
    # train_data_transform = transforms.Compose(test_data_transform_list)
    

    # TRANSFORMS: ["random_resized_crop", "MLC_Policy", "random_flip", "normalize"]
    # if cfg.DATA.TRANSFORM.TWOTYPE.is_twotype == False:
    #     # train_data_transform = build_transform(cfg=cfg, is_train=True, choices=None)
    #     # test_data_transform = build_transform(cfg=cfg, is_train=False, choices=None)

    #     train_data_transform_list = [transforms.Resize((cfg.DATA.TRANSFORM.img_size, cfg.DATA.TRANSFORM.img_size)),
    #                                 transforms.RandomHorizontalFlip(),
    #                                 CutoutPIL(cutout_factor=0.5),
    #                                 RandAugment(),
    #                                 transforms.ToTensor()]

    #     test_data_transform_list = [transforms.Resize((cfg.DATA.TRANSFORM.img_size, cfg.DATA.TRANSFORM.img_size)),
    #                                 transforms.ToTensor()]
    #     train_data_transform_list.append(normalize)
    #     test_data_transform_list.append(normalize)

    #     train_data_transform = transforms.Compose(train_data_transform_list)
    #     test_data_transform = transforms.Compose(test_data_transform_list)
    # else:
    #     train_data_transform = CustomDataAugmentation(size=cfg.DATA.TRANSFORM.TWOTYPE.img_size, min_scale=cfg.DATA.TRANSFORM.TWOTYPE.min_scale)
    #     test_data_transform = transforms.Compose([transforms.Resize((cfg.DATA.TRANSFORM.img_size, cfg.DATA.TRANSFORM.img_size)),
    #                                               transforms.ToTensor(),
    #                                               transforms.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)])


    logger.info('train_data_transform {}'.format(train_data_transform))
    logger.info('test_data_transform {}'.format(test_data_transform))

    # print(cfg.DATA.dataset_dir)
    # # print(source_data)
    # print('========='*100)

    # cfg.DATA.dataset_dir = os.path.join('/media/data/maleilei/MLIC/DDP-VTPMOD' ,cfg.DATA.dataset_dir)
    
    # load data:
    source_data = load_data(cfg.DATA.dataset_dir)

	
    data_handler = HANDLER_DICT[cfg.DATA.dataname]

    train_dataset = data_handler(source_data['train']['images'], source_data['train']['labels'], cfg.DATA.dataset_dir, transform=train_data_transform)
    
    val_dataset = data_handler(source_data['val']['images'], source_data['val']['labels'], cfg.DATA.dataset_dir, transform=test_data_transform)

    logger.info('length of train dataset {}'.format(len(train_dataset)))
    logger.info('length of val dataset {}'.format(len(val_dataset)))

    return train_dataset, val_dataset



def load_data(base_path):
    data = {}
    for phase in ['train', 'val']:
        data[phase] = {}
        data[phase]['labels'] = np.load(os.path.join(base_path, 'formatted_{}_labels.npy'.format(phase)))
        data[phase]['images'] = np.load(os.path.join(base_path, 'formatted_{}_images.npy'.format(phase)))
    return data

class CutoutPIL(object):
    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return x


class TransformPatch_Train(object):
    def __init__(self, args):
        self.n_grid = args.n_grid
        self.image_size = args.image_size

        self.strong = transforms.Compose([
                    transforms.Resize((args.image_size, args.image_size)),
                    CutoutPIL(cutout_factor=0.5),
                    RandAugment(),
                    transforms.ToTensor(),
                ])

    def __call__(self, img):
        strong_list = [self.strong(img)] 
        
        # Append patches
        img = img.resize((self.image_size, self.image_size))

        # To permute the orders of patches
        x_order = np.random.permutation(self.n_grid)
        y_order = np.random.permutation(self.n_grid)

        grid_size_x = img.size[0] // self.n_grid
        grid_size_y = img.size[1] // self.n_grid
        
        for i in x_order:
            for j in y_order:
                x_offset = i * grid_size_x
                y_offset = j * grid_size_y
                patch = img.crop((x_offset, y_offset, x_offset + grid_size_x, y_offset + grid_size_y))
                # Append patches
                strong_list.append(self.strong(patch))
       
        return strong_list


class TransformPatch_Val(object):
    def __init__(self, args):
        self.n_grid = args.n_grid
        self.image_size = args.image_size
        
        self.weak = transforms.Compose([
                        transforms.Resize((args.image_size, args.image_size)),
                        transforms.ToTensor(),
                        # normalize, # no need, toTensor does normalization
                    ])

    def __call__(self, img):
        weak_list = [self.weak(img)]

        # Append patches
        img = img.resize((self.image_size, self.image_size))

        # To permute the order for local patched
        x_order = np.random.permutation(self.n_grid)
        y_order = np.random.permutation(self.n_grid)

        grid_size_x = img.size[0] // self.n_grid
        grid_size_y = img.size[1] // self.n_grid
        
        for i in x_order:
            for j in y_order:
                x_offset = i * grid_size_x
                y_offset = j * grid_size_y
                patch = img.crop((x_offset, y_offset, x_offset + grid_size_x, y_offset + grid_size_y))
                # Append patches
                weak_list.append(self.weak(patch))
        
        return weak_list




