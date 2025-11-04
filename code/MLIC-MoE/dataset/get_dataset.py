# import torch
# import torch.distributed as dist
# import torchvision.transforms as transforms
# # from dataset.cocodataset import CoCoDataset
# from dataset.full_coco14dataset import FullCOCO2014dataset
# from dataset.full_coco17dataset import FullCOCO2017dataset
# from dataset.full_nuswide import FullNusWideDataset
# from dataset.full_pascal_voc07 import FullVoc07Dataset
# from dataset.full_pascal_voc12 import FullVoc12Dataset
# from dataset.full_vg import FullVGDataset
# from dataset.full_cub200 import FullCUB200Dataset
# from dataset.transforms import SLCutoutPIL, CutoutPIL
# from dataset.transforms import MultiScaleCrop
# from dataset.transforms import build_transform
# from dataset.randaugment import RandAugment
# from dataset.two_transformer import CustomDataAugmentation
# import os.path as osp

# def distributedsampler(cfg, train_dataset, val_dataset):
#     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
#     assert cfg.OPTIMIZER.batch_size // dist.get_world_size() == cfg.OPTIMIZER.batch_size / dist.get_world_size(), 'Batch size is not divisible by num of gpus.'
#     train_loader = torch.utils.data.DataLoader(
#         train_dataset,
#         batch_size=cfg.OPTIMIZER.batch_size // dist.get_world_size(), 
#         # batch_size=cfg.OPTIMIZER.batch_size,
#         shuffle=(train_sampler is None),
#         num_workers=cfg.DATA.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    
#     val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
#     val_loader = torch.utils.data.DataLoader(
#         val_dataset, 
#         batch_size=cfg.OPTIMIZER.batch_size // dist.get_world_size(), 
#         # batch_size=cfg.OPTIMIZER.batch_size, 
#         shuffle=False,
#         num_workers=cfg.DATA.num_workers, pin_memory=True, sampler=val_sampler)
#     return train_loader, val_loader, train_sampler


# def get_datasets(cfg, logger):
#     if cfg.DATA.TRANSFORM.crop:
#         train_data_transform_list = [transforms.Resize((cfg.DATA.TRANSFORM.img_size+64, cfg.DATA.TRANSFORM.img_size+64)),
#                                                 MultiScaleCrop(cfg.DATA.TRANSFORM.img_size, scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2),
#                                                 transforms.RandomHorizontalFlip(),
#                                                 transforms.ToTensor()]
#     else:
#         train_data_transform_list = [transforms.Resize((cfg.DATA.TRANSFORM.img_size, cfg.DATA.TRANSFORM.img_size)),
#                                                 RandAugment(),
#                                                 transforms.ToTensor()]

#     test_data_transform_list =  [transforms.Resize((cfg.DATA.TRANSFORM.img_size, cfg.DATA.TRANSFORM.img_size)),
#                                             transforms.ToTensor()]
#     if cfg.DATA.TRANSFORM.cutout and cfg.DATA.TRANSFORM.crop is not True:
#         logger.info("Using Cutout!!!")
#         train_data_transform_list.insert(1, SLCutoutPIL(n_holes=cfg.DATA.TRANSFORM.n_holes, length=cfg.DATA.TRANSFORM.length))
    

#     if cfg.DATA.TRANSFORM.remove_norm is False:
#         if cfg.DATA.TRANSFORM.orid_norm:
#             normalize = transforms.Normalize(mean=[0, 0, 0],
#                                             std=[1, 1, 1])
#             logger.info("mean=[0, 0, 0], std=[1, 1, 1]")
#         else:
#             normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                         std=[0.229, 0.224, 0.225])
#             logger.info("mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]")
#         train_data_transform_list.append(normalize)
#         test_data_transform_list.append(normalize)
#     else:
#         logger.info('remove normalize')

#     # train_data_transform = transforms.Compose(train_data_transform_list)
#     # # train_data_transform = transforms.Compose(test_data_transform_list)
#     # test_data_transform = transforms.Compose(test_data_transform_list)

#     # TRANSFORMS: ["random_resized_crop", "MLC_Policy", "random_flip", "normalize"]
#     if cfg.DATA.TRANSFORM.TWOTYPE.is_twotype == False:
#         # train_data_transform = build_transform(cfg=cfg, is_train=True, choices=None)
#         # test_data_transform = build_transform(cfg=cfg, is_train=False, choices=None)

#         train_data_transform_list = [transforms.Resize((cfg.DATA.TRANSFORM.img_size, cfg.DATA.TRANSFORM.img_size)),
#                                     transforms.RandomHorizontalFlip(),
#                                     CutoutPIL(cutout_factor=0.5),
#                                     RandAugment(),
#                                     transforms.ToTensor()]

#         test_data_transform_list = [transforms.Resize((cfg.DATA.TRANSFORM.img_size, cfg.DATA.TRANSFORM.img_size)),
#                                     transforms.ToTensor()]
#         train_data_transform_list.append(normalize)
#         test_data_transform_list.append(normalize)

#         train_data_transform = transforms.Compose(train_data_transform_list)
#         test_data_transform = transforms.Compose(test_data_transform_list)
#     else:
#         train_data_transform = CustomDataAugmentation(size=cfg.DATA.TRANSFORM.TWOTYPE.img_size, min_scale=cfg.DATA.TRANSFORM.TWOTYPE.min_scale)
#         test_data_transform = transforms.Compose([transforms.Resize((cfg.DATA.TRANSFORM.img_size, cfg.DATA.TRANSFORM.img_size)),
#                                                   transforms.ToTensor(),
#                                                   transforms.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)                                                  ])


#     logger.info('train_data_transform {}'.format(train_data_transform))
#     logger.info('test_data_transform {}'.format(test_data_transform))


#     if cfg.DATA.dataname == 'coco14' or cfg.DATA.dataname == 'COCO2014':
#         # ! config your data path here.
#         dataset_dir = cfg.DATA.dataset_dir
#         dataset_dir = osp.join(dataset_dir, 'COCO2014')
#         train_dir = osp.join(dataset_dir, 'train2014')

#         train_anno_path = osp.join(dataset_dir, 'annotations/instances_train2014.json')
#         train_label_path = './partdata/coco/train_label_vectors.npy'
        
#         test_dir = osp.join(dataset_dir, 'val2014')
#         test_anno_path = osp.join(dataset_dir, 'annotations/instances_val2014.json')
#         test_label_path = './partdata/coco/val_label_vectors.npy'
        
#         train_dataset = FullCOCO2014dataset(
#             dataset_dir=dataset_dir,
#             mode='train',
#             image_dir=train_dir,
#             anno_path=train_anno_path,
#             labels_path=train_label_path,
#             input_transform=train_data_transform,
#         )
#         val_dataset = FullCOCO2014dataset(
#             dataset_dir=dataset_dir,
#             mode='val',
#             image_dir=test_dir,
#             anno_path=test_anno_path,
#             labels_path=test_label_path,
#             input_transform=test_data_transform
#         )

#     elif cfg.DATA.dataname == 'coco17' or cfg.DATA.dataname == 'COCO17':
#         # ! config your data path here.
#         dataset_dir = cfg.DATA.dataset_dir
#         dataset_dir = osp.join(dataset_dir, 'COCO2017')
#         train_dir = osp.join(dataset_dir, 'train2017')

#         train_anno_path = osp.join(dataset_dir, 'annotations/instances_train2017.json')
#         train_label_path = './partdata/coco/train_label_vectors.npy'
        
#         test_dir = osp.join(dataset_dir, 'val2017')
#         test_anno_path = osp.join(dataset_dir, 'annotations/instances_val2017.json')
#         test_label_path = './partdata/coco/val_label_vectors.npy'
        
#         train_dataset = FullCOCO2017dataset(
#             dataset_dir=dataset_dir,
#             mode='train',
#             image_dir=train_dir,
#             anno_path=train_anno_path,
#             labels_path=train_label_path,
#             input_transform=train_data_transform,
#         )
#         val_dataset = FullCOCO2017dataset(
#             dataset_dir=dataset_dir,
#             mode='val',
#             image_dir=test_dir,
#             anno_path=test_anno_path,
#             labels_path=test_label_path,
#             input_transform=test_data_transform
#         )

#     elif cfg.DATA.dataname == 'nuswide' or cfg.DATA.dataname == 'NUSWIDE':
#         dataset_dir = cfg.DATA.dataset_dir
#         root_dir = osp.join(dataset_dir, 'nuswide')
#         img_dir = osp.join(root_dir, 'Flickr')
#         train_anno_path = osp.join(root_dir, 'ImageList', 'TrainImagelist.txt')
#         train_labels_path = osp.join(root_dir, 'Groundtruth', 'Labels_Train.txt')

#         val_anno_path = osp.join(root_dir, 'ImageList', 'TestImagelist.txt')
#         val_labels_path = osp.join(root_dir, 'Groundtruth', 'Labels_Test.txt')
        
#         train_dataset = FullNusWideDataset(
#             root_dir = root_dir,
#             img_dir = img_dir,
#             anno_path = train_anno_path,
#             labels_path = train_labels_path,
#             mode = 'train',
#             transform = train_data_transform
#         )

#         val_dataset = FullNusWideDataset(
#             root_dir = root_dir,
#             img_dir = img_dir,
#             anno_path = val_anno_path,
#             labels_path = val_labels_path,
#             mode = 'val',
#             transform = test_data_transform
#         )

#     elif cfg.DATA.dataname == 'voc2007' or cfg.DATA.dataname == 'VOC2007':
#         dataset_dir = osp.join(cfg.DATA.dataset_dir, 'VOC2007')
#         dup=None
#         train_dataset = FullVoc07Dataset(dataset_dir=dataset_dir,
#                                 img_dir=osp.join(dataset_dir, 'VOCdevkit/VOC2007/JPEGImages'), 
#                                 anno_path = osp.join(dataset_dir, 'VOCdevkit/VOC2007/ImageSets/Main/trainval.txt'), 
#                                 transform = train_data_transform, 
#                                 labels_path = osp.join(dataset_dir, 'VOCdevkit/VOC2007/Annotations'), 
#                                 mode='trainval',
#                                 dup=None)

#         val_dataset = FullVoc07Dataset(dataset_dir=dataset_dir,
#                                 img_dir=osp.join(dataset_dir, 'VOCdevkit/VOC2007/JPEGImages'), 
#                                 anno_path = osp.join(dataset_dir, 'VOCdevkit/VOC2007/ImageSets/Main/test.txt'), 
#                                 transform = test_data_transform, 
#                                 labels_path = osp.join(dataset_dir, 'VOCdevkit/VOC2007/Annotations'),
#                                 mode='test',
#                                 dup=None)
        
#     elif cfg.DATA.dataname == 'voc2012' or cfg.DATA.dataname == 'VOC2012':
#         dataset_dir = osp.join(cfg.DATA.dataset_dir, 'VOC2012')
#         dup=None
#         train_dataset = FullVoc12Dataset(dataset_dir=dataset_dir,
#                                 img_dir=osp.join(dataset_dir, 'VOCdevkit/VOC2012/JPEGImages'), 
#                                 anno_path = osp.join(dataset_dir, 'VOCdevkit/VOC2012/ImageSets/Main/train.txt'), 
#                                 transform = train_data_transform, 
#                                 labels_path = osp.join(dataset_dir, 'VOCdevkit/VOC2012/Annotations'), 
#                                 mode='train',
#                                 label_proportion=cfg.DATA.prob,
#                                 dup=None)

#         val_dataset = FullVoc07Dataset(dataset_dir=dataset_dir,
#                                   img_dir=osp.join(dataset_dir, 'VOCdevkit/VOC2012/JPEGImages'), 
#                                     anno_path = osp.join(dataset_dir, 'VOCdevkit/VOC2012/ImageSets/Main/val.txt'), 
#                                     transform = test_data_transform, 
#                                     labels_path = osp.join(dataset_dir, 'VOCdevkit/VOC2012/Annotations'),
#                                     mode='val',
#                                     label_proportion=1.0,
#                                     dup=None)
#     elif cfg.DATA.dataname == 'vg500' or cfg.DATA.dataname == 'VG500':
#         # dataset_dir = osp.join(cfg.DATA.dataset_dir, 'VG')
#         dataset_dir = cfg.DATA.dataset_dir
#         vg_root = osp.join(dataset_dir, 'VG')
#         train_dir = osp.join(vg_root,'VG_100K')
#         train_list = osp.join(vg_root,'train_list_500.txt')
#         test_dir = osp.join(vg_root,'VG_100K')
#         test_list = osp.join(vg_root,'test_list_500.txt')
#         train_label = osp.join(vg_root,'vg_category_500_labels_index.json')
#         test_label = osp.join(vg_root,'vg_category_500_labels_index.json')

#         train_mode = 'train'
#         val_mode = 'val'

#         train_dataset = FullVGDataset(dataset_dir=vg_root, 
#                                   img_dir = train_dir, 
#                                   img_list = train_list, 
#                                   label_path = train_label, 
#                                   image_transform = train_data_transform, 
#                                   mode = train_mode)
#         val_dataset = FullVGDataset(dataset_dir = vg_root, 
#                                 img_dir = test_dir, 
#                                 img_list = test_list, 
#                                 label_path = test_label, 
#                                 image_transform = test_data_transform, 
#                                 mode = val_mode)
        
#     elif cfg.DATA.dataname == 'cub200' or cfg.DATA.dataname == 'CUB200':

#         dataset_dir = cfg.DATA.dataset_dir
#         dataset_dir = osp.join(dataset_dir, 'CUB_200_2011')
#         img_dir = osp.join(dataset_dir, 'CUB_200_2011', 'images')

#         train_dataset = FullCUB200Dataset(dataset_dir=dataset_dir, 
#                                           img_dir=img_dir, 
#                                           image_transform=train_data_transform, 
#                                           mode='train')

#         val_dataset = FullCUB200Dataset(dataset_dir=dataset_dir, 
#                                         img_dir=img_dir, 
#                                         image_transform=test_data_transform, 
#                                         mode='val')

#     else:
#         raise NotImplementedError("Unknown dataname %s" % cfg.DATA.dataname)

#     print("len(train_dataset):", len(train_dataset)) 
#     print("len(val_dataset):", len(val_dataset))
#     return train_dataset, val_dataset
