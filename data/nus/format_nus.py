import json
import os
import os.path as osp
import argparse
import numpy as np

pp = argparse.ArgumentParser(description='Format nuswide metadata.')
pp.add_argument('--load-path', type=str, default='.', help='Path to a directory containing a copy of the COCO dataset.')
pp.add_argument('--save-path', type=str, default='.', help='Path to output directory.')
args = pp.parse_args()



for split in ['Train', 'Test']:
    
    labels_path = osp.join(args.load_path, 'Groundtruth', 'Labels_{}.txt'.format(split))
    anno_path = osp.join(args.load_path, 'ImageList', '{}Imagelist.txt'.format(split))
    # '/media/data2/MLICdataset/nuswide/ImageList/TrainImagelist.txt',
    img_dir = osp.join(args.load_path, 'Flickr')

    imgnamelist = [line.strip().replace('\\', '/') for line in open(anno_path, 'r')]
    labellist = [line.strip() for line in open(labels_path, 'r')]
    assert len(imgnamelist) == len(labellist)
        
    res = []
    imgname_list = []
    label_list = []

    for idx, (imgname, labelline) in enumerate(zip(imgnamelist, labellist)):
        imgpath = osp.join(img_dir, imgname)
        labels = [int(i) for i in labelline.split(' ')]
        labels = np.array(labels).astype(np.float32)
        if sum(labels) == 0:
            continue

        # imgname_list.append(imgpath)
        imgname_list.append(imgname)
        label_list.append(labels)

    # print(imgname_list[0])
    # print(label_list[0])

    if split == 'Train':
        split = 'train'
    elif split == 'Test':
        split = 'val'
    np.save(os.path.join(args.save_path, 'formatted_' + split + '_labels.npy'), np.array(label_list))
    np.save(os.path.join(args.save_path, 'formatted_' + split + '_images.npy'), np.array(imgname_list))
