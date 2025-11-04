import json
import os
import numpy as np
import argparse

pp = argparse.ArgumentParser(description='Format objects365 metadata.')
pp.add_argument('--load-path', type=str, default='.', help='Path to a directory containing a copy of the VG256 dataset.')
pp.add_argument('--save-path', type=str, default='.', help='Path to output directory.')
args = pp.parse_args()

object365_classes = ['scale', 'tape', 'chicken', 'hurdle', 'game board', 'baozi', 'target', 'plants pot', 'toothbrush', 'projector', 'cheese', 'candy', 'durian', 'dumbbell', 'gas stove', 'lion', 'french fries', 'bench', 'power outlet', 'faucet', 'storage box', 'crab', 'helicopter', 'chainsaw', 'antelope', 'hamimelon', 'jellyfish', 'kettle', 'marker', 'clutch', 'lettuce', 'toilet', 'oven', 'baseball', 'drum', 'hanger', 'toaster', 'bracelet', 'cherry', 'tissue ', 'watermelon', 'basketball', 'cleaning products', 'tent', 'fire hydrant', 'truck', 'rice cooker', 'microscope', 'tablet', 'stuffed animal', 'golf ball', 'CD', 'eggplant', 'bowl', 'desk', 'eagle', 'slippers', 'horn', 'carpet', 'notepaper', 'peach', 'saw', 'surfboard', 'facial cleanser', 'corn', 'folder', 'violin', 'watch', 'glasses', 'shampoo', 'pizza', 'asparagus', 'mushroom', 'steak', 'suitcase', 'table tennis  paddle', 'mango', 'boots', 'necklace', 'noodles', 'volleyball', 'baseball bat', 'nuts', 'stroller', 'pumpkin', 'strawberry', 'pear', 'luggage', 'sandals', 'liquid soap', 'handbag', 'flashlight', 'trombone', 'remote', 'shovel', 'ladder', 'cake', 'pomegranate', 'clock', 'vent', 'cymbal', 'iron', 'okra', 'pasta', 'lantern', 'broom', 'fire extinguisher', 'snowboard', 'rice', 'swing', 'cow', 'van', 'tuba', 'book', 'swan', 'lamp', 'race car', 'egg', 'avocado', 'guitar', 'radio', 'sneakers', 'eraser', 'measuring cup', 'sushi', 'deer', 'parrot', 'scissors', 'balloon', 'tortoise', 'meat balls', 'cat', 'electric drill', 'comb', 'sausage', 'bar soap', 'hamburger', 'pepper', 'router', 'spring rolls', 'american football', 'egg tart', 'tape measure', 'banana', 'gun', 'billiards', 'picture', 'paper towel', 'bus', 'goldfish', 'computer box', 'potted plant', 'ship', 'ambulance', 'dog', 'medal', 'butterfly', 'hair dryer', 'globe', 'french horn', 'board eraser', 'tea pot', 'telephone', 'mop', 'broccoli', 'dolphin', 'chair', 'hat', 'tripod', 'traffic light', 'hot dog', 'pot', 'car', 'dining table', 'crosswalk sign', 'tomato', 'barrel', 'washing machine', 'polar bear', 'tie', 'monkey', 'green beans', 'cucumber', 'cookies', 'suv', 'brush', 'carrot', 'tennis racket', 'helmet', 'sink', 'stool', 'flower', 'radiator', 'fishing rod', 'Life saver', 'lighter', 'bread', 'radish', 'human', 'traffic cone', 'knife', 'grapes', 'cellphone', 'trophy', 'urinal', 'cup', 'paint brush', 'mouse', 'soccer', 'cutting', 'wheelchair', 'Accordion', 'goose', 'red cabbage', 'plate', 'saxophone', 'laptop', 'facial mask', 'onion', 'motorbike', 'canned', 'lobster', 'toiletries', 'earphone', 'flag', 'Bread', 'trumpet', 'parking meter', 'garlic', 'skateboard', 'pie', 'barbell', 'yak', 'stapler', 'tangerine', 'zebra', 'traffic sign', 'bottle', 'hotair balloon', 'sailboat', 'llama', 'blackboard', 'coffee machine', 'flute', 'pencil case', 'ice cream', 'combine with bowl', 'kite', 'microphone', 'fork', 'hoverboard', 'blender', 'skating and skiing shoes', 'nightstand', 'toothpaste', 'poker card', 'fan', 'orange', 'chopsticks', 'pig', 'bathtub', 'glove', 'golf club', 'refrigerator', 'rickshaw', 'candle', 'mirror', 'microwave', 'converter', 'airplane', 'lemon', 'head phone', 'tricycle', 'bear', 'backpack', 'apple', 'trolley', 'tong', 'papaya', 'cello', 'camel', 'binoculars', 'cabbage', 'umbrella', 'cigar', 'pomelo', 'cabinet', 'keyboard', 'horse', 'duck', 'combine with glove', 'pine apple', 'potato', 'air conditioner', 'pliers', 'fire truck', 'hockey stick', 'elephant', 'sports car', 'toy', 'mangosteen', 'rabbit', 'bicycle', 'giraffe', 'screwdriver', 'spoon', 'sheep', 'key', 'wine glass', 'treadmill', 'extension cord', 'shrimp', 'ring', 'boat', 'green vegetables', 'coffee table', 'pitaya', 'shark', 'basket', 'wild bird', 'carriage', 'slide', 'fish', 'frisbee', 'hammer', 'printer', 'plum', 'towel', 'camera', 'speaker', 'pickup truck', 'high heels', 'bow tie', 'pigeon', 'coconut', 'machinery vehicle', 'sofa', 'bed', 'tennis ball', 'dates', 'street lights', 'paddle', 'calculator', 'starfish', 'chips', 'train', 'kiwi fruit', 'belt', 'monitor', 'skis', 'leather shoes', 'sandwich', 'Electronic stove and gas stove', 'penguin', 'surveillance camera', 'cue', 'scallop', 'green onion', 'seal', 'crane', 'donkey', 'pen', 'donut', 'pillow', 'trash bin']


# objv2_ignore_list = [
#     # images exist in annotations but not in image folder.
#     'patch16/objects365_v2_00908726.jpg',
#     'patch6/objects365_v1_00320532.jpg',
#     'patch6/objects365_v1_00320534.jpg',
# ]

def parse_categories(categories):
    category_list = []
    id_to_index = {}
    for i in range(len(categories)):
        category_list.append(categories[i]['name'])
        id_to_index[categories[i]['id']] = i
    print(category_list)
    return (category_list, id_to_index)

# initialize metadata dictionary:
meta = {}
meta['category_id_to_index'] = {}
meta['category_list'] = []


for split in ['test', 'train']:
    
    if split == 'train':
        with open(os.path.join(args.load_path, 'Annotations', 'zhiyuan_objv2_train.json'), 'r') as f:
            D = json.load(f)
    elif split == 'test':
        with open(os.path.join(args.load_path, 'Annotations', 'zhiyuan_objv2_test.json'), 'r') as f:
            D = json.load(f)
    elif split == 'val':
        with open(os.path.join(args.load_path, 'Annotations', 'sample_2020.json'), 'r') as f:
            D = json.load(f)
    else:
        raise ValueError('Invalid split.')
    print('Read Successfully.')
    
    
    if len(meta['category_list']) == 0:
        # parse the category data:
        (meta['category_list'], meta['category_id_to_index']) = parse_categories(D['categories'])
    else:
        # check that category lists are consistent for train2014 and val2014:
        (category_list, id_to_index) = parse_categories(D['categories'])
        assert category_list == meta['category_list']
        assert id_to_index == meta['category_id_to_index']

    image_id_list = sorted(np.unique([str(D['annotations'][i]['image_id']) for i in range(len(D['annotations']))]))
    image_id_list = np.array(image_id_list, dtype=int)
    # sorting as strings for backwards compatibility 
    image_id_to_index = {image_id_list[i]: i for i in range(len(image_id_list))}
    
    num_categories = len(D['categories'])
    num_images = len(image_id_list)
    
    label_matrix = np.zeros((num_images,num_categories))
    image_ids = np.zeros(num_images)
    
    for i in range(len(D['annotations'])):
        
        image_id = int(D['annotations'][i]['image_id'])

        if image_id in [320532, 320534, 908726]:
            continue

        row_index = image_id_to_index[image_id]
    
        category_id = int(D['annotations'][i]['category_id'])
        category_index = int(meta['category_id_to_index'][category_id])
        
        label_matrix[row_index][category_index] = 1
        image_ids[row_index] = int(image_id)
    

    count = 0
    image_ids_list = []
    for imgid in image_ids:
        imgid = int(imgid)
        img_path_v1 = f"Images/" + split + f"/objects365_v1_{str(imgid).zfill(8)}.jpg"
        img_path_v2 = f"Images/" + split + f"/objects365_v2_{str(imgid).zfill(8)}.jpg"
        if os.path.exists(img_path_v1) and os.path.exists(img_path_v2):
            # raise FileNotFoundError(f"Image {imgid} found in both v1 and v2.")
            print(f"Image {imgid} found in both v1 and v2.")
        elif os.path.exists(img_path_v1):
            img_path = img_path_v1
        elif os.path.exists(img_path_v2):
            img_path = img_path_v2
        else:
            count += 1
            print(f"Image {imgid} not found.")
            # raise FileNotFoundError(f"Image {imgid} not found.")
        image_ids_list.append(img_path)
    
    print(count, " images not found.")

    # image_ids = np.array(['{}2014/COCO_{}2014_{}.jpg'.format(split, split, str(int(x)).zfill(12)) for x in image_ids])
    image_ids = np.array(image_ids_list)
    
    if split == 'test':
        split = 'val'
    # save labels and corresponding image ids: 
    np.save(os.path.join(args.save_path, 'formatted_' + split + '_labels.npy'), label_matrix)
    np.save(os.path.join(args.save_path, 'formatted_' + split + '_images.npy'), image_ids)
