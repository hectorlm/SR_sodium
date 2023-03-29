import os
from glob import glob
# import natsort
import json
from collections import Counter
from PIL import Image
import re
import random
from pathlib import Path
from scipy.io import loadmat

from dataloader.utils import *
# from lib.utils import *

# get_imaged_names to my mat file folder
def get_image_names(params,mode='train'):
    # with open(PATH_PARAMETERS) as f:
    #     params = json.load(f)
    # params = params['models_settings']
    if mode == 'train':
        data_dir = params['train_dir']
    if mode == 'test':
        data_dir = params['test_dir']

    file_names = []
    data_path = Path(data_dir)
    # for filename in natsort.natsorted(glob(os.path.join(data_dir, params['image_folder'], '*'))):
    for filename in data_path.glob('*.npz'):
        file_names.append(filename.name)
    return file_names


def load_images(image_names, mode='train'):
    with open(PATH_PARAMETERS) as f:
        params = json.load(f)
    params = params['models_settings']

    if mode == 'train':
        data_dir = params['train_dir']
    if mode == 'test':
        data_dir = params['test_dir']

    resize_w = params['resize_width']
    equalize = bool(params['equalize'])

    images = []
    for image_name in image_names:

        file_name = os.path.join(data_dir, params['image_folder'], image_name)
        image = Image.open(file_name)

        if resize_w is not None:
            orig_w, orig_h = image.size[:2]
            resize_h = int(resize_w / orig_w * orig_h)
            image = np.array(image.resize((resize_w, resize_h), Image.BILINEAR))

        images.append(image)

    return images

def list_labels(file_names, mode='train'):

    with open(PATH_PARAMETERS) as f:
        params = json.load(f)
    params = params['models_settings']

    masks = load_masks(file_names,mode=mode)
    shape_to_label = params['label_to_value']
    label_to_shape = {v:k for k,v in shape_to_label.items()}
    
    labels = set()
    for mask in masks:  
        # mask = rgb2mask(mask)
        labels = labels.union(set([label_to_shape[label] for label in np.unique(mask)]))
        
    return labels

def get_sizes(image_names, mode='train'):

    with open(PATH_PARAMETERS) as f:
        params = json.load(f)
    params = params['models_settings']

    if mode=='train':
        data_dir = params['train_dir']
    if mode=='test':
        data_dir = params['test_dir']

    h = []
    w = []
    
    for image_name in image_names:

        file_name = os.path.join(data_dir, params['image_folder'],  image_name)
        image = np.array(Image.open(file_name))
        
        h.append(image.shape[0])
        w.append(image.shape[1])
        
    d = {'h-range': [min(h), max(h)],
         'w-range': [min(w), max(w)]}
    
    return d

def load_masks(image_names, mode='train'):
    
    with open(PATH_PARAMETERS) as f:
        params = json.load(f)
    params = params['models_settings']

    if mode=='train':
        data_dir = params['train_dir']
    if mode=='test':
        data_dir = params['test_dir']

    resize_w = params['resize_width']

    masks = []
    for image_name in image_names:  
        image_name = re.sub("Img", "Mask", image_name)
        file_name = os.path.join(data_dir, params['mask_folder'],  image_name)
        print(file_name)
        mask = Image.open(file_name)
        if resize_w is not None:
            orig_w, orig_h = mask.size[:2]
            resize_h = int(resize_w/orig_w*orig_h)
            mask = mask.resize((resize_w,resize_h), Image.NEAREST)

        masks.append(np.array(mask))
        
    return masks


## define for my mat files
def dataset_detail(val_ratio):
    with open(PATH_PARAMETERS) as f:
        params = json.load(f)
    params = params['models_settings']
    file_names = get_image_names(params)

    print('length(file_names)')
    print(len(file_names))

    # labels_list = {key for key in params['label_to_value']}
    length_traindata = len(file_names)
    # train_path = os.path.join(params['train_dir'], params['image_folder'])
    train_path = params['train_dir']
    random.shuffle(file_names)
    max_train_index = round(length_traindata * (1 - val_ratio))  # 90% for train

    file_names_train = file_names[0:max_train_index]
    file_names_val = file_names[max_train_index::]

    dataset_detail ={'params':params,
                     'train_path':train_path,
                     'file_names':file_names,
                     'file_names_train':file_names_train,
                     'file_names_val':file_names_val,
                     'length_dataset':length_traindata,
                     'train_num':max_train_index }
    return dataset_detail
