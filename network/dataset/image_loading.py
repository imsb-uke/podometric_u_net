import os

import numpy as np
from skimage.io import imread


def get_file_count(paths, image_format='.tif'):
    total_count = 0
    for path in paths:
        try:
            path_list = [_ for _ in os.listdir(path) if _.endswith(image_format)]
            total_count += len(path_list)
        except OSError:
            print("Directory does not exist. Returned file count for this path will be 0")
    return total_count


# Function to load image
def load_image(img_path):
    img = imread(img_path)
    if img.shape[2] == 4:
        img = img[:, :, :-1]
    # img = np.roll(img, shift=1, axis=2)  # CHECK IMAGE FORMAT
    return img


# Function to load mask
def load_mask(mask_path):
    mask = imread(mask_path)
    return mask


def load_mask_from_img(cfg, img_path, img_name, suffixes):
    a_mask = imread(os.path.join(img_path, img_name + suffixes[0]))
    msk = np.zeros((a_mask.shape[0], a_mask.shape[1], len(suffixes) * cfg.NUMBER_MSK_CHANNELS))
    i = 0
    for suffix in suffixes:
        msk_channel = imread(os.path.join(img_path, img_name + suffix))
        if len(msk_channel.shape) == 2:
            msk_channel = np.expand_dims(msk_channel, axis=-1)
        if len(msk_channel.shape) != 3:
            raise ValueError("Mask must be 3-dim here. Does your mask have 1 or more than 3 dimensions? "
                             "Check the masks.")
        msk[:, :, i:i+cfg.NUMBER_MSK_CHANNELS] = msk_channel
        i += cfg.NUMBER_MSK_CHANNELS
    # print(msk, msk.shape)
    return msk


def load_weights(cfg, img_path, img_name, weight_suffixes):
    a_weights = np.load(os.path.join(img_path, img_name + weight_suffixes[0]))
    weights = np.zeros((a_weights.shape[0], a_weights.shape[1], len(weight_suffixes) * cfg.NUMBER_MSK_CHANNELS))
    i = 0
    for suffix in weight_suffixes:
        weights_channel = np.load(os.path.join(img_path, img_name + suffix))
        if len(weights_channel.shape) == 2:
            weights_channel = np.expand_dims(weights_channel, axis=-1)
        if len(weights_channel.shape) != 3:
            raise ValueError("Weights must be 3-dim here. Has your weights 1 or more than 3 dimensions? Check the weights.")
        weights[:, :, i:i+cfg.NUMBER_MSK_CHANNELS] = weights_channel
        i += cfg.NUMBER_MSK_CHANNELS
    return weights