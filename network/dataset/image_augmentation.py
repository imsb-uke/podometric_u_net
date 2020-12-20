import os
import random
import time

import numpy as np
from skimage.io import imsave, imshow, show

from network.dataset.image_transformations import shifting, flipping, rotation, zoom, signal_reduction


def data_augmentation(image, imagename, mask, maskname, verbose, width_shift_range, height_shift_range,
                      flip_horizontal, flip_vertical, rotation_range,
                      zooming_range, c0_signal_min, c0_signal_max,
                      c1_signal_min, c1_signal_max, c2_signal_min,
                      c2_signal_max, c2_reduce_podo_signal, augm_data_path, save_augm_data, plot_augm_data):
    # Get Arguments for the data augmentation functions
    x_dim, y_dim = image.shape[0], image.shape[1]

    image_orig = image
    mask_orig = mask

    # Do shifting if
    if width_shift_range or height_shift_range:
        shift_range_x = int(width_shift_range * x_dim)
        shift_range_y = int(height_shift_range * y_dim)
        x_shift = random.randint(-shift_range_x, shift_range_x + 1)
        y_shift = random.randint(-shift_range_y, shift_range_y + 1)
        if verbose:
            print("Data augmentation: Shifting by x: ", x_shift, " and y: ", y_shift)
        image = shifting(image, x_shift, y_shift)
        mask = shifting(mask, x_shift, y_shift)

    # Do flipping if
    if flip_horizontal:
        if random.randint(0, 1):
            if verbose:
                print("Data augmentation: horizontal (=left right) flip")
            image = flipping(image, flip_horizontal, 0)
            mask = flipping(mask, flip_horizontal, 0)
    if flip_vertical:
        if random.randint(0, 1):
            if verbose:
                print("Data augmentation: verical (=top bottom) flip")
            image = flipping(image, 0, flip_vertical)
            mask = flipping(mask, 0, flip_vertical)

    # Do rotation if
    if rotation_range:
        rotation_degree = (random.random() * 2 * rotation_range) - rotation_range
        if verbose:
            print("Data augmentation: Rotation by: ", rotation_degree)
        image = rotation(image, rotation_degree)
        mask = rotation(mask, rotation_degree)

    # Do zoom if
    if zooming_range:
        zooming_factor = (random.random() * 2 * zooming_range) - zooming_range + 1
        if verbose:
            print("Data augmentation: Zoom by: ", zooming_factor)
        image = zoom(image, x_dim, y_dim, zooming_factor)
        mask = zoom(mask, x_dim, y_dim, zooming_factor)

    # Do signal reduction if
    if c0_signal_min < 1:
        if c0_signal_min != c0_signal_max:
            signal_reduction_factor = (random.random() + (c0_signal_min / (c0_signal_max - c0_signal_min))) * (
                    c0_signal_max - c0_signal_min)
        else:
            signal_reduction_factor = c0_signal_min
        if verbose:
            print("Data augmentation: Signal reduction by: ", signal_reduction_factor)
        image = signal_reduction(image, 0, signal_reduction_factor)

    if c1_signal_min < 1:
        if c1_signal_min != c1_signal_max:
            signal_reduction_factor = (random.random() + (c1_signal_min / (c1_signal_max - c1_signal_min))) * (
                    c1_signal_max - c1_signal_min)
        else:
            signal_reduction_factor = c1_signal_min
        if verbose:
            print("Data augmentation: Signal reduction by: ", signal_reduction_factor)
        image = signal_reduction(image, 1, signal_reduction_factor)

    if c2_signal_min < 1:
        if c2_signal_min != c2_signal_max:
            signal_reduction_factor = (random.random() + (c2_signal_min / (c2_signal_max - c2_signal_min))) * (
                    c2_signal_max - c2_signal_min)
        else:
            signal_reduction_factor = c2_signal_min
        if verbose:
            print("Data augmentation: Signal reduction by: ", signal_reduction_factor)
        image = signal_reduction(image, 2, signal_reduction_factor)

    if c2_reduce_podo_signal:
        image_podo = image[:, :, 2]
        if len(mask.shape) == 2:
            image_podo[(image_podo * (mask)) > 0.5] = image_podo[(image_podo * (mask)) > 0.5] * 0.75
        else:
            image_podo[(image_podo * (mask[:, :, 1])) > 0.5] = image_podo[(image_podo * (mask[:, :, 1])) > 0.5] * 0.75
        image[:, :, 2] = image_podo

    # Save augmented images if
    if save_augm_data:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        image_out = (255 * image).astype(np.uint8)
        imsave(os.path.join(os.path.join(augm_data_path, "images"), "augm_" + timestr + "_" + imagename), image_out)
        for i in range(mask.shape[2]):
            mask_out = (255 * mask[:, :, i]).astype(np.uint8)
            imsave(os.path.join(os.path.join(augm_data_path, "masks"), "augm_" + timestr + "_" + maskname[i]), mask_out)

    # Plot augmented images if
    if plot_augm_data:
        # Image before augmentation
        imshow(image_orig)
        show()
        # Image after augmentation
        imshow(image)
        show()

    # Future extentions:
    # Do elastic deformation if this is necessary

    return image, mask