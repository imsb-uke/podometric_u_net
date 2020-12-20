import numpy as np
from skimage.transform import resize, rescale

from network.dataset.image_augmentation import data_augmentation
from network.dataset.image_transformations import contrast_stretch_image, histo_equalize_image

# Functions to preprocess image
from network.rescale_array import rescale_array


def paste_slices(tup):
    pos, w, max_w = tup
    wall_min = max(pos, 0)
    wall_max = min(pos + w, max_w)
    block_min = -min(pos, 0)
    block_max = max_w - max(pos + w, max_w)
    block_max = block_max if block_max != 0 else None
    return slice(wall_min, wall_max), slice(block_min, block_max)


def paste(wall, block, loc):
    loc_zip = zip(loc, block.shape, wall.shape)
    wall_slices, block_slices = zip(*map(paste_slices, loc_zip))
    wall[wall_slices] += block[block_slices]


def pad_image(image, target_img_shape):
    new_image = np.zeros((target_img_shape[0], target_img_shape[1], target_img_shape[2]))
    x_pad = int((target_img_shape[0] - image.shape[0]) / 2)
    y_pad = int((target_img_shape[1] - image.shape[1]) / 2)
    paste(new_image, image, (x_pad, y_pad))
    image = new_image
    return image


def norm_image(image, rescale_factor):
    if rescale_factor == "min_max":
        # Norm image values to 0 to 1
        smooth = 1.
        image = (image-np.min(image) + smooth)/(np.max(image)+np.min(image) + smooth)
    else:
        image = image / rescale_factor
    return image


def norm_mask(mask, rescale_factor):
    mask = mask / rescale_factor
    return mask


def resize_image(image, target_img_shape, mask=None, weights=None, verbose=False):
    if verbose:
        print(image.shape, (mask.shape if mask is not None else ', no mask'),
              (weights.shape if weights is not None else ', no weights'))
    if mask is not None:
        if image.shape[0] != mask.shape[0] or image.shape[1] != mask.shape[1]:
            raise ValueError('Image and mask do not have the same shape.')

    # resize
    # case 1: at least one axis of the image is larger than the target size
    # or case 2: both axes of the image are smaller than the target size
    if image.shape[0] > target_img_shape[0] or image.shape[1] > target_img_shape[1] or (
            image.shape[0] < target_img_shape[0] and image.shape[1] < target_img_shape[1]):
        # Find the largest image axis
        if image.shape[0] > image.shape[1]:
            # Scale the image down by a factor s
            s = float(target_img_shape[0] / image.shape[0])
        else:
            # Scale the image down by a factor s
            s = float(target_img_shape[1] / image.shape[1])
        image = rescale(image, s, mode='constant', cval=0, anti_aliasing=True, preserve_range=True, multichannel=True)

        if mask is not None:
            if len(mask.shape) == 4:
                mask = mask[:, :, :, 0]
                mask = rescale(mask, s, mode='constant', cval=0, anti_aliasing=True, preserve_range=True,
                               multichannel=True)
            elif len(mask.shape) == 3:
                mask = rescale(mask, s, mode='constant', cval=0, anti_aliasing=True, preserve_range=True,
                               multichannel=True)
            elif len(mask.shape) == 2:
                mask = rescale(mask, s, mode='constant', cval=0, anti_aliasing=True, preserve_range=True,
                               multichannel=False)
            else:
                raise ValueError("len(mask.shape) is not equal 2, 3 or 4. Check your mask")

        if weights is not None:
            if len(weights.shape) == 3:
                weights = rescale_array(weights, s, mode='constant', cval=0, anti_aliasing=True, multichannel=True)
            elif len(weights.shape) == 2:
                weights = rescale_array(weights, s, mode='constant', cval=0, anti_aliasing=True, multichannel=False)
            else:
                raise ValueError("len(weights.shape) is not equal 2 or 3. Check your weights")
        if verbose:
            print("scale factor: ", s)
            print("after scaling:", image.shape, (mask.shape if mask is not None else ', no mask'),
                  (weights.shape if weights is not None else ', no weights'))

    # case 3: image is equal to target size.
    #         Nothing to do

    # Now all the images have at least one axis in the target_img size
    # If other axis differ from target_img size this has to be adjusted:
    # Fill the border regions with 0 and put the image to the middle
    if image.shape[0] < target_img_shape[0] or image.shape[1] < target_img_shape[1]:
        image = pad_image(image, target_img_shape)
        if mask is not None:
            mask = pad_image(mask, target_img_shape)
        if weights is not None:
            weights = pad_image(weights, target_img_shape)

    if verbose:
        print("resized to:", image.shape, (mask.shape if mask is not None else ', no mask'),
              (weights.shape if weights is not None else ', no weights'))
    return image, mask, weights


def preprocess_image(image, imagename, mask, maskname, target_img_shape, number_msk_channels, rescale_factor,
                     contrast_stretch, histogram_equalization, do_data_augm, **data_augm_args):
    # resize
    if mask is not None:
        image, mask, _ = resize_image(image, target_img_shape, mask)
    else:
        image, _, _ = resize_image(image, target_img_shape)

    # normalise image and mask between 0 and 1
    norm_image(image, rescale_factor)
    if mask is not None:
        norm_mask(mask, 255)

    # contrast stretching
    if contrast_stretch:
        image = contrast_stretch_image(image)

    # histogram equalization
    if histogram_equalization:
        image = histo_equalize_image(image)

    # data_augm
    if do_data_augm:
        if mask is not None:
            image, mask = data_augmentation(image, imagename, mask, maskname, **data_augm_args)
        else:
            raise ValueError("The function data_augmentation has not been implemented for the unsupervised task."
                             "Do you need it?")
    return image, mask
