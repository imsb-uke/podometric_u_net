import numpy as np
from skimage.exposure import rescale_intensity, equalize_hist
from skimage.transform import SimilarityTransform, warp, rotate, rescale, resize


def contrast_stretch_image(img, perc=99.8):
    # Apply it on each channel
    for channel in range(0, img.shape[2]):
        p_lower, p_upper = np.percentile(img[:, :, channel], (100 - perc, perc))
        img[:, :, channel] = rescale_intensity(img[:, :, channel], in_range=(p_lower, p_upper))
    return img


def histo_equalize_image(img):
    for channel in range(0, img.shape[2]):
        img[:, :, channel] = equalize_hist(img[:, :, channel])
    return img


def shifting(image, x_shift, y_shift):
    tform = SimilarityTransform(translation=(x_shift, y_shift))
    shifted_image = warp(image, tform, mode='constant', cval=0)
    return shifted_image


def flipping(image, flip_horizontal, flip_vertical):
    # Do horizontal and/or vertical flipping
    if flip_horizontal:
        image = np.flip(image, axis=1)
    if flip_vertical:
        image = np.flip(image, axis=0)
    return image


def rotation(image, rotation_degree):
    image = rotate(image, rotation_degree)
    return image


def zoom(image, x_dim, y_dim, zooming_factor):
    if len(image.shape) == 3:
        rescaled_image = rescale(image, zooming_factor, mode='reflect', anti_aliasing=True, multichannel=True)
    else:
        rescaled_image = rescale(image, zooming_factor, mode='reflect', anti_aliasing=True, multichannel=False)
    if zooming_factor > 1:
        left = round((rescaled_image.shape[0] - x_dim) / 2)
        right = left + x_dim
        upper = round((rescaled_image.shape[1] - y_dim) / 2)
        lower = upper + y_dim
        cropped_image = rescaled_image[upper:lower, left:right]
    else:
        left = round((x_dim - rescaled_image.shape[0]) / 2)
        right = left + rescaled_image.shape[0]
        upper = round((y_dim - rescaled_image.shape[1]) / 2)
        lower = upper + rescaled_image.shape[1]
        cropped_image = np.zeros(image.shape)
        if len(image.shape) == 2:
            cropped_image[upper:lower, left:right] = rescaled_image
        else:
            cropped_image[upper:lower, left:right, :] = rescaled_image
    return cropped_image


def zoom_resize(image, x_dim, y_dim, zooming_factor):
    if zooming_factor > 1:
        resized_image = resize(image, (round(zooming_factor * x_dim), round(zooming_factor * y_dim)),
                               anti_aliasing=True, preserve_range=True)
        # print(round(zooming_factor * x_dim))
        # print(round(zooming_factor * y_dim))
        left = round((round(zooming_factor * x_dim) - x_dim) / 2)
        upper = round((round(zooming_factor * y_dim) - y_dim) / 2)
        right = left + x_dim
        lower = upper + y_dim
        cropped_image = resized_image[upper:lower, left:right]
    else:
        resized_image = resize(image, (round(zooming_factor * x_dim), round(zooming_factor * y_dim)),
                               anti_aliasing=True, preserve_range=True)
        # print(round(zooming_factor * x_dim))
        # print(round(zooming_factor * y_dim))
        left = round((x_dim - round(zooming_factor * x_dim)) / 2)
        upper = round((y_dim - round(zooming_factor * y_dim)) / 2)
        right = left + round(zooming_factor * x_dim)
        lower = upper + round(zooming_factor * y_dim)
        # print(upper, lower, left, right)
        cropped_image = np.zeros(image.shape)
        if len(image.shape) == 2:
            cropped_image[upper:lower, left:right] = resized_image
        else:
            cropped_image[upper:lower, left:right, :] = resized_image
    return cropped_image


def signal_reduction(image, channel, signal_reduction_factor):
    if channel == 0:
        image[:, :, 0] = signal_reduction_factor * image[:, :, 0]
    if channel == 1:
        image[:, :, 1] = signal_reduction_factor * image[:, :, 1]
    if channel == 2:
        image[:, :, 2] = signal_reduction_factor * image[:, :, 2]
    return image