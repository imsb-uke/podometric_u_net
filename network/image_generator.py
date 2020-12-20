import os
import numpy as np
import random
from skimage.io import imread, imsave, imshow, show
from skimage.transform import resize, rescale, rotate, warp, SimilarityTransform
from skimage.exposure import rescale_intensity, equalize_hist
from skimage.util import crop
import time

from network.dataset.image_loading import load_image, load_mask
from network.dataset.image_preprocessing import preprocess_image


# def data_preprocessing(image, imagename, mask, maskname, target_img_shape, number_msk_channels, rescale,
#                        contrast_stretch, histogram_equalization, do_data_augm, **data_augm_args):
#     # resize
#     # Resize the image if the image axis are larger than target_img_shape[0], ...[1]
#     if image.shape[0] == image.shape[1] and image.shape[0] > target_img_shape[0]:
#         # Scale the image down by a factor s
#         s = float(target_img_shape[0] / image.shape[0])
#         print("scale factor: ", s)
#         image = resize(image, (target_img_shape[0], target_img_shape[1]), anti_aliasing=True, preserve_range=True)
#     elif image.shape[0] > target_img_shape[0] or image.shape[1] > target_img_shape[1]:
#         # Find the larger axis. Find the factor s to scale to x_dim.
#         # Apply it for both dimensions
#         if image.shape[0] > image.shape[1]:
#             s = float(target_img_shape[0] / image.shape[0])
#             image = resize(image, (target_img_shape[0], round(s * image.shape[1])), preserve_range=True)
#         else:
#             s = float(target_img_shape[1] / image.shape[1])
#             print("scale factor: ", s)
#             image = resize(image, (round(s * image.shape[0]), target_img_shape[1]), preserve_range=True)
#
#     # same thing for masks
#     if mask.shape[0] == mask.shape[1] and mask.shape[0] > target_img_shape[0]:
#         s = float(target_img_shape[0] / mask.shape[0])
#         if len(mask.shape) == 4:
#             mask = mask[:, :, :, 0]
#         mask = resize(mask, (target_img_shape[0], target_img_shape[1], mask.shape[2]), anti_aliasing=True, preserve_range=True)
#     elif mask.shape[0] > target_img_shape[0] or mask.shape[1] > target_img_shape[1]:
#         if mask.shape[0] > mask.shape[1]:
#             s = float(target_img_shape[0] / mask.shape[0])
#             if len(mask.shape) == 4:
#                 mask = mask[:, :, :, 0]
#             mask = resize(mask, (target_img_shape[0], round(s * mask.shape[1]), mask.shape[2]), preserve_range=True)
#         else:
#             s = float(target_img_shape[1] / mask.shape[1])
#             # BUG: ValueError: len(output_shape) cannot be smaller than the image dimensions
#             # 1021, 1024 vs (2029, 2034, 2, 3)
#             if len(mask.shape) == 4:
#                 mask = mask[:, :, :, 0]
#             mask = resize(mask, (round(s * mask.shape[0]), target_img_shape[1], mask.shape[2]), preserve_range=True)
#     elif len(mask.shape) == 4:
#         mask = mask[:, :, :, 0]
#
#
#     # Fill the border regions with 0 and put the image to the middle
#     if image.shape[0] < target_img_shape[0] or image.shape[1] < target_img_shape[1]:
#         new_image = np.zeros((target_img_shape[0], target_img_shape[1], target_img_shape[2]))
#         x_pad = int((target_img_shape[0] - image.shape[0]) / 2)
#         y_pad = int((target_img_shape[1] - image.shape[1]) / 2)
#         # print(x_pad,y_pad)
#         paste(new_image, image, (x_pad, y_pad))
#         image = new_image
#         # z-dim is the number of msk channels
#         # new_image = np.zeros((target_img_shape[0], target_img_shape[1], number_msk_channels))
#         # paste(new_image, mask, (x_pad, y_pad))
#         # mask = new_image
#
#     if mask.shape[0] < target_img_shape[0] or mask.shape[1] < target_img_shape[1]:
#         new_image = np.zeros((target_img_shape[0], target_img_shape[1], mask.shape[2]))
#         # WTH soll das hier sein? Zweimal genau das gleiche, also wird x_pad immer 0
#         # wenn target_img_hsape with new_image.shape verglichen wird => muss mask shape sein
#         x_pad = int((target_img_shape[0] - mask.shape[0]) / 2)
#         y_pad = int((target_img_shape[1] - mask.shape[1]) / 2)
#         # print(x_pad,y_pad)
#         paste(new_image, mask, (x_pad, y_pad))
#         mask = new_image
#
#     # rescale
#     if rescale == "min_max":
#         # Norm image values to 0 to 1
#         smooth = 1.
#         image = (image - np.min(image) + smooth) / (np.max(image) + np.min(image) + smooth)
#         mask = mask / 255
#     else:
#         # Norm image values according to the data format
#         image = image / rescale
#         mask = mask / 255
#
#     # contrast stretching
#     if contrast_stretch:
#         image = contrast_stretch_image(image)
#
#     # histogram equalization
#     if histogram_equalization:
#         image = histo_equalize_image(image)
#
#     # data_augm
#     if do_data_augm:
#         image, mask = data_augmentation(image, imagename, mask, maskname, **data_augm_args)
#     return image, mask
#
#
# def data_preprocessing_unsupervised(image, imagename, target_img_shape, number_msk_channels, rescale,
#                                     contrast_stretch, histogram_equalization, do_data_augm, **data_augm_args):
#     # resize
#     # Resize the image if the image axis are larger than target_img_shape[0], ...[1]
#     if image.shape[0] == image.shape[1] and image.shape[0] > target_img_shape[0]:
#         # Scale the image down by a factor s
#         s = float(target_img_shape[0] / image.shape[0])
#         print("scale factor: ", s)
#         image = resize(image, (target_img_shape[0], target_img_shape[1]), anti_aliasing=True, preserve_range=True)
#     elif image.shape[0] > target_img_shape[0] or image.shape[1] > target_img_shape[1]:
#         # Find the larger axis. Find the factor s to scale to x_dim.
#         # Apply it for both dimensions
#         if image.shape[0] > image.shape[1]:
#             s = float(target_img_shape[0] / image.shape[0])
#             print("scale factor: ", s)
#             image = resize(image, (target_img_shape[0], round(s * image.shape[1])), preserve_range=True)
#         else:
#             s = float(target_img_shape[1] / image.shape[1])
#             print("scale factor: ", s)
#             try:
#                 image = resize(image, (round(s * image.shape[0]), target_img_shape[1]), preserve_range=True)
#             except TypeError:
#                 print('TypeError: int object is not subscriptable')
#                 print('The type for image is: ' + str(type(image)))
#                 print('The type for image.size is: ' + str(type(image.size)))
#                 print('The type for image.shape is: ' + str(type(image.shape)))
#                 print('The type for target_img_shape is: ' + str(type(target_img_shape)))
#                 print('The type for s is: ' + str(type(s)))
#     # Fill the border regions with 0 and put the image to the middle
#     if image.shape[0] < target_img_shape[0] or image.shape[1] < target_img_shape[1]:
#         new_image = np.zeros((target_img_shape[0], target_img_shape[1], target_img_shape[2]))
#         x_pad = int((target_img_shape[0] - new_image.shape[0]) / 2)
#         y_pad = int((target_img_shape[1] - new_image.shape[1]) / 2)
#         # print(x_pad,y_pad)
#         paste(new_image, image, (x_pad, y_pad))
#         image = new_image
#
#     # rescale
#     if rescale == "min_max":
#         # Norm image values to 0 to 1
#         smooth = 1.
#         image = (image - np.min(image) + smooth) / (np.max(image) + np.min(image) + smooth)
#     else:
#         # Norm image values according to the data format
#         image = image / rescale
#
#     # contrast stretching
#     if contrast_stretch:
#         image = contrast_stretch_image(image)
#
#     # histogram equalization
#     if histogram_equalization:
#         image = histo_equalize_image(image)
#
#     # data_augm
#     if do_data_augm:
#         raise ValueError("The function data_augmentation has not been implemented for the unsupervised task."
#                          "Do you need it?")
#         # image, mask = data_augmentation(image, imagename, mask, maskname, **data_augm_args)
#     return image
#
#
# def contrast_stretch_image(img, perc=99.8):
#     # Apply it on each channel
#     for channel in range(0, img.shape[2]):
#         p_lower, p_upper = np.percentile(img[:, :, channel], (100 - perc, perc))
#         img[:, :, channel] = rescale_intensity(img[:, :, channel], in_range=(p_lower, p_upper))
#     return img
#
#
# def histo_equalize_image(img):
#     for channel in range(0, img.shape[2]):
#         img[:, :, channel] = equalize_hist(img[:, :, channel])
#     return img
#
#
# def shifting(image, x_shift, y_shift):
#     tform = SimilarityTransform(translation=(x_shift, y_shift))
#     shifted_image = warp(image, tform, mode='constant', cval=0)
#     return shifted_image
#
#
# def flipping(image, flip_horizontal, flip_vertical):
#     # Do horizontal and/or vertical flipping
#     if flip_horizontal:
#         image = np.flip(image, axis=1)
#     if flip_vertical:
#         image = np.flip(image, axis=0)
#     return image
#
#
# def rotation(image, rotation_degree):
#     image = rotate(image, rotation_degree)
#     return image
#
#
# def zoom(image, x_dim, y_dim, zooming_factor):
#     if len(image.shape) == 3:
#         rescaled_image = rescale(image, zooming_factor, mode='reflect', anti_aliasing=True, multichannel=True)
#     else:
#         rescaled_image = rescale(image, zooming_factor, mode='reflect', anti_aliasing=True, multichannel=False)
#     if zooming_factor > 1:
#         left = round((rescaled_image.shape[0] - x_dim) / 2)
#         right = left + x_dim
#         upper = round((rescaled_image.shape[1] - y_dim) / 2)
#         lower = upper + y_dim
#         cropped_image = rescaled_image[upper:lower, left:right]
#     else:
#         left = round((x_dim - rescaled_image.shape[0]) / 2)
#         right = left + rescaled_image.shape[0]
#         upper = round((y_dim - rescaled_image.shape[1]) / 2)
#         lower = upper + rescaled_image.shape[1]
#         cropped_image = np.zeros(image.shape)
#         if len(image.shape) == 2:
#             cropped_image[upper:lower, left:right] = rescaled_image
#         else:
#             cropped_image[upper:lower, left:right, :] = rescaled_image
#     return cropped_image
#
#
# def zoom_resize(image, x_dim, y_dim, zooming_factor):
#     if zooming_factor > 1:
#         resized_image = resize(image, (round(zooming_factor * x_dim), round(zooming_factor * y_dim)),
#                                anti_aliasing=True, preserve_range=True)
#         # print(round(zooming_factor * x_dim))
#         # print(round(zooming_factor * y_dim))
#         left = round((round(zooming_factor * x_dim) - x_dim) / 2)
#         upper = round((round(zooming_factor * y_dim) - y_dim) / 2)
#         right = left + x_dim
#         lower = upper + y_dim
#         cropped_image = resized_image[upper:lower, left:right]
#     else:
#         resized_image = resize(image, (round(zooming_factor * x_dim), round(zooming_factor * y_dim)),
#                                anti_aliasing=True, preserve_range=True)
#         # print(round(zooming_factor * x_dim))
#         # print(round(zooming_factor * y_dim))
#         left = round((x_dim - round(zooming_factor * x_dim)) / 2)
#         upper = round((y_dim - round(zooming_factor * y_dim)) / 2)
#         right = left + round(zooming_factor * x_dim)
#         lower = upper + round(zooming_factor * y_dim)
#         # print(upper, lower, left, right)
#         cropped_image = np.zeros(image.shape)
#         if len(image.shape) == 2:
#             cropped_image[upper:lower, left:right] = resized_image
#         else:
#             cropped_image[upper:lower, left:right, :] = resized_image
#     return cropped_image
#
#
# def signal_reduction(image, channel, signal_reduction_factor):
#     if channel == 0:
#         image[:, :, 0] = signal_reduction_factor * image[:, :, 0]
#     if channel == 1:
#         image[:, :, 1] = signal_reduction_factor * image[:, :, 1]
#     if channel == 2:
#         image[:, :, 2] = signal_reduction_factor * image[:, :, 2]
#     return image
#
#
# def data_augmentation(image, imagename, mask, maskname, verbose, width_shift_range, height_shift_range,
#                       flip_horizontal, flip_vertical, rotation_range,
#                       zooming_range, c0_signal_min, c0_signal_max,
#                       c1_signal_min, c1_signal_max, c2_signal_min,
#                       c2_signal_max, c2_reduce_podo_signal, augm_data_path, save_augm_data, plot_augm_data):
#     # Get Arguments for the data augmentation functions
#     x_dim, y_dim = image.shape[0], image.shape[1]
#
#     image_orig = image
#     mask_orig = mask
#
#     # Do shifting if
#     if width_shift_range or height_shift_range:
#         shift_range_x = int(width_shift_range * x_dim)
#         shift_range_y = int(height_shift_range * y_dim)
#         x_shift = random.randint(-shift_range_x, shift_range_x + 1)
#         y_shift = random.randint(-shift_range_y, shift_range_y + 1)
#         if verbose:
#             print("Data augmentation: Shifting by x: ", x_shift, " and y: ", y_shift)
#         image = shifting(image, x_shift, y_shift)
#         mask = shifting(mask, x_shift, y_shift)
#
#     # Do flipping if
#     if flip_horizontal:
#         if random.randint(0, 1):
#             if verbose:
#                 print("Data augmentation: horizontal (=left right) flip")
#             image = flipping(image, flip_horizontal, 0)
#             mask = flipping(mask, flip_horizontal, 0)
#     if flip_vertical:
#         if random.randint(0, 1):
#             if verbose:
#                 print("Data augmentation: verical (=top bottom) flip")
#             image = flipping(image, 0, flip_vertical)
#             mask = flipping(mask, 0, flip_vertical)
#
#     # Do rotation if
#     if rotation_range:
#         rotation_degree = (random.random() * 2 * rotation_range) - rotation_range
#         if verbose:
#             print("Data augmentation: Rotation by: ", rotation_degree)
#         image = rotation(image, rotation_degree)
#         mask = rotation(mask, rotation_degree)
#
#     # Do zoom if
#     if zooming_range:
#         zooming_factor = (random.random() * 2 * zooming_range) - zooming_range + 1
#         if verbose:
#             print("Data augmentation: Zoom by: ", zooming_factor)
#         image = zoom(image, x_dim, y_dim, zooming_factor)
#         mask = zoom(mask, x_dim, y_dim, zooming_factor)
#
#     # Do signal reduction if
#     if c0_signal_min < 1:
#         if c0_signal_min != c0_signal_max:
#             signal_reduction_factor = (random.random() + (c0_signal_min / (c0_signal_max - c0_signal_min))) * (
#                     c0_signal_max - c0_signal_min)
#         else:
#             signal_reduction_factor = c0_signal_min
#         if verbose:
#             print("Data augmentation: Signal reduction by: ", signal_reduction_factor)
#         image = signal_reduction(image, 0, signal_reduction_factor)
#
#     if c1_signal_min < 1:
#         if c1_signal_min != c1_signal_max:
#             signal_reduction_factor = (random.random() + (c1_signal_min / (c1_signal_max - c1_signal_min))) * (
#                     c1_signal_max - c1_signal_min)
#         else:
#             signal_reduction_factor = c1_signal_min
#         if verbose:
#             print("Data augmentation: Signal reduction by: ", signal_reduction_factor)
#         image = signal_reduction(image, 1, signal_reduction_factor)
#
#     if c2_signal_min < 1:
#         if c2_signal_min != c2_signal_max:
#             signal_reduction_factor = (random.random() + (c2_signal_min / (c2_signal_max - c2_signal_min))) * (
#                     c2_signal_max - c2_signal_min)
#         else:
#             signal_reduction_factor = c2_signal_min
#         if verbose:
#             print("Data augmentation: Signal reduction by: ", signal_reduction_factor)
#         image = signal_reduction(image, 2, signal_reduction_factor)
#
#     if c2_reduce_podo_signal:
#         image_podo = image[:, :, 2]
#         if len(mask.shape) == 2:
#             image_podo[(image_podo * (mask)) > 0.5] = image_podo[(image_podo * (mask)) > 0.5] * 0.75
#         else:
#             image_podo[(image_podo * (mask[:, :, 1])) > 0.5] = image_podo[(image_podo * (mask[:, :, 1])) > 0.5] * 0.75
#         image[:, :, 2] = image_podo
#
#     # Save augmented images if
#     if save_augm_data:
#         timestr = time.strftime("%Y%m%d-%H%M%S")
#         image_out = (255 * image).astype(np.uint8)
#         imsave(os.path.join(os.path.join(augm_data_path, "images"), "augm_" + timestr + "_" + imagename), image_out)
#         for i in range(mask.shape[2]):
#             mask_out = (255 * mask[:, :, i]).astype(np.uint8)
#             imsave(os.path.join(os.path.join(augm_data_path, "masks"), "augm_" + timestr + "_" + maskname[i]), mask_out)
#
#     # Plot augmented images if
#     if plot_augm_data:
#         # Image before augmentation
#         imshow(image_orig)
#         show()
#         # Image after augmentation
#         imshow(image)
#         show()
#
#     # Future extentions:
#     # Do elastic deformation if this is necessary
#
#     return image, mask


# Image generator function
def image_generator(image_path, image_format, mask_foldername, mask_suffix, batch_size, data_percentage, shuffle,
                    **data_gen_args):
    all_image_list = []
    for path in image_path:
        image_list = [_ for _ in os.listdir(path) if _.endswith(image_format)]
        full_image_list = [path + x for x in image_list]
        all_image_list.extend(full_image_list)
    all_image_list.sort()
    print(all_image_list)
    # mask_list = os.listdir(mask_path)
    # mask_list.sort()

    print(len(all_image_list))

    if data_percentage < 1:
        # The following loop is only important for the reduction to also work with pre-augmented data
        suffix_list = []
        for img_name in all_image_list:
            if img_name.find('ANCA') != -1:
                suffix = img_name[img_name.find('ANCA'):]
                suffix_list.append(suffix)
            elif img_name.find('HNE') != -1:
                suffix = img_name[img_name.find('HNE'):]
                suffix_list.append(suffix)
            elif img_name.find('control') != -1:
                suffix = img_name[img_name.find('control'):]
                suffix_list.append(suffix)
        suffix_set = set(suffix_list)
        rand_ind_subset = random.sample(range(len(suffix_set)), int(data_percentage * len(suffix_set)))
        # suffix_sublist = list(suffix_set)[rand_ind_subset]
        suffix_sublist = [list(suffix_set)[index] for index in rand_ind_subset]

        # new_image_list = []
        # for img_name in all_image_list:
        # for suffix in suffix_sublist:
        #     img_name = [s for s in all_image_list if suffix in s]
        #     #    if img_name.find(suffix):
        #     new_image_list.append(img_name)
        new_image_list = [s for s in all_image_list if any(suffix in s for suffix in suffix_sublist)]

        all_image_list = new_image_list

    print(len(all_image_list))

    # Supervised mode:
    if mask_suffix is not "unsupervised":
        i = 0
        while True:
            batch_images = []
            batch_masks = []

            if shuffle:
                batch_paths = np.random.choice(a=all_image_list, size=batch_size)
            else:
                batch_paths = []  # np.ndarray(shape=(1, batch_size))
                for j in range(batch_size):
                    batch_paths.append(all_image_list[i])
                    i = i + 1
                    if i == len(all_image_list):
                        i = 0
                        break

            # print("Elements in batch path:",batch_paths)
            for image_pathname in batch_paths:
                image = load_image(image_pathname)
                # Open masks and put them in one array
                maskfolder = os.path.join(os.path.split(os.path.split(image_pathname)[0])[0], mask_foldername + "/")
                # print(maskfolder)
                # print(image_pathname)
                maskname = []
                maskname.append(os.path.split(image_pathname)[1][0:-4] + mask_suffix[0])
                first_mask = load_mask(os.path.join(maskfolder, maskname[0]))
                mask = first_mask
                mask = np.expand_dims(mask, axis=2)
                for k in range(1, len(mask_suffix)):
                    maskname.append(os.path.split(image_pathname)[1][0:-4] + mask_suffix[k])
                    further_mask = load_mask(os.path.join(maskfolder, maskname[k]))
                    further_mask = np.expand_dims(further_mask, axis=2)
                    mask = np.concatenate((mask, further_mask), axis=2)

                print("")
                print(image_pathname)
                imagename = os.path.split(image_pathname)[1]
                print(maskname)
                # Do preprocessing with normalisation and augmentation
                image, mask = preprocess_image(image, imagename, mask, maskname, **data_gen_args)

                batch_images += [image]
                batch_masks += [mask]

            batch_x = np.array(batch_images)
            batch_y = np.array(batch_masks)

            print(batch_x.shape)
            print(batch_y.shape)
            # print("end of batch")
            yield (batch_x, batch_y)

    # Unsupervised mode:
    elif mask_suffix is "unsupervised":
        # Set mask and maskname so that its boolean value will be false (see more in the function data_preprocessing)
        mask = None
        maskname = None
        i = 0
        while True:
            batch_images = []

            if shuffle:
                batch_paths = np.random.choice(a=all_image_list, size=batch_size)
            else:
                batch_paths = []  # np.ndarray(shape=(1, batch_size))
                for j in range(batch_size):
                    batch_paths.append(all_image_list[i])
                    i = i + 1
                    if i == len(all_image_list):
                        i = 0
                        break

            # print("Elements in batch path:",batch_paths)
            for image_pathname in batch_paths:
                image = load_image(image_pathname)

                print("")
                print(image_pathname)
                imagename = os.path.split(image_pathname)[1]
                # Do preprocessing with normalisation and augmentation
                image = preprocess_image(image, imagename, mask=None, maskname=None, **data_gen_args)

                batch_images += [image]

            batch_x = np.array(batch_images)

            print(batch_x.shape)
            # print("end of batch")
            yield batch_x


if __name__ == "__main__":
    # Run script
    from config.config import Config


    class TrainingConfig(Config):
        # GPUs and IMAGES_PER_GPU needs to be configured here!
        GPUs = ''
        IMAGES_PER_GPU = 4

        ZOOMING_RANGE = 0.5  # high value to check if it works

        WIDTH_SHIFT_RANGE = 0.5
        HEIGHT_SHIFT_RANGE = 0.5
        FLIP_HORIZONTAL = True
        FLIP_VERTICAL = True
        ROTATION_RANGE = 45

        SEGMENTATION_TASK = 'all'  # 'glomerulus' #'all'#'podocytes'

        # NAMES_CLASSES = ['glomerulus', 'podocytes']
        MASK_SUFFIXES = ['_mask_kuh.png', '_mask_rad.png']

        SUPERVISED_MODE = False
        CONTRAST_STRETCHING = True
        HISTOGRAM_EQUALIZATION = True


    cfg = TrainingConfig()

    # Select the masks
    if cfg.SEGMENTATION_TASK == 'glomerulus':
        mask_suffix = []
        mask_suffix.append(cfg.MASK_SUFFIXES[0])
    elif cfg.SEGMENTATION_TASK == 'podocytes':
        mask_suffix = []
        mask_suffix.append(cfg.MASK_SUFFIXES[1])
    elif cfg.SEGMENTATION_TASK == 'all':
        mask_suffix = cfg.MASK_SUFFIXES

    data_path = ['/data/Dataset1/tif/']
    img_train_full = []
    for path in data_path:
        img_train_full.append(os.path.join(path, 'test/images/'))

    data_gen_args = dict(shuffle=False,  # True,
                         target_img_shape=cfg.TARGET_IMG_SHAPE,
                         number_msk_channels=cfg.NUMBER_MSK_CHANNELS * cfg.NUM_OUTPUT_CH,
                         rescale=cfg.RESCALE,
                         contrast_stretch=cfg.CONTRAST_STRETCHING,
                         histogram_equalization=cfg.HISTOGRAM_EQUALIZATION,
                         do_data_augm=False,  # True,
                         verbose=False,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         flip_horizontal=True,
                         flip_vertical=True,
                         rotation_range=45,
                         zooming_range=0.1,
                         # signal_min and signal_max must be in [0,1], signal_max must be greater than signal_min
                         c0_signal_min=1,
                         c0_signal_max=1,
                         c1_signal_min=1,
                         c1_signal_max=1,
                         c2_signal_min=1,
                         c2_signal_max=1,
                         c2_reduce_podo_signal=False,
                         augm_data_path=0,  # Enter a path here if augmented data shall be saved
                         save_augm_data=False,
                         plot_augm_data=False)

    if cfg.SUPERVISED_MODE:
        train_gen = image_generator(img_train_full, cfg.IMAGE_FORMAT, cfg.MASK_FOLDERNAME, mask_suffix, cfg.BATCH_SIZE,
                                    0.1, **data_gen_args)
    else:
        train_gen = image_generator(img_train_full, cfg.IMAGE_FORMAT, cfg.MASK_FOLDERNAME, "unsupervised",
                                    cfg.BATCH_SIZE, 0.1, **data_gen_args)
    next(train_gen)
