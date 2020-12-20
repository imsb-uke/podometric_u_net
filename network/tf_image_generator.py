import os
import tensorflow as tf
import random
import numpy as np
from itertools import cycle, islice

import matplotlib.pyplot as plt


from config.config import Config
from network.dataset.image_preprocessing import resize_image, norm_image, norm_mask

from network.dataset.image_loading import load_image, load_mask_from_img, load_weights
from network.dataset.image_transformations import contrast_stretch_image, histo_equalize_image
from network.dataset.tf_image_augementation import data_augmentation_wgt, data_augmentation


def reshape_for_weighting(cfg, tensor_image, tensor_mask, tensor_weights):
    # Function to include weights for loss weighting in input

    return (tensor_image, tensor_weights), tensor_mask


#####################################################
# Plotting a batch (image, target or prediction)    #
#####################################################

def plot_batch(batch):
    """
    Plots the batch.
    Consists of 2 subplots if the batch does not contain weights.
    Consists of 3 subplots if the batch contains weights.
    The subplots are column-wise ordered
    First column: Images, RGB colormap
    Second columns: Masks, standard colormap (0 for background, 1 for foreground)
    Third columns: Weights, standard colormap (e.g. close to 0 for background, over 10 for pixels inbetween cells.)

    Parameters:
    -----------
    batch: list of arrays:
           [images, masks or predicitions, weights]
           each array has the shape (batch size, img height, img width, channels)

    """

    img_batch = batch[0]
    msk_batch = batch[1]

    #print(img_batch.shape)
    #print(msk_batch.shape)

    img_count = img_batch.shape[0]

    # Create output arrays
    # 1) Output image
    # third dimension with 3 channels since its RGB
    output_img = np.zeros(
        (img_batch.shape[1] * img_count, img_batch.shape[2], img_batch.shape[3]))
    row = 0
    for image_id in range(img_count):
        image = img_batch[image_id, :, :, :]
        #print(image.shape)
        output_img[row * img_batch.shape[1]:(row + 1) * img_batch.shape[1], :, :] = image
        row += 1

    # 2) Masks or predictions
    output_msk = np.zeros((msk_batch.shape[1] * img_count, msk_batch.shape[2] * msk_batch.shape[3]))
    #print("output_msk", output_msk.shape)
    row = 0
    for image_id in range(img_count):
        for j in range(msk_batch.shape[3]):
            mask_ch = msk_batch[image_id, :, :, j]
            mask_ch = mask_ch
            mask_ch = mask_ch.astype(int)
            #mask_ch = np.stack((mask_ch, mask_ch, mask_ch), axis=-1)
            #print(mask_ch.shape)
            #print(msk_batch.shape)
            #print(row * msk_batch.shape[1])
            #print((row + 1) * msk_batch.shape[1])
            #print(j * msk_batch.shape[2])
            #print((j + 1) * msk_batch.shape[2])
            output_msk[row * msk_batch.shape[1]:(row + 1) * msk_batch.shape[1],
            j * msk_batch.shape[2]:(j + 1) * msk_batch.shape[2]] = mask_ch
        row += 1

    if len(batch) == 3:

        # Generator yields img, masks and weights
        #print('Info: Generator yields img, masks, weights.')
        wgt_batch = batch[2]
        #print('Weights shape:', wgt_batch.shape)
        # 3) Weights
        output_wgt = np.zeros((wgt_batch.shape[1] * img_count, wgt_batch.shape[2] * wgt_batch.shape[3]))
        row = 0
        for image_id in range(img_count):
            for k in range(wgt_batch.shape[3]):
                wgt_ch = wgt_batch[image_id, :, :, k]
                wgt_ch = wgt_ch  # * (255 / wgt_ch.max())
                # wgt_ch = wgt_ch.astype(int)
                #wgt_ch = np.stack((wgt_ch, wgt_ch, wgt_ch), axis=-1)

                output_wgt[row * wgt_batch.shape[1]:(row + 1) * wgt_batch.shape[1],
                k * wgt_batch.shape[2]:(k + 1) * wgt_batch.shape[2]] = wgt_ch
            row += 1

        # Plot
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, msk_batch.shape[3],
                                                                                wgt_batch.shape[3]]})

        # Plot weights
        pos = ax3.imshow(output_wgt)
        ax3.set_axis_off()
        ax3.set_title('Weights', fontsize=15)

    else:
        # Generator yields img and masks.
        print('Info: Generator yields img, masks.')
        fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, msk_batch.shape[3]]})

    # Plot images
    pos = ax1.imshow(output_img)
    ax1.set_axis_off()
    ax1.set_title('Images', fontsize=15)

    # Plot masks / predictions
    pos = ax2.imshow(output_msk)
    ax2.set_axis_off()
    ax2.set_title('Masks / Predictions', fontsize=15)

    plt.show()


#####################################################
# Functions to load one random batch for inference  #
#####################################################

def find_msk_paths(cfg, image_path):
    filename = os.path.split(image_path)[1]
    foldername = os.path.split(image_path)[0]
    foldernam = os.path.split(foldername)[0]
    maskfoldername = os.path.join(foldernam, 'masks')
    msk_paths = []
    if cfg.SEGMENTATION_TASK == 'glomerulus':
        msk_paths.append(os.path.join(maskfoldername, filename[:-4] + cfg.MASK_SUFFIXES[0]))
    elif cfg.SEGMENTATION_TASK == 'podocytes':
        msk_paths.append(os.path.join(maskfoldername, filename[:-4] + cfg.MASK_SUFFIXES[1]))
    elif cfg.SEGMENTATION_TASK == 'all':
        for suffix in cfg.MASK_SUFFIXES:
            msk_paths.append(os.path.join(maskfoldername, filename[:-4] + suffix))
    else:
        raise ValueError('cfg.SEGMENTATION_TASK does not match to the implemented values: ',
                         cfg.SEGMENTATION_TASK)
    return msk_paths


def find_weight_paths(cfg, image_path):
    filename = os.path.split(image_path)[1]
    foldername = os.path.split(image_path)[0]
    foldernam = os.path.split(foldername)[0]
    weightsfoldername = os.path.join(foldernam, 'weights')
    wgt_paths = []
    if cfg.SEGMENTATION_TASK == 'glomerulus':
        wgt_paths.append(os.path.join(weightsfoldername, filename[:-4] + cfg.WEIGHTS_SUFFIXES[0]))
    elif cfg.SEGMENTATION_TASK == 'podocytes':
        wgt_paths.append(os.path.join(weightsfoldername, filename[:-4] + cfg.WEIGHTS_SUFFIXES[1]))
    elif cfg.SEGMENTATION_TASK == 'all':
        for suffix in cfg.WEIGHTS_SUFFIXES:
            wgt_paths.append(os.path.join(weightsfoldername, filename[:-4] + suffix))
    else:
        raise ValueError('cfg.SEGMENTATION_TASK does not match to the implemented values: ',
                         cfg.SEGMENTATION_TASK)
    return wgt_paths


def batch_data(array1, array2, multichannel):
    """
    Gets arrays as input and concatenates them at axis 0.

    Parameters:
    -----------
    array1: array (2d or 3d)
    array2: array (2d or 3d)
    multichannel: bool
        True: Batch will have 4 dimensions
        False: Batch will have 3 dimensions

    Return:
    -------
    batched array of shape (batch_size, array_shape)
    """

    # Check if input arrays are valid for this function
    # 2 and 3 dimensions are allowed for multichannel=False
    # 3 and 4 dimensions are allowed for multichannel=True
    if multichannel:
        if len(array1.shape) < 3 or len(array2.shape) < 3 or len(array1.shape) > 4 or len(array2.shape) > 4:
            raise ValueError('The multilayer input array shapes do not match this function',
                             array1.shape, array2.shape)
    else:
        if len(array1.shape) < 2 or len(array2.shape) < 2 or len(array1.shape) > 3 or len(array2.shape) > 3:
            raise ValueError('The input array shapes do not match this function',
                             array1.shape, array2.shape)

    # Check if one array has another shape than the other
    # Make it then the same
    if len(array1.shape) < len(array2.shape):
        array1 = np.expand_dims(array1, axis=0)
    elif len(array1.shape) > len(array2.shape):
        array2 = np.expand_dims(array2, axis=0)

    # For images with multiple channels
    if multichannel:
        # case 1: (x,y,c), (x,y,c)
        if len(array1.shape) == 3:
            array1 = np.expand_dims(array1, axis=0)
            array2 = np.expand_dims(array2, axis=0)
            batched_array = np.concatenate((array1, array2), axis=0)

        # case 2: (b,x,y,c), (1,x,y,c)
        else:
            batched_array = np.concatenate((array1, array2), axis=0)
    else:
        # case 1: (x,y), (x,y)
        if len(array1.shape) == 2:
            array1 = np.expand_dims(array1, axis=0)
            array2 = np.expand_dims(array2, axis=0)
            batched_array = np.concatenate((array1, array2), axis=0)

        # case 2: (b,x,y), (1,x,y)
        else:
            batched_array = np.concatenate((array1, array2), axis=0)


    return batched_array


def concat_msks(msk_paths):
    first_mask = load_image(msk_paths.pop(0))
    mask = first_mask
    mask = np.expand_dims(mask, axis=2)
    for path in msk_paths:
        further_mask = load_image(path)
        further_mask = np.expand_dims(further_mask, axis=2)
        mask = np.concatenate((mask, further_mask), axis=2)
    return mask


def concat_wgts(wgt_paths):
    first_weights = np.load(wgt_paths.pop(0))
    weights = first_weights
    weights = np.expand_dims(weights, axis=2)
    for path in wgt_paths:
        further_weights = np.load(path)
        further_weights = np.expand_dims(further_weights, axis=2)
        weights = np.concatenate((weights, further_weights), axis=2)
    return weights


def load_random_batch(cfg, data_paths):
    """
    Loads a random batch (batch_size, image, masks, weights)

    Parameters:
    -----------
    cfg: contains the cfg.BATCH_SIZE

    data_paths: list containing strings
        paths to the folder where the images, masks and weights are in


    Returns:
    --------
    batch containing the images, masks (and optionally weights)

    """

    train_imgs_dirs = list()
    valid_imgs_dirs = list()
    test_imgs_dirs = list()

    for data_path in data_paths:
        train_imgs_dirs.append(os.path.join(data_path, 'train', 'images'))
        valid_imgs_dirs.append(os.path.join(data_path, 'valid', 'images'))
        #test_imgs_dirs.append(os.path.join(data_path, 'test', 'images'))

    imgs_list = []
    for train_imgs_dir in train_imgs_dirs:
        imgs_list.extend([os.path.join(train_imgs_dir, s) for s in os.listdir(train_imgs_dir)])
    for valid_imgs_dir in valid_imgs_dirs:
        imgs_list.extend([os.path.join(valid_imgs_dir, s) for s in os.listdir(valid_imgs_dir)])
    #for test_imgs_dir in test_imgs_dirs:
        #imgs_list.extend([os.path.join(test_imgs_dir, s) for s in os.listdir(test_imgs_dir)])
    imgs_list.sort()
    # print(imgs_list)
    n_imgs = len(imgs_list)

    # Ziehen ohne Zurücklegen
    # Takes cfg.BATCH_SIZE elements out of len(imgs_list)
    batch_list = random.sample(range(0, n_imgs), cfg.BATCH_SIZE)
    print("batch_list", batch_list)
    # rand_img = random.randrange(0, n_imgs)
    # print(rand_img)
    #img_path = imgs_list[rand_img]

    # Get the image paths of this batch
    batch_imgs_list = [imgs_list[i] for i in batch_list]

    # The first image will be batched with no other image
    first_img = True
    for img_path in batch_imgs_list:
        # Get the image
        print(img_path)
        img = load_image(img_path)
        # Get the corresponding masks
        msk_paths = find_msk_paths(cfg, img_path)
        print(msk_paths)
        msk = concat_msks(msk_paths)
        #print("mask shape", msk.shape)
        # Put it together
        if first_img:
            imgs = np.expand_dims(img, axis=0)
            msks = np.expand_dims(msk, axis=0)
            first_img = False
        else:
            imgs = batch_data(imgs, img, multichannel=True)
            msks = batch_data(msks, msk, multichannel=True)
        #print("masks shape", msks.shape)

    # Normalize it
    imgs = imgs / cfg.RESCALE
    msks = msks / cfg.RESCALE_MSK

    if cfg.WEIGHTING:
        # The first image will be batched with no other image
        first_img = True
        for img_path in batch_imgs_list:
            wgt_paths = find_weight_paths(cfg, img_path)
            #print(wgt_paths)
            weight = concat_wgts(wgt_paths)
            # Put it together
            if first_img:
                wgts = np.expand_dims(weight, axis=0)
                first_img = False
            else:
                wgts = batch_data(wgts, weight, multichannel=True)
            # print(wgts.shape)
        #print("wgts min max shape", np.min(wgts), np.max(wgts), wgts.shape)
        return imgs, msks, wgts
    else:
        return imgs, msks


#####################################################
# tf.data ImageGenerator                            #
#####################################################

class ImageGenerator:
    def __init__(self):
        self.cfg = Config()
        self.data_path = ""
        self.img_path = ""
        self.msk_path = ""
        self.wgt_path = ""

        self.number_of_data = 0
        self.img_path_folder = "images"
        self.wgt_path_folder = "weights"
        self.network_mode = 'training'  # 'validation'

    def set_attributes(self, config, datapath, netw_mode):
        self.cfg = config
        self.data_path = datapath
        self.img_path = os.path.join(datapath, "images")
        self.msk_path = os.path.join(datapath, self.cfg.MASK_FOLDERNAME)
        self.wgt_path = os.path.join(datapath, "weights")
        self.number_of_data = len(os.listdir(self.img_path))
        self.network_mode = netw_mode

    def get_number_of_data(self):
        return self.number_of_data

    def generator(self):
        print("Generator called in", self.network_mode, "mode.")

        img_names = [_ for _ in os.listdir(self.img_path) if _.endswith(self.cfg.IMAGE_FORMAT)]
        img_names.sort()

        if self.cfg.SEGMENTATION_TASK == 'all':
            mask_suffix = self.cfg.MASK_SUFFIXES
            weights_suffix = self.cfg.WEIGHTS_SUFFIXES
        elif self.cfg.SEGMENTATION_TASK in self.cfg.NAMES_CLASSES:
            index = self.cfg.NAMES_CLASSES.index(self.cfg.SEGMENTATION_TASK)
            mask_suffix = []
            mask_suffix.append(self.cfg.MASK_SUFFIXES[index])
            weights_suffix = []
            weights_suffix.append(self.cfg.WEIGHTS_SUFFIXES[index])
        else:
            raise ValueError("The SEGMENTATION_TASK you want to use does not fit to NAMES_CLASSES."
                             "Check cfg.SEGMENTATION_TASK", self.cfg.SEGMENTATION_TASK)

        if self.cfg.TRAINING_DATA_PERCENTAGE < 1 and self.network_mode == 'training':
            random.seed(self.cfg.IMG_GENERATOR_SEED)
            rand_subset = random.sample(range(self.number_of_data), int(self.cfg.TRAINING_DATA_PERCENTAGE * self.number_of_data))
            sublist = [img_names[index] for index in rand_subset]
            img_names = sublist

        if self.cfg.OVERSAMPLING and self.network_mode == 'training':
            print('Number of images before oversampling', len(img_names))
            # Split img_names in "without tag" and "with tag (crescent or lesion)"
            img_names_with_tag = [i for i in img_names if
                                  any(j in i for j in self.cfg.OVERSAMPLING_TAGS)]
            img_names_without_tag = [i for i in img_names if not
                                     any(j in i for j in self.cfg.OVERSAMPLING_TAGS)]

            # Check that tag exists
            if len(img_names_with_tag) == 0:
                raise ValueError(self.cfg.OVERSAMPLING_TAGS, 'could not be found in ', img_names,
                                 'Please check the image names or the tags.')

            # Make list lengths equal
            if len(img_names_with_tag) < len(img_names_without_tag):
                name_of_oversampled_imgs = "img names with tag"
                oversample_factor = len(img_names_without_tag) / len(img_names_with_tag)
                img_names_with_tag = list(islice(cycle(img_names_with_tag), len(img_names_without_tag)))
            else:
                name_of_oversampled_imgs = "img names without tag"
                oversample_factor = len(img_names_with_tag) / len(img_names_without_tag)
                img_names_without_tag = list(islice(cycle(img_names_without_tag), len(img_names_with_tag)))

            # Put together so that img with tag and without tag alternate
            img_names = [None]*(len(img_names_with_tag)+len(img_names_without_tag))
            img_names[::2] = img_names_with_tag
            img_names[1::2] = img_names_without_tag

            print("Oversampling of elements containing ", self.cfg.OVERSAMPLING_TAGS,
                  "\nOversampled", name_of_oversampled_imgs, oversample_factor, "times.")
            print('Number of images after oversampling', len(img_names))

        print(img_names)
        print('mask path:', self.msk_path)

        for i in range(len(img_names)):
            img_name = img_names[i]
            # load images
            img = load_image(os.path.join(self.img_path, img_name))
            msk = load_mask_from_img(self.cfg, self.msk_path, img_name[:-4], mask_suffix)
            if self.cfg.WEIGHTING:
                wgt = load_weights(self.cfg, self.wgt_path, img_name[:-4], weights_suffix)
            else:
                wgt = None

            # resize
            img, msk, wgt = resize_image(img, self.cfg.TARGET_IMG_SHAPE, msk, wgt)

            # norm image to [0,1]
            img = norm_image(img, self.cfg.RESCALE)
            msk = norm_mask(msk, self.cfg.RESCALE_MSK)

            # contrast stretching
            if self.cfg.CONTRAST_STRETCHING:
                img = contrast_stretch_image(img)

            # histogram equalization
            if self.cfg.HISTOGRAM_EQUALIZATION:
                img = histo_equalize_image(img)

            # Add 3d dimension to msk
            # if len(msk.shape) == 2:
            #    msk = np.expand_dims(msk, axis=-1)

            if self.cfg.WEIGHTING:
                yield img, msk, wgt
            else:
                yield img, msk


if __name__ == '__main__':

    class trainingConfig(Config):
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

        WEIGHTING = True
        OVERSAMPLING = True  # False

        USE_AUGM_ON_THE_FLY = False

        # NAMES_CLASSES = ['glomerulus', 'podocytes']
        # MASK_SUFFIXES = ['_mask_kuh.png', '_mask_rad.png']
        # WEIGHTS_SUFFIXES = ['_mask_kuhweights.npy', '_mask_radweights.npy']

        CONTRAST_STRETCHING = False  # True
        HISTOGRAM_EQUALIZATION = False  # True

    config = trainingConfig()
    print("batch size", config.BATCH_SIZE)

    # Training dataset
    train_gen = ImageGenerator()
    data_path = '/data/Dataset1/tif/cv0/train'
    # network_mode = 'training'

    train_gen.set_attributes(config, data_path, 'training')

    if config.WEIGHTING:
        train_dataset = tf.data.Dataset.from_generator(
                        train_gen.generator,
                        (tf.float32, tf.float32, tf.float32),
                        (config.TARGET_IMG_SHAPE,
                        (config.TARGET_IMG_SHAPE[0], config.TARGET_IMG_SHAPE[1], config.NUMBER_MSK_CHANNELS * config.NUM_OUTPUT_CH),
                        (config.TARGET_IMG_SHAPE[0], config.TARGET_IMG_SHAPE[1], config.NUMBER_MSK_CHANNELS * config.NUM_OUTPUT_CH)))
    else:
        train_dataset = tf.data.Dataset.from_generator(
                        train_gen.generator,
                        (tf.float32, tf.float32),
                        (config.TARGET_IMG_SHAPE,
                         (config.TARGET_IMG_SHAPE[0], config.TARGET_IMG_SHAPE[1], config.NUMBER_MSK_CHANNELS * config.NUM_OUTPUT_CH)))

    train_dataset = train_dataset.shuffle(train_gen.get_number_of_data())
    train_dataset = train_dataset.repeat()  # Repeat the input indefinitely.

    # Do data augementation on train_dataset if wished
    if config.USE_AUGM_ON_THE_FLY:
        if config.WEIGHTING:
            train_dataset = train_dataset.map(lambda x, y, z: data_augmentation_wgt(config, x, y, z), num_parallel_calls=4)
        else:
            train_dataset = train_dataset.map(lambda x, y: data_augmentation(config, x, y), num_parallel_calls=4)
        # train_dataset = train_dataset.map(lambda x: data_augmentation(config, x[0], x[1]), num_parallel_calls=4)
    #    tensor_img, tensor_msk = data_augmentation(self.cfg, tensor_img, tensor_msk)

    train_dataset = train_dataset.batch(config.BATCH_SIZE)


    # Validation dataset
    # Here no shuffle is applied to the images, since this is not necessary
    valid_gen = ImageGenerator()
    data_path = '/data/Dataset1/tif/cv0/train'
    # network_mode = 'validation'

    valid_gen.set_attributes(config, data_path, 'validation')

    if config.WEIGHTING:
        valid_dataset = tf.data.Dataset.from_generator(
                        valid_gen.generator,
                        (tf.float32, tf.float32, tf.float32),
                        (config.TARGET_IMG_SHAPE,
                        (config.TARGET_IMG_SHAPE[0], config.TARGET_IMG_SHAPE[1], config.NUMBER_MSK_CHANNELS * config.NUM_OUTPUT_CH),
                        (config.TARGET_IMG_SHAPE[0], config.TARGET_IMG_SHAPE[1], config.NUMBER_MSK_CHANNELS * config.NUM_OUTPUT_CH)))
    else:
        valid_dataset = tf.data.Dataset.from_generator(
                        valid_gen.generator,
                        (tf.float32, tf.float32),
                        (config.TARGET_IMG_SHAPE,
                        (config.TARGET_IMG_SHAPE[0], config.TARGET_IMG_SHAPE[1], config.NUMBER_MSK_CHANNELS * config.NUM_OUTPUT_CH)))

    valid_dataset = valid_dataset.repeat()  # Repeat the input indefinitely.
    valid_dataset = valid_dataset.batch(config.BATCH_SIZE)

    # Create generic iterator
    iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    # Old iterator:
    # iter = dataset.make_initializable_iterator()

    # Initialization options
    train_init_op = iter.make_initializer(train_dataset)
    valid_init_op = iter.make_initializer(valid_dataset)

    el = iter.get_next()

    with tf.Session() as sess:
        # test train
        sess.run(train_init_op)
        # print(sess.run(el)[0].shape, sess.run(el)[1].shape)
        batch = sess.run(el)
        plot_batch(batch)

        #test valid (no shuffle applied here)
        sess.run(valid_init_op)
        batch_val = sess.run(el)
        plot_batch(batch_val)

        #print(sess.run(el)[0].shape, sess.run(el)[1].shape)

    # get_batch = image_generator(config, data_path, network_mode='validation')
    # next(get_batch)
