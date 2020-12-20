import tensorflow as tf
from tensorflow import contrib as tfcontrib


def shift_img(tensor_img, tensor_msk, tensor_wgt, width_shift_range, height_shift_range):
    '''
    :param tensor_img:
    :param tensor_msk:
    :param tensor_wgt:
    :param width_shift_range:
    :param height_shift_range:
    :return:
    ATTENTION: TENSOR HAS SHAPE (?, 1024, 1024, 3). Thus indices 1 and 2 have to be used to get width and height
    '''
    if width_shift_range or height_shift_range:
        # print(tensor_img.shape[1])
        # print(width_shift_range)
        tensor_img_shape = tensor_img.get_shape()
        # print(tensor_img_shape)
        if width_shift_range:
            width_shift_range = tf.random_uniform([],
                                                  -width_shift_range * tensor_img_shape[2].value,
                                                  width_shift_range * tensor_img_shape[2].value)
        if height_shift_range:
            height_shift_range = tf.random_uniform([],
                                                   -height_shift_range * tensor_img_shape[1].value,
                                                   height_shift_range * tensor_img_shape[1].value)
        # Translate both
        tensor_img = tfcontrib.image.translate(tensor_img,
                                             [width_shift_range, height_shift_range])
        tensor_msk = tfcontrib.image.translate(tensor_msk,
                                               [width_shift_range, height_shift_range])
        if tensor_wgt is not None:
            tensor_wgt = tfcontrib.image.translate(tensor_wgt,
                                                   [width_shift_range, height_shift_range])
    return tensor_img, tensor_msk, tensor_wgt


def flip_img(tensor_img, tensor_msk, tensor_wgt, flip_horizontal, flip_vertical):
    # Using loss weighting or not
    if tensor_wgt is not None:
        if flip_horizontal:
            flip_prob = tf.random_uniform([], 0.0, 1.0)
            tensor_img, tensor_msk, tensor_wgt = tf.cond(tf.less(flip_prob, 0.5),
                                                         lambda: (tf.image.flip_left_right(tensor_img),
                                                                  tf.image.flip_left_right(tensor_msk),
                                                                  tf.image.flip_left_right(tensor_wgt)),
                                                         lambda: (tensor_img, tensor_msk, tensor_wgt))
        if flip_vertical:
            flip_prob = tf.random_uniform([], 0.0, 1.0)
            tensor_img, tensor_msk, tensor_wgt = tf.cond(tf.less(flip_prob, 0.5),
                                                         lambda: (tf.image.flip_up_down(tensor_img),
                                                                  tf.image.flip_up_down(tensor_msk),
                                                                  tf.image.flip_up_down(tensor_wgt)),
                                                         lambda: (tensor_img, tensor_msk, tensor_wgt))
    else:
        if flip_horizontal:
            flip_prob = tf.random_uniform([], 0.0, 1.0)
            tensor_img, tensor_msk = tf.cond(tf.less(flip_prob, 0.5),
                                             lambda: (tf.image.flip_left_right(tensor_img),
                                                      tf.image.flip_left_right(tensor_msk)),
                                             lambda: (tensor_img, tensor_msk))
        if flip_vertical:
            flip_prob = tf.random_uniform([], 0.0, 1.0)
            tensor_img, tensor_msk = tf.cond(tf.less(flip_prob, 0.5),
                                             lambda: (tf.image.flip_up_down(tensor_img),
                                                      tf.image.flip_up_down(tensor_msk)),
                                             lambda: (tensor_img, tensor_msk))

    return tensor_img, tensor_msk, tensor_wgt


def rotate_img(tensor_img, tensor_msk, tensor_wgt, rotation_range, verbose=True):
    rotation_degree = tf.random_uniform([], - rotation_range, rotation_range)
    if verbose:
        print("rotation degree:", rotation_degree)
    tensor_img = tf.contrib.image.rotate(tensor_img, rotation_degree)
    tensor_msk = tf.contrib.image.rotate(tensor_msk, rotation_degree)
    if tensor_wgt is not None:
        tensor_wgt = tf.contrib.image.rotate(tensor_wgt, rotation_degree)

    return tensor_img, tensor_msk, tensor_wgt


def zoom_img(tensor_img, tensor_msk, tensor_wgt, original_height, original_width, zooming_range):
    zooming_factor = tf.random_uniform([], - zooming_range, zooming_range)
    # Using loss weighting or not
    if tensor_wgt is not None:
        output_img, output_msk, output_wgt = tf.cond(zooming_factor > 0,
                                                     lambda: zoom_in_wgt(tensor_img, tensor_msk, tensor_wgt,
                                                                         original_height, original_width,
                                                                         zooming_factor),
                                                     lambda: zoom_out_wgt(tensor_img, tensor_msk, tensor_wgt,
                                                                          original_height, original_width,
                                                                          - zooming_factor))
    else:
        output_img, output_msk = tf.cond(zooming_factor > 0,
                                         lambda: zoom_in(tensor_img, tensor_msk, original_height,
                                                         original_width, zooming_factor),
                                         lambda: zoom_out(tensor_img, tensor_msk, original_height,
                                                          original_width, - zooming_factor))
        output_wgt = tensor_wgt

    return output_img, output_msk, output_wgt


def zoom_in_wgt(tensor_img, tensor_msk, tensor_wgt, original_height, original_width, zooming_factor, verbose=True):
    if verbose:
        print("zooming in factor:", zooming_factor)

    # scale = tf.constant(1 - zooming_factor, shape=())
    # box_ind = np.zeros(1)
    # crop_size = (original_height, original_width)
    # boxes = [x1, y1, x2, y2]

    x1 = y1 = 0.5 - (0.5 * (1-zooming_factor))
    x2 = y2 = 0.5 + (0.5 * (1-zooming_factor))

    x1_index = tf.cast(x1*original_height, dtype=tf.int32)
    x2_index = original_height - x1_index
    y1_index = tf.cast(y1*original_width, dtype=tf.int32)
    y2_index = original_width - y1_index

    cropped_img = tf.image.crop_to_bounding_box(tensor_img, x1_index, y1_index, x2_index, y2_index)
    cropped_msk = tf.image.crop_to_bounding_box(tensor_msk, x1_index, y1_index, x2_index, y2_index)

    output_img = tf.image.resize_image_with_pad(cropped_img, original_height, original_width)
    output_msk = tf.image.resize_image_with_pad(cropped_msk, original_height, original_width)

    # if tensor_wgt is not None:
    cropped_wgt = tf.image.crop_to_bounding_box(tensor_wgt, x1_index, y1_index, x2_index, y2_index)
    output_wgt = tf.image.resize_image_with_pad(cropped_wgt, original_height, original_width)
    # else:
    #     output_wgt = tensor_wgt

    return output_img, output_msk, output_wgt


def zoom_in(tensor_img, tensor_msk, original_height, original_width, zooming_factor, verbose=True):
    if verbose:
        print("zooming in factor:", zooming_factor)

    # scale = tf.constant(1 - zooming_factor, shape=())
    # box_ind = np.zeros(1)
    # crop_size = (original_height, original_width)
    # boxes = [x1, y1, x2, y2]

    x1 = y1 = 0.5 - (0.5 * (1-zooming_factor))
    x2 = y2 = 0.5 + (0.5 * (1-zooming_factor))

    x1_index = tf.cast(x1*original_height, dtype=tf.int32)
    x2_index = original_height - x1_index
    y1_index = tf.cast(y1*original_width, dtype=tf.int32)
    y2_index = original_width - y1_index

    cropped_img = tf.image.crop_to_bounding_box(tensor_img, x1_index, y1_index, x2_index, y2_index)
    cropped_msk = tf.image.crop_to_bounding_box(tensor_msk, x1_index, y1_index, x2_index, y2_index)

    output_img = tf.image.resize_image_with_pad(cropped_img, original_height, original_width)
    output_msk = tf.image.resize_image_with_pad(cropped_msk, original_height, original_width)

    return output_img, output_msk


def zoom_out_wgt(tensor_img, tensor_msk, tensor_wgt, original_height, original_width, zooming_factor, verbose=True):
    if verbose:
        print("zooming out factor:", zooming_factor)

    shrinked_height = tf.cast((1 - zooming_factor) * original_height, dtype=tf.int32)
    shrinked_width = tf.cast((1 - zooming_factor) * original_width, dtype=tf.int32)

    shrinked_img = tf.image.resize_image_with_pad(tensor_img, target_height=shrinked_height,
                                                  target_width=shrinked_width)
    shrinked_msk = tf.image.resize_image_with_pad(tensor_msk, target_height=shrinked_height,
                                                  target_width=shrinked_width)

    paddings_width = original_width - shrinked_width
    padding_righter_width = tf.cast(paddings_width / 2, dtype=tf.int32)
    padding_lefter_width = tf.cast(paddings_width / 2, dtype=tf.int32) + tf.cast(paddings_width % 2, dtype=tf.int32)

    paddings_height = original_width - shrinked_width
    padding_upper_height = tf.cast(paddings_height / 2, dtype=tf.int32)
    padding_lower_height = tf.cast(paddings_height / 2, dtype=tf.int32) + tf.cast(paddings_height % 2, dtype=tf.int32)
    paddings = [[padding_upper_height, padding_lower_height],
                [padding_righter_width, padding_lefter_width], [0, 0]]

    output_img = tf.pad(shrinked_img, paddings, mode='CONSTANT', name=None, constant_values=0)
    output_msk = tf.pad(shrinked_msk, paddings, mode='CONSTANT', name=None, constant_values=0)

    if tensor_wgt is not None:
        shrinked_wgt = tf.image.resize_image_with_pad(tensor_wgt, target_height=shrinked_height,
                                                      target_width=shrinked_width)
        output_wgt = tf.pad(shrinked_wgt, paddings, mode='CONSTANT', name=None, constant_values=0)
    else:
        output_wgt = tensor_wgt

    return output_img, output_msk, output_wgt


def zoom_out(tensor_img, tensor_msk, original_height, original_width, zooming_factor, verbose=True):
    if verbose:
        print("zooming out factor:", zooming_factor)

    shrinked_height = tf.cast((1 - zooming_factor) * original_height, dtype=tf.int32)
    shrinked_width = tf.cast((1 - zooming_factor) * original_width, dtype=tf.int32)

    shrinked_img = tf.image.resize_image_with_pad(tensor_img, target_height=shrinked_height,
                                                  target_width=shrinked_width)
    shrinked_msk = tf.image.resize_image_with_pad(tensor_msk, target_height=shrinked_height,
                                                  target_width=shrinked_width)

    paddings_width = original_width - shrinked_width
    padding_righter_width = tf.cast(paddings_width / 2, dtype=tf.int32)
    padding_lefter_width = tf.cast(paddings_width / 2, dtype=tf.int32) + tf.cast(paddings_width % 2, dtype=tf.int32)

    paddings_height = original_width - shrinked_width
    padding_upper_height = tf.cast(paddings_height / 2, dtype=tf.int32)
    padding_lower_height = tf.cast(paddings_height / 2, dtype=tf.int32) + tf.cast(paddings_height % 2, dtype=tf.int32)
    paddings = [[padding_upper_height, padding_lower_height],
                [padding_righter_width, padding_lefter_width], [0, 0]]

    output_img = tf.pad(shrinked_img, paddings, mode='CONSTANT', name=None, constant_values=0)
    output_msk = tf.pad(shrinked_msk, paddings, mode='CONSTANT', name=None, constant_values=0)

    return output_img, output_msk


def change_color(x, hue, saturation, brightness, contrast):
    x = tf.image.random_hue(x, hue)
    x = tf.image.random_saturation(x, 1 - saturation, 1 + saturation)
    x = tf.image.random_brightness(x, brightness)
    x = tf.image.random_contrast(x, 1 - contrast, 1 + contrast)

    return x


def data_augmentation_wgt(cfg, tensor_image, tensor_mask, tensor_weights):

    print('Weights before augmentation', tensor_weights.shape)

    # shifting
    tensor_image, tensor_mask, tensor_weights = shift_img(tensor_image, tensor_mask, tensor_weights,
                                                          cfg.WIDTH_SHIFT_RANGE, cfg.HEIGHT_SHIFT_RANGE)

    # flipping
    tensor_image, tensor_mask, tensor_weights = flip_img(tensor_image, tensor_mask, tensor_weights,
                                                         cfg.FLIP_HORIZONTAL, cfg.FLIP_VERTICAL)

    # rotation
    tensor_image, tensor_mask, tensor_weights = rotate_img(tensor_image, tensor_mask, tensor_weights,
                                                           cfg.ROTATION_RANGE)

    # zooming
    tensor_image, tensor_mask, tensor_weights = zoom_img(tensor_image, tensor_mask, tensor_weights,
                                                         cfg.TARGET_IMG_SHAPE[0], cfg.TARGET_IMG_SHAPE[1],
                                                         cfg.ZOOMING_RANGE)

    # color augmentation
    tensor_image = change_color(tensor_image, cfg.COLOR_HUE, cfg.COLOR_SATURATION, cfg.COLOR_BRIGHTNESS, cfg.COLOR_CONTRAST)

    return tensor_image, tensor_mask, tensor_weights


def data_augmentation(cfg, tensor_image, tensor_mask):

    # shifting
    tensor_image, tensor_mask, tensor_weights = shift_img(tensor_image, tensor_mask, None,
                                                          cfg.WIDTH_SHIFT_RANGE, cfg.HEIGHT_SHIFT_RANGE)

    # flipping
    tensor_image, tensor_mask, tensor_weights = flip_img(tensor_image, tensor_mask, None,
                                                         cfg.FLIP_HORIZONTAL, cfg.FLIP_VERTICAL)

    # rotation
    tensor_image, tensor_mask, tensor_weights = rotate_img(tensor_image, tensor_mask, None,
                                                           cfg.ROTATION_RANGE)

    # zooming
    tensor_image, tensor_mask, tensor_weights = zoom_img(tensor_image, tensor_mask, None,
                                                         cfg.TARGET_IMG_SHAPE[0], cfg.TARGET_IMG_SHAPE[1],
                                                         cfg.ZOOMING_RANGE)

    # color augmentation
    tensor_image = change_color(tensor_image, cfg.COLOR_HUE, cfg.COLOR_SATURATION, cfg.COLOR_BRIGHTNESS, cfg.COLOR_CONTRAST)

    return tensor_image, tensor_mask