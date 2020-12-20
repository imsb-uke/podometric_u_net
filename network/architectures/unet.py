from tensorflow.python.keras import layers


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    """ Convolutional block with two convolutions followed by batch normalisation (if True) and with ReLU activations.
    input_tensor: A tensor. Input tensor on which the convolutional block acts.
    n_filters: An integer. Number of filters in this block.
    kernel_size: An integer. Size of convolutional kernel.
    batchnorm: A bool. Perform batch normalisation after each convolution if True.
    :return: A tensor. The output of the operation.
    """
    # first convolutional layer
    x = layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
                      padding="same")(input_tensor)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # second convolutional layer
    x = layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
                      padding="same")(x)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def define_unet(input_image, cfg):

    # encoder
    encoder_layers = []
    res_connections = []
    input_img = input_image
    filters = cfg.UNET_FILTERS
    for i in range(cfg.UNET_LAYERS):
        l1 = conv2d_block(input_img, filters, kernel_size=3, batchnorm=cfg.BATCH_NORM)
        res_connections.append(l1)
        l2 = layers.MaxPooling2D((2, 2))(l1)
        l_out = layers.Dropout(cfg.DROPOUT_ENC_DEC * 0.5)(l2)

        input_img = l_out
        filters = filters * 2
        encoder_layers.append(l_out)


    # bottom
    l1 = layers.Dropout(cfg.DROPOUT_BOTTOM * 0.5)(encoder_layers[-1])
    l2 = conv2d_block(l1, filters, kernel_size=3, batchnorm=cfg.BATCH_NORM)
    l_out = layers.Dropout(cfg.DROPOUT_BOTTOM * 0.5)(l2)

    bottom_layer = l_out

    # decoder
    decoder_layers = []
    input_img = bottom_layer
    for i in range(cfg.UNET_LAYERS):
        filters = int(filters / 2)
        l1 = layers.Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same')(input_img)
        if cfg.UNET_SKIP:
            # print(res_connections)
            # print(encoder_layers)
            # print(l1)
            l_res = layers.Dropout(cfg.DROPOUT_SKIP * 0.5)(res_connections[-(i + 1)])
            l1 = layers.concatenate([l1, l_res], axis=-1)
        l2 = layers.Dropout(cfg.DROPOUT_ENC_DEC * 0.5)(l1)
        l_out = conv2d_block(l2, filters, kernel_size=3, batchnorm=cfg.BATCH_NORM)

        input_img = l_out
        decoder_layers.append(l_out)

    # output layer(s)
    # if only one mask is used, cfg.NUM_OUTPUT_CH will be 1
    # For two masks it should be 2
    unet_out = layers.Conv2D(cfg.NUM_OUTPUT_CH, (1, 1), name='outputlayer', activation='sigmoid')(decoder_layers[-1])


    hidden_layers = []
    hidden_layers.extend(encoder_layers)
    hidden_layers.extend([bottom_layer])
    hidden_layers.extend(decoder_layers)
    # print(hidden_layers)
    return hidden_layers, unet_out
#   else:
#       return unet_out
