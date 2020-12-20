from tensorflow.python.keras import layers


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    """ Convolutional block with convolution followed by batch normalisation (if True) and with ReLU activations.
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
    return x


def define_cnn_ciresan(input_img, n_filters=48, dropout=0.5, batchnorm=True):
    """ Defines CNN of the Ciresan publication. Cropping is not done here.
    # Predicts the pixel in the middle of the input_image
    inputs: A tensor. Input tensor with the input image or batch of images.
    n_filters: An integer. Number of filters to start with.
    dropout: A float. Percentage of dropout.
    batchnorm: A bool. Perform batch normalisation after each convolution if True.
    :return: A tensor. The output of the network.
    """
    # Contracting path
    # Layer 1: 48 filters, out: 512x512
    # first convolutional layer
    c1 = conv2d_block(input_img, n_filters=n_filters, kernel_size=4, batchnorm=batchnorm)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    # Layer 2: 48 filters, out: 256x256
    c2 = conv2d_block(p1, n_filters=n_filters, kernel_size=5, batchnorm=batchnorm)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Layer 3: 48 filters, out: 128x128
    c3 = conv2d_block(p2, n_filters=n_filters, kernel_size=4, batchnorm=batchnorm)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # Layer 4: 48 filters, out: 64x64
    c4 = conv2d_block(p3, n_filters=n_filters, kernel_size=4, batchnorm=batchnorm)
    p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)

    # Layer 5: 48 filters, centre, out: 64x64
    d5 = layers.Dropout(dropout, noise_shape=None, seed=None)(p4)
    c5 = conv2d_block(d5, n_filters=n_filters, kernel_size=3, batchnorm=batchnorm)
    d5 = layers.Dropout(dropout, noise_shape=None, seed=None)(c5)

    # Layer 6: Fully connected
    c6 = layers.Conv2D(200, (1, 1), activation='relu')(c5)

    # Layer 7: Fully connected  out: 1024x1024
    outputs = layers.Conv2D(2, (1, 1), activation='softmax')(c6)

    return outputs
