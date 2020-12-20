import os

import matplotlib.pyplot as plt
import time
import datetime

import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.utils import multi_gpu_model

import network.dataset.tf_image_augementation
from network.architectures.unet import define_unet
from network.architectures.fc_densenet_tiramisu import define_tiramisu
from config.config import Config
import network.tf_image_generator as img_gen
from network.loss import set_loss, set_metric
from network.dataset.image_loading import get_file_count

from extune.tune.utils import TuneKerasCallback


def train_network(cfg, data_path, callback=None, reporter=None):
    # Start the main script
    print(tf.VERSION)

    start_time = time.time()

    # Load dataset
    # Data directory and specific directories for training and validation
    print(data_path)

    # ToDo: Implement loading of multiple data_paths
    # data_path_name = ""
    # for path in data_path:
    #    pth = os.path.split(path)[0]
    #    pt = os.path.split(pth)[0]
    #    t = os.path.split(pt)[1]
    #    data_path_name += t + "_"

    # Select the training folder:
    train_dir = "train"

    if cfg.USE_PRE_AUGM_FOLDER and cfg.USE_AUGM_ON_THE_FLY:
        print("ERROR. You have to choose to use either augm data folder OR augm on the fly")
        print("train_network will terminate.")
        return

    if cfg.USE_PRE_AUGM_FOLDER:
        train_dir = cfg.AUGM_DATA_FOLDERNAME

    # Training dataset
    dir_train = os.path.join(data_path, "train")
    train_gen = img_gen.ImageGenerator()
    train_gen.set_attributes(cfg, dir_train, 'training')

    print('No msk channels', cfg.NUMBER_MSK_CHANNELS)
    print('No output channels', cfg.NUM_OUTPUT_CH)

    if cfg.WEIGHTING:
        train_dataset = tf.data.Dataset.from_generator(
            train_gen.generator,
            (tf.float32, tf.float32, tf.float32),
            (cfg.TARGET_IMG_SHAPE,
             (cfg.TARGET_IMG_SHAPE[0], cfg.TARGET_IMG_SHAPE[1], cfg.NUMBER_MSK_CHANNELS * cfg.NUM_OUTPUT_CH),
             (cfg.TARGET_IMG_SHAPE[0], cfg.TARGET_IMG_SHAPE[1], cfg.NUMBER_MSK_CHANNELS * cfg.NUM_OUTPUT_CH)))
    else:
        train_dataset = tf.data.Dataset.from_generator(
            train_gen.generator,
            (tf.float32, tf.float32),
            (cfg.TARGET_IMG_SHAPE,
             (cfg.TARGET_IMG_SHAPE[0], cfg.TARGET_IMG_SHAPE[1], cfg.NUMBER_MSK_CHANNELS * cfg.NUM_OUTPUT_CH)))

    train_dataset = train_dataset.shuffle(train_gen.get_number_of_data())
    train_dataset = train_dataset.repeat()  # Repeat the input indefinitely.

    # Augment train_dataset if wished
    if cfg.USE_AUGM_ON_THE_FLY:
        if cfg.WEIGHTING:
            train_dataset = train_dataset.map(lambda x, y, z:
                                              network.dataset.tf_image_augementation.data_augmentation_wgt(cfg,
                                                                                                           x,
                                                                                                           y,
                                                                                                           z),
                                              num_parallel_calls=cfg.DATA_AUG_PARALLEL)
        else:
            train_dataset = train_dataset.map(lambda x, y: network.dataset.tf_image_augementation.data_augmentation(cfg,
                                                                                                                    x,
                                                                                                                    y),
                                              num_parallel_calls=cfg.DATA_AUG_PARALLEL)

    # Reshape output and weights if weighting is used
    if cfg.WEIGHTING:
        train_dataset = train_dataset.map(lambda x, y, z: img_gen.reshape_for_weighting(cfg, x, y, z),
                                          num_parallel_calls=cfg.DATA_AUG_PARALLEL)

    train_dataset = train_dataset.batch(cfg.BATCH_SIZE)
    train_dataset = train_dataset.prefetch(buffer_size=cfg.PREFETCH_BUFFER_SIZE)

    # Validation dataset
    # Here no shuffle is applied to the images, since this is not necessary
    dir_valid = os.path.join(data_path, "valid")
    valid_gen = img_gen.ImageGenerator()
    valid_gen.set_attributes(cfg, dir_valid, 'validation')

    if cfg.WEIGHTING:
        valid_dataset = tf.data.Dataset.from_generator(
            valid_gen.generator,
            (tf.float32, tf.float32, tf.float32),
            (cfg.TARGET_IMG_SHAPE,
             (cfg.TARGET_IMG_SHAPE[0], cfg.TARGET_IMG_SHAPE[1], cfg.NUMBER_MSK_CHANNELS * cfg.NUM_OUTPUT_CH),
             (cfg.TARGET_IMG_SHAPE[0], cfg.TARGET_IMG_SHAPE[1], cfg.NUMBER_MSK_CHANNELS * cfg.NUM_OUTPUT_CH)))
    else:
        valid_dataset = tf.data.Dataset.from_generator(
            valid_gen.generator,
            (tf.float32, tf.float32),
            (cfg.TARGET_IMG_SHAPE,
             (cfg.TARGET_IMG_SHAPE[0], cfg.TARGET_IMG_SHAPE[1], cfg.NUMBER_MSK_CHANNELS * cfg.NUM_OUTPUT_CH)))

    # Reshape output and weights if weighting is used
    if cfg.WEIGHTING:
        valid_dataset = valid_dataset.map(lambda x, y, z: img_gen.reshape_for_weighting(cfg, x, y, z),
                                          num_parallel_calls=cfg.DATA_AUG_PARALLEL)

    valid_dataset = valid_dataset.repeat()  # Repeat the input indefinitely.
    valid_dataset = valid_dataset.batch(cfg.BATCH_SIZE)

    # Create generic iterator
    # data_iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

    # Initialization options
    # train_init_op = data_iter.make_initializer(train_dataset)
    # valid_init_op = data_iter.make_initializer(valid_dataset)

    steps_per_epoch = int(cfg.TRAINING_DATA_PERCENTAGE *
                          (get_file_count([os.path.join(dir_train, "images")], cfg.IMAGE_FORMAT)) / cfg.BATCH_SIZE) + \
                      ((cfg.TRAINING_DATA_PERCENTAGE * get_file_count([os.path.join(dir_train, "images")],
                                                                      cfg.IMAGE_FORMAT))
                       % cfg.BATCH_SIZE > 0)
    validation_steps = int((get_file_count([os.path.join(dir_valid, "images")], cfg.IMAGE_FORMAT)) / cfg.BATCH_SIZE) + \
                       ((get_file_count([os.path.join(dir_valid, "images")], cfg.IMAGE_FORMAT)) % cfg.BATCH_SIZE > 0)

    print("steps_per_epoch: ", steps_per_epoch, "validation_steps: ", validation_steps)

    # loss is set below

    # set the metric
    model_metric = set_metric(cfg.METRIC)
    print("model metric:", model_metric)

    # Define the network model
    inputs = layers.Input(cfg.TARGET_IMG_SHAPE)
    outputs = inputs  # Initialise to be able to use it in if-clause

    if cfg.ARCHITECTURE == 'unet':
        hidden_layers, outputs = define_unet(inputs, cfg)

    elif cfg.ARCHITECTURE == 'fc_densenet_tiramisu56':
        if cfg.TIRAMISU_DROPOUT:
            outputs = define_tiramisu(inputs, n_layers_per_block=[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], growth_rate=12,
                                      dropout_p=0.2)
        else:
            outputs = define_tiramisu(inputs, n_layers_per_block=[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], growth_rate=12,
                                      dropout_p=0)
    elif cfg.ARCHITECTURE == 'fc_densenet_tiramisu67':
        if cfg.TIRAMISU_DROPOUT:
            outputs = define_tiramisu(inputs, n_layers_per_block=[5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], growth_rate=16,
                                      dropout_p=0.2)
        else:
            outputs = define_tiramisu(inputs, n_layers_per_block=[5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], growth_rate=16,
                                      dropout_p=0)
    elif cfg.ARCHITECTURE == 'fc_densenet_tiramisu103':
        if cfg.TIRAMISU_DROPOUT:
            outputs = define_tiramisu(inputs, n_layers_per_block=[4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4], growth_rate=16,
                                      dropout_p=0.2)
        else:
            outputs = define_tiramisu(inputs, n_layers_per_block=[4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4], growth_rate=16,
                                      dropout_p=0)
    else:
        exit("ERROR: No network model type defined")

    # Add an extra input when using loss weighting
    if cfg.WEIGHTING:
        sample_weights = layers.Input((cfg.TARGET_IMG_SHAPE[0], cfg.TARGET_IMG_SHAPE[1],
                                       cfg.NUMBER_MSK_CHANNELS * cfg.NUM_OUTPUT_CH))

        model = models.Model(inputs=(inputs, sample_weights), outputs=[outputs])
    else:
        sample_weights = None
        model = models.Model(inputs=[inputs], outputs=[outputs])

    if cfg.PRETRAINED:
        # Load the weights of the trained model
        load_model = os.path.join(cfg.MODEL_PATH, cfg.PRETRAINED_PATH)

        model.load_weights(load_model)
        # for layer in model.layers[:-1]:
        #    layer.trainable = False

    # If using multiple GPUs, start multi-GPU training
    try:
        parallel_model = multi_gpu_model(model, gpus=cfg.GPU_COUNT)
        print("Training using multiple GPUs.")
    except:
        parallel_model = model
        print("Training using single GPU or CPU.")

    if cfg.LR_DECAY == 'Cosine':
        learning_rate = tf.train.cosine_decay_restarts(
            cfg.LEARNING_RATE,
            tf.train.get_or_create_global_step(),
            first_decay_steps=250,
            m_mul=0.8,
            alpha=cfg.LEARNING_RATE / 10,
            t_mul=2.0)
    else:
        learning_rate = cfg.LEARNING_RATE

    if cfg.OPTIMIZER == 'Adam':
        optim = optimizers.Adam(lr=learning_rate, beta_1=cfg.BETA_1, beta_2=cfg.BETA_2, epsilon=None, amsgrad=False)
    elif cfg.OPTIMIZER == 'RMSprop':
        optim = optimizers.RMSprop(lr=learning_rate)
    elif cfg.OPTIMIZER == 'Amsgrad':
        optim = optimizers.Adam(lr=learning_rate, beta_1=cfg.BETA_1, beta_2=cfg.BETA_2, epsilon=None, amsgrad=True)
    else:
        raise ValueError('Optimiser not recognised. Please choose one among: Adam, RMSprop, Amsgrad.')

    # Include weights for loss weighting if True
    if cfg.WEIGHTING:
        model_loss = set_loss(cfg.LOSS, sample_weights=sample_weights)
        print("config loss:", cfg.LOSS)
        print("model loss:", model_loss)
    else:
        model_loss = set_loss(cfg.LOSS)
        print("config loss:", cfg.LOSS)
        print("model loss:", model_loss)

    # Compile the model twice, to be able to save it later (parallel_model only contains a layer `model`)
    model.compile(optimizer=optim, loss=model_loss, metrics=model_metric)

    parallel_model.compile(optimizer=optim, loss=model_loss, metrics=model_metric)

    parallel_model.summary()

    now = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    save_path = str(os.path.join(cfg.MODEL_PATH, cfg.NAME + '_' + now))
    try:
        os.makedirs(save_path)
    except OSError:
        print("Creation of the directory %s failed." % save_path)
        print("Maybe there is already an existing directory.")
    else:
        print("Successfully created the directory %s ." % save_path)
    cfg.write_cfg_to_log(save_path)

    save_model_path = os.path.join(save_path, 'best_model.hdf5')

    # Create callbacks to save history
    tbCallBack = tf.keras.callbacks.TensorBoard(log_dir=save_path, histogram_freq=0, write_graph=True,
                                                write_images=False)
    if cfg.SAVE_WEIGHTS:
        cp = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path,
                                                monitor='val_' + cfg.METRIC[cfg.MONITORED_METRIC],
                                                save_best_only=True, verbose=1)
    else:
        cp = tf.keras.callbacks.History()

    if cfg.LR_DECAY == 'Plateau':
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_' + cfg.METRIC[cfg.MONITORED_METRIC], factor=0.2,
                                                         patience=5, min_lr=cfg.LEARNING_RATE / 10, verbose=1)
    else:
        reduce_lr = None

    # Train the model with an image generator
    # run_train has no callback or reporter
    if callback is None or reporter is None:
        history = parallel_model.fit(train_dataset.make_one_shot_iterator(),  # train_generator_generator
                                     steps_per_epoch=steps_per_epoch,
                                     epochs=cfg.EPOCHS,
                                     validation_data=valid_dataset.make_one_shot_iterator(),
                                     # valid_generator_generator,
                                     validation_steps=validation_steps,
                                     callbacks=[cp, tbCallBack])

    # run_tune and run_hyperband use callback and reporter
    elif reduce_lr is None:
        history = parallel_model.fit(train_dataset.make_one_shot_iterator(),  # train_generator_generator
                                     steps_per_epoch=steps_per_epoch,
                                     epochs=cfg.EPOCHS,
                                     validation_data=valid_dataset.make_one_shot_iterator(),
                                     # valid_generator_generator,
                                     validation_steps=validation_steps,
                                     callbacks=[callback, cp, tbCallBack, TuneKerasCallback(reporter, cfg)])
    # run_tune and run_hyperband use callback and reporter and reduced learning rate
    else:
        history = parallel_model.fit(train_dataset.make_one_shot_iterator(),  # train_generator_generator
                                     steps_per_epoch=steps_per_epoch,
                                     epochs=cfg.EPOCHS,
                                     validation_data=valid_dataset.make_one_shot_iterator(),
                                     # valid_generator_generator,
                                     validation_steps=validation_steps,
                                     callbacks=[callback, cp, tbCallBack, TuneKerasCallback(reporter, cfg), reduce_lr])

    #####
    # ToDo: Ray Tune seems to ignore the following lines here

    # Save final model
    if cfg.SAVE_WEIGHTS:
        save_model_path = os.path.join(save_path, 'final_model.hdf5')
        model.save(save_model_path)

    # Plot train and validation loss
    metric = history.history[cfg.METRIC[cfg.MONITORED_METRIC]]
    val_metric = history.history['val_' + cfg.METRIC[cfg.MONITORED_METRIC]]

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(cfg.EPOCHS)

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, metric, label='Training ' + cfg.METRIC[cfg.MONITORED_METRIC])
    plt.plot(epochs_range, val_metric, label='Validation ' + cfg.METRIC[cfg.MONITORED_METRIC])
    plt.legend(loc='upper right')
    plt.title('Training and Validation Metric')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training ' + cfg.LOSS)
    plt.plot(epochs_range, val_loss, label='Validation ' + cfg.LOSS)
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.savefig(str(save_path + '/training_curve.png'))

    duration = time.time() - start_time
    print("--- %s seconds ---" % duration)
    duration_std = int(duration / 3600)
    duration_min = int((duration % 3600) / 60)
    duration_sec = int(duration % 60)
    write_txt = open(str(save_path + '/time.txt'), "w")
    write_txt.write(str("Training time: ") + str(duration_std) + "h " + str(duration_min)
                    + "min " + str(duration_sec) + 'sec \n')
    write_txt.close()

    return history


if __name__ == '__main__':
    """
    Test the network and in particularly the loss function
    """
    print("train_network.py")

    from tensorflow import keras

    # Compare dice_loss with seperate dice_loss
    class NetwConfig(Config):
        GPUs = '0'
        IMAGES_PER_GPU = 1
        # BATCH_SIZE = x  # BATCH_SIZE is set to IMAGES_PER_GPU in the config

        SEGMENTATION_TASK = 'all'  # 'all'  #'glomerulus'  # 'podocytes'
        # NAMES_CLASSES = ['rectangles_a', 'rectangles_b']
        # MASK_SUFFIXES = ['_a.png', '_b.png']
        # WEIGHTS_SUFFIXES = ['_weight_a.npy', '_weight_b.npy']

        LOSS = 'TwoLayerCustomBceLoss'  # 'DiceLoss'  #'TwoLayerDiceLoss'  #'BalancedTwoLayerDiceLoss'
        # 'CustomBceLoss'  #'TwoLayerCustomBceLoss'  #'BalancedTwoLayerCustomBceLoss'
        METRIC = ['twolayer_dice_loss', 'twolayer_dice_loss_no_round', 'glom_dice_loss', 'podo_dice_loss']
        # ['twolayer_dice_loss', 'twolayer_dice_loss_no_round', 'dice_loss',
        # 'dice_loss_no_round', 'glom_dice_loss', 'podo_dice_loss']
        WEIGHTING = True  # False
        OVERSAMPLING = True  # False


    cfg = NetwConfig()
    data_path = '/data/Dataset1/tif/'

    ###########################################
    # Load validation dataset with generator  #
    ###########################################
    dir_valid = os.path.join(data_path, "valid")
    valid_gen = img_gen.ImageGenerator()
    valid_gen.set_attributes(cfg, dir_valid, 'validation')

    if cfg.WEIGHTING:
        valid_dataset = tf.data.Dataset.from_generator(
            valid_gen.generator,
            (tf.float32, tf.float32, tf.float32),
            (cfg.TARGET_IMG_SHAPE,
             (cfg.TARGET_IMG_SHAPE[0], cfg.TARGET_IMG_SHAPE[1], cfg.NUMBER_MSK_CHANNELS * cfg.NUM_OUTPUT_CH),
             (cfg.TARGET_IMG_SHAPE[0], cfg.TARGET_IMG_SHAPE[1], cfg.NUMBER_MSK_CHANNELS * cfg.NUM_OUTPUT_CH)))
    else:
        valid_dataset = tf.data.Dataset.from_generator(
            valid_gen.generator,
            (tf.float32, tf.float32),
            (cfg.TARGET_IMG_SHAPE,
             (cfg.TARGET_IMG_SHAPE[0], cfg.TARGET_IMG_SHAPE[1], cfg.NUMBER_MSK_CHANNELS * cfg.NUM_OUTPUT_CH)))

    # Reshape output and weights if weighting is used
    if cfg.WEIGHTING:
        valid_dataset = valid_dataset.map(lambda x, y, z: img_gen.reshape_for_weighting(cfg, x, y, z),
                                          num_parallel_calls=cfg.DATA_AUG_PARALLEL)

    valid_dataset = valid_dataset.repeat()  # Repeat the input indefinitely.
    valid_dataset = valid_dataset.batch(cfg.BATCH_SIZE)

    #####################################
    # Create simple tf model            #
    #####################################
    # Do not do backpropagation to evaulate the loss
    inputs = layers.Input(shape=cfg.TARGET_IMG_SHAPE)
    hid_lay = layers.Conv2D(filters=1, kernel_size=(3, 3), kernel_initializer="Ones",
                            padding="same", trainable=False)(inputs)
    outputs = layers.Conv2D(cfg.NUM_OUTPUT_CH, (1, 1), name='outputlayer',
                            kernel_initializer="Ones", activation="sigmoid", trainable=False)(hid_lay)

    # Add an extra input when using loss weighting
    if cfg.WEIGHTING:
        sample_weights = layers.Input((cfg.TARGET_IMG_SHAPE[0], cfg.TARGET_IMG_SHAPE[1],
                                       cfg.NUMBER_MSK_CHANNELS * cfg.NUM_OUTPUT_CH))
        model = models.Model(inputs=(inputs, sample_weights), outputs=[outputs])
    else:
        sample_weights = None
        model = models.Model(inputs=[inputs], outputs=[outputs])

    #####################################
    # Set loss and metric               #
    #####################################
    # Include weights for loss weighting if True
    if cfg.WEIGHTING:
        model_loss = set_loss(cfg.LOSS, sample_weights=sample_weights)
        print("config loss:", cfg.LOSS)
        print("model loss:", model_loss)
    else:
        model_loss = set_loss(cfg.LOSS)
        print("config loss:", cfg.LOSS)
        print("model loss:", model_loss)
    model_metric = set_metric(cfg.METRIC)

    #####################################
    # Compile keras model               #
    #####################################
    optim = optimizers.Adam(lr=cfg.LEARNING_RATE, beta_1=cfg.BETA_1, beta_2=cfg.BETA_2, epsilon=None, amsgrad=True)
    # use keras.model.evaluate
    model.compile(optimizer=optim, loss=model_loss,
                  metrics=model_metric)

    # Load the weights of the trained model
    # model.load_weights(load_model)

    #####################################
    # Create Callbacks                  #
    #####################################
    cp = tf.keras.callbacks.History()
    tbCallBack = tf.keras.callbacks.TensorBoard(log_dir=cfg.MODEL_PATH,
                                                histogram_freq=0, write_graph=True,
                                                write_images=False)

    #####################################
    # "Train" the model                 #
    #####################################
    history = model.fit(valid_dataset.make_one_shot_iterator(),  # train_generator_generator
                        steps_per_epoch=4,
                        epochs=3,
                        validation_data=valid_dataset.make_one_shot_iterator(),
                        # valid_generator_generator,
                        validation_steps=4,
                        callbacks=[cp, tbCallBack])
