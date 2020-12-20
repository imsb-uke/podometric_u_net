import os
import psutil

import numpy as np

import ray
from ray import tune
from extune.tune.experiment import run_fn

from config.config import Config


class TrainingConfig(Config):
    # GPUs and IMAGES_PER_GPU needs to be configured here!
    GPUs = '0'  # '0'
    IMAGES_PER_GPU = 2  #2 # Since parallel model does not work, BATCH_SIZE is set equal to IMAGES_PER_GPU for any number of GPUs
    GPU_SERVER = True  # False

    SEGMENTATION_TASK = 'all'
    NAME = 'unet_cv0_oversampling'

    # First metric in list will be monitored metric
    # To change this modify MONITORED_METRIC here
    METRIC = ['twolayer_dice_loss', 'twolayer_dice_loss_no_round', 'glom_dice_loss', 'podo_dice_loss']  # 'dice_loss'


cfg = TrainingConfig()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPUs

if cfg.GPU_SERVER:
    max_memory = int(1/8*(psutil.virtual_memory()).total)
    memory = min(80000000000, max_memory)
    num_gpu = 1
else:
    memory = None
    num_gpu = 0

ray.init(ignore_reinit_error=True, num_gpus=num_gpu, object_store_memory=memory)

tune_ex_config = {
    'stop': {'training_iteration': 1000},
    'config': {
        'SEGMENTATION_TASK': tune.grid_search(['all']),  # , 'glomerulus']),  # 'all',  # 'glomeruli'
        'NUM_OUTPUT_CH': 2,
        'UNET_FILTERS': tune.grid_search([32]),  # , 16]),  # 4, 8, 16, 24, 32, 64]),
        'UNET_LAYERS': tune.grid_search([3]),  #, 4]),  # 3, 4, 5]),
        'DROPOUT_ENC_DEC': tune.grid_search([False]),  # , True]),
        'DROPOUT_BOTTOM': tune.grid_search([True]),  # , True]),
        'UNET_SKIP': tune.grid_search([True]),  # , False]),
        'DROPOUT_SKIP': tune.grid_search([False]),  # , True]),
        'BATCH_NORM': tune.grid_search([True]),  # , False]),

        'LOSS': tune.grid_search(['TwoLayerCustomBceLoss']),
                                  # 'balanced_twolayer_dice', 'twolayer_dice'
                                  # 'balanced_twolayer_bce' 'twolayer_bce'
                                  # 'bce', 'gdl', 'dice', 'bce_dice', 'keras_bce'

        'WEIGHTING': tune.grid_search([True]),

        'OVERSAMPLING': tune.grid_search([True, False]),  # True

        'OPTIMIZER': tune.grid_search((['RMSprop'])),  # 'Adam', 'RMSprop', 'Amsgrad'
        'LEARNING_RATE': tune.grid_search(([0.00001])),
        'LR_DECAY': tune.grid_search(['None']),  # , 'Plateau', 'Cosine'

        'TRAINING_DATA_PERCENTAGE': tune.grid_search([1.0]),
        'IMG_GENERATOR_SEED': tune.sample_from(lambda spec: np.random.uniform(1024)),

        'EPOCHS': 1000,

        # If you want to use a pretrained model enter a path here
        'PRETRAINED': False,
        'PRETRAINED_PATH': '',

        'data_path': '/data/Dataset1/tif/cv0',

        'MODEL_PATH': '/source/experiments',

        'SAVE_WEIGHTS': True,  # False

        'cfg': cfg
    },
    'resources_per_trial': {
        'cpu': 8,
        'gpu': num_gpu
    },
    'num_samples': 1
}

run_fn(tune_ex_config)
