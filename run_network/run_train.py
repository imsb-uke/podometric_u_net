from network.train_network import train_network
from config.config import Config


class trainingConfig(Config):
    NAME = 'unet'

    GPU_SERVER = False  # False

    SEGMENTATION_TASK = 'all'  # 'podocytes' # 'glomerulus'

    # If you want to use a pretrained model enter a path here
    # PRETRAINED = True
    # PRETRAINED_PATH = ""

    # PREPROCESSING
    CONTRAST_STRETCHING = False  # True  #
    HISTOGRAM_EQUALIZATION = False  # True  #

    GPUs = ''
    IMAGES_PER_GPU = 1
    EPOCHS = 2

    UNET_FILTERS = 32  # 16, 64
    UNET_LAYERS = 3  # 3, 4, 5
    DROPOUT_ENC_DEC = False  # True
    DROPOUT_BOTTOM = True  # False
    UNET_SKIP = True  # False
    DROPOUT_SKIP = False  # True
    BATCH_NORM = True  # False

    LOSS = 'BalancedTwoLayerCustomBceLoss'  # 'twolayer_bce'  'twolayer_dice' #'BalancedTwoLayerDiceLoss'
    #                                       #'bce' # 'gdl', 'dice', 'bce_dice', 'keras_bce'
    WEIGHTING = False
    OVERSAMPLING = False  # True

    # First metric in list will be monitored metric
    # To change this modify MONITORED_METRIC here
    METRIC = ['glom_dice_loss', 'podo_dice_loss']

    MASK_FOLDERNAME = 'masks'  # 'filtered_masks'

    MODEL_PATH = '/source/results/unittest/'

    SAVE_WEIGHTS = False


# Dataset paths
data_path = '/data/unittest_data/'


cfg = trainingConfig()

train_network(cfg, data_path)
