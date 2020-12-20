from network.test_network import test_network
from config.config import Config


class testConfig(Config):
    ##########################################
    # Change cfg attributes for test_network
    GPUs = ''
    IMAGES_PER_GPU = 1

    # Network
    SEGMENTATION_TASK = 'all'  # Options: 'all', 'podocytes', 'glomerulus'

    # Preprocessing
    CONTRAST_STRETCHING = False  # True
    HISTOGRAM_EQUALIZATION = False  # True

    # Network architecture: should match the trained network
    UNET_FILTERS = 32  # 16  #
    UNET_LAYERS = 3  # 4  #

    DROPOUT_ENC_DEC = False
    DROPOUT_BOTTOM = True  # False  #
    UNET_SKIP = True
    DROPOUT_SKIP = False
    BATCH_NORM = True

    LOSS = 'TwoLayerCustomBceLoss'  # 'bce'  #
    METRIC = ['twolayer_dice_loss', 'glom_dice_loss', 'podo_dice_loss']  # ['dice_loss']  #

    OPTIMIZER = 'RMSprop'  # 'Adam'  #
    LEARNING_RATE = 0.00001  # 0.0001  #
    BETA_1 = 0.9
    BETA_2 = 0.999

    LR_DECAY = 'None'

    MASK_FOLDERNAME = 'masks'  # 'masks'

    # Postprocessing
    FILTER_CLASSES = [800, 3]  # [0, 0]  #

    PREDICTION_THRESHOLD = 0.5

    DO_NOT_THRESHOLD = False
    SAVE_IMAGE_FORMAT = '.tif'  # '.tif'

    UNIT_MICRONS = False

    GLOM_POSTPROCESSING_KEEP_ONLY_LARGEST = True  # False  #
    PODO_POSTPROCESSING_WITH_GLOM = True  # False  #

    # Options
    SUPERVISED_MODE = True  # If no masks available, set False: then no Dice scores will be calculated
    SAVE_IMAGES = True  # Saves images and prediction masks
    GET_object_dc_TP_FN_FP = True  # False  #

    GET_WT1_SIGNAL_FOR_GLOMERULUS = True  # Calculate the median and var WT1 signal inside the predicted glomerulus
    GET_DACH1_SIGNAL_FOR_PODOCYTES = True  # Calculate the median and var DACH1 signal inside the predicted podocytes

    DO_STEREOLOGY_PRED = True  # Calculate stereology on predicitions
    DO_STEREOLOGY_GT = True  # Calculate stereology on ground truth

    ##########################################


# Dataset paths
data_path = ['/data/unittest_data/']

# Enter the path to the trained model here
load_model = '/data/unittest_data/best_model_hh.hdf5'

cfg = testConfig()

# Uncomment to run the network with 99 different threshold values.
# In test_network the foldername should be adjusted as well to have the threshold in the foldername!
# Uncomment therefore a line there.
# for i in range(1, 100):
#    cfg.PREDICTION_THRESHOLD = i / 100
#    test_network(cfg, data_path, load_model)

test_network(cfg, data_path, load_model)

