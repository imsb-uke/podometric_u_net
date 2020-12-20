'''
Unet configuration class
'''
import os


class Config(object):
    NAME = 'Unet_config'

    # Running on a GPU sever? Important for memory allocation from ray tune
    GPU_SERVER = False

    # Which GPUs to use. If GPUs is '', only CPU will be used. GPU count will be calculated via the number of commas
    GPUs = ''

    # How many CPUs to use max
    MAX_CPU_COUNT = 10

    # Number of images to train with on each GPU. A 12GB GPU can typically
    # handle 2 images of 1024x1024px.
    IMAGES_PER_GPU = 2

    # Training steps per epoch is set to the number of training images
    # STEPS_PER_EPOCH = 1000

    # Validation steps per epoch is set to the number of validation images
    # VALIDATION_STEPS = 50

    # Specify architecture to do image segmentation
    ARCHITECTURE = 'unet'  # 'fc_densenet_tiramisu56', 'fc_densenet_tiramisu67', 'fc_densenet_tiramisu103'

    # Unet hyperparameters
    UNET_FILTERS = 16  # 32, 64
    UNET_LAYERS = 4  # 3, 4, 5
    DROPOUT_ENC_DEC = False  # True
    DROPOUT_BOTTOM = False  # True
    UNET_SKIP = True  # False
    DROPOUT_SKIP = False  # True
    BATCH_NORM = True  # False

    # Tiramisu hyperparameters
    TIRAMISU_DROPOUT = True # False

    # Which loss to use
    LOSS = 'balanced_twolayer_dice'  # 'balanced_twolayer_bce'  # 'twolayer_bce' #'bce' # 'gdl', 'dice', 'bce_dice', 'keras_bce'
    METRIC = ['twolayer_dice_loss']  # 'glom_dice_loss', 'podo_dice_loss' #'dice_loss', 'glom_dice_loss', 'podo_dice_loss'
    MONITORED_METRIC = 0
    WEIGHTING = True  # True
    WEIGHT_ZERO_BG = False  # True

    # If oversampling is True, images with the addtional tag 'crescent' or 'lesion'
    # will be sampled as often as all the untagged images in training mode.
    # Therefore the image list will be enlarged so that untagged images are as often in there as untagged images.
    #
    # The number of images per epoch (=steps_per_epoch) won't change
    # steps_per_epoch images will be taken out of the shuffeled oversampled list.
    OVERSAMPLING = False  # True
    OVERSAMPLING_TAGS = ['crescent', 'lesion']

    # How many images of the training set to use
    TRAINING_DATA_PERCENTAGE = 1 # 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
    IMG_GENERATOR_SEED = 243

    # Which image shape will be given as input to the network
    TARGET_IMG_SHAPE = (1024, 1024, 3)

    # Number of mask channels per mask image. Other values than 1 have not been tried out yet!
    # Attention: Number of masks being loaded is specified with SEGMENTATION_TASK!
    NUMBER_MSK_CHANNELS = 1

    # PREPROCESSING:
    RESCALE = 255  # 4096 #255 #4096 #255 #min_max
    RESCALE_MSK = 255

    CONTRAST_STRETCHING = False  # True  #
    HISTOGRAM_EQUALIZATION = False  # True  #

    # Number of epochs
    EPOCHS = 2

    # Other hyperparameters (e.g. for regularization)
    OPTIMIZER = 'Adam'
    LEARNING_RATE = 0.00001
    BETA_1 = 0.9
    BETA_2 = 0.999

    # Hyperparameters for decaying learning rate: None, Plateau, Cosine
    LR_DECAY = 'None'

    # Use a pretrained model
    PRETRAINED = False  # True
    PRETRAINED_PATH = ""

    # For ray tune, do not save weights (will quickly become too big)
    SAVE_WEIGHTS = True

    ###################
    # Dataset config
    ###################
    SEGMENTATION_TASK = 'all'  # 'glomerulus'  # 'podocytes'
    # Names of classes
    # Must match the SEGMENTATION_TASK
    IMAGE_FORMAT = '.tif'
    NAMES_CLASSES = ['glomerulus', 'podocytes']
    MASK_SUFFIXES = ['_mask_glom.tif', '_mask_podo.tif']
    WEIGHTS_SUFFIXES = ['_mask_glomweights.npy', '_mask_podoweights.npy']

    MASK_FOLDERNAME = 'filtered_masks'  # 'masks'

    # Shuffle images in generator for training images:
    # For optimal results this should be greater or equal to the number of images
    # In this project it is set to the number of images in tf_image_generator

    # Augmentation
    AUGM_DATA_FOLDERNAME = 'train_augmented'

    ## Use augm in folder
    USE_PRE_AUGM_FOLDER = False

    ## Use augm on the fly
    USE_AUGM_ON_THE_FLY = True


    ##################
    # Performance optimisation
    # See more info here: https://www.tensorflow.org/alpha/guide/data_performance
    ##################
    DATA_AUG_PARALLEL = 2
    PREFETCH_BUFFER_SIZE = 2
    # tf.data.experimental.AUTOTUNE) could be used as value but leads to freezing of computer

    ##################
    ## data augm param
    ##################
    WIDTH_SHIFT_RANGE = 0.2
    HEIGHT_SHIFT_RANGE = 0.2
    FLIP_HORIZONTAL = True
    FLIP_VERTICAL = True
    ROTATION_RANGE = 45
    ZOOMING_RANGE = 0.1 # values in [0,1)!

    # Random color transformations are applied according to the values here
    COLOR_HUE = 0.08  # adjustes hue randomly in interval [-x, x]
    COLOR_SATURATION = 0.4  # adjustes saturation randomly in interval [-x, x]
    COLOR_BRIGHTNESS = 0.05  # adjustes brightness randomly in interval [-x, x]
    COLOR_CONTRAST = 0.3  # adjustes contrast randomly in interval [-x, x]

    #################
    # POSTPROCESSING
    #################

    # Filter the masks
    # if UNIT_MICRONS == TRUE filter values are converted to pixel values
    # else filter values are assumed to be already pixel values

    # A glomerulus has on our images in average an area of ~ 25000 microns
    # The smalles glomerulus on HH has ~ 11 000 microns?
    # The smalles glomerulus on ANCA has ~ 4 000 microns
    # Divide it by 5
    # -> use micron area of 800 as a filter

    # A podocyte has for our images in average ~ 5 microns in diameter
    # wikipedia.de: For mammals a cell nucleus has  5 - 16 microns
    # That means the area should be about
    # r=2.5, r^2 * pi ~= 18 microns
    #
    # Divide it by 5
    # -> Use micron area of 3 as filter size

    FILTER_CLASSES = [800, 3]

    PREDICTION_THRESHOLD = 0.5
    # DO_NOT_THRESHOLD only works in unsupervised mode, no analysis will be performed
    DO_NOT_THRESHOLD = False  # True

    # If true only largest glom is kept
    GLOM_POSTPROCESSING_KEEP_ONLY_LARGEST = False

    # If true podos with no contact to a segmented glom are excluded
    # Glomerulus mask must be loaded on position 0 (podocyte mask is on position 1)
    PODO_POSTPROCESSING_WITH_GLOM = False


    ##################
    # SAVE RESULTS
    ##################
    MODEL_PATH = '/source/results/unet/'
    SAVE_IMAGE_FORMAT = '.png'

    SUPERVISED_MODE = True  # False  #
    SAVE_IMAGES = True
    UNIT_MICRONS = True
    GET_object_dc_TP_FN_FP = True

    ##################
    # READOUTS
    ##################
    GET_WT1_SIGNAL_FOR_GLOMERULUS = True  # False
    GET_DACH1_SIGNAL_FOR_PODOCYTES = True  # False

    DO_STEREOLOGY_PRED = False  # True  #
    DO_STEREOLOGY_GT = False  # True  #

    # Is set to true in visualize infered data
    VISUALIZATION_MODE = False
    VISUALIZE_ACTIVATIONS = False

    def __init__(self):
        """Compute some attributes out of the attributes above"""

        # Since parallel model does not work, set BATCH_SIZE = IMAGES_PER_GPU
        # if self.GPUs == '':
        #    self.GPU_COUNT = 0
        #    self.BATCH_SIZE = self.IMAGES_PER_GPU
        # else:
        #    self.GPU_COUNT = self.GPUs.count(",") + 1
        #    self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT
        self.BATCH_SIZE = self.IMAGES_PER_GPU

        if self.SEGMENTATION_TASK == 'all':
            self.NUM_OUTPUT_CH = len(self.NAMES_CLASSES)
        else:
            self.NUM_OUTPUT_CH = 1

        if not self.GPU_SERVER and self.MAX_CPU_COUNT > 8:
            self.MAX_CPU_COUNT = 8

    def display(self):
        """Display configuration"""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

    def write_cfg_to_log(self, path):
        """Log configuration"""
        write_log = open(str(os.path.join(path, 'cfg_log.txt')), "w")
        write_log.write("Configurations:\n")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                write_log.write("{:30} {}\n".format(a, getattr(self, a)))
        write_log.write("\n")
        write_log.close()
