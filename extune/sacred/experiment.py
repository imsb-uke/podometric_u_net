import os

# from extune.settings import EXPERIMENTS_DIR, MODULE_DIR
from extune.sacred.utils import config_logger

# from extune.model import model_fn, train_fn, input_fn
from extune.sacred.ingredients import data_ingredient

from network.train_network import train_network
# from network.config import Config

from tensorflow.python.keras.callbacks import Callback

from sacred import Experiment

# from run_hyperband import cfg


# class trainingConfig(Config):
#     # GPUs and IMAGES_PER_GPU needs to be configured here!
#     GPUs = ''
#     IMAGES_PER_GPU = 1
#
#
# cfg = trainingConfig()

# define our sacred experiment (ex) and add our data_ingredient
ex = Experiment('kidney_microscopy', interactive=True, ingredients=[data_ingredient])

# provide the configurable parameters from the JSON config file.
# ex.add_config(os.path.join(MODULE_DIR, 'model', 'config.json'))

@ex.config
def config():
    SEGMENTATION_TASK = 'all' #'podocytes'  #'glomerulus'
    NUM_OUTPUT_CH = 2
    UNET_FILTERS = 16  # 32, 64
    UNET_LAYERS = 4  # 3, 4, 5
    DROPOUT_ENC_DEC = False  # True
    DROPOUT_BOTTOM = False  # True
    UNET_SKIP = True  # False
    DROPOUT_SKIP = False  # True
    BATCH_NORM = True  # False

    LOSS = 'balanced_twolayer_dice'  # 'balanced_twolayer_bce'#'bce'
    METRIC = ['twolayer_dice_loss', 'glom_dice_loss', 'podo_dice_loss']
    MONITORED_METRIC = 0

    LEARNING_RATE = 0.00001
    WEIGHTING = False
    OVERSAMPLING = False
    OPTIMIZER = 'Amsgrad'
    LR_DECAY = 'None'

    TRAINING_DATA_PERCENTAGE = 1.0
    IMG_GENERATOR_SEED = 243

    SAVE_WEIGHTS = True

    EPOCHS = 2

    # If you want to use a pretrained model enter a path here
    PRETRAINED = False
    PRETRAINED_PATH = ""

    MODEL_PATH = '/source/experiments'

    data_path = '/data/Dataset1/tif/'




@ex.capture
def metrics(_run, _config, logs):
    '''
    Arguments:
        _run (sacred _run): the current run
        _config (sacred _config): the current configuration
        logs (keras.logs): the logs from a keras model

    Returns: None
    '''
    _run.log_scalar('loss', float(logs.get('loss')))
    _run.log_scalar('val_loss', float(logs.get('val_loss')))

    _run.log_scalar(_config['METRIC'][_config['MONITORED_METRIC']],
                    float(logs.get(_config['METRIC'][_config['MONITORED_METRIC']])))
    _run.log_scalar('val_'+_config['METRIC'][_config['MONITORED_METRIC']],
                    float(logs.get('val_'+_config['METRIC'][_config['MONITORED_METRIC']])))


class LogMetrics(Callback):
    '''
    A wrapper over the capture method `metrics` to have keras's logs be
    integrated into sacred's log.
    '''
    def on_epoch_end(self, _, logs={}):
        print(logs)
        metrics(logs=logs)


@ex.automain
def main(_log, _run, _config, reporter, cfg):
    '''
    Notes:
        variables starting with _ are automatically passed via sacred due to
        the wrapper.

        I prefer to return, at most, a single value. The returned value will be
        stored in the Observer (file or mongo) and if large weight matricies or
        the model itself, will be very inefficient for storage. Those files
        should be added via 'add_artifact' method.

    Arguments:
        _log (sacred _log): the current logger
        _run (sacred _run): the current run
        _config (sacred _config): the current configuration file
        reporter (keras callback function): the callback function to report to ray tune

    Returns:
        result (float): accuracy if classification, otherwise mean_squared_error
    '''

    cfg.MODEL_PATH = _config['MODEL_PATH']

    cfg.SEGMENTATION_TASK = _config['SEGMENTATION_TASK']
    cfg.NUM_OUTPUT_CH = _config['NUM_OUTPUT_CH']

    cfg.UNET_FILTERS = _config['UNET_FILTERS']
    cfg.UNET_LAYERS = _config['UNET_LAYERS']
    cfg.DROPOUT_ENC_DEC = _config['DROPOUT_ENC_DEC']
    cfg.DROPOUT_BOTTOM = _config['DROPOUT_BOTTOM']
    cfg.UNET_SKIP = _config['UNET_SKIP']
    cfg.DROPOUT_SKIP = _config['DROPOUT_SKIP']
    cfg.BATCH_NORM = _config['BATCH_NORM']

    cfg.LOSS = _config['LOSS']
    cfg.METRIC = _config['METRIC']
    cfg.MONITORED_METRIC = _config['MONITORED_METRIC']

    cfg.LEARNING_RATE = _config['LEARNING_RATE']

    cfg.WEIGHTING = _config['WEIGHTING']
    cfg.OVERSAMPLING = _config['OVERSAMPLING']

    cfg.OPTIMIZER = _config['OPTIMIZER']
    cfg.LR_DECAY = _config['LR_DECAY']

    cfg.TRAINING_DATA_PERCENTAGE = _config['TRAINING_DATA_PERCENTAGE']
    cfg.IMG_GENERATOR_SEED = _config['IMG_GENERATOR_SEED']

    cfg.SAVE_WEIGHTS = _config['SAVE_WEIGHTS']

    cfg.EPOCHS = _config['EPOCHS']

    cfg.PRETRAINED = _config['PRETRAINED']
    cfg.PRETRAINED_PATH = _config['PRETRAINED_PATH']

    # the subdirectory for this particular experiment
    run_dir = os.path.join(cfg.MODEL_PATH, str(_run._id))

    # inform the logger to dump to run_dir
    config_logger(run_dir)

    hist = train_network(cfg, _config['data_path'], LogMetrics(), reporter)

    result = hist.history[_config['METRIC'][_config['MONITORED_METRIC']]][-1]

    return result

