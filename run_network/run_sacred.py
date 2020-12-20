import logging
import os

from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

from extune.sacred.experiment import ex
from config.config import Config


class trainingConfig(Config):
    # GPUs and IMAGES_PER_GPU needs to be configured here!
    GPUs = '0'
    IMAGES_PER_GPU = 1
    GPU_SERVER = False

    SEGMENTATION_TASK = 'all'
    # First metric in list will be monitored metric
    # To change this modify MONITORED_METRIC here
    METRIC = ['twolayer_dice_loss', 'glom_dice_loss', 'podo_dice_loss']  # 'dice_loss'

    MODEL_PATH = '/source/experiments'


cfg = trainingConfig()
print(cfg.LOSS)

# ensure the directory to which we write exists
if not os.path.isdir(os.path.join(cfg.MODEL_PATH)):
    os.makedirs(os.path.join(cfg.MODEL_PATH))
ex.observers.append(FileStorageObserver.create(os.path.join(cfg.MODEL_PATH)))
ex.capture_out_filter = apply_backspaces_and_linefeeds

# use python logger
ex.logger = logging.getLogger(__name__)

# config['reporter'] = reporter
# ex.run(config_updates=config)


run = ex.run(config_updates={'reporter': None, 'cfg': cfg})

print(run.result)
