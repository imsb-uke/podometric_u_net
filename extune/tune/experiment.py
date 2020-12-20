import os
import sys
import logging

import ray
from ray import tune
from ray.tune import run

from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
from extune.sacred.experiment import ex


# wrapper function over sacred experiment
def trainable(config, reporter):
    '''
    Arguments:
        config (dict): Parameters provided from the search algorithm
            or variant generation.
        reporter (Reporter): Handle to report intermediate metrics to Tune.
    '''

    # ensure the directory to which we write exists
    if not os.path.isdir(config['MODEL_PATH']):
        os.makedirs(config['MODEL_PATH'])

    ex.observers.append(FileStorageObserver.create(config['MODEL_PATH']))
    ex.capture_out_filter = apply_backspaces_and_linefeeds

    # use python logger
    ex.logger = logging.getLogger(__name__)

    config['reporter'] = reporter
    ex.run(config_updates=config)


def run_fn(ex_config: dict, scheduler=None, algo=None) -> None:
    '''
    Arguments:
        ex_config (dict): the tune experiment configuration dictionary with all
        values to be specified except `'run'`, which is set to be the trainable
        wrapper over the sacred experiment.
    '''
    tune.register_trainable('trainable', trainable)
    if scheduler is not None and algo is not None:
        tune.run_experiments({
            'kidney_microscopy': {
                **ex_config,
                'run': 'trainable',
                'local_dir': os.path.join(ex_config['config']['MODEL_PATH'], 'ray_results'),
            },
        },
            search_alg=algo,
            scheduler=scheduler)
    else:
        tune.run_experiments({
            'kidney_microscopy': {
                **ex_config,
                'run': 'trainable',
                'local_dir': os.path.join(ex_config['config']['MODEL_PATH'], 'ray_results'),
            },
        })

