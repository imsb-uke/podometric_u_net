import os, pandas as pd, numpy as np
from sacred import Ingredient
# from extune.settings import DATA_DIR
# from extune.utils.misc import make_mapping


data_ingredient = Ingredient('dataset')
# _file = os.path.join(DATA_DIR, 'data.csv')

# @data_ingredient.capture
# def load_data(
#     file:str  = _file,
#     save_dir  = None,
#     use_local = False,
#     _config   = None,
#     _run      = None,
#     _log      = None
# ):
#     '''
#     Arguments:
#         file (str): the data file from which to read. Here assumed to be a csv
#
#         save_dir (str): a directory to save processed data along the way. In
#             combination with `use_local` can used locally cached processed data
#             to save time when running the same pre-processing over and over again.
#
#         use_local (bool): whether or not to load from scratch or load from
#             `save_dir`.
#
#         _log (sacred _log): the current logger
#         _run (sacred _run): the current run
#         _config (sacred _config): the current configuration file
#
#     Returns:
#         data (tuple): a tuple containg (paired_data, map_twin_a, map_twin_b),
#             where:
#             - paired_data (list): a list of tuples containing the positive examples
#             - map_twin_a (dict): from `extune.utils.misc.make_mapping`, a mapping
#                 between the the unique elements in paired_data and their index
#                 for the twin_a data.
#             - map_twin_bm (dict): see map_twin_a
#     '''
#     if use_local and save_dir is not None:
#         twins = np.load(os.path.join(save_dir, 'twins.npy'))
#         map_twin_a = np.load(os.path.join(save_dir, '{}.npy'.format(_config["name_twin_a"])))
#         map_twin_b = np.load(os.path.join(save_dir, '{}.npy'.format(_config["name_twin_b"])))
#         return (twins, map_twin_a, map_twin_b)
#
#     df = pd.read_csv(_file)
#     map_twin_a = make_mapping(df['twin_a'].unique())
#     map_twin_b = make_mapping(df['twin_b'].unique())
#     twins = [tuple(pair) for pair in df[['twin_a', 'twin_b']].values]
#
#     if _config is not None and save_dir is not None and _config['save']:
#         if _log is not None: _log.debug('Saving mappings')
#         np.save(os.path.join(save_dir, '{}.npy'.format(_config["name_twin_a"])), map_twin_a)
#         np.save(os.path.join(save_dir, '{}.npy'.format(_config["name_twin_b"])), map_twin_b)
#         np.save(os.path.join(save_dir, 'twins.npy'), twins)
#         if _log is not None: _log.debug('Mappings saved')
#
#     return (twins, map_twin_a, map_twin_b)
