import os

import numpy as np
import math
import pandas as pd
import time
import datetime

from tensorflow.python.keras import layers
from tensorflow.python.keras import models

from skimage.io import imsave

import network.image_generator as img_gen
import network.evaluation as eval
from network.architectures.unet import define_unet
from network.architectures.fc_densenet_tiramisu import define_tiramisu

from network.loss import dice_coeff_numpy


def get_stereology_readouts(dataset_dict, mask_part, titles, mode='pred'):
    """
    Compute the stereology using the dataset_dict
    Parameters:
    -----------
    dataset_dict:       dict
        contains all the analysis data

    mask_part:          list
        contains the segmentation tasks
        e.g. ['glomerulus', 'podocytes'], ['glomerulus'], ['podocytes']

    titles:             list
        list containing the image names. Used only for indexing

    mode:               string
        use either the prediction readouts ('pred') or the groundtruth readouts ('gt') for stereology

    Returns:
    --------
    stereology_dict:    dict
        contains the computed stereology data
    """
    stereology_dict = {}

    stereology_dict["glomerular_volume_per_million"] = []
    stereology_dict["podocyte_count"] = []
    stereology_dict["podocyte_density"] = []

    if mode == 'pred':
        # Set the variables
        glom_area = dataset_dict['area_preds_%s' % mask_part[0]]
        podo_nuclear_area = dataset_dict['area_preds_%s' % mask_part[1]]
        podo_count = dataset_dict['count_preds_%s' % mask_part[1]]
    elif mode == 'gt':
        # Set the variables
        glom_area = dataset_dict['area_masks_%s' % mask_part[0]]
        podo_nuclear_area = dataset_dict['area_masks_%s' % mask_part[1]]
        podo_count = dataset_dict['count_masks_%s' % mask_part[1]]
    else:
        raise ValueError("mode has to be either 'pred' or 'gt' but is: ", mode)

    for index, title in enumerate(titles):
        # Calculate the stereology glomerular tuft volume, stereology podocyte count, stereology podocyte density for this image
        # Calculate it
        Nv1 = 1 / 1.55
        try:
            Na = podo_count[index] / glom_area[index]
            Vv = podo_nuclear_area[index] / glom_area[index]
            Nv2 = math.pow(Na, 3)
            Nv3 = Nv2 / Vv
        except ZeroDivisionError:
            print('Zero Devision Error in get_stereology_readouts')
            print("index:", index, "title:", title,
                  "\n glom_area", glom_area[index],
                  "podo_count:", podo_count[index], "podo_nucl_area:", podo_nuclear_area[index])
            print('Nv2 is', Nv2, "; Vv is", Vv)
            print('Nv3 will be set to 0')
            Nv3 = 0
        Nv4 = math.sqrt(Nv3)
        Nv = Nv1 * Nv4

        stereology_V_glom = math.pow(glom_area[index], 1.5) * (1.38 / 1.01)
        stereology_V_glom_per_million = stereology_V_glom / 1000000

        stereology_podocyte_count = Nv * stereology_V_glom
        stereology_podocyte_density = stereology_podocyte_count / stereology_V_glom_per_million

        # Insert it in dictionary
        stereology_dict["glomerular_volume_per_million"].append(stereology_V_glom_per_million)
        stereology_dict["podocyte_count"].append(stereology_podocyte_count)
        stereology_dict["podocyte_density"].append(stereology_podocyte_density)

    """ OLD: REMOVE:
        for index, row in data.iterrows():
            # Calculate the stereology glomerular tuft volume, stereology podocyte count, stereology podocyte density for this image
            # print(row)
            # Calculate it
            Na = row['podocyte_count'] / row['glomerulus_area']
            Vv = row['podocyte_nuclear_area'] / row['glomerulus_area']
            Nv1 = 1 / 1.55
            Nv2 = math.pow(Na, 3)
            try:
                Nv3 = Nv2 / Vv
            except ZeroDivisionError:
                print('Zero Devision Error in get_stereology_readouts')
                print("index:", index, "\nrow:", row)
                print('Nv2 is', Nv2, "; Vv is", Vv)
                print('Nv3 will be set to 0')
                Nv3 = 0
            Nv4 = math.sqrt(Nv3)
            Nv = Nv1 * Nv4

            stereology_V_glom = math.pow(row['glomerulus_area'], 1.5) * (1.38 / 1.01)
            stereology_V_glom_per_million = stereology_V_glom / 1000000

            stereology_podocyte_count = Nv * stereology_V_glom
            stereology_podocyte_density = stereology_podocyte_count / stereology_V_glom_per_million

            # Insert it in dataframe
            data.at[index, 'stereology_glomerular_volume_per_million'] = stereology_V_glom_per_million
            data.at[index, 'stereology_podocyte_count'] = stereology_podocyte_count
            data.at[index, 'stereology_podocyte_density'] = stereology_podocyte_density
            """
    return stereology_dict


def get_signal_features(array):
    """
    Computes the mean, var, median, min, max, perc25, perc75 of an array

    Parameters:
    -----------
    array   array
        Array which is used to calculate the mean, var, median, min, max of it.

    Returns:
    --------
    mean_feature    scalar
        dtype('float64')
    var_feature     scalar
        dtype('float64')
    median_feature  scalar
        dtype('float64')
    min_feature     scalar
        dtype('float64')
    max_feature     scalar
        dtype('float64')
    perc25_feature  scalar
        dtype('float64')
    perc75_feature  scalar
        dtype('float64')
    """
    try:
        mean_feature = np.mean(array)
    except ValueError:
        mean_feature = np.NaN
    try:
        var_feature = np.var(array)
    except ValueError:
        var_feature = np.NaN
    try:
        median_feature = np.median(array)
    except ValueError:
        median_feature = np.NaN
    try:
        min_feature = np.min(array)
    except ValueError:
        min_feature = np.NaN
    try:
        max_feature = np.max(array)
    except ValueError:
        max_feature = np.NaN
    try:
        perc25_feature = np.percentile(array, 25)
    except ValueError:
        perc25_feature = np.NaN
    except IndexError:
        perc25_feature = np.NaN
    try:
        perc75_feature = np.percentile(array, 75)
    except ValueError:
        perc75_feature = np.NaN
    except IndexError:
        perc75_feature = np.NaN

    return mean_feature, var_feature, median_feature, min_feature, max_feature, perc25_feature, perc75_feature