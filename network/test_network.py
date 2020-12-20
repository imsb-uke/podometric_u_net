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
from network.dataset.image_loading import get_file_count


def print_full(x):
    pd.set_option('display.max_rows', len(x))
    pd.set_option('display.max_columns', 10)
    print(x)
    pd.reset_option('display.max_rows')
    return


def write_analysis(path, dataset_dict, datasettype, mask_part, start_time, supervised=True):
    """
    Creates a text file which contains a short summary of the dataset_dict data

    Parameters:
    -----------
    path:           string
        path where to save the dataframe
    dataset_dict:   dict
        contains all the analysis data
    datasettype:    string
        adds the name of the subset to the dataframe title
        e.g. 'all', 'train', 'valid', 'test'
    mask_part:      list
        contains the segmentation tasks
        e.g. ['glomerulus', 'podocytes'], ['glomerulus'], ['podocytes']
    start_time:
        time at the start of the script. Used to calculate the duration of the analysis
    supervised:
        (optional)



    Returns:
    --------
    nothing
    """
    for mask_el in mask_part:
        if mask_el == "podocytes":
            filename = datasettype + "_podos.txt"
            filestr = "podos images"
        elif mask_el == "glomerulus":
            filename = datasettype + "_gloms.txt"
            filestr = "gloms images"

        if supervised:
            dc_mean = np.sum(np.array(dataset_dict['dice_coeffs_%s' % mask_el])) / len(dataset_dict['dice_coeffs_%s' % mask_el])
            dc_min = np.min(np.array(dataset_dict['dice_coeffs_%s' % mask_el]))
            dc_max = np.max(np.array(dataset_dict['dice_coeffs_%s' % mask_el]))
            object_dc_mean = np.sum(np.array(dataset_dict['object_dc_%s' % mask_el])) / len(dataset_dict['object_dc_%s' % mask_el])
            object_dc_min = np.min(np.array(dataset_dict['object_dc_%s' % mask_el]))
            object_dc_max = np.max(np.array(dataset_dict['object_dc_%s' % mask_el]))
            pearson = calculate_pearson(dataset_dict['count_masks_%s' % mask_el], dataset_dict['count_preds_%s' % mask_el])

        duration = time.time() - start_time
        duration_std = int(duration / 3600)
        duration_min = int((duration % 3600)/ 60)
        duration_sec = int(duration % 60)

        write_txt = open(str(os.path.join(path, filename)), "w")
        if supervised:
            write_txt.write(str("Mean dice coefficient on pixels of " + filestr + " compared to groundtruth: ") + str(dc_mean) + '\n')
            write_txt.write(str("Min dice coefficient on pixels of " + filestr + " compared to groundtruth: ") + str(dc_min) + '\n')
            write_txt.write(str("Max dice coefficient on pixels of " + filestr + " compared to groundtruth: ") + str(dc_max) + '\n')
            write_txt.write(str("Pearson correlation coefficient on objects of " + filestr + " compared to groundtruth: ") + str(pearson) + '\n')
            write_txt.write(str("Mean dice coeff on objects of " + filestr + " compared to groundtruth: ") + str(object_dc_mean) + '\n')
            write_txt.write(str("Min dice coeff on objects of " + filestr + " compared to groundtruth: ") + str(object_dc_min) + '\n')
            write_txt.write(str("Max dice coeff on objects of " + filestr + " compared to groundtruth: ") + str(object_dc_max) + '\n')
            write_txt.write('\n')
        write_txt.write(str("Test time: ") + str(duration_std) + "h " + str(duration_min)
                        + "min " + str(duration_sec) + 'sec \n')
        write_txt.close()
    return


def write_dataframe(path, dataset_dict, image_list, datasettype, mask_part):
    """
    Creates a pandas dataframe containing the analysis of mask and prediction

    Parameters:
    -----------
    path:           string
        path where to save the dataframe
    dataset_dict:   dict
        contains all the analysis data
    image_list:     list
        contains all the image names
    datasettype:    string
        adds the name of the subset to the dataframe title
        e.g. 'all', 'train', 'valid', 'test'
    mask_part:      list
        contains the segmentation tasks
        e.g. ['glomerulus', 'podocytes'], ['glomerulus'], ['podocytes']

    Returns:
    --------
    nothing

    """
    for mask_el in mask_part:
        titles=[]
        for i in range(len(image_list)):
            # Get rid of .tif and the path before
            image_name = os.path.split(image_list[i])[1]
            titles.append(image_name[:-4])
        df = pd.DataFrame({'Sample name': pd.Series(titles),
                           'GT count': pd.Series(dataset_dict['count_masks_%s' % mask_el]),
                           'Network count': pd.Series(dataset_dict['count_preds_%s' % mask_el]),
                           'GT area': pd.Series(dataset_dict['area_masks_%s' % mask_el]),
                           'Network area': pd.Series(dataset_dict['area_preds_%s' % mask_el]),
                           'Network dice pixel': pd.Series(dataset_dict['dice_coeffs_%s' % mask_el]),
                           'Network dice object': pd.Series(dataset_dict['object_dc_%s' % mask_el]),
                           'Network True pos': pd.Series(dataset_dict['tp_%s' % mask_el]),
                           'Network False pos': pd.Series(dataset_dict['fp_%s' % mask_el]),
                           'Network False neg': pd.Series(dataset_dict['fn_%s' % mask_el])})
        df.to_excel(str(os.path.join(path, datasettype + '_Dataframe_' + mask_el + '.xlsx')))
        # df.to_csv(path + datasettype + '_Dataframe_' + mask_el + '.csv')
    return


def write_readouts(path, dataset_dict, image_list, datasettype, mask_part,
                    do_wt1_signal, do_dach1_signal, do_stereology_pred, do_stereology_gt):
    """
    Creates the csv output which will be used for the classification.
    Dataframe contains optionally the WT1 signal of the glomerulus prediction,
    the DACH1 signal for the podocoyte prediction and
    the stereological calculations.
    """

    titles = []
    for i in range(len(image_list)):
        image_name = os.path.split(image_list[i])[1]
        titles.append(image_name[:-4])

    # Segmentation of only 1 class was applied (e.g. glomerulus or podocytes)
    if len(mask_part) == 1:
        mask_el = mask_part.pop()

        if mask_el == "glomerulus":
            network_area = "glomerulus_area"
            # Add a column if GET_WT1_SIGNAL_FOR_GLOMERULUS = True
            if do_wt1_signal:
                df = pd.DataFrame(
                    {'image_name': pd.Series(titles),
                     network_area: pd.Series(dataset_dict['area_preds_%s' % mask_el]),
                     'mean_WT1_signal_in_glom': pd.Series(dataset_dict['mean_WT1_glom_preds']),
                     'var_WT1_signal_in_glom': pd.Series(dataset_dict['var_WT1_glom_preds']),
                     'median_WT1_signal_in_glom': pd.Series(dataset_dict['median_WT1_glom_preds']),
                     'min_WT1_signal_in_glom': pd.Series(dataset_dict['min_WT1_glom_preds']),
                     'max_WT1_signal_in_glom': pd.Series(dataset_dict['max_WT1_glom_preds']),
                     'perc25_WT1_signal_in_glom': pd.Series(dataset_dict['perc25_WT1_glom_preds']),
                     'perc75_WT1_signal_in_glom': pd.Series(dataset_dict['perc75_WT1_glom_preds'])})
            else:
                df = pd.DataFrame({'image_name': pd.Series(titles),
                                   network_area: pd.Series(dataset_dict['area_preds_%s' % mask_el])})

        elif mask_el == "podocytes":
            network_count = "podocyte_count"
            network_area = "podocyte_nuclear_area"
            # Add a column if GET_DACH1_SIGNAL_FOR_PODOCYTES = True
            if do_dach1_signal:
                df = pd.DataFrame({'image_name': pd.Series(titles),
                                   network_count: pd.Series(dataset_dict['count_preds_%s' % mask_el]),
                                   network_area: pd.Series(dataset_dict['area_preds_%s' % mask_el]),
                                   'mean_DACH1_signal_in_podo':  pd.Series(dataset_dict['mean_DACH1_podo_preds']),
                                   'var_DACH1_signal_in_podo': pd.Series(dataset_dict['var_DACH1_podo_preds']),
                                   'median_DACH1_signal_in_podo': pd.Series(dataset_dict['median_DACH1_podo_preds']),
                                   'min_DACH1_signal_in_podo': pd.Series(dataset_dict['min_DACH1_podo_preds']),
                                   'max_DACH1_signal_in_podo': pd.Series(dataset_dict['max_DACH1_podo_preds']),
                                   'perc25_DACH1_signal_in_podo': pd.Series(dataset_dict['perc25_DACH1_podo_preds']),
                                   'perc75_DACH1_signal_in_podo': pd.Series(dataset_dict['perc75_DACH1_podo_preds'])
                                   })
            else:
                df = pd.DataFrame({'image_name': pd.Series(titles),
                                   network_count: pd.Series(dataset_dict['count_preds_%s' % mask_el]),
                                   network_area: pd.Series(dataset_dict['area_preds_%s' % mask_el])})

        else:
            raise ValueError('The name of the segmentation is not known:', mask_el)

        savepath = str(os.path.join(path, datasettype + '_Dataframe_' + mask_el))
        df.to_csv(savepath + '.csv')
        df.to_excel(savepath + '.xlsx')

    # Segmentation of 2 classes were applied (e.g. glomerulus and podocytes)
    elif len(mask_part) == 2:
        df = pd.DataFrame(
            {'image_name': pd.Series(titles),
             "glomerulus_area": pd.Series(dataset_dict['area_preds_%s' % mask_part[0]]),
             "podocyte_count": pd.Series(dataset_dict['count_preds_%s' % mask_part[1]]),
             "podocyte_nuclear_area": pd.Series(dataset_dict['area_preds_%s' % mask_part[1]])})

        # Add a column if GET_WT1_SIGNAL_FOR_GLOMERULUS = True
        if do_wt1_signal:
            df['mean_WT1_signal_in_glom'] = dataset_dict['mean_WT1_glom_preds']
            df['var_WT1_signal_in_glom'] = dataset_dict['var_WT1_glom_preds']
            df['median_WT1_signal_in_glom'] = dataset_dict['median_WT1_glom_preds']
            df['min_WT1_signal_in_glom'] = dataset_dict['min_WT1_glom_preds']
            df['max_WT1_signal_in_glom'] = dataset_dict['max_WT1_glom_preds']
            df['perc25_WT1_signal_in_glom'] = dataset_dict['perc25_WT1_glom_preds']
            df['perc75_WT1_signal_in_glom'] = dataset_dict['perc75_WT1_glom_preds']


        # Add a column if GET_DACH1_SIGNAL_FOR_PODOCYTES = True
        if do_dach1_signal:
            df['mean_DACH1_signal_in_podo'] = dataset_dict['mean_DACH1_podo_preds']
            df['var_DACH1_signal_in_podo'] = dataset_dict['var_DACH1_podo_preds']
            df['median_DACH1_signal_in_podo'] = dataset_dict['median_DACH1_podo_preds']
            df['min_DACH1_signal_in_podo'] = dataset_dict['min_DACH1_podo_preds']
            df['max_DACH1_signal_in_podo'] = dataset_dict['max_DACH1_podo_preds']
            df['perc25_DACH1_signal_in_podo'] = dataset_dict['perc25_DACH1_podo_preds']
            df['perc75_DACH1_signal_in_podo'] = dataset_dict['perc75_DACH1_podo_preds']

        if do_stereology_pred:
            stereo_dict = get_stereology_readouts(dataset_dict, mask_part, titles, mode='pred')
            # Add it to df
            df['stereology_on_prediction-glomerular_volume_per_million'] = stereo_dict["glomerular_volume_per_million"]
            df['stereology_on_prediction-podocyte_count'] = stereo_dict["podocyte_count"]
            df['stereology_on_prediction-podocyte_density'] = stereo_dict["podocyte_density"]

        if do_stereology_gt:
            stereo_dict = get_stereology_readouts(dataset_dict, mask_part, titles, mode='gt')
            # Add it to df
            df['stereology_on_groundtruth-glomerular_volume_per_million'] = stereo_dict["glomerular_volume_per_million"]
            df['stereology_on_groundtruth-podocyte_count'] = stereo_dict["podocyte_count"]
            df['stereology_on_groundtruth-podocyte_density'] = stereo_dict["podocyte_density"]

        savepath = str(os.path.join(path, datasettype + '_Dataframe_' + mask_part[0] + mask_part[1]))
        df.to_csv(savepath + '.csv')
        df.to_excel(savepath + '.xlsx')
    return


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
        Na = podo_count[index] / glom_area[index]
        Vv = podo_nuclear_area[index] / glom_area[index]
        Nv1 = 1 / 1.55
        Nv2 = math.pow(Na, 3)
        try:
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


def calculate_pearson(list1, list2):
    df = pd.DataFrame({'List 1': pd.Series(list1), 'List 2': pd.Series(list2)})
    corr_df = df.corr(method='pearson')
    pcc = corr_df.loc['List 1', 'List 2']
    return pcc


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


def get_list_of_tasks(task, class_names):
    """
    Get the tasks for this segmentation
    Parameters:
    -----------
    task:            list
        list of chosen segmentation task

    class_names:     list
        list of class names
        first entry: podocytes
        second entry: glomerulus

    Return:
    -------
    list of tasks

    """
    # prefix for data and images
    if task == 'glomerulus':
        task_list = []
        task_list.append(class_names[0])
    elif task == 'podocytes':
        task_list = []
        task_list.append(class_names[1])
    elif task == 'all':
        task_list = class_names
    else:
        raise ValueError('task value does not match to the expected value', task)

    return task_list


def get_list_of_subsets(subset_task):
    """
    Get the subsets to process for this analysis

    Parameters:
    -----------
    subset_task:    string
        string which specifies which subsets to use

    Results:
    --------
    list of subset tasks
    """
    # prefix for data and images
    if subset_task == 'all':
        subset_list = ['train', 'valid', 'test']
    elif subset_task == 'train':
        subset_list = ['train']
    elif subset_task == 'valid':
        subset_list = ['valid']
    elif subset_task == 'test':
        subset_list = ['test']
    else:
        raise ValueError('task value does not match to the expected value', subset_task)

    return subset_list


def only_predict_images(model, cfg, data_pth, load_model, save_path, image_generator, n_images, dataset_name):
    """
    :param model:
    :param cfg:
    :param data_pth:
    :param load_model:
    :param save_path:
    :param image_generator:
    :param n_images:
    :param dataset_name:
    :return:
    """
    start_time = time.time()
    if cfg.SAVE_IMAGES:
        img_save_path = os.path.join(save_path, 'images')
        try:
            os.mkdir(img_save_path)
        except OSError:
            print("Creation of the directory %s failed." % img_save_path)
            print("Maybe there is already an existing directory.")
        else:
            print("Successfully created the directory %s ." % img_save_path)

    # Load the weights of the trained model
    model.load_weights(load_model)

    # prefix for data and images
    mask_part = get_list_of_tasks(cfg.SEGMENTATION_TASK, cfg.NAMES_CLASSES)

    # Get a list with all the image names
    img_full_path = []
    for pth in data_pth:
        img_path = os.path.join(pth, dataset_name + "/images/")
        try:
            img_list = [_ for _ in os.listdir(img_path) if _.endswith(cfg.IMAGE_FORMAT)]
            img_list = [pth + "/" + dataset_name + "/images/" + s for s in img_list]
            img_full_path.extend(img_list)
        except OSError:
            print("Directory does not exist. img_full_path will not be extended for this pth.")

    # Sort the img_full_path list
    img_full_path.sort()
    # img_full_path example ['/data/Dataset1/tif/test/images/A1.tif',
    # '/data/Dataset1/tif/test/images/A2.tif', ... ]

    full_dataset_name = ""
    for pth in data_pth:
        general_dataset_name = os.path.split(os.path.split(os.path.split(pth)[0])[0])[1]
        # example: '/data/Dataset1/tif/' -> 'Dataset1'
        full_dataset_name += general_dataset_name
        full_dataset_name += "_"
    full_dataset_name += dataset_name
    # examples: Dataset1_valid, Dataset2_Dataset1_valid

    # Run across all images in dataset and save results
    pred_dict = {}
    for i in range(n_images):
        img = next(image_generator)
        pred = model.predict(img)
        # Squeeze generator image from (1, 1024, 1024, 3) to (1024, 1024, 3)
        img = img[0, :, :, :]
        # Squeeze prediction from (1, 1024, 1024, ?) to (1024, 1024, ?)
        pred = pred[0, :, :, :]

        if cfg.SAVE_IMAGES:
            img = (255 * img).astype(np.uint8)
            # Changing the LUT from: c0 = DAPI = red, c1 = WT1 = green, c2 = DACH1 = blue
            #                    to: c0 = DAPI = blue, c1 = WT1 = red, c2 = DACH1 = green
            image_name = os.path.split(img_full_path[i])[1]
            suff_pos = image_name.find(cfg.IMAGE_FORMAT)
            sample_name = image_name[:suff_pos]
            imsave(os.path.join(img_save_path, image_name), img[:, :, (1, 2, 0)])
            for index, el in enumerate(mask_part):
                pred_ch = pred[:, :, index]#.astype(np.uint8)
                imsave(os.path.join(img_save_path, sample_name + '_pred_'
                                    + el + cfg.SAVE_IMAGE_FORMAT), pred_ch)

    return



def predict_and_analyse_images(model, cfg, data_pth, load_model, save_path, image_generator, n_images, dataset_name):
    """
    Analyses the network predictions without masks (-> unsupervised mode)
    :param model:
    :param cfg:
    :param data_pth:
    :param load_model:
    :param save_path:
    :param image_generator:
    :param n_images:
    :param dataset_name:
    :return:
    """
    start_time = time.time()
    if cfg.SAVE_IMAGES:
        img_save_path = os.path.join(save_path, 'images')
        try:
            os.mkdir(img_save_path)
        except OSError:
            print("Creation of the directory %s failed." % img_save_path)
            print("Maybe there is already an existing directory.")
        else:
            print("Successfully created the directory %s ." % img_save_path)

    # Load the weights of the trained model
    model.load_weights(load_model)

    # prefix for data and images
    mask_part = get_list_of_tasks(cfg.SEGMENTATION_TASK, cfg.NAMES_CLASSES)

    # filter_th = []

    # filterthstr for images
    filterstr_dict = {}
    for ind, element in enumerate(cfg.NAMES_CLASSES):
        filterstr_dict[element] = "_filter" + str(cfg.FILTER_CLASSES[ind])
        if cfg.UNIT_MICRONS:
            filterstr_dict[element] += 'microns'

    dataset_dict = {}
    for el in mask_part:
        dataset_dict['count_preds_%s' % el] = []
        dataset_dict['area_preds_%s' % el] = []
        if cfg.GET_WT1_SIGNAL_FOR_GLOMERULUS:
            dataset_dict['mean_WT1_glom_preds'] = []
            dataset_dict['var_WT1_glom_preds'] = []
            dataset_dict['median_WT1_glom_preds'] = []
            dataset_dict['min_WT1_glom_preds'] = []
            dataset_dict['max_WT1_glom_preds'] = []
            dataset_dict['perc25_WT1_glom_preds'] = []
            dataset_dict['perc75_WT1_glom_preds'] = []
        if cfg.GET_DACH1_SIGNAL_FOR_PODOCYTES:
            dataset_dict['mean_DACH1_podo_preds'] = []
            dataset_dict['var_DACH1_podo_preds'] = []
            dataset_dict['median_DACH1_podo_preds'] = []
            dataset_dict['min_DACH1_podo_preds'] = []
            dataset_dict['max_DACH1_podo_preds'] = []
            dataset_dict['perc25_DACH1_podo_preds'] = []
            dataset_dict['perc75_DACH1_podo_preds'] = []

    image_dict = {}
    for el in mask_part:
        image_dict['filtered_pred_%s' % el] = []

    # Get a list with all the image names
    img_full_path = []
    for pth in data_pth:
        img_path = os.path.join(pth, dataset_name + "/images/")
        try:
            img_list = [_ for _ in os.listdir(img_path) if _.endswith(cfg.IMAGE_FORMAT)]
            img_list = [pth + "/" + dataset_name + "/images/" + s for s in img_list]
            img_full_path.extend(img_list)
        except OSError:
            print("Directory does not exist. img_full_path will not be extended for this pth.")

    # Sort the img_full_path list
    img_full_path.sort()
    # img_full_path example ['/data/Dataset1/tif/test/images/A1.tif',
    # '/data/Dataset1/tif/test/images/A2.tif', ... ]

    full_dataset_name = ""
    for pth in data_pth:
        general_dataset_name = os.path.split(os.path.split(os.path.split(pth)[0])[0])[1]
        # example: '/data/Dataset1/tif/' -> 'Dataset1'
        full_dataset_name += general_dataset_name
        full_dataset_name += "_"
    full_dataset_name += dataset_name
    # examples: Dataset1_valid, Dataset2_Dataset1_valid

    # Run across all images in dataset and save results
    for i in range(n_images):
        img = next(image_generator)
        pred = model.predict(img)
        # pred = np.squeeze(np.array(pred))
        # pred = np.round(pred)
        pred = pred > cfg.PREDICTION_THRESHOLD

        # Squeeze generator image from (1, 1024, 1024, 3) to (1024, 1024, 3)
        img = img[0, :, :, :]
        # Squeeze prediction from (1, 1024, 1024, ?) to (1024, 1024, ?)
        pred = pred[0, :, :, :]

        #full_dataset_name = ""
        #general_dataset_name = os.path.split(os.path.split(os.path.split(os.path.split(os.path.split(
        #                                     img_full_path[i])[0])[0])[0])[0])[1]
        # example: '/data/Dataset1/tif/test/images/A1.tif' -> 'Dataset1'
        #full_dataset_name += general_dataset_name
        #full_dataset_name += "_"
        #full_dataset_name += dataset_name
        # example: AA101_valid

        # Get the cell (or glom) count for the mask and the prediction:
        for num, el in enumerate(cfg.NAMES_CLASSES):
            if cfg.UNIT_MICRONS:
                this_img_path, this_img_file = os.path.split(img_full_path[i])
                xmicrons, ymicrons = eval.get_micron_info(this_img_path, this_img_file)
                filter_th = cfg.FILTER_CLASSES[num] * xmicrons * ymicrons
                # print(filter_th)
                # print("xmicrons", xmicrons, "ymicrons", ymicrons)
            else:
                filter_th = cfg.FILTER_CLASSES[num]
                # print(filter_th)

            # Ensure that the postprocessing of the right mask is applied
            if cfg.SEGMENTATION_TASK == 'all' or cfg.SEGMENTATION_TASK == el:
                # If only one class is segmented, the mask and prediction array have dimension (x,y,1)
                # Thus set num to 0
                if cfg.SEGMENTATION_TASK == el:
                    num = 0

                # cfg.NAMES_CLASSES = [glomerulus, podocytes]
                # Keep only the largest glom in the prediction
                if el == cfg.NAMES_CLASSES[0] and cfg.GLOM_POSTPROCESSING_KEEP_ONLY_LARGEST is True:
                    labeled_pred, dataset_pred_count, filtered_pred = eval.get_count_and_area(pred[:, :, num],
                                                                                              filter_th,
                                                                                              keep_only_largest_label=True)
                # Remove the podocytes with no contact to the glomerulus prediction
                elif el == cfg.NAMES_CLASSES[1] and cfg.PODO_POSTPROCESSING_WITH_GLOM is True:
                    if cfg.SEGMENTATION_TASK != 'all':
                        raise ValueError('GLOM_POSTPROCESSING DOES NOT WORK FOR THIS SEGMENTATION TASK!')
                    labeled_pred, dataset_pred_count, filtered_pred = eval.get_count_and_area_rmv_podo_outside(cfg,
                                                                                                               pred[:,
                                                                                                               :, num],
                                                                                                               pred[:,
                                                                                                               :, 0],
                                                                                                               filter_th)
                else:
                    labeled_pred, dataset_pred_count, filtered_pred = eval.get_count_and_area(pred[:, :, num],
                                                                                              filter_th)
                image_dict['filtered_pred_%s' % el] = filtered_pred

                # cfg.NAMES_CLASSES = [glomerulus, podocytes]
                # GET_WT1_SIGNAL_FOR_GLOMERULUS
                if el == cfg.NAMES_CLASSES[0] and cfg.GET_WT1_SIGNAL_FOR_GLOMERULUS:
                    # WT1 channel is the second channel
                    signal_image = filtered_pred * img[:, :, 1]
                    # Calculate the signal features and add them to the dict
                    mean_wt1, var_wt1, median_wt1, min_wt1, max_wt1, perc25_wt1, perc75_wt1 = \
                        get_signal_features(signal_image[filtered_pred])

                    dataset_dict['mean_WT1_glom_preds'].append(mean_wt1)
                    dataset_dict['var_WT1_glom_preds'].append(var_wt1)
                    dataset_dict['median_WT1_glom_preds'].append(median_wt1)
                    dataset_dict['min_WT1_glom_preds'].append(min_wt1)
                    dataset_dict['max_WT1_glom_preds'].append(max_wt1)
                    dataset_dict['perc25_WT1_glom_preds'].append(perc25_wt1)
                    dataset_dict['perc75_WT1_glom_preds'].append(perc75_wt1)
                # GET_DACH1_SIGNAL_FOR_PODOCYTES
                elif el == cfg.NAMES_CLASSES[1] and cfg.GET_DACH1_SIGNAL_FOR_PODOCYTES:
                    # DACH1 channel is the third channel
                    signal_image = filtered_pred * img[:, :, 2]
                    # Calculate the signal features and add them to the dict
                    mean_dach1, var_dach1, median_dach1, min_dach1, max_dach1, perc25_dach1, perc75_dach1 = \
                        get_signal_features(signal_image[filtered_pred])

                    dataset_dict['mean_DACH1_podo_preds'].append(mean_dach1)
                    dataset_dict['var_DACH1_podo_preds'].append(var_dach1)
                    dataset_dict['median_DACH1_podo_preds'].append(median_dach1)
                    dataset_dict['min_DACH1_podo_preds'].append(min_dach1)
                    dataset_dict['max_DACH1_podo_preds'].append(max_dach1)
                    dataset_dict['perc25_DACH1_podo_preds'].append(perc25_dach1)
                    dataset_dict['perc75_DACH1_podo_preds'].append(perc75_dach1)

                dataset_dict['count_preds_%s' % el].append(dataset_pred_count)

                if cfg.UNIT_MICRONS:
                    dataset_dict['area_preds_%s' % el].append(np.count_nonzero(labeled_pred) / (xmicrons * ymicrons))
                else:
                    dataset_dict['area_preds_%s' % el].append(np.count_nonzero(labeled_pred))

        if cfg.SAVE_IMAGES:
            img = (255 * img).astype(np.uint8)
            # Changing the LUT from: c0 = DAPI = red, c1 = WT1 = green, c2 = DACH1 = blue
            #                    to: c0 = DAPI = blue, c1 = WT1 = red, c2 = DACH1 = green
            # ToDo: Acces image names via method e.g.: image_valid_generator.filenames[i]
            image_name = os.path.split(img_full_path[i])[1]
            suff_pos = image_name.find(cfg.IMAGE_FORMAT)
            sample_name = image_name[:suff_pos]
            imsave(os.path.join(img_save_path, dataset_name + '_' + image_name), img[:, :, (1, 2, 0)])
            for el in mask_part:
                pred_ch = (255 * image_dict['filtered_pred_%s' % el]).astype(np.uint8)
                imsave(os.path.join(img_save_path, dataset_name + '_' + sample_name + '_pred_' + el +
                                    filterstr_dict[el] + cfg.SAVE_IMAGE_FORMAT), pred_ch)

    if n_images:
        # Create txt file with mean dice coeff and pcc
        write_analysis(save_path, dataset_dict, full_dataset_name, mask_part, start_time, supervised=False)

        # Create short csv file
        if cfg.DO_STEREOLOGY_GT:
            raise ValueError('In unsupervised mode there is no gt for DO_STEREOLOGY_GT.'
                             'Please set DO_STEREOLOGY_GT to false')

        write_readouts(save_path, dataset_dict, img_full_path, full_dataset_name, mask_part,
                        cfg.GET_WT1_SIGNAL_FOR_GLOMERULUS, cfg.GET_DACH1_SIGNAL_FOR_PODOCYTES,
                        cfg.DO_STEREOLOGY_PRED, False)
    return


def predict_and_further_analyse_images(model, cfg, data_pth, load_model, save_path, image_generator, n_images, dataset_name):
    """
    Analyses the network predictions and evaulates them with the masks (-> supervised mode)
    :param model:
    :param cfg:
    :param data_pth:
    :param load_model:
    :param save_path:
    :param image_generator:
    :param n_images:
    :param dataset_name:
    :return:
    """

    start_time = time.time()
    if cfg.SAVE_IMAGES:
        img_save_path = os.path.join(save_path, 'images')
        try:
            os.mkdir(img_save_path)
        except OSError:
            print("Creation of the directory %s failed." % img_save_path)
            print("Maybe there is already an existing directory.")
        else:
            print("Successfully created the directory %s ." % img_save_path)

    # Load the weights of the trained model
    model.load_weights(load_model)

    # prefix for data and images
    mask_part = get_list_of_tasks(cfg.SEGMENTATION_TASK, cfg.NAMES_CLASSES)

    # filter_th = []

    # filterthstr for images
    filterstr_dict = {}
    for ind, element in enumerate(cfg.NAMES_CLASSES):
        filterstr_dict[element] = "_filter" + str(cfg.FILTER_CLASSES[ind])
        if cfg.UNIT_MICRONS:
            filterstr_dict[element] += 'microns'

    dataset_dict = {}
    for el in mask_part:
        dataset_dict['dice_coeffs_%s' % el] = []
        dataset_dict['count_masks_%s' % el] = []
        dataset_dict['area_masks_%s' % el] = []
        dataset_dict['count_preds_%s' % el] = []
        dataset_dict['area_preds_%s' % el] = []
        dataset_dict['object_dc_%s' % el] = []
        dataset_dict['tp_%s' % el] = []
        dataset_dict['fp_%s' % el] = []
        dataset_dict['fn_%s' % el] = []
        if cfg.GET_WT1_SIGNAL_FOR_GLOMERULUS:
            dataset_dict['mean_WT1_glom_preds'] = []
            dataset_dict['var_WT1_glom_preds'] = []
            dataset_dict['median_WT1_glom_preds'] = []
            dataset_dict['min_WT1_glom_preds'] = []
            dataset_dict['max_WT1_glom_preds'] = []
            dataset_dict['perc25_WT1_glom_preds'] = []
            dataset_dict['perc75_WT1_glom_preds'] = []
        if cfg.GET_DACH1_SIGNAL_FOR_PODOCYTES:
            dataset_dict['mean_DACH1_podo_preds'] = []
            dataset_dict['var_DACH1_podo_preds'] = []
            dataset_dict['median_DACH1_podo_preds'] = []
            dataset_dict['min_DACH1_podo_preds'] = []
            dataset_dict['max_DACH1_podo_preds'] = []
            dataset_dict['perc25_DACH1_podo_preds'] = []
            dataset_dict['perc75_DACH1_podo_preds'] = []

    image_dict = {}
    for el in mask_part:
        image_dict['filtered_mask_%s' % el] = []
        image_dict['filtered_pred_%s' % el] = []


    # Get a list with all the image names
    img_full_path = []
    for pth in data_pth:
        img_path = os.path.join(pth, dataset_name + "/images/")
        img_list = [_ for _ in os.listdir(img_path) if _.endswith(cfg.IMAGE_FORMAT)]
        img_list = [pth + "/" + dataset_name + "/images/" + s for s in img_list]
        img_full_path.extend(img_list)

    # Get a list with all the image mask names
    #msk_full_path = []
    #for pth in data_pth:
    #    msk_path = os.path.join(pth, dataset_name + "/masks/")
    #    msk_list = os.listdir(msk_path)
    #    msk_list = [pth + "/" + dataset_name + "/masks/" + s for s in msk_list]
        # print(glom_msk_list)
    #    msk_full_path.extend(msk_list)

    # Sort the img_full_path list
    img_full_path.sort()

    # Get the corresponding lists for the different classes
    masks_lists = {}
    for ind, mask_suff in enumerate(cfg.MASK_SUFFIXES):
        masks_lists[cfg.NAMES_CLASSES[ind]] = []
        for element in img_full_path:
            suff_pos = element.find(cfg.IMAGE_FORMAT)
            element = element[:suff_pos]
            element = element + mask_suff
            element_name = os.path.split(element)[1]
            masks_lists[cfg.NAMES_CLASSES[ind]].append(os.path.join(pth, dataset_name + "/masks/", element_name))

    full_dataset_name = ""
    for pth in data_pth:
        general_dataset_name = os.path.split(os.path.split(os.path.split(pth)[0])[0])[1]
        # example: '/data/Dataset1/tif/' -> 'Dataset1'
        full_dataset_name += general_dataset_name
        full_dataset_name += "_"
    full_dataset_name += dataset_name
    # examples: Dataset1_valid, Dataset2_Dataset1_valid

    # Run across all images in dataset and save results
    for i in range(n_images):
        img, mask = next(image_generator)
        pred = model.predict(img)
        # pred = np.squeeze(np.array(pred))
        # pred = np.round(pred)
        pred = pred > cfg.PREDICTION_THRESHOLD

        # Squeeze generator image from (1, 1024, 1024, 3) to (1024, 1024, 3)
        img = img[0, :, :, :]
        # Squeeze generator mask from (1, 1024, 1024, ?) to (1024, 1024, ?)
        mask = mask[0, :, :, :]
        # Squeeze prediction from (1, 1024, 1024, ?) to (1024, 1024, ?)
        pred = pred[0, :, :, :]

        # Get the cell (or glom) count for the mask and the prediction:
        for num, el in enumerate(cfg.NAMES_CLASSES):
            # Ensure that the postprocessing of the right mask is applied
            if cfg.SEGMENTATION_TASK == 'all' or cfg.SEGMENTATION_TASK == el:

                if cfg.UNIT_MICRONS:
                    this_img_path, this_img_file = os.path.split(img_full_path[i])
                    xmicrons, ymicrons = eval.get_micron_info(this_img_path, this_img_file)
                    filter_th = cfg.FILTER_CLASSES[num] * xmicrons * ymicrons
                    #print(filter_th)
                    # print("xmicrons", xmicrons, "ymicrons", ymicrons)
                else:
                    filter_th = cfg.FILTER_CLASSES[num]
                    #print(filter_th)

                # If only one class is segmented the mask and prediction array have dimension (x,y,1)
                # Thus set num to 0
                if cfg.SEGMENTATION_TASK == el:
                    num = 0

                # cfg.NAMES_CLASSES = [glomerulus, podocytes]
                # Keep only the largest glom in the prediction
                if el == cfg.NAMES_CLASSES[0] and cfg.GLOM_POSTPROCESSING_KEEP_ONLY_LARGEST is True:
                    labeled_mask, dataset_mask_count, filtered_mask = eval.get_count_and_area(mask[:, :, num], filter_th,
                                                                                              keep_only_largest_label=True)
                    labeled_pred, dataset_pred_count, filtered_pred = eval.get_count_and_area(pred[:, :, num], filter_th,
                                                                                              keep_only_largest_label=True)
                # Remove the podocytes with no contact to the glomerulus prediction
                elif el == cfg.NAMES_CLASSES[1] and cfg.PODO_POSTPROCESSING_WITH_GLOM is True:
                    if cfg.SEGMENTATION_TASK != 'all':
                        raise ValueError('GLOM_POSTPROCESSING DOES NOT WORK FOR THIS SEGMENTATION TASK!')
                    labeled_mask, dataset_mask_count, filtered_mask = eval.get_count_and_area_rmv_podo_outside(cfg, mask[:, :, num], mask[:, :, 0], filter_th)
                    labeled_pred, dataset_pred_count, filtered_pred = eval.get_count_and_area_rmv_podo_outside(cfg, pred[:, :, num], pred[:, :, 0], filter_th)

                else:
                    labeled_mask, dataset_mask_count, filtered_mask = eval.get_count_and_area(mask[:, :, num], filter_th)
                    labeled_pred, dataset_pred_count, filtered_pred = eval.get_count_and_area(pred[:, :, num], filter_th)

                image_dict['filtered_mask_%s' % el] = filtered_mask
                image_dict['filtered_pred_%s' % el] = filtered_pred

                # cfg.NAMES_CLASSES = [glomerulus, podocytes]
                # GET_WT1_SIGNAL_FOR_GLOMERULUS
                if el == cfg.NAMES_CLASSES[0] and cfg.GET_WT1_SIGNAL_FOR_GLOMERULUS:
                    # WT1 channel is the second channel
                    signal_image = img[:, :, 1]
                    # Calculate the signal features and add them to the dict
                    mean_wt1, var_wt1, median_wt1, min_wt1, max_wt1, perc25_wt1, perc75_wt1 = \
                        get_signal_features(signal_image[filtered_pred])

                    dataset_dict['mean_WT1_glom_preds'].append(mean_wt1)
                    dataset_dict['var_WT1_glom_preds'].append(var_wt1)
                    dataset_dict['median_WT1_glom_preds'].append(median_wt1)
                    dataset_dict['min_WT1_glom_preds'].append(min_wt1)
                    dataset_dict['max_WT1_glom_preds'].append(max_wt1)
                    dataset_dict['perc25_WT1_glom_preds'].append(perc25_wt1)
                    dataset_dict['perc75_WT1_glom_preds'].append(perc75_wt1)

                # GET_DACH1_SIGNAL_FOR_PODOCYTES
                elif el == cfg.NAMES_CLASSES[1] and cfg.GET_DACH1_SIGNAL_FOR_PODOCYTES:
                    # DACH1 channel is the third channel
                    signal_image = img[:, :, 2]
                    # Calculate the signal features and add them to the dict
                    mean_dach1, var_dach1, median_dach1, min_dach1, max_dach1, perc25_dach1, perc75_dach1 = \
                        get_signal_features(signal_image[filtered_pred])

                    dataset_dict['mean_DACH1_podo_preds'].append(mean_dach1)
                    dataset_dict['var_DACH1_podo_preds'].append(var_dach1)
                    dataset_dict['median_DACH1_podo_preds'].append(median_dach1)
                    dataset_dict['min_DACH1_podo_preds'].append(min_dach1)
                    dataset_dict['max_DACH1_podo_preds'].append(max_dach1)
                    dataset_dict['perc25_DACH1_podo_preds'].append(perc25_dach1)
                    dataset_dict['perc75_DACH1_podo_preds'].append(perc75_dach1)

                dataset_dict['count_masks_%s' % el].append(dataset_mask_count)
                dataset_dict['count_preds_%s' % el].append(dataset_pred_count)

                # Calculate the dice coefficient
                dataset_dice = dice_coeff_numpy(filtered_mask, filtered_pred)
                dataset_dict['dice_coeffs_%s' % el].append(dataset_dice)

                if cfg.UNIT_MICRONS:
                    dataset_dict['area_masks_%s' % el].append(np.count_nonzero(labeled_mask) / (xmicrons * ymicrons))
                    dataset_dict['area_preds_%s' % el].append(np.count_nonzero(labeled_pred) / (xmicrons * ymicrons))
                else:
                    dataset_dict['area_masks_%s' % el].append(np.count_nonzero(labeled_mask))
                    dataset_dict['area_preds_%s' % el].append(np.count_nonzero(labeled_pred))

                # Get the object_dc-score, TP, FP, FN
                if cfg.GET_object_dc_TP_FN_FP:
                    object_dc, tp, fp, fn = eval.coregistrate_and_get_object_dc_score(labeled_pred, dataset_pred_count,
                                                                    labeled_mask, dataset_mask_count)
                else:
                    object_dc, tp, fp, fn = 0, [], [], []
                dataset_dict['object_dc_%s' % el].append(object_dc)
                dataset_dict['tp_%s' % el].append(len(tp))
                dataset_dict['fp_%s' % el].append(len(fp))
                dataset_dict['fn_%s' % el].append(len(fn))

        if cfg.SAVE_IMAGES:
            img = (255 * img).astype(np.uint8)
            # Changing the LUT from: c0 = DAPI = red, c1 = WT1 = green, c2 = DACH1 = blue
            #                    to: c0 = DAPI = blue, c1 = WT1 = red, c2 = DACH1 = green
            # ToDo: Acces image names via method e.g.: image_valid_generator.filenames[i]
            image_name = os.path.split(img_full_path[i])[1]
            suff_pos = image_name.find(cfg.IMAGE_FORMAT)
            sample_name = image_name[:suff_pos]
            imsave(os.path.join(img_save_path, dataset_name + '_' + sample_name + cfg.SAVE_IMAGE_FORMAT),
                   img[:, :, (1, 2, 0)])
            for el in mask_part:
                mask_ch = (255 * image_dict['filtered_mask_%s' % el]).astype(np.uint8)
                pred_ch = (255 * image_dict['filtered_pred_%s' % el]).astype(np.uint8)
                imsave(os.path.join(img_save_path, dataset_name + '_' + sample_name + '_pred_' + el + filterstr_dict[el] + cfg.SAVE_IMAGE_FORMAT), pred_ch)
                imsave(os.path.join(img_save_path, dataset_name + '_' + sample_name + '_mask_' + el + filterstr_dict[el] + cfg.SAVE_IMAGE_FORMAT), mask_ch)

    # Create pandas data frame
    write_dataframe(save_path, dataset_dict, img_full_path, full_dataset_name, mask_part)

    # Create txt file with mean dice coeff and pcc
    write_analysis(save_path, dataset_dict, full_dataset_name, mask_part, start_time, supervised=True)

    # Create short csv file
    write_readouts(save_path, dataset_dict, img_full_path, full_dataset_name, mask_part,
                    cfg.GET_WT1_SIGNAL_FOR_GLOMERULUS, cfg.GET_DACH1_SIGNAL_FOR_PODOCYTES,
                    cfg.DO_STEREOLOGY_PRED, cfg.DO_STEREOLOGY_GT)
    return


def test_network(cfg, data_path, load_model):

    print(data_path)
    data_path_name = ""
    for path in data_path:
        pth = os.path.split(path)[0]
        pt = os.path.split(pth)[0]
        t = os.path.split(pt)[1]
        data_path_name += t + "_"

    # Select the masks
    if cfg.SEGMENTATION_TASK == 'glomerulus':
        mask_suffix = []
        mask_suffix.append(cfg.MASK_SUFFIXES[0])
    elif cfg.SEGMENTATION_TASK == 'podocytes':
        mask_suffix = []
        mask_suffix.append(cfg.MASK_SUFFIXES[1])
    elif cfg.SEGMENTATION_TASK == 'all':
        mask_suffix = cfg.MASK_SUFFIXES

    # Data generators to read in the images for testing
    # Path to the images

    #if cfg.EVALUATION_MODE == 'supervised':
    #    msk_train_full = []
    #    msk_valid_full = []
    #    msk_test_full = []
    #    for path in data_path:
    #        msk_train_full.append(os.path.join(path, 'train/masks/'))
    #        msk_valid_full.append(os.path.join(path, 'valid/masks/'))
    #        msk_test_full.append(os.path.join(path, 'test/masks/'))

    img_train_full = []
    img_valid_full = []
    img_test_full = []

    for path in data_path:
        img_train_full.append(os.path.join(path, 'train/images/'))
        img_valid_full.append(os.path.join(path, 'valid/images/'))
        img_test_full.append(os.path.join(path, 'test/images/'))

    data_gen_args = dict(shuffle=False, # Shuffle has to be False. Otherwise add_glomerulus_gt_mask_preprocessing_to_analysis will fail
                         target_img_shape=cfg.TARGET_IMG_SHAPE,
                         number_msk_channels=cfg.NUMBER_MSK_CHANNELS * cfg.NUM_OUTPUT_CH,
                         rescale_factor=cfg.RESCALE,
                         contrast_stretch=cfg.CONTRAST_STRETCHING,
                         histogram_equalization=cfg.HISTOGRAM_EQUALIZATION,
                         do_data_augm=False,
                         verbose=True,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         flip_horizontal=True,
                         flip_vertical=True,
                         rotation_range=45,
                         zooming_range=0.1,
                         c0_signal_min=1,
                         # signal_min and signal_max must be in [0,1], signal_max must be greater than signal_min
                         c0_signal_max=1,
                         c1_signal_min=1,
                         c1_signal_max=1,
                         c2_signal_min=1,
                         c2_signal_max=1,
                         c2_reduce_podo_signal=False,
                         augm_data_path=0,  # Enter a path here if augmented data shall be saved
                         save_augm_data=False,
                         plot_augm_data=False)

    if cfg.SUPERVISED_MODE:
        train_generator = img_gen.image_generator(img_train_full, cfg.IMAGE_FORMAT, cfg.MASK_FOLDERNAME, mask_suffix,
                                                  cfg.BATCH_SIZE, 1, **data_gen_args)
        valid_generator = img_gen.image_generator(img_valid_full, cfg.IMAGE_FORMAT, cfg.MASK_FOLDERNAME, mask_suffix,
                                                  cfg.BATCH_SIZE, 1, **data_gen_args)
        test_generator = img_gen.image_generator(img_test_full, cfg.IMAGE_FORMAT, cfg.MASK_FOLDERNAME, mask_suffix,
                                                 cfg.BATCH_SIZE, 1, **data_gen_args)
    else:
        train_generator = img_gen.image_generator(img_train_full, cfg.IMAGE_FORMAT, cfg.MASK_FOLDERNAME, "unsupervised",
                                                  cfg.BATCH_SIZE, 1, **data_gen_args)
        valid_generator = img_gen.image_generator(img_valid_full, cfg.IMAGE_FORMAT, cfg.MASK_FOLDERNAME, "unsupervised",
                                                  cfg.BATCH_SIZE, 1, **data_gen_args)
        test_generator = img_gen.image_generator(img_test_full, cfg.IMAGE_FORMAT, cfg.MASK_FOLDERNAME, "unsupervised",
                                                 cfg.BATCH_SIZE, 1, **data_gen_args)

    n_img_train = int(get_file_count(img_train_full, cfg.IMAGE_FORMAT))
    n_img_valid = int(get_file_count(img_valid_full, cfg.IMAGE_FORMAT))
    n_img_test = int(get_file_count(img_test_full, cfg.IMAGE_FORMAT))


    # Define the network model
    inputs = layers.Input(shape=cfg.TARGET_IMG_SHAPE)
    outputs = inputs  # Initialise to be able to use it in if-clause
    if cfg.ARCHITECTURE == 'unet':
        hidden_layers, outputs = define_unet(inputs, cfg)
    elif cfg.ARCHITECTURE == 'fc_densenet_tiramisu103':
        outputs = define_tiramisu(inputs)
        outputs = outputs
    elif cfg.ARCHITECTURE == 'fc_densenet_tiramisu56':
        outputs = define_tiramisu(inputs, n_layers_per_block=[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], growth_rate=12)
        outputs = outputs
    elif cfg.ARCHITECTURE == 'fc_densenet_tiramisu67':
        outputs = define_tiramisu(inputs, n_layers_per_block=[5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], growth_rate=16)
        outputs = outputs

    else:
        exit("ERROR: No network model type defined")

    print(cfg.ARCHITECTURE)

    if cfg.VISUALIZE_ACTIVATIONS is True:
        out_list = []
        out_list.extend(hidden_layers)
        model = models.Model(inputs=[inputs], outputs=hidden_layers + [outputs])
    else:
        model = models.Model(inputs=[inputs], outputs=[outputs])

    print(load_model)

    if cfg.VISUALIZATION_MODE:
        print("test_network is run in visualization mode")
        return model, train_generator, n_img_train, valid_generator, n_img_valid, test_generator, n_img_test
    else:
        print("test_network is run in normal mode")

        load_model_folder = os.path.split(load_model)[0]
        now = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        #save_path = os.path.join(load_model_folder, "th0-" + str(cfg.PREDICTION_THRESHOLD)[2:] + "_results_" + now)
        save_path = os.path.join(load_model_folder, "results_" + now)

        try:
            os.mkdir(save_path)
        except OSError:
            print("Creation of the directory %s failed." % save_path)
            print("Maybe there is already an existing directory.")
        else:
            print("Successfully created the directory %s ." % save_path)

        cfg.write_cfg_to_log(save_path)

        if cfg.SUPERVISED_MODE:
            predict_and_further_analyse_images(model, cfg, data_path, load_model, save_path, train_generator, n_img_train, "train")
            predict_and_further_analyse_images(model, cfg, data_path, load_model, save_path, valid_generator, n_img_valid, "valid")
            predict_and_further_analyse_images(model, cfg, data_path, load_model, save_path, test_generator, n_img_test, "test")
        else:
            if cfg.DO_NOT_THRESHOLD:
                only_predict_images(model, cfg, data_path, load_model, save_path, train_generator, n_img_train, "train")
                only_predict_images(model, cfg, data_path, load_model, save_path, valid_generator, n_img_valid, "valid")
                only_predict_images(model, cfg, data_path, load_model, save_path, test_generator, n_img_test, "test")
            else:
                #predict_and_analyse_images(model, cfg, data_path, load_model, save_path, train_generator, n_img_train, "train")
                #predict_and_analyse_images(model, cfg, data_path, load_model, save_path, valid_generator, n_img_valid, "valid")
                predict_and_analyse_images(model, cfg, data_path, load_model, save_path, test_generator, n_img_test, "test")

        return
