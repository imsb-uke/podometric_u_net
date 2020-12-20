import os

import numpy as np
import pandas as pd
import time


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
        if mask_el == 'podocytes':
            filename = datasettype + '_podos.txt'
            filestr = 'podos images'
        elif mask_el == 'glomerulus':
            filename = datasettype + '_gloms.txt'
            filestr = 'gloms images'
        else:
            filename = datasettype + 'unknown.txt'
            filestr = 'unknown type'

        write_txt = open(str(os.path.join(path, filename)), "w")

        if supervised:
            dc_mean = np.sum(np.array(dataset_dict['dice_coeffs_%s' % mask_el])) / len(dataset_dict['dice_coeffs_%s'
                                                                                                    % mask_el])
            dc_min = np.min(np.array(dataset_dict['dice_coeffs_%s' % mask_el]))
            dc_max = np.max(np.array(dataset_dict['dice_coeffs_%s' % mask_el]))
            object_dc_mean = np.sum(np.array(dataset_dict['object_dc_%s' % mask_el])) / len(dataset_dict['object_dc_%s'
                                                                                                         % mask_el])
            object_dc_min = np.min(np.array(dataset_dict['object_dc_%s' % mask_el]))
            object_dc_max = np.max(np.array(dataset_dict['object_dc_%s' % mask_el]))
            pearson = calculate_pearson(dataset_dict['count_masks_%s' % mask_el], dataset_dict['count_preds_%s'
                                                                                               % mask_el])

            write_txt.write(str("Mean dice coefficient on pixels of " + filestr + " compared to groundtruth: ") +
                            str(dc_mean) + '\n')
            write_txt.write(str("Min dice coefficient on pixels of " + filestr + " compared to groundtruth: ") +
                            str(dc_min) + '\n')
            write_txt.write(str("Max dice coefficient on pixels of " + filestr + " compared to groundtruth: ") +
                            str(dc_max) + '\n')
            write_txt.write(str("Pearson correlation coefficient on objects of " + filestr +
                                " compared to groundtruth: ") + str(pearson) + '\n')
            write_txt.write(str("Mean dice coeff on objects of " + filestr + " compared to groundtruth: ") +
                            str(object_dc_mean) + '\n')
            write_txt.write(str("Min dice coeff on objects of " + filestr + " compared to groundtruth: ") +
                            str(object_dc_min) + '\n')
            write_txt.write(str("Max dice coeff on objects of " + filestr + " compared to groundtruth: ") +
                            str(object_dc_max) + '\n')
            write_txt.write('\n')

        duration = time.time() - start_time
        duration_std = int(duration / 3600)
        duration_min = int((duration % 3600) / 60)
        duration_sec = int(duration % 60)

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
        titles = []
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