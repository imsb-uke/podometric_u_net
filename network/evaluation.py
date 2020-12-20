'''
This script has functions in it which are used in network which evaluate images.
If this script here is run it returns the object_dc-score of each segmented object by the predicition with respect to the groundtruth
'''
import os
import skimage
import scipy
import numpy as np
import matplotlib.pyplot as plt


#####################################
# Plotting functions                #
#####################################

def plot_img_and_hist(image, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram.
       Source: https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_equalize.html#sphx-glr-auto-examples-color-exposure-plot-equalize-py
    """
    image = skimage.img_as_float(image)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = skimage.exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf


def plot_img_and_segmentations(imgs_dict, names_list, color_list):
    fig, axs = plt.subplots(1, len(names_list), figsize=(5 * len(names_list),5))
    #plt.title('Visualization of data and prediction')
    for ax, img_name, colormap in zip(axs, names_list, color_list):
        pic = imgs_dict[img_name]
        ax.imshow(pic, cmap=colormap)
        ax.axis('off')
        ax.set_title(img_name.capitalize())

    plt.show()
    return


def plot_img_and_segm_overlayed(img, msks_dict, msk_names_list, color_list, change_bg_color_list):
    fig, axs = plt.subplots(len(msk_names_list), 1, figsize=(15, 15 * len(msk_names_list)))

    for ax, msk_name, colormap, change_bg in zip(axs, msk_names_list, color_list, change_bg_color_list):
        ax.imshow(img)
        if change_bg:
            overlay_mask = msks_dict[msk_name]
        else:
            overlay_mask = np.ma.masked_array(msks_dict[msk_name], msks_dict[msk_name] == 0)
        ax.imshow(overlay_mask, colormap, alpha=0.5)
        ax.axis('off')
        ax.set_title(msk_name.capitalize())
    plt.show()


def plot_segmentations_dice(imgs_dict, names_list, label_list):
    fig, axs = plt.subplots(1, len(names_list), figsize=(len(names_list) * 10, 10))

    handles = label_list

    # plt.title('Visualization of data and prediction')
    for ax, msk_name, in zip(axs, names_list):
        pic = imgs_dict[msk_name]
        ax.imshow(pic * 255)
        ax.axis('off')
        subtitle = msk_name + " comparison"
        ax.set_title(subtitle.capitalize())
        ax.legend(handles=handles)
    plt.show()
    return


####################################
# Metric, Micron extraction        #
####################################


def dice_coeff_numpy(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    score = (2 * intersection + 1.) / (y_true.sum() + y_pred.sum() + 1.)
    return score


def get_micron_info(pathtofile, filename):
    """
    Returns the pixel per micron ratio for x and y.
    Works with .tif images from ImageJ

    Parameters:
    -----------
    pathtofile:     string
        path of the folder where the file is in
    filename:       string
        name of the file

    Returns:
    --------
    (pix mic x, pix mic y)      tuple
        Tuple with the pixel per micron ratio for x and y
    """
    # Load microns unit
    with skimage.external.tifffile.TiffFile(os.path.join(pathtofile, filename)) as tif:
        metadata = tif.info()
        # Find info about pixels per micron
        x_pos = metadata.find("* 282 x_resolution")
        y_pos = metadata.find("* 283 y_resolution")
        pixel_per_micron_x = float(metadata[x_pos + 25: x_pos + 32]) * 0.000001
        pixel_per_micron_y = float(metadata[y_pos + 25: y_pos + 32]) * 0.000001

    if pixel_per_micron_x != pixel_per_micron_y:
        print("Error. The resolution in micron in x and y are different. ",
              "Please check the image. If there is no error in the image, this has to be implemented!",
              "get_micron_info will return nothing.")
        return
    return (pixel_per_micron_x, pixel_per_micron_y)


####################################
# Area analyis of images           #
####################################


def get_zero_area_in_img(image, area_threshold=0.1):
    """
    Finds the sliced away area in an image

    Parameters:
    -----------
    image:              array
        with shape e.g. (1024, 1024, 3)
        values in [0,1]

    area_threshold:     float
        values in [0,1]
        percentage of zero_area size necessary to define it as cropped_img_area
    Returns:
    --------
    cropped_img_area:    array
        with same shape as image
        values: True or False

    """
    # Reduce image to grayscale image
    grayscale_image = skimage.color.rgb2gray(image)

    # Set all values which are 0 to 1 in a new array
    cropped_img_area = np.zeros(grayscale_image.shape)
    cropped_img_area[grayscale_image == 0] = 1

    # Find connected components
    labelled_image, count_image = scipy.ndimage.label(cropped_img_area)

    refined_cropped_img_area = cropped_img_area.copy()

    # Filter out all connected components with size smaller or equal area_threshold
    for label in range(1, count_image + 1):
        if len(refined_cropped_img_area[labelled_image == label]) <= area_threshold * cropped_img_area.size:
            refined_cropped_img_area[labelled_image == label] = 0
            # count_refined_mask -= 1

    # Return a boolean array
    final_cropped_img_area = np.array(refined_cropped_img_area > 0)

    # Debug:
    if np.max(final_cropped_img_area) > 0:
        print("zero area in image detected")
        print("Percentage of cropped area:", np.sum(final_cropped_img_area) / final_cropped_img_area.size)

    return final_cropped_img_area


def get_count_and_area(mask, filter_th, keep_only_largest_label=False, verbose=False):
    labelled_mask, count_mask = scipy.ndimage.label(mask)

    # Keep only the biggest connected component
    if keep_only_largest_label:
        refined_mask = mask.copy()
        len_largest_label = 0
        id_largest_label = 0
        for label in range(1, count_mask + 1):
            if len(refined_mask[labelled_mask == label]) > len_largest_label:
                len_largest_label = len(refined_mask[labelled_mask == label])
                id_largest_label = label
        refined_mask[:] = 0

        refined_mask[labelled_mask == id_largest_label] = 1
        count_mask = 1
        if verbose:
            print(refined_mask.shape, refined_mask.min(), refined_mask.max())
            print("Kept only the largest region and set count_mask to 1.")
    else:
        # count_refined_mask = count_mask
        refined_mask = mask.copy()

    # Filter out all connected components with size smaller or equal filter_th
    for label in range(1, count_mask + 1):
        if len(refined_mask[labelled_mask == label]) <= filter_th:
            refined_mask[labelled_mask == label] = 0
            # count_refined_mask -= 1

    # refined_mask has to be relabeled now.
    relabelled_mask, recounted_mask = scipy.ndimage.label(refined_mask)
    if recounted_mask < count_mask and verbose:
        print("Removed ", count_mask - recounted_mask, " regions because they are smaller or equal ", filter_th,
              " pixels.")
    filtered_mask = np.array(relabelled_mask > 0)

    return relabelled_mask, recounted_mask, filtered_mask


def get_count_and_area_rmv_podo_outside(cfg, mask, filter_mask, filter_th, verbose=False):
    # Outputs the labelled_mask, the mask_count and the filtered_mask
    # The mask is labeled, then cropped by the filter_mask
    # Afterwards, all labels which are contained in the mask are not removed in the labelled_mask

    labelled_mask, count_mask = scipy.ndimage.label(mask)

    if cfg.GLOM_POSTPROCESSING_KEEP_ONLY_LARGEST is True:
        labeled_filter_mask, dataset_filter_mask_count, filtered_filter_mask = get_count_and_area\
            (filter_mask, cfg.FILTER_CLASSES[0], keep_only_largest_label=True, verbose=verbose)
    else:
        labeled_filter_mask, dataset_filter_mask_count, filtered_filter_mask = get_count_and_area\
            (filter_mask, cfg.FILTER_CLASSES[0], verbose=verbose)

    labelled_mask_copy = labelled_mask.copy()
    labelled_mask_copy2 = labelled_mask.copy()
    labelled_mask_copy[filtered_filter_mask == 0] = 0
    if verbose:
        print(labelled_mask_copy.max(), labelled_mask_copy.min())
    labels_not_cropped = np.unique(labelled_mask_copy)
    labels_not_cropped = np.trim_zeros(labels_not_cropped)
    if verbose:
        print(labels_not_cropped)
    final_mask = np.isin(labelled_mask_copy2, labels_not_cropped)
    if verbose:
        print(final_mask.max(), final_mask.min())

    return get_count_and_area(final_mask, filter_th, verbose=verbose)


def image_to_label_image(img):
    label, count = scipy.ndimage.label(img)
    return label, count


def coregistrate_and_get_object_dc_score(label_pred, count_pred, label_mask, count_mask, verbose=0):
    def dice_coeff_with_intersect_matrix(matrix, tensor):
        intersection_matrices = matrix * tensor
        intersection_sum_array = np.sum(intersection_matrices, axis=(1,2))
        score_array = (2 * intersection_sum_array + 1.) / (np.sum(matrix) + np.sum(tensor, axis=(1,2)) + 1.)
        return score_array, intersection_sum_array

    def get_true_positives_and_false_negatives_all_cells():
        true_positives = []
        false_negatives = []
        array_dim = label_pred.shape
        prediction_array = np.empty((count_pred, array_dim[0], array_dim[1]))
        score_arrays = np.zeros((count_mask, count_pred))
        for i in range(count_pred):
            prediction_array[i,:,:] = np.array([label_pred == i+1])
        if verbose:
            print(prediction_array.shape)
            print(np.max(prediction_array))
            print(np.min(prediction_array))
        for k in range(1, count_mask + 1):
            score_arr, intersection_sum_arr = dice_coeff_with_intersect_matrix(np.array([label_mask == k]),
                                                                               prediction_array)
            if verbose:
                print("Intersection array: ")
                print(intersection_sum_arr)
                print("Score array: ")
                print(score_arr)
            if np.max(intersection_sum_arr) == 0:
                if verbose:
                    print("cell ", k, " in the groundtruth colocalizes with no cell in the prediction")
                false_negatives.append((k, 0))
            elif np.max(intersection_sum_arr > 0):
                score_arrays[k-1, :] = score_arr
        cells_to_process = min(count_mask - len(false_negatives), count_pred)
        while cells_to_process:
            i, j = np.unravel_index(score_arrays.argmax(), score_arrays.shape)
            cell_mask = i + 1
            cell_pred = j + 1
            if verbose:
                print("Cells to process: ", cells_to_process)
                print("cell ", cell_mask, " in groundtruth colocalizes the BEST with cell ", cell_pred,
                      " in the prediction")
            true_positives.append((cell_mask, cell_pred, np.max(score_arrays)))
            score_arrays[i, :] = 0
            score_arrays[:, j] = 0
            cells_to_process -= 1

        true_positives.sort()
        list_tp= [x[0] for x in true_positives]
        list_mask = list(range(1, count_mask + 1))
        for element in false_negatives:
            list_mask.remove(element[0])
        additional_false_negs = list(set(list_mask) - set(list_tp))
        additional_false_negs = [(x, 0) for x in additional_false_negs]
        additional_false_negs.sort()
        if verbose:
            print("The cells ", additional_false_negs, " in the groundtruth colocalize with prediction cells that "
                                                       "match better to other cells. Thus this cells will be counted "
                                                       "as false negative.")
        false_negatives = false_negatives + additional_false_negs

        return true_positives, false_negatives

    def get_false_positives(tp):
        list_tp = [x[1] for x in tp]
        list_pred = list(range(1, count_pred + 1))
        false_positives = list(set(list_pred) - set(list_tp))
        false_positives = [(0, x) for x in false_positives]
        false_positives.sort()
        return false_positives

    if np.max(label_pred) > 0:
        # True positives, false negatives
        tp, fn = get_true_positives_and_false_negatives_all_cells()
        # False positives
        fp = get_false_positives(tp)
    else:
        print("Warning. label_pred is a zero array. Thus TP = 0, FP = 0.")
        tp, fp = [], []
        fn = [(k, 0) for k in range(1,count_mask+1)]
    # object_dc-score
    if len(tp) > 0:
        object_dc_score = (2 * len(tp)) / (len(fp) + len(fn) + 2 * len(tp))
    else:
        object_dc_score = 0

    return object_dc_score, tp, fp, fn


def run_script():
    import yaml
    with open("config/parameters_train.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    path = "/source/"
    mask = skimage.io.imread(path + 'groundtruth/podocytes/A_mask_podo.tif')
    pred = skimage.io.imread(path + 'imagej/podocytes/A_mask_podo.tif')

    label_pred, count_pred = image_to_label_image(pred)
    label_mask, count_mask = image_to_label_image(mask)
    print("The pred image has ", count_pred, " cells.")
    print("The mask image has ", count_mask, " cells.")
    object_dc, tp, fp, fn = coregistrate_and_get_object_dc_score(label_pred, count_pred, label_mask, count_mask, verbose=1)
    print("The object_dc-score is: ", object_dc)
    print("There are ", len(tp), " TP cells: ", tp)
    print("There are ", len(fp), " FP cells: ", fp)
    print("There are ", len(fn), " FN cells: ", fn)

    return


if __name__ == '__main__':
    from config import Config

    # Uncomment to test object_dv, tp, fp, fn
    #run_script()

    # Uncomment to do no testing of Remove podocytes outside glom
    #"""

    cfg = Config()

    # Create a dict containing the masks
    msks_dict = {}
    mask_list = cfg.NAMES_CLASSES

    # Load img and masks
    path = '/data/test_postprocessing'
    img = skimage.io.imread(os.path.join(path, 'images', 'A.tif'))
    mask_glom_name = 'A_mask_glom.tif'
    mask_podo_name = 'A_mask_podo.tif'
    mask_glom = skimage.io.imread(os.path.join(path, 'masks', mask_glom_name))
    mask_podo = skimage.io.imread(os.path.join(path, 'masks', mask_podo_name))

    # Display img and masks
    msks_dict[mask_list[0]] = mask_glom
    msks_dict[mask_list[1]] = mask_podo
    plot_img_and_segm_overlayed(img[:, :, (1,2,0)], msks_dict, mask_list, ['Set1', 'hot'], [False, True])

    # Remove podocytes outside glom
    filter_th = 0
    relabelled_mask, recounted_mask, filtered_mask = get_count_and_area_rmv_podo_outside(
        cfg, mask_podo, mask_glom, filter_th, verbose=False)

    # Plot img and processed masks
    msks_dict[mask_list[0]] = mask_glom
    msks_dict[mask_list[1]] = filtered_mask
    plot_img_and_segm_overlayed(img[:, :, (1, 2, 0)], msks_dict, mask_list, ['Set1', 'hot'], [False, True])
