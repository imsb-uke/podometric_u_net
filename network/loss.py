import numpy as np
import tensorflow as tf
from tensorflow.python.keras import losses

from tensorflow.python.ops.losses import losses_impl

from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import optimizers
from network.architectures.unet import define_unet
from config.config import Config
import network.tf_image_generator as tf_img_gen


#####################################################
# Generalized dice loss and helper functions        #
#####################################################

def labels_to_one_hot(ground_truth, num_classes=1):
    """
    Converts ground truth labels to one-hot, sparse tensors.
    Used extensively in segmentation losses.

    :param ground_truth: ground truth categorical labels (rank `N`)
    :param num_classes: A scalar defining the depth of the one hot dimension
        (see `depth` of `tf.one_hot`)
    :return: one-hot sparse tf tensor
        (rank `N+1`; new axis appended at the end)
    """
    # read input/output shapes
    if isinstance(num_classes, tf.Tensor):
        num_classes_tf = tf.to_int32(num_classes)
    else:
        num_classes_tf = tf.constant(num_classes, tf.int32)
    input_shape = tf.shape(ground_truth)
    output_shape = tf.concat(
        [input_shape, tf.reshape(num_classes_tf, (1,))], 0)

    if num_classes == 1:
        # need a sparse representation?
        return tf.reshape(ground_truth, output_shape)

    # squeeze the spatial shape
    ground_truth = tf.reshape(ground_truth, (-1,))
    # shape of squeezed output
    dense_shape = tf.stack([tf.shape(ground_truth)[0], num_classes_tf], 0)

    # create a rank-2 sparse tensor
    ground_truth = tf.to_int64(ground_truth)
    ids = tf.range(tf.to_int64(dense_shape[0]), dtype=tf.int64)
    ids = tf.stack([ids, ground_truth], axis=1)
    one_hot = tf.SparseTensor(
        indices=ids,
        values=tf.ones_like(ground_truth, dtype=tf.float32),
        dense_shape=tf.to_int64(dense_shape))

    # resume the spatial dims
    one_hot = tf.sparse_reshape(one_hot, output_shape)
    return one_hot


def generalised_dice_loss(prediction,
                          ground_truth,
                          weight_map=None,
                          type_weight='Square'):
    """
    Function to calculate the Generalised Dice Loss defined in
        Sudre, C. et. al. (2017) Generalised Dice overlap as a deep learning
        loss function for highly unbalanced segmentations. DLMIA 2017

    :param prediction: the logits
    :param ground_truth: the segmentation ground truth
    :param weight_map:
    :param type_weight: type of weighting allowed between labels (choice
        between Square (square of inverse of volume),
        Simple (inverse of volume) and Uniform (no weighting))
    :return: the loss
    """
    prediction = tf.cast(prediction, tf.float32)
    if len(ground_truth.shape) == len(prediction.shape):
        ground_truth = ground_truth[..., -1]
    one_hot = labels_to_one_hot(ground_truth, tf.shape(prediction)[-1])

    if weight_map is not None:
        num_classes = prediction.shape[1].value
        # weight_map_nclasses = tf.reshape(
        #     tf.tile(weight_map, [num_classes]), prediction.get_shape())
        weight_map_nclasses = tf.tile(
            tf.expand_dims(tf.reshape(weight_map, [-1]), 1), [1, num_classes])
        ref_vol = tf.sparse_reduce_sum(
            weight_map_nclasses * one_hot, reduction_axes=[0])

        intersect = tf.sparse_reduce_sum(
            weight_map_nclasses * one_hot * prediction, reduction_axes=[0])
        seg_vol = tf.reduce_sum(
            tf.multiply(weight_map_nclasses, prediction), 0)
    else:
        ref_vol = tf.sparse_reduce_sum(one_hot, reduction_axes=[0])
        intersect = tf.sparse_reduce_sum(one_hot * prediction,
                                         reduction_axes=[0])
        seg_vol = tf.reduce_sum(prediction, 0)
    if type_weight == 'Square':
        weights = tf.reciprocal(tf.square(ref_vol))
    elif type_weight == 'Simple':
        weights = tf.reciprocal(ref_vol)
    elif type_weight == 'Uniform':
        weights = tf.ones_like(ref_vol)
    else:
        raise ValueError("The variable type_weight \"{}\""
                         "is not defined.".format(type_weight))
    new_weights = tf.where(tf.is_inf(weights), tf.zeros_like(weights), weights)
    weights = tf.where(tf.is_inf(weights), tf.ones_like(weights) *
                       tf.reduce_max(new_weights), weights)
    generalised_dice_numerator = \
        2 * tf.reduce_sum(tf.multiply(weights, intersect))
    # generalised_dice_denominator = \
    #     tf.reduce_sum(tf.multiply(weights, seg_vol + ref_vol)) + 1e-6
    generalised_dice_denominator = tf.reduce_sum(
        tf.multiply(weights, tf.maximum(seg_vol + ref_vol, 1)))
    generalised_dice_score = \
        generalised_dice_numerator / generalised_dice_denominator
    generalised_dice_score = tf.where(tf.is_nan(generalised_dice_score), 1.0,
                                      generalised_dice_score)
    return 1 - generalised_dice_score


def generalised_dice_loss_helper():
    def gdl(y_true, y_pred):
        return generalised_dice_loss(y_pred, y_true)
    return gdl


#####################################################
# DiceLoss class built on Keras Loss                #
#####################################################

class CustomBceLoss(losses.Loss):
    """CrossEntropyLoss class.
        To be implemented by subclasses:
        * `call()`: Contains the logic for loss calculation using `y_true`, `y_pred`.
        Example subclass implementation:
        ```
        class MeanSquaredError(Loss):
        def call(self, y_true, y_pred):
            y_pred = ops.convert_to_tensor(y_pred)
            y_true = math_ops.cast(y_true, y_pred.dtype)
            return K.mean(math_ops.square(y_pred - y_true), axis=-1)
        ```
        Args:
        reduction: Type of `tf.losses.Reduction` to apply to loss. Default value is
          `SUM_OVER_BATCH_SIZE`.
        name: Optional name for the op.
        """

    def __init__(self,
                 sample_weights,
                 reduction=losses_impl.ReductionV2.NONE,
                 name=None):
        self.reduction = reduction
        self.name = name
        self.sample_weights = sample_weights

    def call(self, y_true, y_pred):
        """Invokes the `CrossEntropyLoss` instance.
        Args:
          y_true: Ground truth values.
          y_pred: The predicted values.
        Returns:
          Mean squared error losses.
        """
        if self.sample_weights is not None:
            return custom_bce(y_true, y_pred, self.sample_weights)
        else:
            return custom_bce(y_true, y_pred, None)


class TwoLayerCustomBceLoss(losses.Loss):
    """TwoLayerCrossEntropyLoss class.
        To be implemented by subclasses:
        * `call()`: Contains the logic for loss calculation using `y_true`, `y_pred`.
        Example subclass implementation:
        ```
        class MeanSquaredError(Loss):
        def call(self, y_true, y_pred):
            y_pred = ops.convert_to_tensor(y_pred)
            y_true = math_ops.cast(y_true, y_pred.dtype)
            return K.mean(math_ops.square(y_pred - y_true), axis=-1)
        ```
        Args:
        reduction: Type of `tf.losses.Reduction` to apply to loss. Default value is
          `SUM_OVER_BATCH_SIZE`.
        name: Optional name for the op.
        """

    def __init__(self,
                 sample_weights,
                 reduction=losses_impl.ReductionV2.NONE,
                 name=None):
        self.reduction = reduction
        self.name = name
        self.sample_weights = sample_weights

    def call(self, y_true, y_pred):
        """Invokes the `CrossEntropyLoss` instance.
        Args:
          y_true: Ground truth values.
          y_pred: The predicted values.
        Returns:
          Mean squared error losses.
        """
        if self.sample_weights is not None:
            return custom_2layer_bce(y_true, y_pred, self.sample_weights)
        else:
            return custom_2layer_bce(y_true, y_pred, None)


class BalancedTwoLayerCustomBceLoss(losses.Loss):
    """BalancedTwoLayerCrossEntropyLoss class.
        To be implemented by subclasses:
        * `call()`: Contains the logic for loss calculation using `y_true`, `y_pred`.
        Example subclass implementation:
        ```
        class MeanSquaredError(Loss):
        def call(self, y_true, y_pred):
            y_pred = ops.convert_to_tensor(y_pred)
            y_true = math_ops.cast(y_true, y_pred.dtype)
            return K.mean(math_ops.square(y_pred - y_true), axis=-1)
        ```
        Args:
        reduction: Type of `tf.losses.Reduction` to apply to loss. Default value is
          `SUM_OVER_BATCH_SIZE`.
        name: Optional name for the op.
        """

    def __init__(self,
                 sample_weights,
                 reduction=losses_impl.ReductionV2.NONE,
                 name=None):
        self.reduction = reduction
        self.name = name
        self.sample_weights = sample_weights

    def call(self, y_true, y_pred):
        """Invokes the `CrossEntropyLoss` instance.
        Args:
          y_true: Ground truth values.
          y_pred: The predicted values.
        Returns:
          Mean squared error losses.
        """
        if self.sample_weights is not None:
            return custom_balanced_2layer_bce(y_true, y_pred, self.sample_weights)
        else:
            return custom_balanced_2layer_bce(y_true, y_pred, None)


class DiceLoss(losses.Loss):
    """DiceLoss class.
    To be implemented by subclasses:
    * `call()`: Contains the logic for loss calculation using `y_true`, `y_pred`.
    Example subclass implementation:
    ```
    class MeanSquaredError(Loss):
    def call(self, y_true, y_pred):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        return K.mean(math_ops.square(y_pred - y_true), axis=-1)
    ```
    Args:
    reduction: Type of `tf.losses.Reduction` to apply to loss. Default value is
      `SUM_OVER_BATCH_SIZE`.
    name: Optional name for the op.
    """

    def __init__(self,
                 sample_weights,
                 reduction=losses_impl.ReductionV2.NONE,
                 name=None):
        self.reduction = reduction
        self.name = name
        self.sample_weights = sample_weights

    def call(self, y_true, y_pred):
        """Invokes the `DiceLoss` instance.
        Args:
          y_true: Ground truth values.
          y_pred: The predicted values.
        Returns:
          Mean squared error losses.
        """
        if self.sample_weights is not None:
            return dice_loss(y_true, y_pred, self.sample_weights, round=False)
        else:
            return dice_loss(y_true, y_pred, round=False)


class TwoLayerDiceLoss(losses.Loss):
    """TwoLayerDiceLoss class.
    To be implemented by subclasses:
    * `call()`: Contains the logic for loss calculation using `y_true`, `y_pred`.
    Example subclass implementation:
    ```
    class MeanSquaredError(Loss):
    def call(self, y_true, y_pred):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        return K.mean(math_ops.square(y_pred - y_true), axis=-1)
    ```
    Args:
    reduction: Type of `tf.losses.Reduction` to apply to loss. Default value is
      `SUM_OVER_BATCH_SIZE`.
    name: Optional name for the op.
    """

    def __init__(self,
                 sample_weights,
                 reduction=losses_impl.ReductionV2.NONE,
                 name=None):
        self.reduction = reduction
        self.name = name
        self.sample_weights = sample_weights

    def call(self, y_true, y_pred):
        """Invokes the `TwoLayerDiceLoss` instance.
        Args:
          y_true: Ground truth values.
          y_pred: The predicted values.
        Returns:
          Two-layered dice losses.
        """
        if self.sample_weights is not None:
            return twolayer_dice_loss(y_true, y_pred, self.sample_weights, round=False)
        else:
            return twolayer_dice_loss(y_true, y_pred, round=False)


class BalancedTwoLayerDiceLoss(losses.Loss):
    """BalancedTwoLayerDiceLoss class.
    To be implemented by subclasses:
    * `call()`: Contains the logic for loss calculation using `y_true`, `y_pred`.
    Example subclass implementation:
    ```
    class MeanSquaredError(Loss):
    def call(self, y_true, y_pred):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        return K.mean(math_ops.square(y_pred - y_true), axis=-1)
    ```
    Args:
    reduction: Type of `tf.losses.Reduction` to apply to loss. Default value is
      `SUM_OVER_BATCH_SIZE`.
    name: Optional name for the op.
    """

    def __init__(self,
                 sample_weights,
                 reduction=losses_impl.ReductionV2.NONE,
                 name=None):
        self.reduction = reduction
        self.name = name
        self.sample_weights = sample_weights

    def call(self, y_true, y_pred):
        """Invokes the `TwoLayerDiceLoss` instance.
        Args:
          y_true: Ground truth values.
          y_pred: The predicted values.
        Returns:
          Two-layered dice losses.
        """
        if self.sample_weights is not None:
            return balanced_twolayer_dice_loss(y_true, y_pred, self.sample_weights, round=False)
        else:
            return balanced_twolayer_dice_loss(y_true, y_pred, round=False)


#####################################################
# Dice coeffs, losses and help functions            #
#####################################################

def dice_coeff(y_true, y_pred, weights=None, round=True):
    """Dice coefficient between a prediction and a target tensor.
    # Arguments
        y_true: A tensor.
        y_pred: A tensor with the same shape as `y_true`.
        weights: (Optional) weight matrix to weight the regions differently.
    # Returns
        A floating point value.
    """
    smooth = 1.

    # Flatten the two tensors to be compared
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    if round:
        y_true_f = tf.round(y_true_f)
        y_pred_f = tf.round(y_pred_f)

    if weights is not None:
        # Weight the different parts of the sums according the weight matrix
        weights_f = tf.reshape(weights, [-1])
        intersection = tf.reduce_sum(weights_f * y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (tf.reduce_sum(weights_f * y_true_f) +
                                                tf.reduce_sum(weights_f * y_pred_f) + smooth)
        # Warning: With this configuration the Weighting of the background has no influence at all,
        # since it is multiplied with 0. -> generalized dice loss ?
    else:
        # Without weighting
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score


def twolayer_dice_coeff(y_trues, y_preds, weights=None):
    return - twolayer_dice_loss(y_trues, y_preds, weights) + 1


def glom_dice_coeff(y_trues, y_preds, weights=None):
    return - glom_dice_loss(y_trues, y_preds, weights) + 1


def podo_dice_coeff(y_trues, y_preds, weights=None):
    return - podo_dice_loss(y_trues, y_preds, weights) + 1


def dice_coeff_numpy(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    score = (2 * intersection + 1.) / (y_true.sum() + y_pred.sum() + 1.)
    return score


def dice_loss(y_true, y_pred, weights=None, round=True):
    """Dice loss (1 - dice coefficient) between a prediction and a target tensor.
    # Arguments
        y_true: A tensor.
        y_pred: A tensor with the same shape as `y_true`.
        weights: (Optional) weight matrix to weight the regions differently.
    # Returns
        A floating point value.
    """
    d_loss = 1 - dice_coeff(y_true, y_pred, weights, round)
    return d_loss


def dice_loss_no_round(y_true, y_pred, weights=None):
    """
    Wrapper for metric dice_loss_no_round
    """
    return dice_loss(y_true, y_pred, weights=weights, round=False)


# def separate_dice_loss(y_trues, y_preds, weights=None):
#    d_loss_1 = glom_dice_loss(y_trues, y_preds, weights)
#    d_loss_2 = podo_dice_loss(y_trues, y_preds, weights)
#
#    return d_loss_1, d_loss_2


def twolayer_dice_loss(y_true, y_pred, weights=None, round=True):
    """Dice coefficient between a prediction and a target tensor.
    # Arguments
        y_true: A tensor.
        y_pred: A tensor with the same shape as `y_true`.
        weights: (Optional) weight matrix to weight the regions differently.
    # Returns
        A floating point value.
    """
    ml_dice_loss = (glom_dice_loss(y_true, y_pred, weights, round) + podo_dice_loss(y_true, y_pred, weights, round)) / 2
    return ml_dice_loss


def twolayer_dice_loss_no_round(y_true, y_pred, weights=None):
    """
    Wrapper for metric twolayer_dice_loss_no_round
    """
    return twolayer_dice_loss(y_true, y_pred, weights=None, round=False)


def balanced_twolayer_dice_loss(y_true, y_pred, weights=None, round=True):
    """Dice coefficient between a prediction and a target tensor.
    # Arguments
        y_true: A tensor.
        y_pred: A tensor with the same shape as `y_true`.
        weights: (Optional) weight matrix to weight the regions differently.
    # Returns
        A floating point value.
    """
    bal_2layer_dice_loss = (glom_dice_loss(y_true, y_pred, weights, round) / podo_dice_loss(y_true, y_pred, weights, round))\
                            * glom_dice_loss(y_true, y_pred, weights, round) + \
                           (podo_dice_loss(y_true, y_pred, weights, round) / glom_dice_loss(y_true, y_pred, weights, round))\
                            * podo_dice_loss(y_true, y_pred, weights, round)
    return bal_2layer_dice_loss


def glom_dice_loss(y_trues, y_preds, weights=None, round=True):
    """Dice loss (1 - dice coefficient) between a prediction and a target tensor.
    # Masks are splitted in their two parts (glom and podo)
    # Arguments
        y_true: A tensor.
        y_pred: A tensor with the same shape as `y_true`.
        weights: (Optional) weight matrix to weight the regions differently.
    # Returns
        A floating point value.
    """
    y_true_1, y_true_2 = tf.split(y_trues, 2, axis=3)
    y_pred_1, y_pred_2 = tf.split(y_preds, 2, axis=3)
    if weights is not None:
        weights_1, weights_2 = tf.split(weights, 2, axis=3)
    else:
        weights_1 = None
    # print(y_true_1.shape, y_true_2.shape, y_pred_1.shape, y_pred_2.shape)
    # d_loss_1 = 1 - dice_coeff(tf.squeeze(y_true_1), tf.squeeze(y_pred_1), weights)
    d_loss_1 = 1 - dice_coeff(y_true_1, y_pred_1, weights_1, round)
    return d_loss_1


def podo_dice_loss(y_trues, y_preds, weights=None, round=True):
    """Dice loss (1 - dice coefficient) between a prediction and a target tensor.
    # Masks are splitted in their two parts (glom and podo)
    # Arguments
        y_true: A tensor.
        y_pred: A tensor with the same shape as `y_true`.
        weights: (Optional) weight matrix to weight the regions differently.
    # Returns
        A floating point value.
    """
    y_true_1, y_true_2 = tf.split(y_trues, 2, axis=3)
    y_pred_1, y_pred_2 = tf.split(y_preds, 2, axis=3)
    if weights is not None:
        weights_1, weights_2 = tf.split(weights, 2, axis=3)
    else:
        weights_2 = None
    # d_loss_2 = 1 - dice_coeff(tf.squeeze(y_true_2), tf.squeeze(y_pred_2), weights)
    d_loss_2 = 1 - dice_coeff(y_true_2, y_pred_2, weights_2, round)
    return d_loss_2


def dice_loss_helper(round=False):
    """Helper function
    # Arguments

    # Returns
        A function.
    """
    def own_dice(y_true, y_pred):
        weights = None
        return dice_loss(y_true, y_pred, weights, round)
    return own_dice


def twolayer_dice_helper(round=False):
    def own_twolayer_dice(y_true, y_pred):
        weights = None
        return twolayer_dice_loss(y_true, y_pred, weights, round)
    return own_twolayer_dice


def balanced_2layer_dice_helper(round=False):
    def own_balanced_dice(y_true, y_pred):
        weights = None
        return balanced_twolayer_dice_loss(y_true, y_pred, weights, round)
    return own_balanced_dice


#####################################################
# Own binary-crossentropy losses and help functions #
#####################################################

def custom_bce(target, output, weights, name=None):
    """Binary crossentropy between an output tensor and a target tensor.
    # Arguments
        target: A tensor with the same shape as `output`.
        output: A tensor.
        weights: (Optional) weight matrix to weight the regions differently.
        from_logits: Whether `output` is expected to be a logits tensor.
            By default, we consider that `output`
            encodes a probability distribution.
    # Returns
        A tensor.
    """
    # Note: tf.nn.sigmoid_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    # transform back to logits

    _epsilon = tf.convert_to_tensor(tf.keras.backend.epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
    output = tf.log(output / (1 - output))

    with tf.name_scope(name, "logistic_loss", [output, target]) as name:
        logits = tf.convert_to_tensor(output, name="logits")
        labels = tf.convert_to_tensor(target, name="labels")
        try:
            labels.get_shape().merge_with(logits.get_shape())
        except ValueError:
            raise ValueError("logits and labels must have the same shape (%s vs %s)" %
                             (logits.get_shape(), labels.get_shape()))

        # The logistic loss formula from above is
        #   x - x * z + log(1 + exp(-x))
        # For x < 0, a more numerically stable formula is
        #   -x * z + log(1 + exp(x))
        # Note that these two expressions can be combined into the following:
        #   max(x, 0) - x * z + log(1 + exp(-abs(x)))
        # To allow computing gradients at zero, we define custom versions of max and
        # abs functions.
        zeros = tf.zeros_like(logits, dtype=logits.dtype)
        cond = (logits >= zeros)
        relu_logits = tf.where(cond, logits, zeros)
        neg_abs_logits = tf.where(cond, -logits, logits)
        if weights is None:
            return tf.reduce_sum(tf.add(relu_logits - logits * labels,
                   tf.log1p(tf.exp(neg_abs_logits)), name=name)) \
                   / tf.cast(tf.reduce_prod(tf.shape(logits)), tf.float32)
        else:
            return tf.reduce_sum(weights * tf.add(relu_logits - logits * labels,
                   tf.log1p(tf.exp(neg_abs_logits)), name=name)) \
                   / tf.cast(tf.reduce_prod(tf.shape(logits)), tf.float32)


def custom_bce_helper():
    """Helper function
    # Arguments

    # Returns
        A function.
    """
    def own_bce(y_true, y_pred):
        return losses.binary_crossentropy(y_true, y_pred)
    return own_bce


def custom_2layer_bce(target, output, weights=None, name=None):
    """Binary crossentropy for multilayer y (e.g. (1,1024,1024,2)
       Use this function as a wrapper to call custom_bce"""
    target_1, target_2 = tf.split(target, 2, axis=3)
    pred_1, pred_2 = tf.split(output, 2, axis=3)

    bce1 = custom_bce(target_1, pred_1, weights, name)
    bce2 = custom_bce(target_2, pred_2, weights, name)
    return bce1 + bce2


def custom_balanced_2layer_bce(target, output, weights=None, name=None):
    """Binary crossentropy for multilayer y (e.g. (1,1024,1024,2)
       Use this function as a wrapper to call custom_bce"""
    target_1, target_2 = tf.split(target, 2, axis=3)
    pred_1, pred_2 = tf.split(output, 2, axis=3)

    bce1 = custom_bce(target_1, pred_1, weights, name)
    bce2 = custom_bce(target_2, pred_2, weights, name)

    return (bce1 / bce2) * bce1 + (bce2 / bce1) * bce2


def custom_2layer_bce_helper():
    """Helper function
    # Arguments

    # Returns
        A function.
    """
    def own_2layerbce(y_true, y_pred):
        return custom_2layer_bce(y_true, y_pred)
    return own_2layerbce


def custom_balanced_2layer_bce_helper():
    """Helper function
    # Arguments

    # Returns
        A function.
    """
    def own_bal_2layerbce(y_true, y_pred):
        return custom_balanced_2layer_bce(y_true, y_pred)
    return own_bal_2layerbce


#####################################################
# Loss functions combining different losses         #
#####################################################

def bce_dice_loss(y_true, y_pred):
    """Combination of binary cross-entropy and dice loss.
    # Arguments
        y_true: A tensor.
        y_pred: A tensor with the same shape as `y_true`.

    # Returns
        A floating point value.
    """
    bd_loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return bd_loss


def bce_dice_loss_helper():
    """Helper function to be able to use weights in bce_dice_loss.

    # Returns
        A function.
    """
    def bce_dice(y_true, y_pred):
        return bce_dice_loss(y_true, y_pred)
    return bce_dice


def cce_dice_loss(y_true, y_pred):
    """Combination of categorical cross-entropy and dice loss.
    # Arguments
        y_true: A tensor.
        y_pred: A tensor with the same shape as `y_true`.
    # Returns
        A floating point value.
    """
    cd_loss = losses.categorical_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return cd_loss


#####################################################
# Set loss and metric to use in the model           #
#####################################################

def set_loss(loss_name, sample_weights=None):
    # Basic losses without weighting

    # Losses for one class segmentation
    if loss_name == 'bce':
        model_loss = custom_bce_helper()
    elif loss_name == 'dice':
        model_loss = dice_loss_helper()
    # Losses for two class segmentation
    elif loss_name == 'twolayer_bce':
        model_loss = custom_2layer_bce_helper()
    elif loss_name == 'balanced_twolayer_bce':
        model_loss = custom_balanced_2layer_bce_helper()
    elif loss_name == 'twolayer_dice':
        model_loss = twolayer_dice_helper()
    elif loss_name == 'balanced_twolayer_dice':
        model_loss = balanced_2layer_dice_helper()
    elif loss_name == 'bce_dice':
        model_loss = bce_dice_loss_helper()
    elif loss_name == 'gdl':
        model_loss = generalised_dice_loss_helper()
    elif loss_name == 'keras_bce':
        model_loss = 'binary_crossentropy'

    # Losses which are able to include and load the weights,
    # which were created by get_weights.py

    # Losses for one class segmentation
    elif loss_name == 'CustomBceLoss':
        model_loss = CustomBceLoss(sample_weights)
    elif loss_name == 'DiceLoss':
        model_loss = DiceLoss(sample_weights)
    # Losses for two class segmentation
    elif loss_name == 'TwoLayerCustomBceLoss':
        model_loss = TwoLayerCustomBceLoss(sample_weights)
    elif loss_name == 'BalancedTwoLayerCustomBceLoss':
        model_loss = BalancedTwoLayerCustomBceLoss(sample_weights)
    elif loss_name == 'TwoLayerDiceLoss':
        model_loss = TwoLayerDiceLoss(sample_weights)
    elif loss_name == 'BalancedTwoLayerDiceLoss':
        model_loss = BalancedTwoLayerDiceLoss(sample_weights)

    else:
        raise ValueError("loss you want to use is not defined in set_loss")
    return model_loss


def set_metric(metric_list):
    model_metric = []
    for metric_name in metric_list:
        # For all the metrics, weighting is not considered

        # Metrics only for two class segmentation
        if metric_name == 'twolayer_dice_loss':
            model_metric.append(twolayer_dice_loss)
        elif metric_name == 'twolayer_dice_loss_no_round':
            model_metric.append(twolayer_dice_loss_no_round)
        # Metrics for one class segmentation
        elif metric_name == 'dice_loss':
            model_metric.append(dice_loss)
        elif metric_name == 'dice_loss_no_round':
            model_metric.append(dice_loss_no_round)
        # Metrics for class1 and class2 for two class segmentation
        elif metric_name == 'glom_dice_loss':
            model_metric.append(glom_dice_loss)
        elif metric_name == 'podo_dice_loss':
            model_metric.append(podo_dice_loss)

        else:
            print(metric_list)
            raise ValueError("metric you want to use is not defined in set_metric")
    return model_metric


if __name__ == '__main__':
    """
    Test the implemented loss function
    """

    print("loss.py")

    # Compare dice_loss with seperate dice_loss
    class lossConfig(Config):
        GPUs = '0'
        IMAGES_PER_GPU = 1
        # BATCH_SIZE = x  # BATCH_SIZE is set to IMAGES_PER_GPU in the config

        SEGMENTATION_TASK = 'all'  #'all'  #'glomerulus'  # 'podocytes'
        NAMES_CLASSES = ['rectangles_a', 'rectangles_b']
        MASK_SUFFIXES = ['_a.png', '_b.png']
        WEIGHTS_SUFFIXES = ['_weight_a.npy', '_weight_b.npy']

        UNET_FILTERS = 4  # 32, 64
        UNET_LAYERS = 2  # 3, 4, 5

        DROPOUT_SKIP = True
        BATCH_NORM = True

        LOSS = 'TwoLayerCustomBceLoss'  # 'DiceLoss'  #'TwoLayerDiceLoss'  #'BalancedTwoLayerDiceLoss'
        #                                 'CustomBceLoss'  #'TwoLayerCustomBceLoss'  #'BalancedTwoLayerCustomBceLoss'
        WEIGHTING = True  # True

    cfg = lossConfig()
    data_path = ['/data/Dataset1/']
    # load_model = "best_model.hdf5"

    # load keras model
    inputs = layers.Input(shape=cfg.TARGET_IMG_SHAPE)
    outputs = inputs  # Initialise to be able to use it in if-clause
    hidden_layers, outputs = define_unet(inputs, cfg)

    # Add an extra input when using loss weighting
    if cfg.WEIGHTING:
        sample_weights = layers.Input((cfg.TARGET_IMG_SHAPE[0], cfg.TARGET_IMG_SHAPE[1],
                                       cfg.NUMBER_MSK_CHANNELS * cfg.NUM_OUTPUT_CH))
        model = models.Model(inputs=(inputs, sample_weights), outputs=[outputs])
    else:
        sample_weights = None
        model = models.Model(inputs=[inputs], outputs=[outputs])

    # Set loss
    # Include weights for loss weighting if True
    if cfg.WEIGHTING:
        model_loss = set_loss(cfg.LOSS, sample_weights=sample_weights)
        print("config loss:", cfg.LOSS)
        print("model loss:", model_loss)
    else:
        model_loss = set_loss(cfg.LOSS)
        print("config loss:", cfg.LOSS)
        print("model loss:", model_loss)

    # Set metric
    model_metric = set_metric(['dice_loss', 'dice_loss_no_round'])

    optim = optimizers.Adam(lr=cfg.LEARNING_RATE, beta_1=cfg.BETA_1, beta_2=cfg.BETA_2, epsilon=None, amsgrad=True)

    # use keras.model.evaluate
    model.compile(optimizer=optim, loss=model_loss,
                  metrics=model_metric)

    # Load the weights of the trained model
    # model.load_weights(load_model)

    if cfg.WEIGHTING:
        print("cfg.BATCH_SIZE", cfg.BATCH_SIZE)
        img, mask, wgt = tf_img_gen.load_random_batch(cfg, data_path)
        print(img.shape, mask.shape, wgt.shape)
        pred = model.predict((img, wgt))
    else:
        img, mask = tf_img_gen.load_random_batch(cfg, data_path)
        print(img.shape, mask.shape)
        pred = model.predict(img)

    print("pred shape", pred.shape)
    print("pred min max", np.min(pred), np.max(pred))
    # Threshold prediction
    dice_calculation_numpy = dice_coeff_numpy(mask, pred)
    pred = pred > cfg.PREDICTION_THRESHOLD
    dice_calculation_numpy_thresholded = dice_coeff_numpy(mask, pred)
    print("pred min max after", np.min(pred), np.max(pred))

    # Plot image, mask and prediction. First row: image, mask; Second row: image, prediction
    x_batch = np.concatenate((img, img), axis=0)
    y_batch = np.concatenate((mask, pred), axis=0)
    if cfg.WEIGHTING:
        evalu1 = model.evaluate((img, wgt), mask, batch_size=cfg.BATCH_SIZE)
        wgt_batch = np.concatenate((wgt, wgt), axis=0)
        batch = [x_batch, y_batch, wgt_batch]
    else:
        evalu1 = model.evaluate(img, mask, batch_size=cfg.BATCH_SIZE)
        batch = [x_batch, y_batch]
    print("model loss and metric:", evalu1)
    print("dice calculation numpy:", dice_calculation_numpy)
    print("dice calculation numpy thresholded:", dice_calculation_numpy_thresholded)
    tf_img_gen.plot_batch(batch)

