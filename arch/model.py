# u-net model with up-convolution or up-sampling and weighted binary-crossentropy as loss func

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    concatenate,
    Conv2DTranspose,
    BatchNormalization,
    Dropout,
)

# from keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K

# from tensorflow.keras import ops

# import tensorflow_addons as tfa
from tensorflow.keras import optimizers

import tensorflow as tf

from tensorflow.keras.regularizers import l2

import tensorflow_io as tfio
import numpy as np

from losses import *

def conv_block(
    inputs, n_filters, dropout_prob=0, max_pooling=True, w_decay=0, norm=True
):
    """
    Convolutional downsampling block

    Arguments:
        inputs -- Input tensor
        n_filters -- Number of filters for the convolutional layers
        dropout_prob -- Dropout probability
        max_pooling -- Use MaxPooling2D to reduce the spatial dimensions of the output volume
    Returns:
        next_layer, skip_connection --  Next layer and skip connection outputs
    """
    if norm:
        inputs = BatchNormalization()(inputs)

    conv = Conv2D(
        n_filters,  # Number of filters
        3,  # Kernel size
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(w_decay),
    )(inputs)
    conv = Conv2D(
        n_filters,  # Number of filters
        3,  # Kernel size
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(w_decay),
    )(conv)

    # if dropout_prob > 0 add a dropout layer, with the variable dropout_prob as parameter
    if dropout_prob > 0:
        conv = Dropout(dropout_prob)(conv)

    # if max_pooling is True add a MaxPooling2D with 2x2 pool_size
    if max_pooling:
        next_layer = MaxPooling2D(2, strides=2)(conv)
    else:
        next_layer = conv

    skip_connection = conv

    return next_layer, skip_connection


def upsampling_block(expansive_input, contractive_input, n_filters, norm=False):
    """
    Convolutional upsampling block

    Arguments:
        expansive_input -- Input tensor from previous layer
        contractive_input -- Input tensor from previous skip layer
        n_filters -- Number of filters for the convolutional layers
    Returns:
        conv -- Tensor output
    """

    up = Conv2DTranspose(
        n_filters, 3, strides=2, padding="same"  # number of filters  # Kernel size
    )(expansive_input)

    merge = concatenate(
        [up, contractive_input], axis=3
    )  # Merge the previous output and the contractive_input

    if norm:
        merge = BatchNormalization()(merge)

    conv = Conv2D(
        n_filters,  # Number of filters
        3,  # Kernel size
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(merge)
    conv = Conv2D(
        n_filters,  # Number of filters
        3,  # Kernel size
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv)

    return conv



def unet_model(
    n_classes,
    tile_width,
    tile_height,
    n_bands,
    n_blocks,
    class_weight_list,
    n_filters_start=64,
    w_decay=1e-5,
    droprate=0.3,
    drop_multiplier=None,
    weight_multiplier=None,
    filter_growth=2,
    normalize_inputs=True,
    optimizer="adam",
    loss_function="categorical_crossentropy",
    metrics=["categorical_accuracy"],
):
    inputs = Input((tile_width, tile_height, n_bands))
    x = inputs  # Initialize x to be the input tensor
    n_filters = n_filters_start
    contracting_blocks = []  # List to store the contractive blocks

    if drop_multiplier is None:
        drop_multiplier = [
            1.0
        ] * n_blocks  # Default: no modification to weight decay rates

    if weight_multiplier is None:
        weight_multiplier = [
            1.0
        ] * n_blocks  # Default: no modification to dropout rates

    for i in range(n_blocks):
        if normalize_inputs:
            norm = False if i == 0 else True
        else:
            norm = False
        if i < n_blocks - 1:
            x, skip = conv_block(
                inputs=x,
                n_filters=n_filters,
                w_decay=w_decay * weight_multiplier[i],
                dropout_prob=droprate * drop_multiplier[i],
                norm=norm,
            )
            contracting_blocks.append(skip)
        else:  # the bottleneck step has no max pooling and bacth normalization and output for skip connection is not saved
            x, _ = conv_block(
                inputs=x,
                n_filters=n_filters,
                w_decay=w_decay * weight_multiplier[i],
                dropout_prob=droprate * drop_multiplier[i],
                max_pooling=False,
                norm=False,
            )
        n_filters *= (
            filter_growth  # Double the number of filters in each downsampling block
        )

    contracting_blocks.reverse()
    n_filters //= filter_growth
    for i in range(n_blocks - 1):
        norm = False
        if normalize_inputs:
            norm = True
            if i == n_blocks - 2:
                norm = False
        n_filters //= (
            filter_growth  # Halve the number of filters in each upsampling block
        )
        x = upsampling_block(x, contracting_blocks[i], n_filters=n_filters, norm=norm)
        # x = upsampling_block(x, contracting_blocks.pop(), n_filters=n_filters)

    # x = Conv2D(n_filters,
    #         3,
    #         activation='relu',
    #         padding='same',
    #         kernel_initializer='he_normal')(x)

    outputs = Conv2D(n_classes, 1, activation="softmax", padding="same")(x)

    model = Model(inputs=inputs, outputs=outputs)

    class_weights = class_weight_list

    weighted_dice_loss = WeightedDiceLoss(class_weights)
    weighted_jaccard_loss = WeightedJaccardLoss(class_weights)

    alpha = class_weight_list
    # Initialize loss function with computed alpha
    weighted_cfce = CategoricalFocalCrossentropy(alpha=alpha)

    losses_dict = {
        "categorical_focal_crossentropy": cfce,
        "weighted_categorical_focal_crossentropy": weighted_cfce,
        "tversky_loss": tversky_loss,
        "dice_loss": dice_loss,
        "weighted_dice_loss": weighted_dice_loss,
        "cce_dice": cce_dice_loss,
        "focal_dice": cfce_dice_loss,
        "jaccard_loss": jaccard_loss,
        "weighted_jaccard_loss": weighted_jaccard_loss,
        "cce_jaccard": cce_jaccard_loss,
        "focal_jaccard": cfce_jaccard_loss,
    }

    loss = losses_dict.get(loss_function, f"{loss_function}")

    model.compile(
        optimizer=optimizer,  # "adam",  # optimizers.experimental.SGD(learning_rate=learning_rate), The learning rate. Defaults to .: model.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-4)) model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3)
        loss=loss,  # "categorical_crossentropy",  # weighted_binary_crossentropy,
        metrics=metrics,   # [
        #     "categorical_accuracy",
        #     # tfa.metrics.F1Score(
        #     #     num_classes=n_classes, average=None
        #     # ),  # , average="weighted"),
        #     # tfa.metrics.MultiLabelConfusionMatrix,
        #     f1_score,
        # ],
    )  # metrics.Precision(), metrics.Recall()]) #sparse_ optimizer="rmsprop" #f1, loss="categorical_crossentropy"

    return model

    # def weighted_binary_crossentropy(y_true, y_pred):
    #     class_loglosses = K.mean(K.binary_crossentropy(y_true, y_pred), axis=[0, 1, 2])
    #     return K.sum(class_loglosses * K.constant(class_weights))

    # f1 = tfa.metrics.F1Score(num_classes=n_classes, average=None)

    # Configure the model for training.
    # We use the "sparse" version of categorical_crossentropy
    # because our target data is integers.

    # metrics.Precision(), metrics.Recall()]) #sparse_ optimizer="rmsprop" #f1, loss="categorical_crossentropy"
    # More about metrics https://keras.io/guides/training_with_built_in_methods/


# def recall_m(y_true, y_pred):
#     y_true = K.ones_like(y_true)
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

#     recall = true_positives / (all_positives + K.epsilon())
#     return recall


# def precision_m(y_true, y_pred):
#     y_true = K.ones_like(y_true)
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     precision = true_positives / (predicted_positives + K.epsilon())
#     return precision


# def f1_score(y_true, y_pred):
#     precision = precision_m(y_true, y_pred)
#     recall = recall_m(y_true, y_pred)
#     return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
