import argparse
import logging
import os
import glob
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    accuracy_score,
    classification_report,
    jaccard_score,
)

# import tensorflow_addons as tfa
import tensorflow_io as tfio
import numpy as np

import json
from tensorboard.plugins.hparams import api as hp

# from sklearn.model_selection import train_test_split
# from keras.utils.np_utils import to_categorical

from model import unet_model
from data_handling import *

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("train_unet")


def get_dataset_from_args_lst(args_lst):
    file_lst = [
        fn.replace(",", "").replace("[", "").replace("]", "") for fn in args_lst
    ]

    pth_lst = [f"{args.data_dir}/{filename}" for filename in file_lst]

    ds = get_dataset(
        pth_lst,
        im_shape=[args.im_sz, args.im_sz, args.n_bands],
        categories=args.categories,
    )

    return ds


def get_predictions_and_labels(model, dataset):
    y_true = []
    y_pred = []

    for batch in dataset:
        inputs, labels = batch
        predictions = model.predict(inputs)
        y_true.extend(np.argmax(labels, axis=-1))
        y_pred.extend(np.argmax(predictions, axis=-1))

    return np.array(y_true), np.array(y_pred)


def get_labels(dataset):
    y_true = []
    for batch in dataset:
        inputs, labels = batch
        y_true.extend(np.argmax(labels, axis=-1))
    return y_true


# Compute class weights based on their frequency
def compute_class_weights(y_true):
    num_classes = tf.cast(tf.shape(y_true)[1], tf.float32)  # Cast to float32
    class_counts = tf.cast(tf.reduce_sum(y_true, axis=0), tf.float32)  # Cast to float32
    total_samples = tf.cast(tf.reduce_sum(class_counts), tf.float32)  # Cast to float32
    epsilon = tf.cast(1e-7, tf.float32)  # Cast to float32
    class_weights = total_samples / (num_classes * class_counts + epsilon)
    class_weights /= tf.reduce_sum(class_weights)  # Normalize weights
    return class_weights


# Function to calculate performance metrics
def calculate_performance_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true.flatten(), y_pred.flatten())
    precision, recall, fscore, support = precision_recall_fscore_support(
        y_true.flatten(), y_pred.flatten(), average="weighted", zero_division=0
    )
    conf_matrix = confusion_matrix(y_true.flatten(), y_pred.flatten())

    return accuracy, support, precision, recall, fscore, conf_matrix


def get_performance_measures(ds, model, set_name="", logger=logger):
    lst_arr = list(ds.as_numpy_iterator())
    logger.info("Stored as numpy")

    logger.info("predicting test data on model")
    pred = model.predict(x=ds)

    del ds

    y = lst_arr[0][1]
    for i in range(1, len(lst_arr)):
        y = np.concatenate([y, lst_arr[i][1]], axis=0)
    logger.info("iterated over all samples stored in one array")

    del lst_arr

    y_true = np.argmax(y, axis=-1)
    logger.info("Ground truth in categories calculated")
    del y

    y_pred = np.argmax(pred, axis=-1)
    logger.info("Predictions in categories calculated")

    accuracy = accuracy_score(y_true=y_true.flatten(), y_pred=y_pred.flatten())
    precision, recall, fscore, support = precision_recall_fscore_support(
        y_true=y_true.flatten(), y_pred=y_pred.flatten()
    )
    conf_matrix = confusion_matrix(y_true=y_true.flatten(), y_pred=y_pred.flatten())
    jaccard_score_iou = jaccard_score(
        y_true=y_true.flatten(), y_pred=y_pred.flatten(), average="weighted"
    )  # NN
    jaccard_score_iou_class = jaccard_score(
        y_true=y_true.flatten(), y_pred=y_pred.flatten(), average=None
    )  # NN

    logger.info("Performance metrics on set calculated")

    logger.info(f"{set_name} accuracy: {accuracy}")
    logger.info(f"{set_name} f-score: {fscore}")
    logger.info(f"{set_name} precision: {precision}")
    logger.info(f"{set_name} recall: {recall}")
    logger.info(f"{set_name} support: {support}")
    logger.info(f"{set_name} Confusion Matrix: {conf_matrix}")
    logger.info(f"{set_name} Jaccard index weight-averaged: {jaccard_score_iou}")
    logger.info(f"{set_name} Jaccard index per class: {jaccard_score_iou_class}")


# def get_dataset_partitions_tf(
#     ds,
#     ds_size,
#     train_split=0.8,
#     val_split=0.1,
#     test_split=0.1,
#     shuffle=True,
#     shuffle_size=10000,
# ):
#     assert (train_split + test_split + val_split) == 1

#     if shuffle:
#         # Specify seed to always have the same split distribution between runs
#         ds = ds.shuffle(shuffle_size, seed=12)

#     train_size = int(train_split * ds_size)
#     val_size = int(val_split * ds_size)

#     ds_train = ds.take(train_size)
#     val_ds = ds.skip(train_size).take(val_size)
#     test_ds = ds.skip(train_size).skip(val_size)

#     return ds_train, val_ds, test_ds


if __name__ == "__main__":
    # https://keras.io/examples/keras_recipes/tfrecord/

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.

    parser.add_argument("--categories", type=int, default=10)
    parser.add_argument("--im_sz", type=int, default=64)
    parser.add_argument("--n_bands", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument(
        "--train_file_lst", nargs="*", type=str, default=["train.tfrecords"]
    )
    parser.add_argument(
        "--validation_file_lst", nargs="*", type=str, default=["validation.tfrecords"]
    )
    parser.add_argument("--test_file_lst", nargs="*", type=str, default=None)

    parser.add_argument("--test_2_file_lst", nargs="*", type=str, default=None)
    parser.add_argument("--test_3_file_lst", nargs="*", type=str, default=None)

    parser.add_argument("--n_blocks", type=int, default=5)

    # parser.add_argument("--train_file_lst",  type=list, default=["train.tfrecords"])
    # parser.add_argument("--validation_file_lst", type=list, default=["validation.tfrecords"])

    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument(
        "--loss_function",
        type=str,
        default="categorical_crossentropy",
        help="Input a loss function, can be a named loss from tensorflow 2.14 or one of the following: categorical_focal_crossentropy, weighted_categorical_focal_crossentropy, dice_loss, weighted_dice_loss, cce_dice, focal_dice, jaccard_loss, weighted_jaccard_loss, cce_jaccard, focal_jaccard",
    )
    # TODO parser.add_argument("--metrics", nargs="*", type=str, default=["categorical_accuracy", "f1_score"])

    # parser.add_argument("--class_weights", type=str, default=None) # TODO check if is called somewhere
    parser.add_argument(
        "--class_weight_list",
        nargs="*",
        type=float,
        default=None,
        help="A list of class weights to be used during model training. If not provided, class weights will be balanced by default.",
    )
    parser.add_argument("--droprate", type=float, default=0.25)
    parser.add_argument("--w_decay", type=float, default=1e-3)

    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument(
        "--normalization",
        default=True,
        type=lambda x: True if x.lower() == "true" else False,
        metavar="CP",
        help="Normalize all layers input",
    )

    # parser.add_argument("--class_weights", type=str, default=None)

    # input data and model directories
    parser.add_argument("--model_dir", type=str, default="/opt/ml/model")

    parser.add_argument(
        "--data_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAINING")
    )
    # parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))

    parser.add_argument(
        "--tf_logs_path",
        type=str,
        required=True,
        help="Path used for writing TensorFlow logs. Can be S3 bucket.",
    )

    args, _ = parser.parse_known_args()

    MODEL_INPUT_CHANNEL = "in_weights"  # FIXME hardcoded should be same as input dictionary in sm estimator fot method.

    logger.info("######################################")
    logger.info(f"Output dir: {os.environ['SM_MODEL_DIR']}")
    model_output_dir = f"{os.environ['SM_MODEL_DIR']}/unet/1"
    model_input_dir = f"/opt/ml/input/data/{MODEL_INPUT_CHANNEL}"  # f"{os.environ['SM_MODEL_IN_DIR']}"

    weights_in_path = f"{model_input_dir}/model_weights.hdf5"

    weights_path = f"{model_output_dir}/model_weights.hdf5"

    try:
        os.makedirs(model_output_dir, exist_ok=True)  # args.model_dir, exist_ok=True)
    except:
        logger.info("could not make directory")

    # if args.class_weights is None:
    #     args.class_weights = [1] * args.categories
    # train_pth_lst = [
    #     f"{args.data_dir}/{fn.replace(',', '').replace('[', '').replace(']', '')}"
    #     for fn in args.train_file_lst
    # ]

    train_pth_lst = [
        os.path.join(args.data_dir, fn.strip()) for fn in args.train_file_lst
    ]

    ds_train = get_dataset(
        train_pth_lst,
        im_shape=[args.im_sz, args.im_sz, args.n_bands],
        categories=args.categories,
    )
    y_train = get_labels(ds_train)

    if args.class_weight_list is None:
        # Compute class weights
        args.class_weight_list = compute_class_weights(y_train)
        print("Computed class weights:", args.class_weight_list)

    # macro_f1_score = tfa.metrics.F1Score(num_classes=args.categories, average='macro', threshold=0.5, name="macro_f1_score")

    model = unet_model(
        n_classes=args.categories,
        tile_width=args.im_sz,
        tile_height=args.im_sz,
        n_bands=args.n_bands,
        n_blocks=args.n_blocks,
        class_weight_list=args.class_weight_list,
        n_filters_start=64,
        filter_growth=2,
        # upconv=True,
        # class_weights=[1] * args.categories,
        drop_multiplier=None,
        weight_multiplier=None,
        droprate=args.droprate,
        w_decay=args.w_decay,
        # learning_rate=args.learning_rate,# in optimizer
        normalize_inputs=args.normalization,
        optimizer="adam",
        loss_function=args.loss_function,
        metrics=[
            "categorical_accuracy",
            # tf.keras.metrics.F1Score(name='f1_score'), # BUGGED
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.MeanIoU(num_classes=args.categories, name="meanIoU"),
            tf.keras.metrics.FalseNegatives(name="false_negatives"),
            tf.keras.metrics.TruePositives(name="true_positives"),
        ],
    )

    if os.path.isfile(weights_in_path):
        model.load_weights(weights_in_path)
        print("Model weights succesfully loaded!!!")
    # model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_weights_only=True, save_best_only=True)
    # early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
    # reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=0.00001)
    model_checkpoint_loss = ModelCheckpoint(
        weights_path, monitor="val_loss", save_best_only=True
    )
    model_checkpoint_acc = ModelCheckpoint(
        weights_path, monitor="val_acc", save_best_only=True
    )
    # csv_logger = CSVLogger("log_unet.csv", append=True, separator=";")

    print(args.train_file_lst)
    print(type(args.train_file_lst))

    # train_pth_lst = glob.glob(f"{args.data_dir}/{args.train_file}")

    # TRAINING_FILENAMES, VALIDATION_FILENAMES = train_test_split(
    #     tf.io.gfile.glob(r'data\tfrecords\ld_train*.tfrec'),
    #     test_size=0.3, random_state=101
    # )

    # train_file_lst = [
    #     fn.replace(",", "").replace("[", "").replace("]", "")
    #     for fn in args.train_file_lst
    # ]

    # train_pth_lst = [f"{args.data_dir}/{filename}" for filename in train_file_lst]
    # print(train_pth_lst)

    # ds_train = get_dataset(
    #     train_pth_lst,
    #     im_shape=[args.im_sz, args.im_sz, args.n_bands],
    #     categories=args.categories,
    # )

    # ds_train, val_ds, test_ds = get_dataset_partitions_tf(ds=ds, ds_size=ds_size)
    # validation_pth_lst = glob.glob(f"{args.data_dir}/{args.validation_file_lst}")

    # validation_file_lst = [
    #     fn.replace(",", "").replace("[", "").replace("]", "")
    #     for fn in args.validation_file_lst
    # ]

    # dev_pth_lst = [f"{args.data_dir}/{filename}" for filename in validation_file_lst]
    # print(dev_pth_lst)

    dev_pth_lst = [
        os.path.join(args.data_dir, fn.strip()) for fn in args.validation_file_lst
    ]


    ds_dev = get_dataset(
        dev_pth_lst,
        im_shape=[args.im_sz, args.im_sz, args.n_bands],
        categories=args.categories,
    )

    #### TENSORBOARD ####
    # Parameters Initialization  #FIXME if we want to log all of them we need to set in the training script
    HP_CATEGORIES = hp.HParam("categories", hp.IntInterval(1, 100))
    HP_IM_SZ = hp.HParam("im_sz", hp.IntInterval(1, 100))
    HP_N_BANDS = hp.HParam("n_bands", hp.IntInterval(0, 100))
    HP_EPOCHS = hp.HParam("epochs", hp.IntInterval(1, 100))
    HP_BATCH_SIZE = hp.HParam("batch_size", hp.IntInterval(0, 1000))
    HP_DROPRATE = hp.HParam("droprate", hp.RealInterval(0.0, 1.0))
    HP_LR = hp.HParam("learning_rate", hp.RealInterval(0.0, 1.0))
    HP_W_DECAY = hp.HParam("w_decay", hp.RealInterval(0.0, 1.0))
    HP_OPTIMIZER = hp.HParam("optimizer", hp.Discrete(["sgd", "adam", "rmsprop"]))
    HP_LARGE_KERNEL = hp.HParam("large_kernel_size", hp.IntInterval(0, 100))
    HP_SMALL_KERNEL = hp.HParam("small_kernel_size", hp.IntInterval(0, 100))
    HP_N_BLOCKS = hp.HParam("n_blocks", hp.IntInterval(0, 100))
    HP_LOSS_FUNCTION = hp.HParam(
        "loss_function",
        hp.Discrete(
            [
                "categorical_crossentropy",
                "categorical_focal_crossentropy",
                "weighted_categorical_focal_crossentropy",
                "tversky_loss",
                "dice_loss",
                "weighted_dice_loss",
                "cce_dice",
                "focal_dice",
                "jaccard_loss",
                "weighted_jaccard_loss",
                "cce_jaccard",
                "focal_jaccard",
            ]
        ),
    )

    METRIC_ACCURACY = hp.Metric("categorical_accuracy", display_name="Accuracy")
    METRIC_IOU = hp.Metric("mean_iou", display_name="meanIoU")
    # METRIC_MACRO_F1_SCORE = hp.Metric("macro_f1_score", display_name="MacroAvgF1Score")
    METRIC_PRECISION = hp.Metric("precision", display_name="precision")
    METRIC_RECALL = hp.Metric("recall", display_name="recall")
    METRIC_FALSE_NEGATIVES = hp.Metric("false_negatives", display_name="FalseNegatives")
    METRIC_TRUE_POSITIVES = hp.Metric("true_positives", display_name="TruePositives")
    METRIC_LOSS = hp.Metric("loss", display_name="Loss")

    # Set logs directory
    job_name = json.loads(os.environ.get("SM_TRAINING_ENV"))["job_name"]
    logs_dir = f"{args.tf_logs_path}/{job_name}"

    # callback for tensorboard (training)
    tensorboard = TensorBoard(
        log_dir=logs_dir,
        update_freq="epoch",
        write_graph=True,
        write_images=True,
    )

    # Configuration of hyperparameters to visualize in TensorBoard
    hp.hparams_config(
        hparams=[
            HP_CATEGORIES,
            HP_IM_SZ,
            HP_N_BANDS,
            HP_EPOCHS,
            HP_BATCH_SIZE,
            HP_DROPRATE,
            HP_LR,
            HP_W_DECAY,
            HP_OPTIMIZER,
            HP_LARGE_KERNEL,
            HP_SMALL_KERNEL,
            HP_N_BLOCKS,
            HP_LOSS_FUNCTION,
        ],
        metrics=[
            METRIC_ACCURACY,
            METRIC_IOU,
            METRIC_PRECISION,
            METRIC_RECALL,
            METRIC_FALSE_NEGATIVES,
            METRIC_TRUE_POSITIVES,
            METRIC_LOSS,
        ],  # METRIC_F1_SCORE,
    )

    hparams = {
        HP_CATEGORIES: args.categories,
        HP_IM_SZ: args.im_sz,
        HP_N_BANDS: args.n_bands,
        HP_EPOCHS: args.epochs,
        HP_BATCH_SIZE: args.batch_size,
        HP_DROPRATE: args.droprate,
        HP_LR: args.learning_rate,
        HP_W_DECAY: args.w_decay,
        HP_OPTIMIZER: args.optimizer,
        # HP_LARGE_KERNEL: args.large_kernel_size, #FIXME add argument to the parser with default
        # HP_SMALL_KERNEL: args.small_kernel_size,
        HP_N_BLOCKS: args.n_blocks,
        HP_LOSS_FUNCTION: args.loss_function,
    }

    # callback for tensorboard (hyperparameters log)
    tensorboard_hp = hp.KerasCallback(
        writer=logs_dir, hparams=hparams, trial_id=job_name
    )

    history = model.fit(
        ds_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=ds_dev,  # optinal  validation_split=0.33
        verbose=2,
        shuffle=True,
        callbacks=[
            model_checkpoint_loss,
            # model_checkpoint_acc,
            # csv_logger,
            tensorboard,
            tensorboard_hp,
        ],
    )

    # model_name = "test"
    # model.save_weights(
    #     os.path.join(args.sm_model_dir, model_name), overwrite=True, save_format="tf"
    # )  # args.sm_model_dir)

    # try:
    #     model.save(model_output_dir)  # "/opt/ml/model")"/opt/ml/model")#
    #     logger.info("#### MODEL SAVED")
    #     logger.info(glob.glob(f"{model_output_dir}/*.*"))
    try:
        tf.saved_model.save(model, model_output_dir)
    except:
        logger.info("failed")

    ### Performance measures:

    # get_performance_measures(ds=ds_dev, model=model, set_name="Final dev set")

    y_true, y_pred = get_predictions_and_labels(model, ds_dev)

    # Calculate performance metrics
    (
        accuracy,
        support,
        precision,
        recall,
        fscore,
        conf_matrix,
    ) = calculate_performance_metrics(y_true, y_pred)

    report_str = classification_report(
        y_true.flatten(), y_pred.flatten(), zero_division=0
    )
    jaccard_score_iou = jaccard_score(
        y_true=y_true.flatten(), y_pred=y_pred.flatten(), average="weighted"
    )  # NN

    #     logger.info(f"dev accuracy: {accuracy}")
    #     logger.info(f"dev f-score: {fscore}")
    #     logger.info(f"dev precision: {precision}")
    #     logger.info(f"dev recall: {recall}")
    #     logger.info(f"dev support: {support}")
    #     logger.info(f"dev confusion matrix: {conf_matrix}")
    #     logger.info(f"dev report: {report_str}")
    logger.info(f"dev Jaccard index weight-averaged: {jaccard_score_iou}")

    # Record the Trained model´s metrics and confusion matrix in Tensorboard as a table
    metrics_dict = {
        "Dev Accuracy": accuracy,
        "Dev F-Score": fscore,
        "Dev Precision": precision,
        "Dev Recall": recall,
        "Dev Support": support,
        "Dev IoU": jaccard_score_iou,
    }

    report_dict = classification_report(
        y_true.flatten(), y_pred.flatten(), zero_division=0, output_dict=True
    )

    def tb_metrics_summary(metrics, step):
        valid_metrics = {
            name: value for name, value in metrics.items() if value is not None
        }
        if not valid_metrics:
            return  # Skip logging if there are no valid metrics
        table = "\n".join(
            [f"{name} | {value}" for name, value in valid_metrics.items()]
        )
        preamble = f"The metrics of the trained model after {step} epochs are:\n\n"
        result = f"{preamble}Metric | Value\n---|---\n{table}"
        tf.summary.text("metrics_trained_model", result, step)

    def tb_classification_report(report_dict, step):
        headers = [""] + list(report_dict["0"].keys())
        classes_rows = [
            [key]
            + [
                (
                    format(report_dict[key][metric], ".4f")
                    if isinstance(report_dict[key][metric], float)
                    else report_dict[key][metric]
                )
                for metric in headers[1:]
            ]
            for key in sorted(report_dict.keys())
            if key.isnumeric()
        ]
        avg_keys = ["macro avg", "weighted avg"]
        metrics_rows = [
            [key]
            + [
                (
                    format(report_dict[key][metric], ".4f")
                    if isinstance(report_dict[key][metric], float)
                    else report_dict[key][metric]
                )
                for metric in headers[1:]
            ]
            for key in avg_keys
        ]
        accuracy_key = "accuracy"
        accuracy_row = [
            [accuracy_key]
            + [" "] * (len(headers) - 2)
            + [
                format(report_dict[accuracy_key], ".4f"),
                f"({report_dict[accuracy_key]:.0f})",
            ]
        ]
        column_widths = [
            max(
                len(str(headers[i])),
                max(
                    len(str(row[i]))
                    for row in classes_rows + accuracy_row + metrics_rows
                ),
            )
            for i in range(len(headers))
        ]

        header_row = (
            "|"
            + "|".join(
                "{:^{width}}".format(header, width=width)
                for header, width in zip(headers, column_widths)
            )
            + "|"
        )
        separator_row = "|" + " |" * len(column_widths)

        classes_rows_formatted = [
            "|"
            + "|".join(
                "{:<{width}}".format(str(data), width=width) + "|"
                for data, width in zip(row, column_widths)
            )
            for row in classes_rows
        ]
        accuracy_row_formatted = [
            "|"
            + "|".join(
                "{:<{width}}".format(str(data), width=width) + "|"
                for data, width in zip(accuracy_row[0], column_widths)
            )
        ]
        metrics_rows_formatted = [
            "|"
            + "|".join(
                "{:<{width}}".format(str(data), width=width) + "|"
                for data, width in zip(row, column_widths)
            )
            for row in metrics_rows
        ]

        markdown_table = "{}\n{}\n{}\n{}\n{}\n{}\n{}".format(
            header_row,
            separator_row,
            "\n".join(classes_rows_formatted),
            separator_row,
            "\n".join(metrics_rows_formatted),
            separator_row,
            "\n".join(accuracy_row_formatted),
        )
        markdown_table = markdown_table.replace("||", "|")

        preamble = f"The classification report after {step} epochs is:\n\n"
        result = tf.strings.join([preamble, markdown_table], separator="\n")
        tf.summary.text("classification_report", result, step)

    def tb_confusion_matrix(matrix, num_classes, step):
        header = (
            "| Actual/Predicted | "
            + " | ".join([f"Class {i}" for i in range(num_classes)])
            + " |"
        )
        separator = "|------------------|" + "---------|" * num_classes
        body = ""

        for i in range(num_classes):
            row = (
                f"| Class {i}           | "
                + " | ".join([f"{matrix[i][j]}" for j in range(num_classes)])
                + " |"
            )
            body += row + "\n"

        header_tensor = tf.strings.join([header], separator="\n")
        separator_tensor = tf.strings.join([separator], separator="\n")
        body_tensor = tf.strings.join([body], separator="\n")
        markdown_table = tf.strings.join(
            [header_tensor, separator_tensor, body_tensor], separator="\n"
        )

        preamble = (
            f"Confusion Matrix after {step} epochs with {num_classes} Classes:\n\n"
        )
        result = tf.strings.join([preamble, markdown_table])
        tf.summary.text("confusion_matrix", result, step)

    # def tb_model_dir(model_output_dir, step): #FIXME
    #     preamble = f"Model saved in:\n"
    #     dir = f"{model_output_dir}"
    #     result = tf.strings.join([preamble, dir])
    #     tf.summary.text("model_directory", result, step)

    def tb_sm_job(job_name, step):
        preamble = "View the SageMaker Training Job on the page:\n"
        link = f"<https://eu-central-1.console.aws.amazon.com/sagemaker/home?region=eu-central-1#/jobs/{job_name}>"
        result = tf.strings.join([preamble, link], separator="\n")
        tf.summary.text("sagemaker_training_job", result, step)

    writer = tf.summary.create_file_writer(logs_dir + "/text/")

    with writer.as_default():
        # with tf.name_scope("model_directory"):
        #         tb_model_dir(args.model_dir, args.epochs)
        with tf.name_scope("metrics_summary"):
            tb_metrics_summary(metrics_dict, args.epochs)
        with tf.name_scope("confusion_matrix"):
            tb_confusion_matrix(conf_matrix, args.categories, args.epochs)
        with tf.name_scope("classification_report"):
            tb_classification_report(report_dict, args.epochs)
        with tf.name_scope("sagemaker_training_job"):
            tb_sm_job(job_name, args.epochs)

    writer.close()

    for i, test_file_lst in enumerate(
        [args.test_file_lst, args.test_2_file_lst, args.test_3_file_lst]
    ):
        if test_file_lst != None:
            set_name = f"test {i}"

            y_true, y_pred = get_predictions_and_labels(
                model, get_dataset_from_args_lst(args_lst=args.test_file_lst)
            )

            # Calculate performance metrics
            (
                accuracy,
                support,
                precision,
                recall,
                fscore,
                conf_matrix,
            ) = calculate_performance_metrics(y_true, y_pred)
            jaccard_score_iou = jaccard_score(
                y_true=y_true.flatten(), y_pred=y_pred.flatten(), average="weighted"
            )
            jaccard_score_iou_class = jaccard_score(
                y_true=y_true.flatten(), y_pred=y_pred.flatten(), average=None
            )

            logger.info(f"{set_name} accuracy: {accuracy}")
            logger.info(f"{set_name} f-score: {fscore}")
            logger.info(f"{set_name} precision: {precision}")
            logger.info(f"{set_name} recall: {recall}")
            logger.info(f"{set_name} support: {support}")
            logger.info(f"{set_name} Confusion Matrix: {conf_matrix}")
            logger.info(
                f"{set_name} Jaccard index weight-averaged: {jaccard_score_iou}"
            )
            logger.info(
                f"{set_name} Jaccard index per class: {jaccard_score_iou_class}"
            )

            writer_test = tf.summary.create_file_writer(
                logs_dir + "/test/" + f"set_{i}/"
            )

            with writer_test.as_default():
                with tf.name_scope("Test metrics summary:"):
                    tb_confusion_matrix(conf_matrix, args.categories, args.epochs)
                    # Format Jaccard scores per class
                    jaccard_class = "\n".join(
                        [
                            f"Class {k}: {jaccard_score_iou_class[k]:.4f}. "
                            for k in range(args.categories)
                        ]
                    )
                    # Joining all results
                    results = tf.strings.join(
                        [
                            f"{set_name} accuracy: {accuracy}",
                            "\n",
                            f"{set_name} f-score: {fscore}",
                            "\n",
                            f"{set_name} precision: {precision}",
                            "\n",
                            f"{set_name} recall: {recall}",
                            "\n",
                            f"{set_name} support: {support}",
                            "\n",
                            f"{set_name} Jaccard index weight-averaged: {jaccard_score_iou}",
                            "\n",
                            f"{set_name} Jaccard index per class:\n{jaccard_class}",
                            "\n",
                        ],
                        separator="\n",
                    )
                    tf.summary.text("Test metrics", results, step=args.epochs)
            writer_test.close()

    # if args.test_2_file_lst != None:
    #     get_performance_measures(
    #         ds=get_dataset_from_args_lst(args_lst=args.test_2_file_lst),
    #         model=model,
    #         set_name="Test-2 set",
    #     )

    # if args.test_3_file_lst != None:
    #     get_performance_measures(
    #         ds=get_dataset_from_args_lst(args_lst=args.test_3_file_lst),
    #         model=model,
    #         set_name="Test-3 set",
    #     )

    # try:
    #     model.save(model_output_dir)  # "/opt/ml/model")"/opt/ml/model")#
    #     logger.info("#### MODEL SAVED")
    #     logger.info(glob.glob(f"{model_output_dir}/*.*"))
    # except:
    #     logger.info("failed")

    # try:
    #     logger.info("#### MODEL Type")
    #     logger.info(f"type model: {type(model)}")
    # except:
    #     logger.info("failed")

    # try:
    #     logger.info("predicting train data on model")
    #     pred = model.predict(x=ds_train)
    #     lst_arr = list(ds_train.as_numpy_iterator())
    #     logger.info("Stored as numpy")

    #     del ds_train

    #     y = lst_arr[0][1]
    #     for i in range(1, len(lst_arr)):
    #         # if i//100 == 0:
    #         # logger.info(f"Iteration: {i}")

    #         y = np.concatenate([y, lst_arr[i][1]], axis=0)
    #     logger.info("iterated over all samples stored in one array")

    #     del lst_arr

    #     y_true = np.argmax(y, axis=-1)
    #     logger.info("Ground truth in categories calculated")

    #     del y
    #     y_pred = np.argmax(pred, axis=-1)
    #     logger.info("Predictions in categories calculated")

    # precision, recall, fscore, support = precision_recall_fscore_support(
    #     y_true=y_true.flatten(), y_pred=y_pred.flatten()
    # )
    #     logger.info("Performance metrics validation set calculated")

    #     logger.info(f"train categorical f-score: {fscore}")
    #     logger.info(f"train categorical precision: {precision}")
    #     logger.info(f"train categorical recall: {recall}")
    #     logger.info(f"train categorical support: {support}")

    # conf_matrix = confusion_matrix(y_true=y_true.flatten(), y_pred=y_pred.flatten())
    #     logger.info("Confusion matrix of validation set calculated")

    #     logger.info(f"Confusion Matrix: {conf_matrix}")

    #     print(f"validation categorical f-score: {fscore}")
    #     print(f"validation categorical precision: {precision}")
    #     print(f"validation categorical recall: {recall}")
    #     print(f"validation categorical support: {support}")
    #     print(f"Confusion Matrix: {conf_matrix}")

    # except:
    #     logger.info("did not work")

    # try:

    #     test_1_pth_lst = [f"{args.data_dir}/{filename}" for filename in test_1_file_lst]
    #     print(test_1_pth_lst)

    #     ds_test_1 = get_dataset(
    #         test_1_pth_lst,
    #         im_shape=[args.im_sz, args.im_sz, arg.n_bands],
    #         categories=args.categories,
    #     )

    #     logger.info("predicting test data on model")
    #     pred = model.predict(x=ds_test)

    #     lst_arr = list(ds_test.as_numpy_iterator())
    #     logger.info("Stored as numpy")

    #     del ds_test_1

    #     y = lst_arr[0][1]
    #     for i in range(1, len(lst_arr)):
    #         # if i//100 == 0:
    #         # logger.info(f"Iteration: {i}")

    #         y = np.concatenate([y, lst_arr[i][1]], axis=0)
    #     logger.info("iterated over all samples stored in one array")

    #     del lst_arr

    #     y_true = np.argmax(y, axis=-1)
    #     logger.info("Ground truth in categories calculated")

    #     del y
    #     y_pred = np.argmax(pred, axis=-1)
    #     logger.info("Predictions in categories calculated")

    #     precision, recall, fscore, support = precision_recall_fscore_support(
    #         y_true=y_true.flatten(), y_pred=y_pred.flatten()
    #     )
    #     logger.info("Performance metrics validation set calculated")

    #     logger.info(f"validation categorical f-score: {fscore}")
    #     logger.info(f"validation categorical precision: {precision}")
    #     logger.info(f"validation categorical recall: {recall}")
    #     logger.info(f"validation categorical support: {support}")
    #     accuracy_score()

    #     conf_matrix = confusion_matrix(y_true=y_true.flatten(), y_pred=y_pred.flatten())
    #     logger.info("Confusion matrix of validation set calculated")

    #     logger.info(f"Confusion Matrix: {conf_matrix}")

    #     print(f"validation categorical f-score: {fscore}")
    #     print(f"validation categorical precision: {precision}")
    #     print(f"validation categorical recall: {recall}")
    #     print(f"validation categorical support: {support}")
    #     print(f"Confusion Matrix: {conf_matrix}")

    # except:
    #     logger.info("did not work")

    #     test_2_pth_lst = [f"{args.data_dir}/{filename}" for filename in test_2_file_lst]

    #     ds_test_2 = get_dataset(
    #         test_2_pth_lst,
    #         im_shape=[args.im_sz, args.im_sz, args.n_bands],
    #         categories=args.categories,
    #     )

    #     ds_test_3 = get_dataset(
    #         test_3_pth_lst,
    #         im_shape=[args.im_sz, args.im_sz, args.n_bands],
    #         categories=args.categories,
    #     )
