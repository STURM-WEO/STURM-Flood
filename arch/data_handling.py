# from osgeo import gdal
# import tensorflow as tf
# from tqdm import tqdm
# from tf.keras.utils.np_utils import to_categorical
import tensorflow as tf
# import tensorflow_addons as tfa
import numpy as np


### Raster Functions


# def read_image(image_pth):
#     ds = gdal.Open(image_pth)
#     arr = ds.ReadAsArray()
#     arr = arr.astype(np.int16)  # float32)
#     arr = arr.transpose(1, 2, 0)
#     return arr


# def read_mask(mask_pth):
#     ds = gdal.Open(mask_pth)
#     arr = ds.ReadAsArray()
#     arr[arr == 999] = 3
#     arr = to_categorical(arr)
#     # arr = np.expand_dims(arr, -1)
#     arr = arr.astype(np.int16)  # float32)

#     return arr


### Saving Tensorflow Dataset Functions


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):  # if value ist tensor
        value = value.numpy()  # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a floast_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array


def parse_tfr_element(element, im_shape, categories):

    # use the same structure as above; it's kinda an outline of the structure we now want to create
    data = {
        "raw_image": tf.io.FixedLenFeature([], tf.string),
        "height": tf.io.FixedLenFeature([], tf.int64),
        "width": tf.io.FixedLenFeature([], tf.int64),
        "depth": tf.io.FixedLenFeature([], tf.int64),
        "mask": tf.io.FixedLenFeature([], tf.string),
        "categories": tf.io.FixedLenFeature([], tf.int64),
    }

    content = tf.io.parse_single_example(element, data)

    raw_image = content["raw_image"]
    # height = content["height"]
    # width = content["width"]
    # depth = content["depth"]
    mask = content["mask"]
    # categories = content["categories"]

    height = im_shape[0]
    width = im_shape[1]
    depth = im_shape[2]
    categories = categories

    image = tf.io.parse_tensor(raw_image, out_type=tf.float32)#tf.ensure_shape(tf.io.parse_tensor(raw_image, out_type=tf.int16), ())  # float32)#
    image = tf.reshape(image, shape=tf.stack([height, width, depth])) 

    mask = tf.io.parse_tensor(mask, out_type=tf.int16)  # float32)#
    mask = tf.reshape(mask, shape=tf.stack([height, width, categories]))#, [tf.cast(height, np.int16), width, categories])  # [height,width,1])
    mask = tf.cast(mask, tf.float32)

    image, mask = data_augmentation(image, mask, im_shape, augment=True)

    # image = tf.io.parse_tensor(raw_image, out_type=tf.int16) #tf.ensure_shape(tf.io.parse_tensor(raw_image, out_type=tf.int16), ())  # float32)#
    # image = tf.reshape(image, shape=[height, width, depth])  # [height,width,depth])

    # mask = tf.io.parse_tensor(mask, out_type=tf.int16) #tf.ensure_shape(tf.io.parse_tensor(mask, out_type=tf.int16), ())  # float32)#
    # mask = tf.reshape(mask, shape=[height, width, categories])  # [height,width,1])
    # mask = tf.cast(mask, tf.float32)

    return (image, mask)


### Load dataset functions


# data_augmentation = tf.keras.Sequential([
#   layers.RandomFlip("horizontal_and_vertical"),
#   layers.RandomRotation(0.4),
# ])

@tf.function
# def data_augmentation(datapoint, augment=True): # im_shape,

#     input_image = datapoint[0]
#     input_mask = datapoint[1]
def data_augmentation(input_image, input_mask, im_shape, augment=True): # im_shape,

   
    # augmentation
    if augment:
        # zoom in a bit
        # if tf.random.uniform(()) > 0.5:
        #     input_image = tf.image.central_crop(input_image, 0.75)
        #     input_mask = tf.image.central_crop(input_mask, 0.75)
        #     # resize
        #     input_image = tf.image.resize(input_image, im_shape)
        #     input_mask = tf.image.resize(input_mask, im_shape)
        
        # # random brightness adjustment illumination
        # input_image = tf.image.random_brightness(input_image, 0.3)
        # # random contrast adjustment
        # input_image = tf.image.random_contrast(input_image, 0.2, 0.5)
        
        # flipping random horizontal or vertical
        if tf.random.uniform(()) > 0.5:
            input_image = tf.image.flip_left_right(input_image)
            input_mask = tf.image.flip_left_right(input_mask)
        if tf.random.uniform(()) > 0.5:
            input_image = tf.image.flip_up_down(input_image)
            input_mask = tf.image.flip_up_down(input_mask)

        # # rotation in 30° steps
        # rot_factor = tf.cast(tf.random.uniform(shape=[], maxval=12, dtype=tf.int32), tf.float32)
        # angle = np.pi/12*rot_factor
        # input_image = tfa.image.rotate(input_image, angle)
        # input_mask = tfa.image.rotate(input_mask, angle)

    return input_image, input_mask


def load_dataset(filenames, im_shape, categories):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed
    dataset = tf.data.TFRecordDataset(
        filenames
    )  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(
        ignore_order
    )  # uses data as soon as it streams in, rather than in its original order
    # dataset = dataset.map(parse_tfr_element)#, fn_kwargs={im_shape:im_shape, categories:categories})
    dataset = dataset.map(lambda x: parse_tfr_element(x, im_shape, categories))
    # dataset = dataset.map(lambda x: (data_augmentation(x))) #FIXME , im_shape same transformation for image and masks look up online
    # returns a dataset of (image, mask) pairs
    return dataset


def get_dataset(filenames, im_shape, categories, batch_size=12):
    dataset = load_dataset(filenames, im_shape, categories)
    # dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=48)  # tf.data.AUTOTUNE)#32)#
    dataset = dataset.batch(batch_size)
    return dataset
