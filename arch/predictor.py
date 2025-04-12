import tensorflow as tf
import numpy as np
import math
import argparse

def predict(x, model, patch_sz=160, n_classes=5):
    img_height = x.shape[0]
    img_width = x.shape[1]
    n_channels = x.shape[2]
    # make extended img so that it contains integer number of patches
    npatches_vertical = math.ceil(img_height / patch_sz)
    npatches_horizontal = math.ceil(img_width / patch_sz)
    # npatches_vertical = math.floor(img_height / patch_sz)
    # npatches_horizontal = math.floor(img_width / patch_sz)

    extended_height = patch_sz * npatches_vertical
    extended_width = patch_sz * npatches_horizontal
    ext_x = np.zeros(
        shape=(extended_height, extended_width, n_channels), dtype=np.float32
    )
    # fill extended image with mirrors:
    ext_x[:img_height, :img_width, :] = x
    for i in range(img_height, extended_height):
        ext_x[i, :, :] = ext_x[2 * img_height - i - 1, :, :]
    for j in range(img_width, extended_width):
        ext_x[:, j, :] = ext_x[:, 2 * img_width - j - 1, :]

    # now we assemble all patches in one array
    patches_list = []
    for i in range(0, npatches_vertical):
        for j in range(0, npatches_horizontal):
            x0, x1 = i * patch_sz, (i + 1) * patch_sz
            y0, y1 = j * patch_sz, (j + 1) * patch_sz
            patches_list.append(ext_x[x0:x1, y0:y1, :])
    # model.predict() needs numpy array rather than a list
    patches_array = np.asarray(patches_list)
    # predictions:
    patches_predict = model.predict(patches_array, batch_size=4)
    prediction = np.zeros(
        shape=(extended_height, extended_width, n_classes), dtype=np.float32
    )
    for k in range(patches_predict.shape[0]):
        i = k // npatches_horizontal
        j = k % npatches_vertical
        x0, x1 = i * patch_sz, (i + 1) * patch_sz
        y0, y1 = j * patch_sz, (j + 1) * patch_sz
        # print(f"k is: {k}")
        # print(f"shape patches predict: {patches_predict.shape}")
        # print(f"shape patches predict at k: {patches_predict[k, :, :, :].shape}")

        prediction[x0:x1, y0:y1, :] = patches_predict[k, :, :, :]
    return prediction[:img_height, :img_width, :]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_pth", type=str, default="./model/saved_model.pb")
    parser.add_argument("--image_pth", type=str, default="./image.tif")
    args, _ = parser.parse_known_args()

    model = args.model_dir
    image = load_image(args.image_dir)

    model.predict(patches_array, batch_size=4)

    # predict(
    #     x=
    #     model=tf.keras.models.load_model(args.model_dir),
    #     )
