# -----------------------------
# Visualization functions
# -----------------------------


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from IPython.display import display
import numpy as np
import rasterio
import os
import sys

sys.path.append('../utils')
from utils.utility import preprocess_mask

# Color settings
contingency_colors = {
    'TP': [26, 87, 128],
    'FP': [143, 9, 50],
    'FN': [246, 185, 104],
    'TN': [240, 240, 240]
}
water_mask_cmap = ListedColormap(['#f0f0f0', '#1a5780'])

# Contrast stretch for RGB image
def normalize_rgb(rgb_array, low=2, high=98):
    rgb_stretched = np.zeros_like(rgb_array)
    for i in range(3):
        p_low, p_high = np.percentile(rgb_array[:, :, i], (low, high))
        rgb_stretched[:, :, i] = np.clip(
            (rgb_array[:, :, i] - p_low) / (p_high - p_low + 1e-6), 0, 1)
    return rgb_stretched

# Visualize tile and prediction with ground-truth if available
def visualize_tile(tile, dataset_name, composite_dirs, output_dir, mask_dirs, is_s2=True, with_gt=True):
    img_path = os.path.join(composite_dirs[dataset_name], tile)
    tile_stem = tile.split('.')[0]
    pred_path = os.path.join(
        output_dir, f"{dataset_name.lower()}_{tile_stem}_binary.tif")

    with rasterio.open(img_path) as src:
        img = src.read(out_dtype=np.float32)
    with rasterio.open(pred_path) as src:
        prediction = src.read(1)

    if with_gt:
        mask_path = os.path.join(mask_dirs[dataset_name], tile)
        with rasterio.open(mask_path) as src:
            mask = preprocess_mask(src.read(1))

    # 2 subplots (no GT) or 3 subplots (with GT)
    fig, axes = plt.subplots(1, 3 if with_gt else 2,
                             figsize=(20 if not with_gt else 30, 10))

    if is_s2:
        rgb_image = np.dstack((img[2], img[1], img[0]))
        rgb_image = normalize_rgb(rgb_image)
        axes[0].imshow(rgb_image)
    else:
        axes[0].imshow(img[0], cmap='gray', vmin=np.min(
            img[0]), vmax=np.max(img[0]))
    axes[0].axis('off')
    axes[0].set_title(f"{dataset_name} {tile}", fontsize=24)

    if with_gt:
        axes[1].imshow(mask, cmap=water_mask_cmap, vmin=0, vmax=1)
        axes[1].axis('off')
        axes[1].set_title("Binary Water Mask", fontsize=24)

        contingency_map = np.full(
            (*mask.shape, 3), contingency_colors['TN'], dtype=np.uint8)
        contingency_map[(prediction == 1) & (mask == 1)
                        ] = contingency_colors['TP']
        contingency_map[(prediction == 1) & (mask == 0)
                        ] = contingency_colors['FP']
        contingency_map[(prediction == 0) & (mask == 1)
                        ] = contingency_colors['FN']
        axes[2].imshow(contingency_map)
        axes[2].axis('off')
        axes[2].set_title("Contingency Map", fontsize=24)
    else:
        axes[1].imshow(prediction, cmap=water_mask_cmap, vmin=0, vmax=1)
        axes[1].axis('off')
        axes[1].set_title("Inference", fontsize=24)

    plt.tight_layout()
    return fig


def save_visualizations(df, dataset_name, visualization_dir, composite_dirs, output_dir, mask_dirs, is_s2=True, show_inline=True, with_gt=True, max_display=5):
    for i, (_, row) in enumerate(df.iterrows()):
        tile = row['tile_id']
        tile_stem = tile.split('.')[0]
        fig = visualize_tile(tile, dataset_name, composite_dirs,
                             output_dir, mask_dirs, is_s2=is_s2, with_gt=with_gt)

        fig_path = f"{visualization_dir}/{dataset_name}_{tile_stem}.png"
        fig.savefig(fig_path, dpi=600, bbox_inches='tight')

        if show_inline and i < max_display:
            display(fig)

        plt.close(fig)