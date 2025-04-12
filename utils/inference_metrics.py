import sys
import os
import numpy as np
import rasterio
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score
sys.path.append('../utils')
from utils.utility import write_geotiff, preprocess_mask


def run_inference(tile, dataset, model, composite_dirs, mask_dirs, output_dir, score_threshold=0.5, with_gt=True):
    composite_path = os.path.join(composite_dirs[dataset], tile)
    mask_path = os.path.join(mask_dirs[dataset], tile)
    tile_stem = tile.split('.')[0]
    prob_output_path = os.path.join(
        output_dir, f"{dataset.lower()}_{tile_stem}_probs.tif")
    binary_output_path = os.path.join(
        output_dir, f"{dataset.lower()}_{tile_stem}_binary.tif")

    # Load composite
    with rasterio.open(composite_path) as src:
        input_data = src.read(out_dtype=np.float32).transpose(1, 2, 0)
        reference_profile = src.profile.copy()
        reference_transform = src.transform
        reference_crs = src.crs

    prediction_probs = model.predict(np.expand_dims(input_data, axis=0))[0]
    binary_prediction = (
        prediction_probs[:, :, 1] > score_threshold).astype(np.uint8)

    # Save prediction outputs
    write_geotiff(prob_output_path, prediction_probs, reference_profile,
                  transform=reference_transform, crs=reference_crs, dtype=rasterio.float32)
    write_geotiff(binary_output_path, binary_prediction, reference_profile,
                  transform=reference_transform, crs=reference_crs, dtype=rasterio.uint8)

    if with_gt:
        with rasterio.open(mask_path) as src:
            mask = preprocess_mask(src.read(1))
        return calculate_metrics(mask, binary_prediction)
    return None

def calculate_metrics(gt, pred):
    f1 = f1_score(gt.flatten(), pred.flatten(), zero_division=1)
    precision = precision_score(gt.flatten(), pred.flatten(), zero_division=1)
    recall = recall_score(gt.flatten(), pred.flatten(), zero_division=1)
    iou = jaccard_score(gt.flatten(), pred.flatten(), zero_division=1)
    tp = np.sum((pred == 1) & (gt == 1))
    tn = np.sum((pred == 0) & (gt == 0))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    return {'F1': f1, 'Precision': precision, 'Recall': recall, 'IoU': iou, 'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}