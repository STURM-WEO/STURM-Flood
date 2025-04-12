# -----------------------------
# Utility functions
# -----------------------------
import rasterio
import numpy as np

def preprocess_mask(mask_array):
    return np.where(mask_array > 0, 1, 0)


def write_geotiff(filename, array, profile, transform, crs, dtype=rasterio.float32):
    profile.update(
        dtype=dtype,
        count=array.shape[2] if array.ndim == 3 else 1,
        transform=transform,
        crs=crs
    )

    with rasterio.open(filename, 'w', **profile) as dst:
        if array.ndim == 3:
            for i in range(array.shape[2]):
                dst.write(array[:, :, i], i + 1)
        else:
            dst.write(array, 1)