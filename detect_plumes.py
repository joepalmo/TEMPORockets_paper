import numpy as np
import pandas as pd
import xarray as xr
import skimage
from math import radians, sin, cos, atan2, sqrt

def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2*R*atan2(sqrt(a), sqrt(1 - a))

def detect_single_plume(
    data: xr.DataArray,
    bbox: tuple,
    source: tuple,
    threshold: float,
    min_plume_size: int = 0,
    snr_threshold: float = 5.,
    distance_threshold: float = 26.,
    max_busy_frac: float = 0.2,       # skip scene if too many > threshold
) -> xr.DataArray:
    """
    Detect a single plume in a known bounding box and originating from a known source.

    Args:
        data (xr.DataArray): Input 2D satellite data.
        bbox (tuple): Bounding box as (lat_min, lat_max, lon_min, lon_max).
        source (tuple): Source location as (lat, lon).
        threshold (float): Threshold for plume detection.
        min_plume_size (int): Minimum size of the detected plume.
        snr_threshold (float, optional): Minimum SNR for detected plumes.

    Returns:
        xr.DataArray: Binary mask of the detected plume.
    """
    lon_min, lon_max, lat_min, lat_max = bbox
    source_lat, source_lon = source
    
    # Subset data to the bounding box
    roi = data.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))

    vals = roi.values
    finite = np.isfinite(vals)
    if not finite.any():
        raise ValueError("ROI has no finite data.")

    # (1) Busy scene check using the global threshold (from 2×2° region)
    # get data from 2x2 degree region
    large_roi = data.sel(latitude=slice(lat_min-0.75, lat_max+0.75), longitude=slice(lon_min-0.75, lon_max+0.75))
    frac_active = ((large_roi.values > threshold)).mean()
    if frac_active > max_busy_frac:
        raise ValueError(f"Scene too busy: {frac_active:.1%} > {max_busy_frac:.0%} above global threshold.")

    # Apply thresholding to identify plume
    plume_mask = roi.values > threshold

    # Label connected regions
    labeled_plumes = skimage.measure.label(plume_mask, background=0)
    
    # Check if any plumes were detected
    unique_labels, counts = np.unique(labeled_plumes, return_counts=True)
    if len(unique_labels) <= 1:  # Only background is present
        raise ValueError("No plumes detected within the bounding box.")
    
    # check if more than 50% of pixels are non nan
    # if counts[1] / (roi.size - np.isnan(roi).sum()) < 0.95:
    #     raise ValueError("More than 95% of pixels are nan.")

    # Calculate background noise
    background_values = roi.values[~plume_mask]
    if background_values.size == 0:
        raise ValueError("No background values found for SNR calculation.")
    background_noise = np.nanstd(background_values)  # Noise level (standard deviation)

    # Collect plume information in a DataFrame
    plume_data = []
    for label in unique_labels[1:]:  # Skip the background label
        # Get indices of the current label
        indices = np.argwhere(labeled_plumes == label)
        
        # Calculate centroid
        centroid_lat = np.nanmean(roi.latitude.values[indices[:, 0]])
        centroid_lon = np.nanmean(roi.longitude.values[indices[:, 1]])
        
        # Compute distance from the source
        distance = np.sqrt((centroid_lat - source_lat) ** 2 + (centroid_lon - source_lon) ** 2)
        
        # Calculate enhancement and mean signal
        plume_values = roi.values[indices[:, 0], indices[:, 1]]
        enhancement = np.nansum(plume_values)  # Total enhancement
        mean_signal = np.nanmean(plume_values)
        
        # Calculate SNR
        snr = mean_signal / background_noise
        
        # Store data
        plume_data.append({
            "label": label,
            "centroid_lat": centroid_lat,
            "centroid_lon": centroid_lon,
            "distance": distance,
            "enhancement": enhancement,
            "mean_signal": mean_signal,
            "snr": snr,
            "size": len(indices)  # Number of pixels in the plume
        })

    # Create a DataFrame
    plumes_df = pd.DataFrame(plume_data)

    # remove plumes that are too small
    plumes_df = plumes_df[plumes_df["size"] >= min_plume_size]

    # Sort plumes by SNR
    # plumes_df.sort_values("snr", ascending=False, inplace=True)

    # Filter plumes by distance
    filtered_df = plumes_df[
        (plumes_df["distance"] < distance_threshold) 
    ]

    # Filter plumes by SNR
    filtered_df = filtered_df[
        (filtered_df["snr"] > snr_threshold)
    ]
    
    if filtered_df.empty:
        raise ValueError(f"No plumes met the SNR threshold of {snr_threshold} and distance criteria. Highest SNR: {plumes_df['snr'].max()}")

    if len(filtered_df) > 1:
        filtered_df = filtered_df[filtered_df["size"]>=15]
        filtered_df = filtered_df[filtered_df["snr"]>6]
        if len(filtered_df) > 1:
            print(filtered_df.to_string())
            raise ValueError(f"Ambiguous scene: Multiple ({len(filtered_df)}) plume candidates after gates and filtering.")
        if len(filtered_df) == 0:
            raise ValueError(f"Ambiguous scene: No plume candidates after gates and filtering.")
        else:
            pass

    # Select the best plume -- there should only be one now
    best_plume = filtered_df.loc[filtered_df["snr"].idxmax()]

    # if there is a plume that is larger than 100 pixels, select that one
    # otherwise select the one with the highest mean signal
    # large_plumes = filtered_df[filtered_df["size"] > 100]
    # if not large_plumes.empty:
    #     best_plume = large_plumes.loc[large_plumes["mean_signal"].idxmax()]
    # else:
    #     best_plume = filtered_df.loc[filtered_df["mean_signal"].idxmax()]

    
    # Extract the label of the best plume
    best_label = best_plume["label"]
    best_plume_mask = (labeled_plumes == best_label)

    print(f"Detected plume: with SNR={best_plume['snr']:.2f}, distance={best_plume['distance']:.2f} degrees, enhancement={best_plume['enhancement']:.2f}")

    # Return the detected plume as an xarray DataArray
    result = xr.DataArray(
        best_plume_mask,
        coords=roi.coords,
        dims=roi.dims,
        attrs={"description": "Binary mask of the detected plume",
                 "snr": best_plume["snr"],
                 "distance": best_plume["distance"],
                 "enhancement": best_plume["enhancement"],
                 "mean_signal": best_plume["mean_signal"],
                 "size": best_plume["size"]}
                 
    )
    
    return result


def detect_single_plume1(
    data: xr.DataArray,
    bbox: tuple,                       # (lat_min, lat_max, lon_min, lon_max)
    source: tuple,                     # (lat, lon)
    threshold: float,                  # from larger 2×2° region
    min_plume_size: int = 0,
    snr_threshold: float = 5.,
    distance_threshold: float = 26.,
    # NEW: knobs + optional global bg stats
    z_thr: float = 2.0,                # robust z threshold for masking
    max_busy_frac: float = 0.2,       # skip scene if too many > threshold
    k_total: float = 6.0,
    k_mean: float  = 1.5,
    k_peak: float  = 3.0,
    global_bg_med: float | None = None,
    global_bg_sigma: float | None = None,
) -> xr.DataArray:

    lon_min, lon_max, lat_min, lat_max = bbox
    source_lat, source_lon = source
    roi = data.sel(latitude=slice(lat_min, lat_max),
                   longitude=slice(lon_min, lon_max))
    vals = roi.values
    finite = np.isfinite(vals)
    if not finite.any():
        raise ValueError("ROI has no finite data.")

    # (1) Busy scene check using the global threshold (from 2×2° region)
    frac_active = ((vals > threshold)).mean()
    print(frac_active)
    if frac_active > max_busy_frac:
        raise ValueError(f"Scene too busy: {frac_active:.1%} > {max_busy_frac:.0%} above global threshold.")

    # (2) Robust background for z-scores:
    #     Prefer global stats from the larger region; fallback to ROI median/MAD.
    if (global_bg_med is not None) and (global_bg_sigma is not None) and np.isfinite(global_bg_sigma) and global_bg_sigma > 0:
        bg_med   = float(global_bg_med)
        bg_sigma = float(global_bg_sigma)
    else:
        pool = vals[finite]
        bg_med  = np.nanmedian(pool)
        mad     = np.nanmedian(np.abs(pool - bg_med))
        bg_sigma = 1.4826*mad if mad > 0 else np.nanstd(pool)
        if not np.isfinite(bg_sigma) or bg_sigma <= 0:
            raise ValueError("Background sigma not finite/positive.")

    # (3) Stricter masking using global (or ROI-fallback) z-scores
    z = (vals - bg_med) / bg_sigma
    plume_mask = (z > z_thr) & finite

    labeled_plumes = skimage.measure.label(plume_mask.astype(np.uint8), background=0, connectivity=2)
    unique_labels = np.unique(labeled_plumes)
    if unique_labels.size <= 1:
        raise ValueError("No plumes detected within the ROI.")

    # Legacy SNR for reporting/filtering
    background_noise = np.nanstd(vals[~plume_mask & finite]) if (~plume_mask & finite).any() else bg_sigma

    # (4) Magnitude gates
    n_min = max(min_plume_size, 8)
    rows = []
    for lab in unique_labels[1:]:
        idx = np.argwhere(labeled_plumes == lab)
        if idx.shape[0] < n_min:
            continue

        blob = vals[idx[:, 0], idx[:, 1]]
        anom = blob - bg_med
        anom = anom[np.isfinite(anom)]
        pos  = anom[anom > 0]
        if pos.size == 0:
            continue

        n = pos.size
        total_anom = np.nansum(pos)
        mean_anom  = np.nanmean(pos)
        peak_anom  = np.nanmax(pos)
        # print('total_anom')
        # print(total_anom, k_total * bg_sigma * np.sqrt(n))
        # print('mean_anom')
        # print(mean_anom,  k_mean  * bg_sigma)
        # print('peak_anom')
        # print(peak_anom,  k_peak  * bg_sigma)

        if not (total_anom > k_total * bg_sigma * np.sqrt(n) and
                mean_anom  > k_mean  * bg_sigma and
                peak_anom  > k_peak  * bg_sigma):
            continue

        cy = np.nanmean(roi.latitude.values[idx[:, 0]])
        cx = np.nanmean(roi.longitude.values[idx[:, 1]])
        dist = np.sqrt((cy - source_lat)**2 + (cx - source_lon)**2)

        mean_signal = np.nanmean(blob)
        snr = mean_signal / (background_noise if background_noise > 0 else 1e-9)

        rows.append({
            "label": lab,
            "centroid_lat": cy,
            "centroid_lon": cx,
            "distance": dist,
            "enhancement": float(np.nansum(blob)),
            "mean_signal": mean_signal,
            "snr": snr,
            "size": idx.shape[0]
        })

    if not rows:
        raise ValueError("No plumes passed magnitude gates.")

    df = pd.DataFrame(rows)
    df = df[(df["size"] >= min_plume_size) &
            (df["distance"] < distance_threshold) &
            (df["snr"] > snr_threshold)]
            
    if df.empty:
        raise ValueError("No plumes met distance/SNR criteria after gates.")
    
    if len(df) > 1:
        raise ValueError(f"Ambiguous scene: Multiple ({len(df)}) plume candidates after gates and filtering.")

    large = df[df["size"] > 100]
    best = large.loc[large["mean_signal"].idxmax()] if not large.empty \
           else df.loc[df["mean_signal"].idxmax()]

    best_mask = (labeled_plumes == int(best["label"]))
    out = xr.DataArray(
        best_mask,
        coords=roi.coords, dims=roi.dims,
        attrs={
            "description": "Binary mask of the detected plume",
            "snr": float(best["snr"]),
            "distance": float(best["distance"]),
            "enhancement": float(best["enhancement"]),
            "mean_signal": float(best["mean_signal"]),
            "size": int(best["size"]),
            # provenance
            "z_thr": float(z_thr),
            "bg_median": float(bg_med),
            "bg_sigma": float(bg_sigma),
            "busy_frac": float(frac_active),
            "k_total": float(k_total),
            "k_mean": float(k_mean),
            "k_peak": float(k_peak),
        }
    )
    return out
