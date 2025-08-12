import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import os


def convert_timestamp_series_to_epoch(series):
    """
    Convert a pandas datetime-like Series to seconds since the Unix epoch.

    Parameters
    ----------
    series : pd.Series
        Series of datetime-like values.

    Returns
    -------
    np.ndarray
        Array of integer seconds since 1970-01-01.
    """
    return ((series - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).values

def compute_event_matching(residuals, ground_truth_times, threshold, matching_max_days):
    """
    Match predicted anomaly timestamps to ground-truth events within ±matching_max_days
    and compute event-level precision/recall metrics.

    Parameters
    ----------
    residuals : pd.Series
        Residual values indexed by timestamp.
    ground_truth_times : pd.Series
        Ground-truth event timestamps (datetime-like).
    threshold : float
        Residual threshold above which points are marked as anomalies.
    matching_max_days : float
        Maximum days for a predicted anomaly to be matched to a ground-truth event.

    Returns
    -------
    tuple
        (precision, recall, f1, f2, TP, FP, FN, anomaly_times)
    """
    matching_max_distance_seconds = timedelta(days=matching_max_days).total_seconds()
    anomaly_times = residuals[residuals > threshold].index
    gt_seconds = convert_timestamp_series_to_epoch(ground_truth_times)
    pred_seconds = convert_timestamp_series_to_epoch(anomaly_times)

    matched_gt = set()
    matched_preds = set()

    for i, pred_time in enumerate(pred_seconds):
        closest_idx = None
        closest_dist = float("inf")
        for j, gt_time in enumerate(gt_seconds):
            if j in matched_gt:
                continue
            dist = abs(pred_time - gt_time)
            if dist <= matching_max_distance_seconds and dist < closest_dist:
                closest_idx = j
                closest_dist = dist
        if closest_idx is not None:
            matched_preds.add(i)
            matched_gt.add(closest_idx)

    TP = len(matched_preds)
    FP = len(pred_seconds) - TP
    FN = len(gt_seconds) - TP

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    f2 = 5 * precision * recall / (4 * precision + recall) if (4 * precision + recall) > 0 else 0

    return precision, recall, f1, f2, TP, FP, FN, anomaly_times


def compute_precision_recall_curve_from_residuals(
    residuals,
    ground_truth_times,
    matching_max_days=1.0,
    satellite="",
    feature="",
    suffix="",
    save_path="images",
    selection_metric="f2",
    model_type="arima"
):
    
    """
    Sweep thresholds over residuals to generate a precision–recall curve, pick the best point
    using `selection_metric`, and save the PR plot.

    Parameters
    ----------
    residuals : pd.Series
        Residual values indexed by timestamp.
    ground_truth_times : pd.Series
        Ground-truth event timestamps.
    matching_max_days : float, default 1.0
        Matching tolerance in days.
    satellite : str
        Satellite name (for filenames/labels).
    feature : str
        Feature name (for filenames/labels).
    suffix : str
        Extra identifier for filenames.
    save_path : str, default "images"
        Root directory for saving plots.
    selection_metric : {"f1","f2"}, default "f2"
        Metric used to select the best threshold.
    model_type : {"arima","xgb"}, default "arima"
        Used to route outputs under images/{model_type}/...

    Returns
    -------
    (pd.DataFrame, pd.Series)
        DataFrame of PR points per threshold, and the best row (as Series).
    """
    residuals = residuals.copy().dropna()

    if not suffix:
        suffix = f"{satellite}_{feature}_mdays{int(matching_max_days)}"
    thresholds = np.sort(residuals.unique())[::-1]  # high to low thresholds

    results = []

    for threshold in thresholds:
        precision, recall, f1, f2, TP, FP, FN, anomaly_times = compute_event_matching(
            residuals, ground_truth_times, threshold, matching_max_days
        )

        results.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "f2": f2,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "anomalies" : anomaly_times
        })

    df = pd.DataFrame(results)

    # keep only non-zero PR pairs
    df = df[(df["precision"] > 0) | (df["recall"] > 0)]

    if selection_metric not in df.columns:
        raise ValueError(f"Invalid selection metric: {selection_metric}")
    best_idx = df[selection_metric].idxmax()
    best = df.loc[best_idx]
    anomaly_times = best["anomalies"] 

    # Build plot name and folder
    satellite_clean = satellite.replace(" ", "_")
    feature_clean = feature.replace(" ", "_").lower()
    plot_name = f"prplot_{satellite_clean}_{feature_clean}_{suffix}"
    base_dir = save_path
    model_base_dir = os.path.join(base_dir,model_type)
    prplot_path = os.path.join(model_base_dir, "prplots", satellite)
    os.makedirs(prplot_path, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(df["recall"], df["precision"], marker='o', label="PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve – {satellite} | {feature} | {suffix}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(prplot_path, plot_name + ".png"), dpi=300)
    plt.show()
    plt.close()

    return df,best



def sweep_thresholds_and_plot(
    residuals,
    ground_truth_times,
    matching_max_days,
    satellite,
    feature,
    mode,
    save_path="images",
    selection_metric="f2",
    summary_log_path=None,
    model_type="arima"  
):
    
    """
    Compute PR curve, save curve CSVs and top/best variants, and update a summary log.

    Outputs
    -------
    images/{model_type}/prplots/{satellite}/prplot_*.png
    {model_type}_logs/prcurves/*.csv
    {model_type}_logs/prcurves_top5/*_top5.csv
    {model_type}_logs/prcurves_best/*_best.csv
    """

    suffix = f"{mode}_mdays{int(matching_max_days)}"
    df_pr, best_metrics = compute_precision_recall_curve_from_residuals(
        residuals=residuals,
        ground_truth_times=ground_truth_times,
        matching_max_days=matching_max_days,
        satellite=satellite,
        feature=feature,
        suffix=suffix,
        save_path=save_path,
        selection_metric=selection_metric,
        model_type=model_type
    )

    if model_type not in ["arima", "xgb"]:
        raise ValueError("Unknown model type for log path!")
    abs_save_path = os.path.abspath(save_path)
    log_base_dir = abs_save_path.replace("images", f"{model_type}_logs") 
    if summary_log_path is None:
        summary_log_path = os.path.join(log_base_dir, "pr_summary.csv")
    curve_log_path = os.path.join(log_base_dir, "prcurves")
    top5_log_path = os.path.join(log_base_dir, "prcurves_top5")
    best_log_path = os.path.join(log_base_dir, "prcurves_best")
    for path in [curve_log_path, top5_log_path, best_log_path]:
        os.makedirs(path, exist_ok=True)

    plot_name = f"{satellite}_{feature}_{suffix}"
    df_pr.to_csv(os.path.join(curve_log_path, plot_name + ".csv"), index=False)
    df_pr.sort_values(by=selection_metric, ascending=False).head(5).to_csv(
        os.path.join(top5_log_path, plot_name + "_top5.csv"), index=False
    )
    best_metrics.to_frame().T.to_csv(
        os.path.join(best_log_path, plot_name + "_best.csv"), index=False
    )

    # Summary CSV
    log_best_pr_summary(
        summary_log_path=summary_log_path,
        satellite=satellite,
        feature=feature,
        mode=mode,
        mdays=matching_max_days,
        best_metrics=best_metrics
    )

    return df_pr, best_metrics


def log_best_pr_summary(summary_log_path, satellite, feature, mode, mdays, best_metrics):
    """
    Append or update the best PR result for a satellite–feature–mode–mdays combo
    in the summary CSV.

    Parameters
    ----------
    summary_log_path : str
    satellite : str
    feature : str
    mode : str
    mdays : float
    best_metrics : pd.Series
    """
    os.makedirs(os.path.dirname(summary_log_path), exist_ok=True)

    row = {
        'satellite': satellite,
        'feature': feature,
        'mode': mode,
        'mdays': mdays,
        'threshold': best_metrics['threshold'],
        'precision': best_metrics['precision'],
        'recall': best_metrics['recall'],
        'TP': best_metrics['TP'],
        'FP': best_metrics['FP'],
        'FN': best_metrics['FN'],
        'f1': best_metrics['f1'],
        'f2': best_metrics['f2']
    }

    if os.path.exists(summary_log_path):
        df_existing = pd.read_csv(summary_log_path)
        df_existing = df_existing[~(
            (df_existing['satellite'] == satellite) &
            (df_existing['feature'] == feature) &
            (df_existing['mode'] == mode) &
            (df_existing['mdays'] == mdays)
        )]
        df_updated = pd.concat([df_existing, pd.DataFrame([row])], ignore_index=True)
    else:
        df_updated = pd.DataFrame([row])

    numeric_cols = ["threshold", "precision", "recall", "f1", "f2"]
    df_updated[numeric_cols] = df_updated[numeric_cols].round(4)
    df_updated.to_csv(summary_log_path, index=False)






