import os
import pandas as pd
import matplotlib.pyplot as plt

def resolve_log_mode(mode):
    
    """
    Map alias mode names to the actual mode names used in filenames/logs.

    Parameters
    ----------
    mode : str
        Mode key (e.g., "xgb_raw", "raw").

    Returns
    -------
    str
        Resolved mode name used in PR curve filenames.
    """
    alias_map = {
        "xgb_raw": "raw",
        "xgb_cleaned": "cleaned"
    }
    return alias_map.get(mode, mode)

def plot_combined_pr_curves(
    satellite,
    feature,
    modes,
    match_days_list,
    log_dirs,
    save_root,
    mode_labels=None,
    model_type=None
):
    
    """
    Plot combined precision–recall curves for multiple modes at two matching-window sizes.

    Reads per-mode PR curve CSVs from:
        {log_dirs[mode]}/prcurves/{satellite}_{feature}_{mode}_mdays{mdays}.csv

    Saves a single 1x2 subplot figure to:
        {save_root}/{satellite}/{feature}_{model_tag}.png

    Parameters
    ----------
    satellite : str
        Satellite name (as used in PR filenames).
    feature : str
        Feature name (as used in PR filenames).
    modes : list[str]
        Mode keys to plot (e.g., ["raw", "cleaned"] or ["xgb_raw", "xgb_cleaned"]).
    match_days_list : list[int|float]
        Exactly two matching-window sizes (e.g., [3, 5]).
    log_dirs : dict[str, str]
        Map from mode -> base log directory containing 'prcurves' subfolder.
    save_root : str
        Root folder to save the output figure.
    mode_labels : dict[str, str], optional
        Custom labels per mode for the legend; falls back to defaults if None.
    model_type : str, optional
        Tag used in the figure title and output filename (e.g., "ARIMA", "XGB").
    """

    assert len(match_days_list) == 2, "Expected 2 matching days for subplot."

    default_labels = {
        "raw": "Univar Raw",
        "cleaned": "Univar Cleaned",
        "smoothed": "Smoothed (win=7)",
        "multivar_raw": "Multivar Raw",
        "multivar_cleaned": "Multivar Cleaned",
        "xgb_raw": "XGB Univar Raw",
        "xgb_cleaned": "XGB Univar Cleaned",
        "arima_raw": "ARIMA Raw",
        "arima_cleaned": "ARIMA Cleaned",
        "arima_smoothed": "ARIMA Smoothed"
    }
    mode_labels = mode_labels or default_labels
    satellite_clean = satellite.replace(" ", "_")
    feature_clean = feature.replace(" ", "_")

    # Collect PR dataframes by matching window then by mode
    data = {mdays: {} for mdays in match_days_list}
    for mdays in match_days_list:
        for mode in modes:
            real_mode = resolve_log_mode(mode)
            fname = f"{satellite}_{feature}_{real_mode}_mdays{mdays}"
            log_dir = log_dirs.get(mode)
            if not log_dir:
                print(f"[SKIP] No log dir for mode: {mode}")
                continue
            fpath = os.path.join(log_dir, "prcurves", fname + ".csv")
            if not os.path.exists(fpath):
                print(f"[SKIP] Missing PR curve: {fpath}")
                continue
            df = pd.read_csv(fpath)
            df = df[(df["precision"] > 0) | (df["recall"] > 0)]
            if df.empty:
                print(f"[SKIP] Empty PR curve after filtering: {fpath}")
                continue
            data[mdays][mode] = df

    if not any(data[mdays] for mdays in match_days_list):
        print(f"[SKIP] No data to plot for {satellite} – {feature}")
        return

    fig, axs = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    all_handles = []
    all_labels = []

    for i, mdays in enumerate(match_days_list):
        ax = axs[i]

        # Dynamic y-limit upper bound based on data present in this panel
        if data[mdays]:
            max_prec = max([df["precision"].max() for df in data[mdays].values()])
            upper_y = min(1.0, max(0.1, round(max_prec + 0.05, 2)))
        else:
            upper_y = 1.0

        for mode, df in data[mdays].items():
            label = mode_labels.get(mode, mode)
            line, = ax.plot(df["recall"], df["precision"], label=label, linewidth=2)
            if label not in all_labels:
                all_handles.append(line)
                all_labels.append(label)

        ax.set_title(f"Matching Window: ±{mdays} Days")
        ax.set_xlabel("Recall")
        if i == 0:
            ax.set_ylabel("Precision")
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, upper_y])

    fig.legend(
        all_handles, all_labels,
        loc="upper center",
        ncol=min(len(all_labels), 4),
        bbox_to_anchor=(0.5, 1.08),
        frameon=False
    )

    model_str = f"{model_type} " if model_type else ""
    fig.suptitle(f"{model_str}Precision–Recall Curves for {satellite} – {feature.replace('_', ' ')}", y=1.12)

    save_folder = os.path.join(save_root, satellite_clean)
    os.makedirs(save_folder, exist_ok=True)
    model_tag = model_type.lower() if model_type else "model"
    save_path = os.path.join(save_folder, f"{feature_clean}_{model_tag}.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {save_path}")