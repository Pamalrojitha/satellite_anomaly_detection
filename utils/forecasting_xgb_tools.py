import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from precision_recall_tools import sweep_thresholds_and_plot 


def exclude_initial_points(df, num_points=30):
    """
    Drop the first `num_points` rows from a time-indexed DataFrame.
    """
    return df.iloc[num_points:]

def rescale_feature(df, feature_name):
    """
    Rescale select features for readability.

    If `feature_name` is one of ["Brouwer mean motion", "eccentricity"],
    multiply by 1e6 in-place and print a short note.
    """
    if feature_name in ["Brouwer mean motion", "eccentricity"]:
        df[feature_name] *= 1e6
        print(f"Rescaled {feature_name} ×1e6")
    return df

def interpolate_daily(df):
    """
    Resample to daily frequency using linear interpolation.

    Returns a new DataFrame on a daily DateTimeIndex with freq='D'.
    """
    df.index = pd.to_datetime(df.index)
    daily_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    df_resampled = df.reindex(df.index.union(daily_index)).sort_index()
    df_resampled = df_resampled.interpolate('linear').loc[daily_index]
    df_resampled.index.freq = 'D'
    return df_resampled

def clean_outliers_by_iqr(df, iqr_factor=3.0, verbose=True):
    """
    Replace IQR-based outliers with NaN and interpolate them.

    Returns a tuple of:
      - cleaned DataFrame
      - boolean mask DataFrame marking where outliers were flagged
    """

    df_cleaned = df.copy()
    outlier_mask_total = pd.DataFrame(False, index=df.index, columns=df.columns)
    total_flagged = 0

    for feature in df.columns:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - iqr_factor * IQR
        upper = Q3 + iqr_factor * IQR

        mask = (df[feature] < lower) | (df[feature] > upper)
        flagged = mask.sum()

        df_cleaned.loc[mask, feature] = np.nan
        outlier_mask_total[feature] = mask

        total_flagged += flagged
        if verbose:
            print(f"{feature}: {flagged} outliers flagged")

    df_cleaned = df_cleaned.interpolate('time')
    if verbose:
        print(f"Total interpolated outliers: {total_flagged}")

    return df_cleaned, outlier_mask_total

def generate_lag_features(df, feature_name, num_lags=3):
    """
    Build univariate lag features for `feature_name`.

    Returns:
      X (DataFrame): columns feature_lag_1..num_lags (NaNs dropped)
      y (Series): target aligned to X.index
    """

    df_lagged = pd.DataFrame({
        f"{feature_name}_lag_{i}": df[feature_name].shift(i)
        for i in range(1, num_lags + 1)
    })
    y = df[feature_name]
    X = df_lagged.dropna()
    y = y.loc[X.index]
    return X, y

def make_subfolder(base, satellite, cleaned=False):
    """
    Create a satellite/mode-specific subfolder under `base`.

    cleaned can be:
      - True / False for univariate modes
      - "multivar_cleaned" / "multivar_raw" for multivariate modes
    """

    if cleaned == True:
        folder = os.path.join(base, f"{satellite}_cleaned")
    elif cleaned == "multivar_cleaned":
        folder = os.path.join(base, f"{satellite}_multivar_cleaned")
    elif cleaned == "multivar_raw":
        folder = os.path.join(base, f"{satellite}_multivar_raw")
    else:
        folder = os.path.join(base, satellite)

    os.makedirs(folder, exist_ok=True)
    return folder


def plot_training_curves(model, satellite, feature, save_path, cleaned=False):
    """
    Save training vs validation MAE curves from an XGBRegressor `evals_result()`.
    """
    save_dir = make_subfolder(save_path, satellite, cleaned)
    results = model.evals_result()
    plt.figure(figsize=(8, 4.5))
    plt.plot(results.get('validation_0', {}).get('mae', []), label="Train MAE")
    plt.plot(results.get('validation_1', {}).get('mae', []), label="Val MAE")
    plt.xlabel("Boosting Rounds")
    plt.ylabel("MAE")
    plt.title(f"{satellite} - {feature}")
    plt.legend()
    plt.tight_layout()
    filename = f"{satellite}_{feature}_{str(cleaned).lower()}_training_curve.png".replace(" ", "_").lower()
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.show()
    plt.close()


def plot_forecast_results(observed, predicted, satellite, feature, save_path,
                          ground_truth_times, residuals, anomaly_times, threshold,
                          zoom_range=None, cleaned=False):
    """
    Plot observed vs predicted, with residuals and anomaly markers, for full and zoom views.
    """

    save_dir = make_subfolder(save_path, satellite, cleaned)

    def _plot(x_obs, x_pred, x_res, anomaly_idx, label, filename_suffix, zoom_range=None):
        fig, ax1 = plt.subplots(figsize=(12, 4.5))
        ax1.plot(x_obs, label="Observed", color='blue')
        ax1.plot(x_pred, label="Predicted", color='orange')
        ax1.set_xlabel("Date")
        ax1.set_ylabel(f"{feature}", color='black')
        ax1.tick_params(axis='y', labelcolor='black')

        ax2 = ax1.twinx()
        abs_res = x_res.abs()
        ax2.plot(abs_res, label="|Residuals|", color='green')
        ax2.axhline(threshold, color='red', linestyle='--', label="Threshold")
        ax2.scatter(anomaly_idx, abs_res.loc[anomaly_idx], color='black', marker='x', s=60,
                    linewidths=2, label="Anomalies")
        ax2.set_ylabel("|Residual|", color='black')
        ax2.tick_params(axis='y', labelcolor='black')

        shown_label = False
        for ts in ground_truth_times:
            if x_obs.index[0] <= ts <= x_obs.index[-1]:
                ax1.axvline(ts, linestyle=':', color='brown', alpha=0.7, linewidth=2.0,
                            label="Ground Truth" if not shown_label else "")
                shown_label = True

        if filename_suffix == "zoom" and zoom_range:
            start_str = pd.to_datetime(zoom_range[0]).date()
            end_str = pd.to_datetime(zoom_range[1]).date()
            ax1.set_title(f"{satellite} | {feature}\nForecast and Residuals: Zoomed ({start_str} to {end_str})", pad=10)
        else:
            ax1.set_title(f"{satellite} | {feature}\nForecast and Residuals: Full Timeline", pad=10)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, bbox_to_anchor=(1.14, 1), loc="upper left", frameon=True)

        fig.tight_layout(rect=[0, 0, 0.88, 1])
        filename = f"{feature}_{filename_suffix}.png".replace(" ", "_").lower()
        plt.savefig(os.path.join(save_dir, filename), dpi=300)
        plt.show()
        plt.close()

    _plot(observed, predicted, residuals, anomaly_times, "Full Timeline", "full")
    if zoom_range:
        start, end = zoom_range
        obs_zoom = observed.loc[start:end]
        pred_zoom = predicted.loc[start:end]
        res_zoom = residuals.loc[start:end]
        anom_zoom = anomaly_times[(anomaly_times >= start) & (anomaly_times <= end)]
        _plot(obs_zoom, pred_zoom, res_zoom, anom_zoom, "Zoomed", "zoom", zoom_range=zoom_range)

def evaluate_and_log_precision_recall(
    residuals,
    ground_truth_times,
    satellite,
    feature,
    mode,
    log_dir,
    matching_max_days=1.0,
    selection_metric="f2"
):
    """
    Run threshold sweep, save PR curve CSV/plots via sweep_thresholds_and_plot, and print best stats.

    Returns
    -------
    (pd.DataFrame, pd.Series): PR curve dataframe and best metrics row.
    """

    df_pr, best_metrics = sweep_thresholds_and_plot(
    residuals=residuals,
    ground_truth_times=ground_truth_times,
    matching_max_days=matching_max_days,
    satellite=satellite,
    feature=feature,
    mode=mode,
    save_path="../images",
    selection_metric=selection_metric,
    summary_log_path=os.path.join(log_dir, "pr_summary.csv"),
    model_type="xgb"  
)

    best_threshold = best_metrics["threshold"]
    prec = best_metrics["precision"]
    recall = best_metrics["recall"]
    f1 = best_metrics["f1"]
    f2 = best_metrics["f2"]
    TP = best_metrics["TP"]
    FP = best_metrics["FP"]
    FN = best_metrics["FN"]

    mode_tag = "[CLEANED]" if mode == "cleaned" else "[RAW]"
    print(f"{mode_tag} | mdays={matching_max_days} | Threshold={best_threshold:.4f} | P={prec:.2f} | R={recall:.2f} | F1={f1:.2f} | F2={f2:.2f}")
    print(f"{mode_tag} TP={TP}, FP={FP}, FN={FN}")

    return df_pr, best_metrics

def generate_multivariate_lag_features(df, input_features, target_feature, num_lags):
    """
    Build multivariate lag features for `input_features` to predict `target_feature`.

    Returns:
      X_lagged (DataFrame) and y (Series) aligned on the same index.
    """

    X_lagged = pd.DataFrame(index=df.index)
    for feature in input_features:
        for lag in range(1, num_lags + 1):
            X_lagged[f"{feature}_lag{lag}"] = df[feature].shift(lag)

    y = df[target_feature].copy()
    return X_lagged, y

def run_multivariate_forecasting(
    df,
    satellite,
    maneuver_times,
    input_features,
    target_features,
    date_range,
    num_lags,
    matching_days_list,
    output_dirs,
    hyperparam_grid
):
    """
    Train XGB multivariate forecasters (raw and, if outliers exist, cleaned),
    log PR curves for ±matching_days_list, and save forecasts/residuals/curves.

    Notes:
      - Uses 80/20 split for early stopping, then refits on full data.
      - Saves residuals and forecasts under images/xgb/... respecting mode.
      - PR curves and logs are written under xgb_logs/.
    """

    df = df.copy()
    for feat in input_features:
        df = rescale_feature(df, feat)
    df = interpolate_daily(df)
    df = exclude_initial_points(df)

    df_cleaned, outlier_mask = clean_outliers_by_iqr(df, iqr_factor=5.0, verbose=False)
    outlier_detected = outlier_mask.any(axis=1).any()

    modes = [("multivar_raw", df)]
    if outlier_detected:
        modes.append(("multivar_cleaned", df_cleaned))
        overlap_count = sum(
            any((outlier_mask.index[outlier_mask.any(axis=1)] >= gt - pd.Timedelta(days=1)) &
                (outlier_mask.index[outlier_mask.any(axis=1)] <= gt + pd.Timedelta(days=1)))
            for gt in maneuver_times
        )
        overlap_log_path = os.path.join(output_dirs["log"], "outlier_overlap_log.csv")
        exists = os.path.exists(overlap_log_path)
        pd.DataFrame([{
            "Satellite": satellite,
            "Feature": "MULTIVARIATE",
            "Outliers_Flagged": outlier_mask.any(axis=1).sum(),
            "Outliers_Overlapping_with_Manoeuvres": overlap_count
        }]).to_csv(overlap_log_path, mode='a', index=False, header=not exists)

    for target in target_features:
        for mode_tag, df_used in modes:
            print(f"\n=== MULTIVARIATE: {satellite} | Target = {target} | Mode = {mode_tag} ===")

            X_lag, y = generate_multivariate_lag_features(df_used, input_features, target, num_lags)
            valid_idx = X_lag.dropna().index.intersection(y.dropna().index)
            X_lag, y = X_lag.loc[valid_idx], y.loc[valid_idx]

            if len(y) < 10:
                print(f"Skipping {satellite} - {target} - {mode_tag} (not enough data)")
                continue

            best_model, best_params, best_iter = None, None, None
            lowest_mae = float("inf")
            split = y.index[int(len(y) * 0.8)]

            for params in hyperparam_grid:
                model = XGBRegressor(**params, early_stopping_rounds=10, eval_metric="mae", verbosity=0)
                model.fit(
                    X_lag[X_lag.index < split], y[y.index < split],
                    eval_set=[
                        (X_lag[X_lag.index < split], y[y.index < split]),
                        (X_lag[X_lag.index >= split], y[y.index >= split])
                    ],
                    verbose=False
                )
                val_mae = model.evals_result()['validation_1']['mae'][model.best_iteration]
                if val_mae < lowest_mae:
                    best_model, best_params, best_iter = model, params, model.best_iteration
                    lowest_mae = val_mae

            print(f"[{mode_tag.upper()}] Best params for {satellite} | {target}: {best_params}, Best iter: {best_iter}")

            plot_training_curves(best_model, satellite, target, output_dirs["curve"], cleaned=mode_tag)

            final_params = best_params.copy()
            final_params["n_estimators"] = best_iter + 1
            final_model = XGBRegressor(**final_params, eval_metric="mae", verbosity=0)
            final_model.fit(X_lag, y)
            y_pred = pd.Series(final_model.predict(X_lag), index=y.index)
            residuals = (y - y_pred).abs()

            # Save residuals and forecasts
            res_dir = make_subfolder(output_dirs["residual"], satellite, cleaned=mode_tag)
            residuals.to_csv(os.path.join(res_dir, f"{target}_residuals.csv".replace(" ", "_").lower()))

            fc_dir = make_subfolder(output_dirs["forecast"], satellite, cleaned=mode_tag)
            pd.DataFrame({"observed": y, "predicted": y_pred}).to_csv(
                os.path.join(fc_dir, f"{target}_forecast.csv".replace(" ", "_").lower())
            )

            # PR curve sweep for each match window
            best_f2 = -1
            best_metrics = None
            best_day = None
            for mday in matching_days_list:
                df_pr, metrics = sweep_thresholds_and_plot(
                    residuals=residuals,
                    ground_truth_times=maneuver_times,
                    matching_max_days=mday,
                    satellite=satellite,
                    feature=target,
                    mode=mode_tag,
                    save_path="../images", 
                    selection_metric="f2",
                    summary_log_path=os.path.join(output_dirs["log"], "pr_summary.csv"),
                    model_type="xgb"
                )
                if metrics is not None and metrics["f2"] > best_f2:
                    best_f2 = metrics["f2"]
                    best_metrics = metrics
                    best_day = mday

                print(f"[{mode_tag.upper()}] mdays={mday} | Threshold={metrics['threshold']:.4f} | "
                      f"P={metrics['precision']:.2f} | R={metrics['recall']:.2f} | F1={metrics['f1']:.2f} | F2={metrics['f2']:.2f}")
                print(f"[{mode_tag.upper()}] TP={metrics['TP']}, FP={metrics['FP']}, FN={metrics['FN']}")

            if best_metrics is None:
                print(f"[{mode_tag.upper()}] No valid threshold found for {target}")
                continue

            print(f"\n[{mode_tag.upper()}] Using BEST mdays = ±{int(best_day)} for {target} (F2 = {best_f2:.2f})")

            plot_forecast_results(
                observed=y, predicted=y_pred, satellite=satellite, feature=target,
                save_path=output_dirs["forecast"],
                ground_truth_times=maneuver_times,
                residuals=residuals,
                anomaly_times=residuals[residuals > best_metrics["threshold"]].index,
                threshold=best_metrics["threshold"],
                zoom_range=date_range,
                cleaned=mode_tag
            )







