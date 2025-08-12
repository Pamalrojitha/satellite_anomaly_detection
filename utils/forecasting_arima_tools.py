import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import os
import warnings
import time
import precision_recall_tools as prt

warnings.filterwarnings("ignore")

# Set global plot style
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 12,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11
})

def ensure_dir(path):
    """Ensure a directory exists (create if missing)."""
    os.makedirs(path, exist_ok=True)

def load_feature_data(path, feature_name, scale=True):
    """
    Load a specific feature from CSV, scaling small-magnitude features.

    Parameters
    ----------
    path : str
        Path to the CSV file with time-indexed features.
    feature_name : str
        Column name to load.
    scale : bool, default True
        Multiply certain features by 1e6 for readability.

    Returns
    -------
    pd.DataFrame
        DataFrame with the selected feature.
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df[[feature_name]]
    if scale and feature_name in ["eccentricity", "Brouwer mean motion"]:
        df = df  * 1e6
    return df


def interpolate_daily(df):
    """Resample to daily frequency using linear interpolation."""
    df.index = pd.to_datetime(df.index)
    daily_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    df_resampled = df.reindex(df.index.union(daily_index)).sort_index()
    df_resampled = df_resampled.interpolate('linear').loc[daily_index]
    df_resampled.index.freq = 'D'
    return df_resampled

def exclude_initial_points(df, num_points = 30):
    """Exclude the first `num_points` rows from a DataFrame."""
    return df.iloc[num_points:]


def clean_outliers_by_iqr(df, iqr_factor=3.0, verbose=True):
    """
    Replace IQR outliers with NaN and interpolate them.

    Parameters
    ----------
    df : pd.DataFrame
        Input time series.
    iqr_factor : float
        Multiplier for IQR to define outlier bounds.
    verbose : bool
        Print counts of flagged outliers.

    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        Cleaned DataFrame and boolean mask of flagged positions.
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


def plot_original_vs_interpolated(original, interpolated, feature_name, date_range=None):
    """Plot raw vs interpolated series."""
    plt.figure(figsize=(14, 5))
    if date_range:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    else:
        start = interpolated.index.min()
        end = start + pd.DateOffset(days=30)

    plt.plot(interpolated.loc[start:end].index, interpolated.loc[start:end][feature_name], label="Interpolated", linewidth=2)
    plt.plot(original.loc[start:end].index, original.loc[start:end][feature_name], label="Original", linestyle='--', marker='o', color='orange')
    plt.title(f"{feature_name}: Interpolated vs Original ({start.date()} to {end.date()})")
    plt.xlabel("Date")
    plt.ylabel(feature_name)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def run_adf_test(series, verbose=True):
    """
    Run ADF test on original and differenced series to determine ARIMA d parameter.
    Returns a list of candidate d values (usually [0] or [1]).
    """
    
    series = series.dropna()
    if series.nunique() <= 1:
        if verbose:
            print("Series is constant — ADF test skipped.")
        return [0]

    adf_stat_0, adf_p_0, *_ = adfuller(series)
    if verbose:
        print(f"ADF p-value (d=0): {adf_p_0:.5f}")

    if adf_p_0 < 0.05:
        if verbose:
            print("Series is stationary at d=0")
        return [0]

    series_diff = series.diff().dropna()
    adf_stat_1, adf_p_1, *_ = adfuller(series_diff)
    if verbose:
        print(f"ADF p-value (d=1): {adf_p_1:.5f}")

    if adf_p_1 < 0.05:
        if verbose:
            print("Series is stationary after differencing → d=1")
        return [1]
    else:
        if verbose:
            print("Warning: Series may still not be stationary after d=1")
        return [1]  


def grid_search_arima(df, d_values, max_order=15, patience=25, satellite=None, feature=None, suffix=None):
    """
    Grid search ARIMA(p,d,q) minimizing AIC.
    Saves top-5 AIC table to arima_logs if satellite, feature, suffix provided.
    """
    best_model = None
    best_order = None
    best_aic = float("inf")
    aic_results = []
    start_time = time.time()

    for d in d_values:
        no_improvement = 0
        for p in range(max_order):
            for q in range(max_order):
                if no_improvement >= patience:
                    break
                try:
                    model = ARIMA(df, order=(p, d, q))
                    res = model.fit()
                    aic = res.aic
                    aic_results.append({"p": p, "d": d, "q": q, "aic": aic})
                    if aic < best_aic:
                        best_aic = aic
                        best_model = res
                        best_order = (p, d, q)
                        no_improvement = 0
                    else:
                        no_improvement += 1
                except:
                    aic_results.append({"p": p, "d": d, "q": q, "aic": np.nan})
                    no_improvement += 1

    end_time = time.time()
    print(f"Best ARIMA Order: {best_order}, AIC: {best_aic:.2f}, Models Tried: {len(aic_results)}, Time: {round(end_time-start_time, 1)}s")
    
    # Save top-5 AIC table
    aic_df = pd.DataFrame(aic_results)
    aic_df_sorted = aic_df.sort_values(by="aic").dropna().head(5)

    if satellite and feature and suffix:
        save_name = f"aic_top5_{satellite}_{feature}_{suffix}.csv".replace(" ", "_")
        save_path = os.path.join("..", "arima_logs", save_name)
        aic_df_sorted.to_csv(save_path, index=False)

    return aic_df, best_model, best_order, best_aic


def plot_residual_diagnostics(residuals, feature_name, satellite="", suffix="raw", save_path="images"):
    """Plot and save residual time series and Q-Q plot for ARIMA residuals."""
    base_dir = os.path.join(save_path, "arima", "diagnostics", satellite)
    ensure_dir(os.path.join(base_dir, "residuals"))
    ensure_dir(os.path.join(base_dir, "qqplots"))
    
    filename_base = f"{satellite}_{feature_name}_{suffix}".replace(" ", "_")

    # Plot residuals over time
    plt.figure(figsize=(14, 4))
    residuals.plot()
    plt.title(f"Residuals from ARIMA - {feature_name}")
    plt.axhline(0, linestyle='--', color='gray')
    plt.xlabel("Date")         
    plt.ylabel("Residuals") 
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "residuals", f"{filename_base}_residuals.png"), dpi=300)
    plt.show()

    # Q-Q plot
    sm.qqplot(residuals, line='s')
    plt.title(f"{satellite}-{feature_name}:Q-Q Plot of Residuals")
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "qqplots", f"{filename_base}_qqplot.png"), dpi=300)
    plt.show()



def evaluate_and_log_precision_recall_arima(
    residuals,
    ground_truth_times,
    satellite,
    feature,
    mode,
    matching_max_days,
    save_path,
    summary_log_path,
    selection_metric="f2"
):
    
    """
    Generate PR curves from ARIMA residuals and log best results.

    Returns
    -------
    (pd.DataFrame, pd.Series)
        PR curve dataframe and best metrics row.
    """

    from precision_recall_tools import sweep_thresholds_and_plot

    df_pr, best_metrics = sweep_thresholds_and_plot(
        residuals=residuals,
        ground_truth_times=ground_truth_times,
        matching_max_days=matching_max_days,
        satellite=satellite,
        feature=feature,
        mode=mode,
        save_path=save_path,
        selection_metric=selection_metric,
        summary_log_path=summary_log_path,
        model_type="arima"
    )

    best_threshold = best_metrics["threshold"]
    prec = best_metrics["precision"]
    recall = best_metrics["recall"]
    f1 = best_metrics["f1"]
    f2 = best_metrics["f2"]
    TP = best_metrics["TP"]
    FP = best_metrics["FP"]
    FN = best_metrics["FN"]

    mode_labels = {"raw": "[RAW]", "cleaned": "[CLEANED]", "smoothed": "[SMOOTHED]"}
    mode_tag = mode_labels.get(mode, f"[{mode.upper()}]")
    print(f"{mode_tag} | mdays={matching_max_days} | Threshold={best_threshold:.4f} | P={prec:.2f} | R={recall:.2f} | F1={f1:.2f} | F2={f2:.2f}")
    print(f"{mode_tag} TP={TP}, FP={FP}, FN={FN}")

    return df_pr, best_metrics


def make_subfolder(base, satellite, cleaned=False, mode=None):
    """Make and return a subfolder path based on satellite and mode/cleaned flag."""
    if mode:
        folder_name = f"{satellite}_{mode}"
    else:
        folder_name = f"{satellite}_cleaned" if cleaned else satellite

    folder = os.path.join(base, folder_name)
    os.makedirs(folder, exist_ok=True)
    return folder


def plot_forecast_results_arima(
    observed,
    predicted,
    satellite,
    feature,
    save_path,
    ground_truth_times,
    residuals,
    anomaly_times,
    threshold,
    zoom_range=None,
    cleaned=False,
    mode = None
):
    """Plot ARIMA forecasts with residuals, anomalies, and ground truth markers."""

    save_dir = make_subfolder(os.path.join(save_path, "arima", "forecast"), satellite, cleaned, mode)

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
        mode_tag = f"_{mode}" if mode else ("_cleaned" if cleaned else "_raw")
        filename = f"{feature}{mode_tag}_{filename_suffix}.png".replace(" ", "_").lower()

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


def run_arima_on_smoothed_series(
    df_interp,
    satellite,
    feature,
    maneuver_times,
    date_range,
    matching_days_list,
    output_path,
    summary_log_path,
    aic_log_path,
    window=7
):
    """
    Run ARIMA on smoothed (rolling mean) series, detect anomalies, and plot results.
    """

    smoothed_series = df_interp[feature].rolling(window=window, min_periods=1, center=True).mean()
    df_smoothed = pd.DataFrame({feature: smoothed_series})

    d_values = run_adf_test(df_smoothed[feature])
    aic_df, model, order, _ = grid_search_arima(
        df_smoothed, d_values, satellite=satellite, feature=feature, suffix="smoothed"
    )
    aic_df["Satellite"] = satellite
    aic_df["Feature"] = feature
    aic_df[["Satellite", "Feature", "p", "d", "q", "aic"]].to_csv(aic_log_path, mode="a", header=False, index=False)

    if model:
        fitted = model.fittedvalues
        observed = df_smoothed.squeeze()
        residuals = observed - fitted

        plot_residual_diagnostics(residuals, feature, satellite=satellite, suffix="smoothed", save_path=output_path)

        best_candidates = []
        for mdays in matching_days_list:
            print(f"\n[SMOOTHED] Computing PR Curve – {satellite} | {feature} | ±{mdays} days")
            df_pr, best_metrics = evaluate_and_log_precision_recall_arima(
                residuals=residuals.abs(),
                ground_truth_times=maneuver_times,
                satellite=satellite,
                feature=feature,
                mode="smoothed",
                matching_max_days=mdays,
                save_path=output_path,
                summary_log_path=summary_log_path,
                selection_metric="f2"
            )
            best_metrics["mdays"] = mdays
            best_candidates.append(best_metrics)

        best_overall = max(best_candidates, key=lambda x: x["f2"])
        print(f"\n[SMOOTHED] Using BEST mdays = ±{int(best_overall['mdays'])} for anomaly detection (F2 = {best_overall['f2']:.2f})")
        threshold = best_overall["threshold"]

        prt.log_best_pr_summary(
            summary_log_path,
            satellite=satellite,
            feature=feature,
            mode="smoothed",
            mdays=best_overall["mdays"],
            best_metrics=best_overall
        )

        plot_forecast_results_arima(
            observed=observed,
            predicted=fitted,
            residuals=residuals,
            satellite=satellite,
            feature=feature,
            save_path=output_path,
            ground_truth_times=maneuver_times,
            anomaly_times=best_overall["anomalies"],
            threshold=threshold,
            zoom_range=date_range,
            cleaned=False,
            mode = "smoothed"
        )
    else:
        print(f"[SMOOTHED] ARIMA model fitting failed for {satellite} | {feature}")

