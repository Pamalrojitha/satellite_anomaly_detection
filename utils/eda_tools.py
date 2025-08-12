import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os
import plotly.graph_objects as go
import plotly.io as pio
import re


plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 12,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11
})

pio.renderers.default = 'browser'

def load_satellite_data(csv_path):
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['Unnamed: 0'], errors='coerce')
    df.drop(columns=['Unnamed: 0'], inplace=True)
    df.set_index('timestamp', inplace=True)
    return df

def summarize_data(df, satellite_name, log_path=None):
    print(df.describe())
    print(df.isna().sum())

    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        # Compute NA counts as a single-row DataFrame
        na_counts = df.isna().sum().to_frame().T
        na_counts.insert(0, "Satellite", satellite_name)

        # Write or append to CSV
        if os.path.exists(log_path):
            existing = pd.read_csv(log_path)
            updated = pd.concat([existing, na_counts], ignore_index=True)
            updated.to_csv(log_path, index=False)
        else:
            na_counts.to_csv(log_path, index=False)


def detect_iqr_outliers(df, satellite_name, iqr_factor=1.5, show_top_n=10, log_path=None):
    """
    Detect IQR-based outliers and log counts per feature in wide format (one row per satellite).

    Args:
        df (DataFrame): Time series data
        satellite_name (str): Name of the satellite
        iqr_factor (float): IQR multiplier
        show_top_n (int): Number of top outliers to display
        log_path (str or None): CSV log path for writing outlier counts

    Returns:
        Dict of DataFrames with outlier rows per feature
    """
    outlier_records = {}
    counts_dict = {}

    for feature in df.columns:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - iqr_factor * IQR
        upper = Q3 + iqr_factor * IQR

        mask = (df[feature] < lower) | (df[feature] > upper)
        outliers = df.loc[mask, [feature]].copy()
        count = mask.sum()
        counts_dict[feature] = count

        print(f"{feature}: {count} outliers flagged")

        if not outliers.empty:
            outliers["outlier_score"] = outliers[feature].apply(
                lambda x: abs(x - Q1) if x < Q1 else abs(x - Q3)
            )
            top_outliers = outliers.sort_values(by="outlier_score", ascending=False).head(show_top_n)
            print(top_outliers[[feature]])

        outlier_records[feature] = outliers.drop(columns=["outlier_score"], errors='ignore')

    # Logging to wide-format CSV
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        wide_row = pd.DataFrame([counts_dict])
        wide_row.insert(0, "Satellite", satellite_name)

        if os.path.exists(log_path):
            existing = pd.read_csv(log_path)
            updated = pd.concat([existing, wide_row], ignore_index=True)
            updated.to_csv(log_path, index=False)
        else:
            wide_row.to_csv(log_path, index=False)

    return outlier_records



def load_maneuver_file(file_path):
    colspecs = [(0,5), (6,10), (11,14), (15,17), (18,20), (21,25), (26,29), (30,32), (36,39), (40,43), (44,46)]
    colnames = ['satellite_id','start_year','start_day','start_hour','start_minute','end_year','end_day','end_minute',
                'maneuver_type','parameter_type','burn_number']
    df = pd.read_fwf(file_path, colspecs=colspecs, names=colnames, header=None)
    df['start_time'] = df.apply(lambda row: datetime(row['start_year'], 1, 1) + timedelta(days=row['start_day']-1,
                                                                                         hours=row['start_hour'],
                                                                                         minutes=row['start_minute']), axis=1)
    cleaned = df[['start_time']].drop_duplicates().sort_values('start_time')
    output_dir = "../satellite_data/cleaned_man"
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"cleaned_man_{os.path.splitext(os.path.basename(file_path))[0]}.csv"
    cleaned.to_csv(os.path.join(output_dir, file_name), index=False)

    return cleaned

def load_fengyun_maneuver_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    records = []
    for line in lines:
        parts = line.strip().split('"')
        if len(parts) >= 3:
            info = parts[0].split()
            start = pd.to_datetime(parts[1].replace(" CST", ""))
            end = pd.to_datetime(parts[3].replace(" CST", ""))
            records.append({'maneuver_type': info[0], 'satellite_id': info[1], 'start_time': start, 'end_time': end})
    df = pd.DataFrame(records)
    cleaned = df[['start_time']].drop_duplicates().sort_values('start_time')
    output_dir = "../satellite_data/cleaned_man"
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(file_path)
    clean_name = re.sub(r'\.txt.*$','',base_name)
    file_name = f"cleaned_man_{clean_name}.csv"
    cleaned.to_csv(os.path.join(output_dir, file_name), index=False)

    return cleaned


def plot_time_series_with_maneuvers(df, man_df, satellite_name, save_path="images"):
    for feature in df.columns:
        plt.figure(figsize=(12, 5))
        plt.plot(df.index, df[feature], label=feature, color='b', linewidth=2.0)
        # Interpolate for maneuver points
        interpolated = df[feature].reindex(df.index.union(man_df['start_time'])).interpolate('time').loc[man_df['start_time']].values
        plt.scatter(man_df['start_time'], interpolated, color='red', marker='x', s=60, linewidths=2, label='maneuver start')
        plt.title(f"{satellite_name}: {feature} over time with actual maneuvers")
        plt.xlabel("Time")
        plt.ylabel(feature)
        plt.legend()
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/{satellite_name}_{feature.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
        plt.show()


def plot_time_series_for_range(df, man_df, start_date, end_date, satellite_name, save_path="images"):
    df_range = df[(df.index >= start_date) & (df.index <= end_date)]
    man_range = man_df[(man_df['start_time'] >= start_date) & (man_df['start_time'] <= end_date)]
    for feature in df_range.columns:
        plt.figure(figsize=(12, 5))
        plt.plot(df_range.index, df_range[feature], label=feature, color='b')
        interpolated = df_range[feature].reindex(df_range.index.union(man_range['start_time'])).interpolate('time').loc[man_range['start_time']].values
        plt.scatter(man_range['start_time'], interpolated, color='red', marker='x', s=60, linewidths=2,label='maneuver start')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xlabel("Time")
        plt.ylabel(feature)
        plt.title(f"{satellite_name}: {feature} from {start_date} to {end_date} with maneuvers")
        plt.legend()
        os.makedirs(save_path, exist_ok=True)
        filename = f"{satellite_name}_{start_date[:7]}_to_{end_date[:7]}_{feature.replace(' ', '_')}.png"
        plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
        plt.show()


def interactive_plot(df, man_df):
    for feature in df.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df[feature], mode='lines', name=feature, line=dict(color='blue')))
        valid_times = man_df['start_time'][(man_df['start_time'] >= df.index.min()) & (man_df['start_time'] <= df.index.max())]
        interpolated = df[feature].reindex(df.index.union(valid_times)).interpolate('time').loc[valid_times].values
        fig.add_trace(go.Scatter(x=valid_times, y=interpolated, mode='markers', name='maneuver start',
                                 marker=dict(color='red', symbol='x', size=10)))
        fig.update_layout(title=f"{feature} over time with maneuvers", xaxis_title="Time", yaxis_title="Value",
                          hovermode='x unified', showlegend=True)
        fig.update_yaxes(autorange=True)
        fig.show()