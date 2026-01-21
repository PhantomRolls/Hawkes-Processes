import yaml
import os
import csv
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import json

plt.rcParams.update({
                "font.family": "serif",
                "font.size": 9,
                "axes.titlesize": 9,
                "axes.labelsize": 9,
                "legend.fontsize": 8,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
                "axes.linewidth": 0.8,
            })

def load_config(path="config.yaml"):
        with open(path, "r") as f:
            return yaml.safe_load(f)

def load_real_data(start, end, path):
    T = (datetime.strptime(end, "%Y-%m-%d %H:%M:%S") - datetime.strptime(start, "%Y-%m-%d %H:%M:%S")).total_seconds()
    df = pd.read_csv(path)
    df['ets'] = pd.to_datetime(df['ets'], format='%Y%m%d:%H:%M:%S.%f')
    df_trades = df.loc[df['etype'] == 'T'].copy()
    df_trades = df_trades.sort_values('ets')
    df_trades.loc[:, 'N'] = range(1, len(df_trades)+1)
    start = pd.Timestamp(start)
    end   = pd.Timestamp(end)
    df_zoom = df_trades.loc[(df_trades['ets'] >= start) & 
                            (df_trades['ets'] <= end)].copy()
    df_zoom.loc[:, 't'] = df_zoom['ets']
    t0 = start 
    df_zoom.loc[:, 't_sec'] = (df_zoom['t'] - t0).dt.total_seconds()
    df_zoom = df_zoom.sort_values('t_sec')
    events_real = df_zoom['t_sec'].values
    return events_real, T

def save_results_to_csv(result, csv_path):
    """
    Append one result dictionary to a CSV file,
    creating the directory and header if needed.
    """

    # Ensure directory exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    fieldnames = [
        "generator",
        "method",
        "alpha_level",
        "M",
        "KS",
        "CvM",
        "AD",
        "time_seconds",
    ]

    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(result)


def save_params_json(key, params, json_path):
    def to_json_serializable(obj):
        if isinstance(obj, dict):
            return {k: to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_json_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        else:
            return obj
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
    else:
        data = {}

    data[key] = params
    data = to_json_serializable(data)
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
        
def plot_counting_process(events, T=None, start_hour=9):
    events = np.sort(np.asarray(events))

    if events.size == 0:
        raise ValueError("events is empty")

    if T is None:
        T = events[-1]

    events = events[events <= T]

    # Step function
    x = np.concatenate(([0.0], events, [T]))
    y = np.concatenate(([0], np.arange(1, len(events) + 1), [len(events)]))

    # Convert seconds to hours of day
    x_hours = start_hour + x / 3600.0

    plt.figure(figsize=(6.8, 2.6))
    plt.step(
        x_hours, y,
        where="post",
        color="black",
        linewidth=1.2
    )

    plt.ylabel("$N(t)$")

    # Nice hour ticks
    hour_min = np.floor(x_hours.min())
    hour_max = np.ceil(x_hours.max())
    plt.xticks(
        np.arange(hour_min, hour_max + 1),
        [f"{int(h):02d}:00" for h in range(int(hour_min), int(hour_max) + 1)]
    )

    plt.tick_params(axis="both", labelsize=8)
    plt.tight_layout(pad=0.4)
    plt.show()

def plot_multiple_days(list_of_events, T=None, start_hour=9):
    plt.figure(figsize=(6.8, 2.6))

    grays = ["black", "gray",  "silver"]
    linestyles = ["-", "--", ":", "-."]
    n = len(list_of_events)

    for i, events in enumerate(list_of_events):
        events = np.sort(np.asarray(events))

        if T is None:
            T_use = events[-1]
        else:
            T_use = T

        x = np.concatenate(([0], events, [T_use]))
        y = np.concatenate(([0], np.arange(1, len(events) + 1), [len(events)]))
        x_hours = start_hour + x / 3600.

        color = grays[i % len(grays)]
        style = linestyles[(i // len(grays)) % len(linestyles)]

        plt.step(x_hours, y, where="post", linewidth=1.2,
                 color=color, linestyle=style)

    plt.ylabel("$N(t)$")
    hour_min = start_hour
    hour_max = np.floor(start_hour + T_use/3600.)

    plt.xticks(
        np.arange(hour_min, hour_max+1),
        [f"{int(h):02d}:00" for h in range(int(hour_min), int(hour_max+1))]
    )

    plt.tight_layout()
    plt.show()


def plot_two_counting_processes(events1, events2, T=None):
    events1 = np.sort(np.asarray(events1))
    events2 = np.sort(np.asarray(events2))

    if T is None:
        T = max(events1.max(), events2.max())

    times = np.linspace(0, T, 1000)

    N1 = np.searchsorted(events1, times, side="right")
    N2 = np.searchsorted(events2, times, side="right")

    plt.figure(figsize=(12, 4))
    plt.plot(times, N1, label="estimated", color="blue")
    plt.plot(times, N2, label="real", color="red")

    plt.xlabel("Time")
    plt.ylabel("N(t)")
    plt.title("Counting processes N1(t) and N2(t)")
    plt.legend()
    plt.tight_layout()
    plt.show() 

def plot_interarrival_distribution(events_real, bins=50, density=False, logx=True, ax=None):
    events = np.asarray(events_real)

    inter = np.diff(events)

    inter_pos = inter[inter > 0]

    if logx:
        min_i = inter_pos.min()
        max_i = inter_pos.max()
        log_bins = np.logspace(np.log10(min_i), np.log10(max_i), bins)
        used_bins = log_bins
    else:
        used_bins = bins

    if ax is None:
        ax = plt.gca()
    ax.hist(inter_pos, bins=used_bins, density=density, color='0.7', edgecolor='0.2', linewidth=0.3)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel("Inter-Arrival Time" + (" (log)" if logx else ""))
    ax.set_ylabel("Density" if density else "Frequency")


def plot_bic(scores, J_max, criterion, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.gca()
        ax.plot(
            range(1, J_max + 1),
            scores,
            lw=0.9,
            marker='o',
            ms=2.8,
            color="k"  
        )
        ax.set_xlabel("Number of Components")
        ax.set_ylabel(criterion.upper())


from scipy.stats import lognorm

def plot_gmm_interarrival_counts(
    events_real,
    gmm,
    bins=None,
    ax=None,
    plot_components=False,
    plot_total=True,
):
    events = np.asarray(events_real)
    inter = np.diff(events)
    inter = inter[inter > 0]

    if ax is None:
        ax = plt.gca()
        
    if bins is None:
        inter = np.diff(events)
        inter = inter[inter > 0]
        bins = np.logspace(np.log10(inter.min()), np.log10(inter.max()), 80)

    bin_edges = bins if np.ndim(bins) > 0 else np.histogram_bin_edges(inter, bins=bins)

    N = len(inter)
    counts_total = np.zeros(len(bin_edges) - 1)

    # Comptes attendus par bin
    for j in range(gmm.n_components):
        mu = gmm.means_[j, 0]
        sigma = np.sqrt(gmm.covariances_[j, 0, 0])
        w = gmm.weights_[j]

        # Probabilité par bin
        cdf = lognorm.cdf(bin_edges, s=sigma, scale=np.exp(mu))
        probs = np.diff(cdf)

        counts = N * w * probs
        counts_total += counts

        if plot_components:
            ax.step(
                bin_edges[:-1],
                counts,
                where="post",
                lw=2,
                label=f"Comp {j+1} | τ={np.exp(mu):.2e}",
                color="red",
                alpha=1
            )

    if plot_total:
        bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
        x = np.logspace(np.log10(bin_edges[0]), np.log10(bin_edges[-1]), 2000)
        y = np.interp(np.log10(x), np.log10(bin_centers), counts_total)
        ax.plot(
            x,
            y,
            lw=1.2,
            color="k",
            label="GMM total"
        )

        # ax.step(
        #     bin_edges[:-1],
        #     counts_total,
        #     where="post",
        #     lw=2.5,
        #     color="red",
        #     linestyle="-",
        #     label="GMM total"
        # )



from scipy.stats import lognorm

def annotate_gmm_weights(
    events_real,
    gmm,
    ax,
    bins=None,
    fontsize=7.5,
    alpha=0.9,
    y_mult=1.10,        
    stack_mult=1.18,  
    min_weight=0.0    
):
    events = np.asarray(events_real)
    inter = np.diff(events)
    inter = inter[inter > 0]
    N = len(inter)

    if bins is None:
        inter = np.diff(events)
        inter = inter[inter > 0]
        bins = np.logspace(np.log10(inter.min()), np.log10(inter.max()), 80)
        
    bin_edges = np.asarray(bins, dtype=float)
    if bin_edges.ndim == 0:
        bin_edges = np.histogram_bin_edges(inter, bins=bins)

    used_bins = {}  # k -> combien de labels déjà posés sur ce bin

    for j in range(gmm.n_components):
        w = float(gmm.weights_[j])
        if w < min_weight:
            continue

        mu = float(gmm.means_[j, 0])
        sigma = float(np.sqrt(gmm.covariances_[j, 0, 0]))

        # Counts attendus par bin pour CETTE composante (exactement comme le step)
        cdf = lognorm.cdf(bin_edges, s=sigma, scale=np.exp(mu))
        probs = np.diff(cdf)
        counts = N * w * probs

        if not np.any(np.isfinite(counts)) or counts.max() <= 0:
            continue

        kmax = int(np.nanargmax(counts))

        # x au centre "log" du bin (mieux en échelle log)
        x_left, x_right = bin_edges[kmax], bin_edges[kmax + 1]
        x_peak = np.sqrt(x_left * x_right)  # moyenne géométrique
        y_peak = float(counts[kmax])

        # si plusieurs labels tombent sur le même bin, on empile légèrement
        n_here = used_bins.get(kmax, 0)
        used_bins[kmax] = n_here + 1
        y = y_peak * (y_mult * (stack_mult ** n_here))

        ax.text(
            x_peak,
            y,
            fr"$\pi_{j+1}={w:.2f}$",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            alpha=alpha,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.45)
        )

def plot_bic_distrib(events, info):
        fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))
        fig.subplots_adjust(top=0.88, bottom=0.20, wspace=0.45)
        plot_bic(info["scores"], info["J_max"], info["criterion"], ax=axes[0])
        plot_interarrival_distribution(events, ax=axes[1])
        annotate_gmm_weights(events, info["gmm"], ax=axes[1])
        plot_gmm_interarrival_counts(events, gmm=info["gmm"], ax=axes[1], plot_components=False, plot_total=True)

        plt.show()
        
def qq_plot(x):
    import scipy.stats as stats

    x = np.asarray(x)
    x = x[np.isfinite(x)]
    n = len(x)

    # Quantiles théoriques Exp(1)
    q_theo = stats.expon.ppf((np.arange(1, n + 1) - 0.5) / n)

    # Quantiles empiriques
    q_emp = np.sort(x)

    plt.figure(figsize=(3.2, 3.0))

    # Points empiriques
    plt.scatter(
        q_theo,
        q_emp,
        s=14,
        facecolors="none",
        edgecolors="black",
        linewidths=0.7,
        label="Transformed inter-arrival times"
    )

    # Droite de référence
    plt.plot(
        q_theo,
        q_theo,
        linestyle="-",
        color="black",
        linewidth=1.5,
        label="Reference line: Exp(1)"
    )

    plt.xlabel("Theoretical quantiles of Exp(1)")
    plt.ylabel("Empirical quantiles")

    plt.tight_layout()
    plt.show()

    
    

def result_table(path, value, log=True):
    # Charger le fichier JSON
    with open(path, "r") as f:
        data = json.load(f)
    records = []
    for key, v in data.items():
        date = v["date"]
        start, end = v["interval"]
        hour = f"{start}-{end}"
        if value == "tests":
            val = int(v["KS"] or v["CvM"] or v["AD"])
        elif value == "J":
            params = v.get("estimated_params", {})
            val = params["J"]
        if value == "branching_ratio":
            branching_ratios = v.get("branching_ratios", {})
            val = sum(branching_ratios)
        if value == "branching_ratios":
            branching_ratios = v.get("branching_ratios", [])
            val = np.array(branching_ratios).round(2)
        elif value == "beta0":
            params = v.get("estimated_params", {})
            if log:
                val = round(np.log(params["betas"][0]),2)
            else:
                val = round(params["betas"][0], 2)
        elif value == "beta1":
            params = v.get("estimated_params", {})
            if log:
                val = round(np.log(params["betas"][1]),2)
            else:
                val = round(params["betas"][1], 2)
        elif value == "beta2":
            params = v.get("estimated_params", {})
            if log:
                val = round(np.log(params["betas"][2]),2)
            else:
                val = round(params["betas"][2], 2)
        records.append({
            "date": date,
            "hour": hour,
            "values": val
        })
    df = pd.DataFrame(records)
    table = (
        df
        .pivot(index="hour", columns="date", values="values")
        .sort_index()
    )
    table.attrs["name"] = f"{value}"
    return table

def analyze_table(df):    
    if df.attrs["name"] == "beta0" or df.attrs["name"] == "beta1" or df.attrs["name"] == "beta2":
        fig, axes = plt.subplots(1, 1, figsize=(6.8, 2.0))
        print(df)
        median = df.stack().median()
        print("Median :", median, " | ", np.exp(df.stack().median()))
        # df.mean(axis=0).plot(ax=axes[0], label="mean")
        # df.median(axis=0).plot(ax=axes[0], label="median")
        # axes[0].set_title("Daily Average Log(Beta)")
        # axes[0].set_ylabel("Beta")
        # axes[0].set_xlabel("Date")
        # axes[0].legend()
        
        df_plot = df.T.copy()
        dates = pd.to_datetime(df_plot.index)

        # positions entières
        x = np.arange(len(dates))

        # plot avec positions
        axes.plot(x, df_plot.values, color="0.5", alpha=0.6, linewidth=1.0)
        axes.plot(x, df_plot.median(axis=1).values, color="black", linewidth=2.0)

        # labels EXACTEMENT ceux du df
        labels = dates.strftime("%m-%d")
        axes.set_xticks(x)
        axes.set_xticklabels(labels)

        axes.set_ylabel(r"$\log(\beta_0)$")
        axes.set_xlabel(None)

        plt.tight_layout()
        plt.show()



    
        # df.T.boxplot(ax=axes[2])
        # axes[2].set_title("Hourly Log(Beta) Dispersion")
        # axes[2].set_ylabel("Beta")
        

        # labels = [str(h).split(":")[0] for h in df.index]

        # axes[2].set_xticks(range(1, len(labels) + 1))
        # axes[2].set_xticklabels(labels, rotation=90)

    elif df.attrs["name"] == "tests":
        print(df)
        n_rejections = df.values.sum()
        print(f"Number of rejections: {n_rejections} "
            f"({round(n_rejections/df.size*100,1)} %)")

        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(["white", "0.05"])

        fig, ax = plt.subplots(figsize=(6.8, 2.6))  # largeur article (2 colonnes)

        ax.imshow(
            df.values,
            cmap=cmap,
            aspect="auto",
            interpolation="nearest",
            vmin=0,
            vmax=1
        )

        # --- Axes
        ax.set_xlabel("Day")
        ax.set_ylabel("Testing window")

        ax.set_xticks(np.arange(len(df.columns)))
        ax.set_xticklabels(df.columns, rotation=45, ha="right")

        # Nettoyage des labels horaires (09:00:00-10:00:00 -> 09–10)
        clean_windows = [
            f"{str(h).split(':')[0]}–{str(h).split('-')[1].split(':')[0]}"
            if isinstance(h, str) and '-' in h else str(h)
            for h in df.index
        ]
        ax.set_yticks(np.arange(len(df.index)))
        ax.set_yticklabels(clean_windows)

        # --- Séparateurs fins entre cellules (IMPORTANT)
        ax.set_xticks(np.arange(-0.5, df.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, df.shape[0], 1), minor=True)

        ax.grid(
            which="minor",
            color="0.8",
            linestyle="-",
            linewidth=0.6
        )

        ax.tick_params(which="minor", bottom=False, left=False)

        plt.tight_layout()
        plt.show()


    elif df.attrs["name"] == "J" or df.attrs["name"] == "branching_ratio" or df.attrs["name"] == "branching_ratios":
        print(df)
        
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_beta_stability(dfs, beta_labels=(0, 1, 2)):
    n = len(beta_labels)
    fig, axes = plt.subplots(
        n, 1,
        figsize=(6.8, 1.6 * n),
        sharex=True
    )

    if n == 1:
        axes = [axes]

    for ax, j in zip(axes, beta_labels):
        df = dfs[f"beta{j}"]

        # transpose: index = dates
        df_plot = df.T.copy()
        dates = pd.to_datetime(df_plot.index)
        x = np.arange(len(dates))

        # all intraday windows
        ax.plot(
            x,
            df_plot.values,
            color="0.5",
            alpha=0.6,
            linewidth=1.0
        )

        # median across windows
        ax.plot(
            x,
            df_plot.median(axis=1).values,
            color="black",
            linewidth=2.0
        )

        ax.set_ylabel(rf"$\log(\beta_{j})$")
        ax.set_xlim(x[0], x[-1])

    # x-axis labels (dates)
    labels = dates.strftime("%m-%d")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels)

    axes[-1].set_xlabel(None)

    plt.tight_layout()
    plt.show()


