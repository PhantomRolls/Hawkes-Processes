from pointprocess.utils.io import result_table, analyze_table

method = "khmaladze"
duration = "1h"
H0 = "multiexp_fixed_betas"

path = f"results/multiexp_naive.json"

table = result_table(path, "beta0")

# analyze_table(table)



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_test_decisions_comparison(df_left, df_right,
                                   label_left="Naive",
                                   label_right="Transformation-based"):

    assert df_left.shape == df_right.shape
    assert (df_left.columns == df_right.columns).all()
    assert (df_left.index == df_right.index).all()

    cmap = ListedColormap(["white", "0.05"])

    fig, axes = plt.subplots(
        1, 2,
        figsize=(7.6, 3.0),
        sharey=True
    )

    for ax, df, title in zip(
        axes,
        [df_left, df_right],
        [label_left, label_right]
    ):
        ax.imshow(
            df.values,
            cmap=cmap,
            aspect="auto",
            interpolation="nearest",
            vmin=0,
            vmax=1
        )

        ax.set_title(title, fontsize=10, pad=6)

        # Light grid
        ax.set_xticks(np.arange(-0.5, df.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, df.shape[0], 1), minor=True)
        ax.grid(which="minor", color="0.75", linewidth=0.90)
        ax.tick_params(which="minor", bottom=False, left=False)

    short_dates = [d[5:] for d in df_left.columns] 
    # ---- X axis
    axes[0].set_xticks(np.arange(len(df_left.columns)))
    axes[0].set_xticklabels(short_dates, rotation=45, ha="right")
    axes[1].set_xticks(np.arange(len(df_left.columns)))
    axes[1].set_xticklabels(short_dates, rotation=45, ha="right")

    # ---- Y axis ONLY on the left
    def clean_hour_label(h):
        if isinstance(h, str) and "-" in h:
            start, end = h.split("-")
            start_h = start.split(":")[0]
            end_h = end.split(":")[0]
            return f"{start_h}:00–{end_h}:00"
        return str(h)


    clean_windows = [clean_hour_label(h) for h in df_left.index]

    axes[0].set_yticks(np.arange(len(df_left.index)))
    axes[0].set_yticklabels(clean_windows)

    # ---- REMOVE Y ticks on the right panel
    axes[1].tick_params(axis="y", left=False, labelleft=False)

    plt.tight_layout()
    plt.show()



def plot_test_decisions_grid(
    df_pairs,
    h0_labels,
    method_labels=("Naive", "Khmaladze"),
):
    n_rows = len(df_pairs)
    n_cols = 2

    # --- checks
    assert len(h0_labels) == n_rows
    for df_left, df_right in df_pairs:
        assert df_left.shape == df_right.shape
        assert (df_left.columns == df_right.columns).all()
        assert (df_left.index == df_right.index).all()

    cmap = ListedColormap(["white", "0.05"])

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(7.6, 1.6 * n_rows),
        sharex=True,
        sharey=False,   # 🔴 IMPORTANT : axes Y indépendants
    )

    if n_rows == 1:
        axes = np.array([axes])  # ensure 2D

    # ---------- helpers ----------
    def fmt_hhmm(x):
        from datetime import time
        import pandas as pd
        if isinstance(x, (pd.Timestamp, time)):
            return x.strftime("%H:%M")
        s = str(x)
        return s[:5]

    def clean_window_label(h):
        if isinstance(h, str) and "-" in h:
            start, end = h.split("-", 1)
            return f"{fmt_hhmm(start)}–{fmt_hhmm(end)}"
        return fmt_hhmm(h)

    def subsample_ticks(labels, step):
        ticks = np.arange(len(labels))
        return ticks[::step], [labels[i] for i in ticks[::step]]

    # ---------- main loop ----------
    for i, ((df_left, df_right), h0_label) in enumerate(zip(df_pairs, h0_labels)):
        for j, (df, title) in enumerate(zip((df_left, df_right), method_labels)):
            ax = axes[i, j]

            ax.imshow(
                df.values,
                cmap=cmap,
                aspect="auto",
                interpolation="nearest",
                vmin=0,
                vmax=1
            )

            # --- column titles (top row only)
            if i == 0:
                ax.set_title(title, fontsize=10, pad=6)

            # --- row label (left column only)
            if j == 0:
                ax.set_ylabel(h0_label, fontsize=9)

            # --- grid
            ax.set_xticks(np.arange(-0.5, df.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, df.shape[0], 1), minor=True)
            ax.grid(which="minor", color="0.75", linewidth=0.9)
            ax.tick_params(which="minor", bottom=False, left=False)

            # --- Y ticks: propres à chaque H0
            if j == 0:
                windows = [clean_window_label(h) for h in df.index]
                n_w = len(windows)

                # --- subsample Y labels if too many windows
                if n_w >= 20:
                    step = 3
                elif n_w >= 10:
                    step = 2 if n_w >= 10 else 1
                else:
                    step = 1

                yticks, ylabels = subsample_ticks(windows, step)

                ax.set_yticks(yticks)
                ax.set_yticklabels(ylabels, fontsize=8)

            else:
                ax.tick_params(axis="y", left=False, labelleft=False)

            # --- X ticks only on bottom row
            if i < n_rows - 1:
                ax.tick_params(axis="x", bottom=False, labelbottom=False)

    # ---- X labels (bottom row only)
    short_dates = [d[5:] for d in df_pairs[0][0].columns]
    for ax in axes[-1, :]:
        ax.set_xticks(np.arange(len(short_dates)))
        ax.set_xticklabels(short_dates, rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    plt.show()



df_naive_exp = result_table(f"results/exp_naive.json", "tests")
df_khm_exp = result_table(f"results/exp_khmaladze.json", "tests")

df_naive_pl = result_table(f"results/pl_naive.json", "tests")
df_khm_pl = result_table(f"results/pl_khmaladze.json", "tests")



df_naive_multiexp = result_table(f"results/multiexp_naive.json", "tests")
df_khm_multiexp = result_table(f"results/multiexp_khmaladze.json", "tests")

df_naive_30min = result_table(f"results/multiexp_naive_30min.json", "tests")
df_kmh_30min = result_table(f"results/multiexp_khmaladze_30min.json", "tests")

df_naive_2h = result_table(f"results/multiexp_naive_2h.json", "tests")
df_kmh_2h = result_table(f"results/multiexp_khmaladze_2h.json", "tests")

df_naive_4h = result_table(f"results/multiexp_naive_4h.json", "tests")
df_kmh_4h = result_table(f"results/multiexp_khmaladze_4h.json", "tests")

df_naive_15min = result_table(f"results/multiexp_naive_15min.json", "tests")
df_kmh_15min = result_table(f"results/multiexp_khmaladze_15min.json", "tests")

df_pairs = [(df_naive_multiexp, df_khm_multiexp),(df_naive_exp, df_khm_exp),(df_naive_pl,  df_khm_pl),]

h0_labels = [r"$H_0^{MEH}$",r"$H_0^{Exp}$",r"$H_0^{PL}$",]


df_pairs2 = [(df_naive_15min, df_kmh_15min),(df_naive_30min, df_kmh_30min),(df_naive_multiexp, df_khm_multiexp),(df_naive_2h, df_kmh_2h),(df_naive_4h, df_kmh_4h)]

h0_labels2 = ["15 min", "30 min","1 h","2h", "4h"]


plot_test_decisions_grid(df_pairs2,h0_labels2, method_labels=("Naive", "Khmaladze"))

analyze_table(df_kmh_2h)