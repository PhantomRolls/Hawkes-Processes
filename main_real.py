import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pointprocess.simulation.hawkes_exp import HawkesExp
from pointprocess.simulation.hawkes_pl import HawkesPL
from pointprocess.simulation.hawkes_multiexp import HawkesMultiExp
from pointprocess.montecarlo.monte_carlo_real import monte_carlo_real
from pointprocess.testing.one_run import one_run
from pointprocess.utils.io import load_real_data, qq_plot, plot_two_counting_processes, plot_counting_process
from pointprocess.utils.io import load_config

dates = ["2017-01-17", "2017-01-18", "2017-01-19", "2017-01-20", "2017-01-23", "2017-01-24", "2017-01-25", "2017-01-26", "2017-01-27", "2017-01-30", "2017-01-31", "2017-02-01"]  
intervals_1h = [("10:00:00", "11:00:00"), ("11:00:00", "12:00:00"), ("12:00:00", "13:00:00"), ("13:00:00", "14:00:00"), ("14:00:00", "15:00:00"), ("15:00:00", "16:00:00")]
intervals_30min = [
    ("10:00:00", "10:30:00"),
    ("10:30:00", "11:00:00"),
    ("11:00:00", "11:30:00"),
    ("11:30:00", "12:00:00"),
    ("12:00:00", "12:30:00"),
    ("12:30:00", "13:00:00"),
    ("13:00:00", "13:30:00"),
    ("13:30:00", "14:00:00"),
    ("14:00:00", "14:30:00"),
    ("14:30:00", "15:00:00"),
    ("15:00:00", "15:30:00"),
    ("15:30:00", "16:00:00"),
]

intervals_15min = [
    ("10:00:00", "10:15:00"),
    ("10:15:00", "10:30:00"),
    ("10:30:00", "10:45:00"),
    ("10:45:00", "11:00:00"),
    ("11:00:00", "11:15:00"),
    ("11:15:00", "11:30:00"),
    ("11:30:00", "11:45:00"),
    ("11:45:00", "12:00:00"),
    ("12:00:00", "12:15:00"),
    ("12:15:00", "12:30:00"),
    ("12:30:00", "12:45:00"),
    ("12:45:00", "13:00:00"),
    ("13:00:00", "13:15:00"),
    ("13:15:00", "13:30:00"),
    ("13:30:00", "13:45:00"),
    ("13:45:00", "14:00:00"),
    ("14:00:00", "14:15:00"),
    ("14:15:00", "14:30:00"),
    ("14:30:00", "14:45:00"),
    ("14:45:00", "15:00:00"),
    ("15:00:00", "15:15:00"),
    ("15:15:00", "15:30:00"),
    ("15:30:00", "15:45:00"),
    ("15:45:00", "16:00:00"),
]

intervals_2h = [("10:00:00", "12:00:00"), ("11:00:00", "13:00:00"), ("12:00:00", "14:00:00"), ("13:00:00", "15:00:00"), ("14:00:00", "16:00:00"), ("15:00:00", "17:00:00")]

intervals_4h = [("10:00:00", "14:00:00"), ("11:00:00", "15:00:00"), ("12:00:00", "16:00:00")]
# monte_carlo_real(dates, intervals_1h, H0="multiexp_fixed_betas", method="naive_rtc", J=3)
    
 


# date = "2017-01-20"
# start = "10:00:00"
# end = "11:00:00"

# start_ = date + " " + start
# end_ = date + " " + end
# events_real, T = load_real_data(start=start_, end=end_, path=f"data/{date}.csv")



# ks_reject, ad_reject, cvm_reject, estimated_params, x = one_run(
#                 events=events_real,
#                 T=T,
#                 H0="multiexp_fixed_betas",
#                 method="naive_rtc",
#                 alpha_level=0.05, plot=True, J=3)

# print(ks_reject, ad_reject, cvm_reject)

# print(estimated_params)
# # events_sim = HawkesPL(estimated_params).events

# # plot_two_counting_processes(events_sim, events_real)
# # plot_counting_process(events_real)
# # qq_plot(x)


config = load_config("config.yaml")
process_params = config[HawkesMultiExp.__name__]
events_sim = HawkesMultiExp(process_params).events
T = process_params["T"]

ks_reject, ad_reject, cvm_reject, estimated_params, x_sim = one_run(
                events=events_sim,
                T=T,
                H0="pl",
                method="naive_rtc",
                alpha_level=0.05, plot=False, J=3)


print(ks_reject, ad_reject, cvm_reject)

def qq_plot_compare(x_sim, x_real):
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as stats

    def clean_sort_and_qexp(x):
        x = np.asarray(x)
        x = x[np.isfinite(x)]
        x = np.sort(x)
        n = len(x)
        q_theo = stats.expon.ppf((np.arange(1, n + 1) - 0.5) / n)
        return q_theo, x

    # Quantiles théoriques + empiriques pour chaque échantillon
    q_theo_sim, q_emp_sim = clean_sort_and_qexp(x_sim)
    q_theo_real, q_emp_real = clean_sort_and_qexp(x_real)

    # Limites communes pour comparer proprement
    x_max = max(q_theo_sim.max(), q_theo_real.max())
    y_max = max(q_emp_sim.max(), q_emp_real.max())

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))
    
    fig.subplots_adjust(top=0.88, bottom=0.15, wspace=0.35)

    # ----- Simulated -----
    axes[0].scatter(
        q_theo_sim,
        q_emp_sim,
        s=14,
        facecolors="none",
        edgecolors="black",
        linewidths=0.7,
    )
    axes[0].plot(
        q_theo_sim,
        q_theo_sim,
        linestyle="-",
        color="black",
        linewidth=1.5,
    )
    axes[0].set_title("Simulated")
    axes[0].set_xlabel("Theoretical quantiles of Exp(1)")
    axes[0].set_ylabel("Empirical quantiles")
    axes[0].set_xlim(0, x_max)
    axes[0].set_ylim(0, y_max)

    # ----- Real data -----
    axes[1].scatter(
        q_theo_real,
        q_emp_real,
        s=14,
        facecolors="none",
        edgecolors="black",
        linewidths=0.7,
    )
    axes[1].plot(
        q_theo_real,
        q_theo_real,
        linestyle="-",
        color="black",
        linewidth=1.5,
    )
    axes[1].set_title("Real data")
    axes[1].set_xlabel("Theoretical quantiles of Exp(1)")
    axes[1].set_xlim(0, x_max)
    axes[1].set_ylim(0, y_max)

    plt.show()

qq_plot_compare(x_sim, x_sim)

