import numpy as np
import matplotlib.pyplot as plt
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

class PointProcess:
    def plot(self):
        self.lambda_values = self._intensity_on_grid(self.times, self.params, self.events)
        cumul = np.arange(1, len(self.events) + 1)
        fig, ax1 = plt.subplots(figsize=(6.8, 2.6))
        if len(self.events):
            (line1,) = ax1.step(self.events, cumul, where='post',
                                label="N(t)", color='black', lw=0.9)
        else:
            (line1,) = ax1.plot([], [], label="N(t)", color='k')
        ax1.set_xlabel(r"t")
        ax1.set_ylabel(r"Cumulative count $N(t)$")
        ax1.set_xlim(0, self.T)
        
        ax2 = ax1.twinx()
        (line2,) = ax2.plot(self.times, self.lambda_values,
                            label=r"$\lambda^*(t)$", alpha=0.6, lw=0.8, color='0.5')
        ax2.set_ylabel(r"Intensity  $\lambda^*(t)$")
        ax1.legend([line1, line2], ["N(t)", r"$\lambda^*(t)$"], loc="upper left", frameon=False)
        fig.tight_layout()
        plt.show()