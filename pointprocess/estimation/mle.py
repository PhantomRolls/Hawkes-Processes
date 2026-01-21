from scipy.optimize import minimize
from pointprocess.estimation.likelihoods.exp import hawkes_exp_loglik
from pointprocess.estimation.likelihoods.pl import hawkes_pl_loglik
from pointprocess.estimation.likelihoods.multiexp import hawkes_multiexp_loglik
import numpy as np
from sklearn.mixture import GaussianMixture
from pointprocess.utils.io import plot_bic_distrib

def estimate_betas_gmm(
    event_times,
    n_components=None,    
    J_max=6,
    min_intervals=200,
    random_state=42,
    sort_desc=True,
    criterion="bic",          # "aic" or "bic"
):

    t = np.asarray(event_times, dtype=float)
    dt = np.diff(t)
    dt = dt[dt > 0]

    if len(dt) < max(min_intervals, J_max * 20):
        raise ValueError("Pas assez d'inter-temps positifs pour un GMM robuste.")

    log_dt = np.log(dt).reshape(-1, 1)

    if n_components is None:
        scores = []
        models = []

        for k in range(1, J_max + 1):
            gmm = GaussianMixture(
                n_components=k,
                random_state=random_state,
                covariance_type="full",
            )
            gmm.fit(log_dt)

            score = gmm.bic(log_dt) if criterion == "bic" else gmm.aic(log_dt)
            scores.append(score)
            models.append(gmm)

        def choose_k_elbow(bics, min_gain=50):
            for i in range(J_max):
                gain = bics[i] - bics[i+1]
                if gain < min_gain:
                    return i
            return J_max

        best_idx = choose_k_elbow(scores)
        gmm = models[best_idx]
        n_components = gmm.n_components
    else:
        gmm = GaussianMixture(
            n_components=n_components,
            random_state=random_state,
            covariance_type="full",
        )
        gmm.fit(log_dt)
        scores = None
        

    log_means = gmm.means_.reshape(-1)
    taus = np.exp(log_means)
    betas = 1.0 / taus

    weights = gmm.weights_.reshape(-1)
    covs = gmm.covariances_.reshape(n_components, -1)

    if sort_desc:
        order = np.argsort(betas)[::-1]
        betas = betas[order]
        taus = taus[order]
        weights = weights[order]
        log_means = log_means[order]
        covs = covs[order]

    info = {
        "n_components": n_components,
        "taus": taus,
        "weights": weights,
        "log_means": log_means,
        "covariances": covs,
        "n_intervals": len(dt),
        "criterion": criterion,
        "gmm": gmm,
        "scores": scores,
        "J_max": J_max
    }
    
        
    return betas, info

def fit_hawkes(events, T, H0, x0=None, plot=False, J=None):
    events = np.asarray(events, float)
    n = events.size

    dt = np.diff(events) if n > 1 else np.zeros(1)
    tail = T - events

    if H0 == "exp":
        if x0 is None:
            x0 = np.array([0.5, 0.8, 1.0])
        
        bounds = [(1e-8,None), (0,None), (1e-8,None)]
        
        def obj(p):
            return -hawkes_exp_loglik(p, events, T, dt, tail)

    # ---- POWER-LAW ----
    elif H0 == "pl":
        L = T / 10
        
        if x0 is None:
            x0 = np.array([0.5, 0.5, 1.5])
        
        bounds = [(1e-8,None), (0,None), (1e-8,None)]
        
        def obj(p):
            return -hawkes_pl_loglik(p[0], p[1], p[2], events, T, L, tail)

    # ---- MULTI-EXP ----
    elif H0 == "multiexp":
        J = 3
        if x0 is None:
            mu0 = 0.5
            alpha0 = np.full(J, 0.5/J)
            beta0  = np.linspace(0.5, 2.0, J)
            x0 = np.concatenate(([mu0], alpha0, beta0))
        
        bounds = [(1e-8,None)] + [(0,None)]*J + [(1e-8,None)]*J
        
        alpha_idx = slice(1, 1+J)
        beta_idx  = slice(1+J, 1+2*J)
        
        def obj(p):
            mu = p[0]
            alphas = p[alpha_idx]
            betas  = p[beta_idx]
            return -hawkes_multiexp_loglik(mu, alphas, betas, events, T, dt, tail)
    
    elif H0 == "multiexp_fixed_betas":
        betas_fixed, info = estimate_betas_gmm(events, n_components=3)              # Important : choose if you fix the betas or estimate them
        # betas_fixed = np.array([216459,  766,  0.24])

        if plot:
            plot_bic_distrib(events, info)

        if x0 is None:
            mu0 = 0.5
            alpha0 = np.full(J, 0.5/J)
            x0 = np.concatenate(([mu0], alpha0))
        
        bounds = [(1e-8,None)] + [(0,None)]*J
        
        alpha_idx = slice(1, 1+J)
        betas_fixed = betas_fixed
        
        def obj(p):
            mu = p[0]
            alphas = p[alpha_idx]
            return -hawkes_multiexp_loglik(mu, alphas, betas_fixed, events, T, dt, tail)

    else:
        raise ValueError("Unknown H0")

    res = minimize(
        obj,
        x0=x0,
        bounds=bounds,
        method="L-BFGS-B",
        options={"maxiter": 200, "ftol": 1e-6}
    )
    
    p = res.x

    if H0 == "exp":
        params = {
            "mu": p[0],
            "alpha": p[1],
            "beta": p[2],
            "T": T
        }

    elif H0 == "pl":
        params = {
            "mu": p[0],
            "alpha": p[1],
            "beta": p[2],
            "T": T
        }

    elif H0 == "multiexp":
        mu = p[0]
        J = (len(p) - 1) // 2
        alphas = p[1:1+J]
        betas  = p[1+J:1+2*J]
        params = {
            "mu": mu,
            "alphas": alphas,
            "betas": betas,
            "T": T,
            "J": J
        }
        
    elif H0 == "multiexp_fixed_betas":
        mu = p[0]
        J = len(p) - 1
        alphas = p[1:1+J]

        params = {
            "mu": mu,
            "alphas": alphas,
            "betas": betas_fixed,
            "T": T,
            "J": J
        }

    # attach dict to result object for convenience
    res.params_dict = params

    return res

