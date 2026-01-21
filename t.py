from pointprocess.utils.io import result_table, analyze_table, plot_beta_stability



df_naive_multiexp = result_table(f"results/multiexp_naive.json", "beta0")
df_khm_multiexp = result_table(f"results/multiexp_khmaladze.json", "beta0")

df_naive_30min = result_table(f"results/multiexp_naive_30min.json", "tests")
df_kmh_30min = result_table(f"results/multiexp_khmaladze_30min.json", "tests")

df_naive_2h = result_table(f"results/multiexp_naive_2h.json", "tests")
df_kmh_2h = result_table(f"results/multiexp_khmaladze_2h.json", "tests")

df_naive_4h = result_table(f"results/multiexp_naive_4h.json", "tests")
df_kmh_4h = result_table(f"results/multiexp_khmaladze_4h.json", "tests")

df_naive_15min = result_table(f"results/multiexp_naive_15min.json", "tests")
df_kmh_15min = result_table(f"results/multiexp_khmaladze_15min.json", "tests")


beta0 = result_table(f"results/betas_not_fixed.json", "beta0")
beta1 = result_table(f"results/betas_not_fixed.json", "beta1")
beta2 = result_table(f"results/betas_not_fixed.json", "beta2")

plot_beta_stability(dfs={"beta0": beta0, "beta1": beta1, "beta2": beta2})
