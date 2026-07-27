import os
import sys
import time

import pandas as pd

from scripts.extract_exp_stats import get_experiment_df

TRY_LOADING_FROM_CACHE = True

def main():
    # 1. Load trial-level data
    # if no args, use default
    if len(sys.argv) == 1:
        sys.argv.append("ablation")
    exp_dir = sys.argv[1]
    trials_out = f"experiments/{exp_dir}/all_trials.csv"
    # Available OOD methods
    # ood_method_file_names = ["embeddingcenterscore.csv", "energyscore.csv", "genscore.csv", "mahalanobisscore.csv",
    # "maxlogitscore.csv", "mspscore.csv", "nearestneighborscore.csv"]
    ood_method_file_names = ["nearestneighborscore.csv", "mahalanobisscore.csv"]
    ood_types = ["near", "far"]

    t0 = time.perf_counter()
    load_csv = True
    if TRY_LOADING_FROM_CACHE and os.path.exists(trials_out):
        df = pd.read_csv(trials_out)
        cached_configs = df["config"].unique().tolist()
        configs = [dname for dname in os.listdir(f"experiments/{exp_dir}") if os.path.isdir(os.path.join(f"experiments/{exp_dir}", dname))]
        missing_configs = [c for c in configs if c not in cached_configs]
        if not missing_configs:
            load_csv = False
            print(f"WARNING: Loading existing trials from {trials_out}")
        else:
            print(f"WARNING: Existing trials at {trials_out} missing configs: {missing_configs}. Recomputing all trials.")
    if load_csv:
        df = get_experiment_df(exp_dir, ood_method_file_names=ood_method_file_names)
        if len(df) != 0:
            df.to_csv(trials_out, index=False)

        print(f"Loaded {len(df)} trial rows in {time.perf_counter() - t0:.2f} seconds")
        print(f"Saved {len(df)} trial rows to {trials_out}")

    valid_ood_methods = []
    for fname in ood_method_file_names:
        m = fname.replace("score.csv", "")
        required_cols = [f"{m}_auroc_near", f"{m}_auroc_far", f"{m}_fpr95_near", f"{m}_fpr95_far",]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(
                f"[WARN] Skipping OOD method '{m}' – missing columns: {missing}",
                file=sys.stderr,
            )
        else:
            valid_ood_methods.append(m)

    if not valid_ood_methods:
        raise ValueError("No valid OOD methods found in dataframe for the requested list.")

    # 2. Aggregate within dataset/config
    base_metrics = ["f1_last", "f1_best", "f1_mean"]

    ood_metrics = []
    for m in valid_ood_methods:
        ood_metrics.extend([f"{m}_auroc_near", f"{m}_auroc_far", f"{m}_fpr95_near", f"{m}_fpr95_far",])

    metrics = base_metrics + ood_metrics
    agg_ds = df.groupby(["config", "dataset"])[metrics].agg(["mean", "std"]).reset_index()

    # Flatten column names
    agg_ds.columns = ["_".join(col).rstrip("_") for col in agg_ds.columns.values]
    agg_ds = agg_ds.round(2)

    # 3. Aggregate across datasets per config
    agg_cfg = agg_ds.groupby("config").mean(numeric_only=True).reset_index().round(2)

    # 4. Pretty-print LaTeX table
    print("\nLaTeX-style table rows:")

    # Display names for methods in headers
    method_display = {
        "nearestneighbor": "kNN",
        "mahalanobis": "Maha",
        "energy": "Energy",
        "maxlogit": "MaxLogit",
        "msp": "MSP",
        "gen": "GEN",
        "embeddingcenter": "EmbCenter",
    }

    # --- Print column headers ---
    headers = ["Variant"]
    for ood_type in ood_types:
        for m in valid_ood_methods:
            label = method_display.get(m, m)
            headers.append(f"{ood_type}-OOD {label} (AUROC/FPR@95)")

    headers.extend(["F1$_{\\text{best}}$", "F1$_{\\text{mean}}$", "F1$_{\\text{last}}$",])

    print(" & ".join(headers) + " \\\\")
    print("\\midrule")
    # --------------------------------

    for _, row in agg_cfg.iterrows():
        vals = []
        # Add config name first (escape underscores for LaTeX safety)
        config_name = row["config"].replace("_", "\\_")
        vals.append(config_name)

        for ood_type in ood_types:  # near and/ot far
            for m in valid_ood_methods:   # all methods
                vals.append(f"{row[f'{m}_auroc_{ood_type}_mean']:.2f} / {row[f'{m}_fpr95_{ood_type}_mean']:.2f}")

        # F1 metrics with std in \sd{}
        vals.append(f"{row['f1_best_mean']:.2f}\\sd{{{row['f1_best_std']:.2f}}}")
        vals.append(f"{row['f1_mean_mean']:.2f}\\sd{{{row['f1_mean_std']:.2f}}}")
        vals.append(f"{row['f1_last_mean']:.2f}\\sd{{{row['f1_last_std']:.2f}}}")

        print(" & ".join(vals) + " \\\\")
    t2 = time.perf_counter()
    print(f"\nCompleted in {t2 - t0:.2f} seconds")


if __name__ == "__main__":
    main()
