# postprocess/ablation_table_per_OOD_group.py
import os
import sys
import time

import pandas as pd

from scripts.extract_exp_stats import get_experiment_df

TRY_LOADING_FROM_CACHE = True

# ---- Fixed groups (exactly as requested) ----
GROUP_ORDER = ["HAR", "AUDIO", "MOTION", "ECG", "SPECTRO", "OTHER"]
GROUP_SET = set(GROUP_ORDER)

# Display names for methods
METHOD_DISPLAY = {
    "nearestneighbor": "kNN",
    "mahalanobis": "Maha",
    "msp": "MSP",
    "energy": "Energy",
    "maxlogit": "MaxLogit",
    "gen": "GEN",
    "embeddingcenter": "EmbCenter",
}


def map_type_to_group(t: str) -> str:
    if not isinstance(t, str):
        return "OTHER"
    key = t.strip().upper()
    return key if key in GROUP_SET else "OTHER"


def avg(series: pd.Series) -> float:
    s = series.dropna()
    return float(s.mean()) if len(s) else float("nan")


def pair_cell(auroc: float, fpr: float) -> str:
    # AUROC↑ / FPR@95↓
    if pd.isna(auroc) or pd.isna(fpr):
        return r"-- / --"
    return f"{auroc:.2f} / {fpr:.2f}"


def build_rows_for_archive(df: pd.DataFrame, archive: str, method_prefix: str) -> list[str]:
    """
    Returns LaTeX rows:
      Method & Near & HAR & AUDIO & MOTION & ECG & SPECTRO & OTHER & F1_last \\

    method_prefix is e.g. 'nearestneighbor', 'mahalanobis', 'msp', ...
    and expects columns:
      <method_prefix>_auroc_near, <method_prefix>_auroc_far,
      <method_prefix>_fpr95_near, <method_prefix>_fpr95_far
    """
    near_auc = f"{method_prefix}_auroc_near"
    near_fpr = f"{method_prefix}_fpr95_near"
    far_auc = f"{method_prefix}_auroc_far"
    far_fpr = f"{method_prefix}_fpr95_far"

    missing = [
        col
        for col in [near_auc, near_fpr, far_auc, far_fpr, "f1_last", "type", "archive", "config"]
        if col not in df.columns
    ]
    if missing:
        raise ValueError(f"Required columns for method '{method_prefix}' are missing: {missing}")

    df_arch = df[df["archive"].str.upper() == archive.upper()].copy()
    # map dataset type to one of the fixed groups
    df_arch["group"] = df_arch["type"].apply(map_type_to_group)

    rows: list[str] = []
    for config, dcfg in df_arch.groupby("config", sort=False):
        # Near-OOD: mean across datasets
        near_cell = pair_cell(avg(dcfg[near_auc]), avg(dcfg[near_fpr]))

        # Far-OOD per fixed group
        far_cells = []
        for g in GROUP_ORDER:
            dgrp = dcfg[dcfg["group"] == g]
            far_cells.append(pair_cell(avg(dgrp[far_auc]), avg(dgrp[far_fpr])))

        f1_last = avg(dcfg["f1_last"])
        method = config.replace("_", r"\_")
        row = " & ".join([method, near_cell, *far_cells, f"{f1_last:.2f}"]) + r" \\"
        rows.append(row)

    return rows


def print_block(df: pd.DataFrame, archive: str, method_prefix: str):
    """
    Emits a LaTeX block for one archive with a header row that includes the group names.
    """
    arch_label = "UCR (Univariate)" if archive.upper() == "UCR" else "UEA (Multivariate)"
    header = (
        r"\midrule"
        + "\n"
        + rf"\multicolumn{{9}}{{c}}{{\textbf{{{arch_label}}}}} \\"
        + "\n"
        + r"\midrule"
        + "\n"
        +
        # Column header row (prints the column names)
        r"\textbf{Method} & \textbf{Near-OOD} & "
        + " & ".join(rf"\textbf{{{g}}}" for g in GROUP_ORDER)
        + r" & \textbf{F1$_\text{ID\_last}$} \\"
    )
    print(header)
    for row in build_rows_for_archive(df, archive, method_prefix):
        print(row)


def filter_valid_methods(df: pd.DataFrame, methods: list[str]) -> list[str]:
    valid = []
    for m in methods:
        required_cols = [
            f"{m}_auroc_near",
            f"{m}_auroc_far",
            f"{m}_fpr95_near",
            f"{m}_fpr95_far",
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(
                f"[WARN] Skipping OOD method '{m}' – missing columns: {missing}",
                file=sys.stderr,
            )
        else:
            valid.append(m)
    if not valid:
        raise ValueError("No valid OOD methods found in dataframe for the requested list.")
    return valid


def main():
    # Load input dir
    if len(sys.argv) == 1:
        sys.argv.append("ablation")
    exp_dir = sys.argv[1]
    trials_out = f"experiments/{exp_dir}/all_trials.csv"
    # Available OOD methods
    # ood_method_file_names = ["embeddingcenterscore.csv", "energyscore.csv", "genscore.csv", "mahalanobisscore.csv",
    # "maxlogitscore.csv", "mspscore.csv", "nearestneighborscore.csv"]
    ood_method_file_names = ["nearestneighborscore.csv", "mahalanobisscore.csv"]

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

    # Parse and validate which detectors to include
    requested_methods = [fname.replace("score.csv", "") for fname in ood_method_file_names]
    methods = filter_valid_methods(df, requested_methods)

    # Print LaTeX rows for each selected OOD method
    for m in methods:
        display = METHOD_DISPLAY.get(m, m)
        print(f"\n% ================== {display} ({m}_*) ==================")
        print_block(df, "UCR", m)
        print_block(df, "UEA", m)


if __name__ == "__main__":
    main()
