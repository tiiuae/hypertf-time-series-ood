import argparse
import os

import pandas as pd

# Map dataset name -> type
from ood_detection.utils.data.info import DATASETS_TYPE


def extract_trial_info(
    trial_path: str,
    experiment: str,
    config: str,
    f1_col: str = "f1_prot",
    ood_metrics: list = ("auroc", "fpr95"),
    ood_method_file_names: list[str] = None,
):
    """
    Extract trial information from a single experiment trial folder.

    Parameters:
        trial_path (str): The path to the trial folder.
        experiment (str): The name of the experiment.
        config (str): The name of the configuration.
        f1_col (str): The column name in the val_metrics.csv file to use for F1 calculation.
            If not provided, the function will search for any column with "f1" in its name.
        ood_metrics (list): The list of OOD metrics to extract.
        ood_method_file_names (list[str]): The list of OOD method file names to extract from. (e.g., ["nearestneighborscore.csv", "mahalanobisscore.csv"])

    Returns:
        dict: A dictionary containing the extracted information.
    """
    parts = trial_path.split(os.sep)
    archive = parts[-3]
    dataset = parts[-2]
    trial_id = parts[-1]
    dataset_type = DATASETS_TYPE.get(dataset, "UNKNOWN")

    row = {
        "experiment": experiment,
        "config": config,
        "archive": archive,
        "dataset": dataset,
        "type": dataset_type,
        "trial": trial_id.replace("trial_", ""),
        "f1_last": None,
        "f1_best": None,
        "f1_mean": None,
    }

    # ---------- F1 ----------
    val_file = os.path.join(trial_path, "metrics", "val_metrics.csv")
    if os.path.exists(val_file):
        df = pd.read_csv(val_file)
        if not df.empty:
            # If the requested f1_col is not there, search for any column with "f1" in its name
            candidate_col = None
            if f1_col in df.columns:
                candidate_col = f1_col
            else:
                f1_candidates = [c for c in df.columns if "f1" in c.lower()]
                if f1_candidates:
                    candidate_col = f1_candidates[0]  # take the first match
                    print(f"⚠️  Using fallback F1 column '{candidate_col}' in {trial_path}")
                else:
                    print(f"❌ No F1 column found in {trial_path}")

            if candidate_col is not None:
                f1_vals = df[candidate_col].astype(float).tolist()
                row["f1_last"] = f1_vals[-1]
                row["f1_best"] = max(f1_vals)
                row["f1_mean"] = sum(f1_vals) / len(f1_vals)

    # ---------- OOD ----------
    def read_ood_by_group(ood_file: str, metric: str):
        """Return dict with near/far/all values for given metric."""
        if not os.path.exists(ood_file):
            print(f"❌ OOD file not found: {ood_file}")
            return {"near": None, "far": None, "all": None}
        df = pd.read_csv(ood_file)
        if "ood_dataset" not in df.columns or metric not in df.columns:
            return {"near": None, "far": None, "all": None}

        # Drop All_Averaged and All_Combined rows (case-insensitive)
        df = df[~df["ood_dataset"].str.lower().isin(["all_averaged", "all_combined"])]

        values_near, values_far = [], []
        for _, r in df.iterrows():
            ds = str(r["ood_dataset"])
            val = float(r[metric])
            ds_type = DATASETS_TYPE.get(ds, "UNKNOWN")

            if ds_type == dataset_type:
                values_near.append(val)
            else:
                values_far.append(val)

        def safe_mean(vals):
            return float(sum(vals) / len(vals)) if vals else None

        return {
            "near": safe_mean(values_near),
            "far": safe_mean(values_far),
            "all": safe_mean(values_near + values_far),
        }

    ood_method_file_paths = [os.path.join(trial_path, "metrics", "ood_metrics", fn) for fn in ood_method_file_names]

    for m in ood_metrics:
        for ood_method_file in ood_method_file_paths:
            ood_vals = read_ood_by_group(ood_method_file, m)
            for g in ["near", "far", "all"]:
                ood_method = os.path.basename(ood_method_file).replace("score.csv", "")
                row[f"{ood_method}_{m}_{g}"] = ood_vals[g]

    return row


def collect_trial_rows(experiment_root: str, experiment_name: str, ood_method_file_names: list[str]):
    rows = []
    for config in sorted(os.listdir(experiment_root)):
        config_path = os.path.join(experiment_root, config)
        if not os.path.isdir(config_path):
            continue

        for archive in sorted(os.listdir(config_path)):
            archive_path = os.path.join(config_path, archive)
            if not os.path.isdir(archive_path):
                continue

            for dataset in sorted(os.listdir(archive_path)):
                dataset_path = os.path.join(archive_path, dataset)
                if not os.path.isdir(dataset_path):
                    continue

                for trial in sorted(os.listdir(dataset_path)):
                    trial_path = os.path.join(dataset_path, trial)
                    if not os.path.isdir(trial_path) or not trial.startswith("trial_"):
                        continue
                    row = extract_trial_info(trial_path, experiment_name, config, ood_method_file_names=ood_method_file_names)
                    rows.append(row)
    return rows


def get_experiment_df(experiment_name: str, ood_method_file_names: list[str] = None) -> pd.DataFrame:
    """Return DataFrame with all trial rows for a given experiment name."""
    if ood_method_file_names is None:
        ood_method_file_names = [
            "embeddingcenterscore.csv",
            "energyscore.csv",
            "genscore.csv",
            "mahalanobisscore.csv",
            "maxlogitscore.csv",
            "mspscore.csv",
            "nearestneighborscore.csv",
        ]
        print(f"⚠️  No OOD method file names provided. Using default set. {ood_method_file_names}")
    experiment_root = os.path.join("experiments", experiment_name)
    if not os.path.isdir(experiment_root):
        raise ValueError(f"Experiment not found: {experiment_root}")
    rows = collect_trial_rows(experiment_root, experiment_name, ood_method_file_names=ood_method_file_names)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment_name", type=str, help="Name of experiment under experiments/ (e.g., ablation, pooling)"
    )
    args = parser.parse_args()

    experiment_root = os.path.join("experiments", args.experiment_name)

    if not os.path.isdir(experiment_root):
        raise ValueError(f"Experiment not found: {experiment_root}")

    rows = collect_trial_rows(experiment_root, args.experiment_name)
    df = pd.DataFrame(rows)

    out_csv = os.path.join(experiment_root, "all_trials.csv")
    df.to_csv(out_csv, index=False)

    print(f"✅ Saved {len(df)} trial rows to {out_csv}")
    print(df.head())
