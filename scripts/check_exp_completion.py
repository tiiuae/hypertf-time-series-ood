import glob
import os
import os.path as osp
import shutil

import pandas as pd

import ood_detection.utils.data.info as data_info


def extract_epoch_metrics(logpath: str, epoch: int, ood_methods: list[str]) -> dict[str, dict[str, float]] | None:
    result = {}

    # Open log file
    with open(logpath, encoding="utf-8") as f:
        lines = f.readlines()

    # Extract loader and dataset info (handles dashes in dataset names)
    dataset = loader = None
    for line in lines:
        if "Loading data..." in line and "Loader:" in line and "Dataset:" in line:
            parts = line.split("Loader:")[1].strip().split(", Dataset:")
            if len(parts) == 2:
                loader = parts[0].strip()
                dataset = parts[1].strip().split(",")[0]
            break

    if not loader or not dataset:
        return
    result["dataset_info"] = {"loader": loader, "dataset": dataset}

    # --- Find Val metrics for epoch ---
    val_metrics_path = osp.dirname(osp.dirname(logpath)) + "/metrics/val_metrics.csv"
    try:
        val_metrics_df = pd.read_csv(val_metrics_path, sep=",", usecols=["epoch"])
    except BaseException:
        return
    if epoch not in val_metrics_df["epoch"].values:
        return
    result["val_metrics"] = val_metrics_df

    # --- Find OOD metrics for epoch ---
    result["ood_metrics"] = {}
    for method in ood_methods:
        ood_metrics_path = osp.dirname(osp.dirname(logpath)) + f"/metrics/ood_metrics/{method}.csv"
        try:
            ood_metrics_df = pd.read_csv(ood_metrics_path, sep=",", usecols=["epoch", "ood_dataset"])
        except BaseException:
            return
        if epoch not in ood_metrics_df["epoch"].values:
            return
        if "All_Averaged" not in ood_metrics_df["ood_dataset"].values:
            return
        result["ood_metrics"][method] = ood_metrics_df
    return result


def verify_exp(
    exp_dir,
    epoch_to_chk: int,
    print_not_done: bool,
    delete_incomplete_exp: bool,
    repeat_per_exp_type: int = 1,
    ood_methods: list[str] = None,
    ucr_dsets: list[str] = None,
    uea_dsets: list[str] = None,
    datasets_to_skip: list[str] = None,
):
    for ds in datasets_to_skip:
        if ds in ucr_dsets:
            ucr_dsets.remove(ds)
        if ds in uea_dsets:
            uea_dsets.remove(ds)

    paths = sorted(glob.glob(exp_dir + "/**/train.txt", recursive=True))
    done_dsets = []
    for path in paths:
        complete = False
        result = extract_epoch_metrics(path, epoch_to_chk, ood_methods)

        if result:
            dset = path.split("/")[-4]
            assert result["dataset_info"]["dataset"] == dset

            ldr = result["dataset_info"]["loader"]
            epochs = result["val_metrics"]["epoch"].values

            if ldr == "UCR":
                assert dset in ucr_dsets, f"{dset} not in ucr_dsets: {ucr_dsets}"
                if epoch_to_chk in epochs:
                    complete = True
            elif ldr == "UEA":
                assert dset in uea_dsets, f"{dset} not in uea_dsets: {uea_dsets}"
                if epoch_to_chk in epochs:
                    complete = True
            else:
                raise ValueError("Unknown experiment directory")
            if complete:
                done_dsets.append(dset)

        if delete_incomplete_exp and not complete:
            # two levels up from train.txt
            exp_path = os.path.dirname(os.path.dirname(path))
            shutil.rmtree(exp_path)

    if "UCR" in exp_dir:
        all_dsets = ucr_dsets * repeat_per_exp_type
    elif "UEA" in exp_dir:
        all_dsets = uea_dsets * repeat_per_exp_type
    else:
        raise ValueError(f"Unknown experiment directory {exp_dir}")

    done = len(done_dsets)
    total = len(all_dsets)

    for dset in done_dsets:  # remove completed dsets from all_dsets
        all_dsets.remove(dset)
    not_done = all_dsets

    if done != total:
        print(f"\033[91m{exp_dir} is not verified. {done}/{total} experiments done\033[0m")
        if print_not_done:
            print(f"Not done: {not_done}")
    else:
        print(f"\033[92m{exp_dir} is verified. All {done}/{total} experiments done")


def main():
    ########################################################################
    print_not_done = 0
    delete_incomplete_exp = 0
    repeat_per_exp_type = {"UCR": 3, "UEA": 3}
    epoch_to_verify = {"UCR": 1, "UEA": 100}

    # WARNING: Either choose subsets or all the datasets
    ucr_dsets = list(data_info.UCR_DATASETS_TYPE.keys())
    uea_dsets = list(data_info.UEA_DATASETS_TYPE.keys())

    # Optional datasets to skip
    datasets_to_skip = []

    exps = [
        # UCR experiments
        "experiments/main_table/hyper_tf/UCR",
        "experiments/main_table/hypertf_aux_outlier_near_aux/UCR",
        "experiments/main_table/hypertf_aux_outlier_far_aux/UCR",
        "experiments/main_table/hypertf_mixup_oe_near_aux/UCR",
        "experiments/main_table/hypertf_mixup_oe_far_aux/UCR",
        # UEA experiments
        "experiments/main_table/hyper_tf/UEA",
        "experiments/main_table/hypertf_aux_outlier_near_aux/UEA",
        "experiments/main_table/hypertf_aux_outlier_far_aux/UEA",
        "experiments/main_table/hypertf_mixup_oe_near_aux/UEA",
        "experiments/main_table/hypertf_mixup_oe_far_aux/UEA",
    ]
    ood_methods = [
        # "maxlogitscore",
        "nearestneighborscore",
        "mahalanobisscore",
    ]
    ########################################################################

    for exp in exps:
        if "UCR" in exp:
            dset = "UCR"
        elif "UEA" in exp:
            dset = "UEA"
        else:
            raise ValueError(f"Unknown experiment type for {exp}. Please specify UCR, or UEA.")

        try:
            verify_exp(
                exp,
                epoch_to_verify[dset],
                print_not_done,
                delete_incomplete_exp,
                repeat_per_exp_type[dset],
                ood_methods,
                ucr_dsets,
                uea_dsets,
                datasets_to_skip,
            )
        except Exception as e:
            print(f"Error verifying {exp}: {e}")


if __name__ == "__main__":
    main()
