import argparse
import time

import ood_detection.utils.data.info as data_info
from scripts.exp_scripts.exp_utils import gpu_worker, launch_workers

start = time.perf_counter()
dataset_to_cfg = {}

parser = argparse.ArgumentParser(description="Run main table experiments with hypertuning for SupCon loss.")
parser.add_argument("--ngpus", type=int, default=1, help="Number of GPUs to use for parallel execution. (default: %(default)s)")
parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3], help="List of random seeds to run for each experiment. (default: %(default)s)")
parser.add_argument("--print_cmds_only", action="store_true", help="If set, the generated commands will be printed instead of executed.")
args = parser.parse_args()


for ds in list(data_info.UCR_DATASETS_TYPE.keys()):
    dataset_to_cfg[ds] = "configs/auxiliary_oe/ucr_aux_contrastive_id_oe_mixup.yaml"
for ds in list(data_info.UEA_DATASETS_TYPE.keys()):
    dataset_to_cfg[ds] = "configs/auxiliary_oe/uea_aux_contrastive_id_oe_mixup.yaml"

# Use nested exp_name structure
# Final experiments will be tuples of (exp_name, *params_to_chk, seed)
# (exp_name, same_outlier_exposure_sub_type, oe_sampling_feat_strategy, feat_strategy_args)
EXPERIMENTS = [
    ("main_table/hypertf_mixup_oe_near_aux", True, "id_oe_mixup", {"use_hybrid_feature_seed": False}),
    ("main_table/hypertf_mixup_oe_far_aux", False, "id_oe_mixup", {"use_hybrid_feature_seed": False}),
]


def build_cmd(
    gpu_id, exp_name, same_outlier_exposure_sub_type, oe_sampling_feat_strategy, feat_strategy_args, seed, dataset
):
    """
    The gpu_id, exp_name, ..., seed, dataset parameters must always be passed
    """
    cfg_path = dataset_to_cfg[dataset]
    cmd = [
        "python",
        "train.py",
        "--cfg",
        cfg_path,
        "-n",
        exp_name,
        "--dataset",
        dataset,
        "-o",
        f"gpu={gpu_id}",
        "model.args.global_pooling=max_avg",
        f"seed={seed}",
        "trainer.args.epochs=1",  # TODO - change back to 100
        "trainer.args.checkpoint_save_freq=100",
        "trainer.args.save_best_loss_checkpoint=False",
        "trainer.args.plot_metrics=False",
        "ood_eval.eval_freq=100",
        "ood_eval.data.type=far",  # should always be far
        "ood_eval.data.far_ood_subtype=all",
        "dataset.args.crop_scale=[0.75,0.99]",
        f"dataset.args.same_outlier_exposure_sub_type={same_outlier_exposure_sub_type}",
        f"dataset.args.oe_sampling.feature_strategy.type={oe_sampling_feat_strategy}",
        f"dataset.args.oe_sampling.feature_strategy.args={feat_strategy_args}",
        "model.args.input_projection.enabled=True",
        "model.args.input_projection.dim=64",
        "model.args.input_projection.training_mask_probability=0.0",
        "model.args.input_projection.norm_type=layer",
        "loss.args.losses[0].args.temperature=0.0125",
        "loss.args.losses[3].args.temperature=0.0125",
        "loss.args.losses[3].args.base_temperature=0.025",
        "loss.args.losses[3].args.feature_dim=128",
        "loss.args.losses[3].args.detach_id=True",
        "loss.args.losses[3].args.queue_capacity=0",
        "loss.args.lambdas=[1.0,0.0,0.0,1.0,0.0]",
    ]

    return cmd


# ---------- No changes should be made below ----------


# launch gpu workers
launch_workers(
    build_cmd=build_cmd,
    gpu_worker=gpu_worker,
    run_exps=not args.print_cmds_only,
    dataset_to_cfg=dataset_to_cfg,
    seeds_to_run=args.seeds,
    experiments=EXPERIMENTS,
    datasets_to_skip=None,
    top_log_dir="logs",
    ngpus=args.ngpus,
)

print("✅ All jobs completed. Logs saved in nested logs/{exp_name}/ directories.")
print(f"Total time taken: {time.perf_counter() - start:.2f} seconds")
