from collections.abc import Callable
from functools import partial
from itertools import product
import os
import subprocess
from threading import Lock, Thread
import time

import psutil

# Set critical environment variables for consistent performance
os.environ["OMP_NUM_THREADS"] = "1"  # Prevent CPU oversubscription
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


def set_cpu_affinity(gpu_id, ngpus):
    """Set CPU affinity"""
    try:
        cpu_count = psutil.cpu_count()
        cores_per_gpu = cpu_count // ngpus
        start_core = gpu_id * cores_per_gpu
        end_core = start_core + cores_per_gpu
        p = psutil.Process()
        p.cpu_affinity(list(range(start_core, end_core)))
    except Exception as excep:
        print(
            f"{excep}. Warning: Failed to set CPU affinity for GPU {gpu_id}. This may lead to suboptimal performance.",
            flush=True,
        )
        pass


def get_next_job(queue_lock, job_queue):
    with queue_lock:
        return job_queue.pop(0) if job_queue else None


def build_log_path(exp_args, dataset, gpu_id, top_log_dir):
    """
    Construct a log filename from experiment arguments, dataset, gpu_id, and
    top-level log directory.

    Args:
        exp_args (list): List of experiment arguments, where the first element is
            the experiment name and the last element is the seed.
        dataset (str): Dataset name.
        gpu_id (int): GPU ID.
        top_log_dir (str): Top-level log directory.

    Returns:
        str: Log filename.
    """
    exp_name = exp_args[0]
    seed = exp_args[-1]
    other_args = exp_args[1:-1]

    log_dir = os.path.join(top_log_dir, exp_name)
    os.makedirs(log_dir, exist_ok=True)

    # Format intermediate args as arg1_val1, arg2_val2, ...
    other_parts = [f"arg{i + 1}_{v}" for i, v in enumerate(other_args)]
    log_filename = f"{dataset}_" + "_".join(other_parts) + f"_seed_{seed}_gpu_{gpu_id}.log"

    return os.path.join(log_dir, log_filename)


def gpu_worker(gpu_id: int, build_cmd: Callable, run_exps: bool, job_queue: list, top_log_dir: str, ngpus: int):
    """
    Worker function for each GPU. Given a queue of jobs, executes each job by running a subprocess for each experiment.

    Args:
        gpu_id: The ID of the current GPU.
        build_cmd: A function taking in the GPU ID, experiment arguments, and dataset name, and returning a list of strings representing the command to run.
        run_exps: Whether to actually run the experiments or just print the commands.
        job_queue: A list of jobs, where each job is a tuple of (experiment arguments, dataset name).
        top_log_dir: The top directory for all logs.
        ngpus: The total number of GPUs.

    Returns:
        None
    """
    set_cpu_affinity(gpu_id, ngpus)
    queue_lock = Lock()

    while True:
        job = get_next_job(queue_lock, job_queue)
        if job is None:
            break

        exp_args, dataset = job
        exp_name = exp_args[0]
        log_path = build_log_path(exp_args, dataset, gpu_id, top_log_dir)

        print(f"[GPU {gpu_id}] Starting {exp_name} on {dataset} -> {log_path}", flush=True)

        cmd = build_cmd(gpu_id, *exp_args, dataset)
        print(" ".join(cmd))
        if run_exps:
            with open(log_path, "w", encoding="utf-8") as log_file:
                log_file.write(f"[GPU {gpu_id}] Starting {exp_name} on {dataset} -> {log_path}\n")
                process = subprocess.Popen(
                    cmd, stdout=log_file, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True
                )
                process.wait()


def launch_workers(
    build_cmd: Callable,
    gpu_worker: Callable,
    run_exps: bool,
    dataset_to_cfg: dict[str, str],
    seeds_to_run: list[int],
    experiments: list[tuple],
    datasets_to_skip: list[str],
    top_log_dir: str,
    ngpus: int,
):
    """
    Launches multiple threads to run experiments on different GPUs.

    Args:
        build_cmd: A function that takes in gpu_id, exp_name, *params_to_chk, seed, dataset and returns a command to run the experiment.
        gpu_worker: The worker thread function.
        run_exps: Whether to run the experiments or just print the commands.
        dataset_to_cfg: A dictionary mapping dataset name to its configuration.
        seeds_to_run: A list of seeds to run the experiments with.
        experiments: A list of experiment tuples in the format (exp_name, *params_to_chk).
        datasets_to_skip: A list of dataset names to skip.
        top_log_dir: The directory to save the logs.
        ngpus: The number of GPUs to use.
    """
    os.makedirs(top_log_dir, exist_ok=True)

    # remove skipped datasets
    for ds in datasets_to_skip if datasets_to_skip is not None else []:
        if ds in dataset_to_cfg:
            dataset_to_cfg.pop(ds)
    datasets = list(dataset_to_cfg.keys())

    # repeat with different seeds
    experiments_with_seed = []
    for _seed in seeds_to_run:
        for exp in experiments:
            experiments_with_seed.append((*exp, _seed))

    # job_queue is in format (exp_name, *params_to_chk, seed, dataset)
    job_queue = list(product(experiments_with_seed, datasets))

    threads = []
    for gpu in range(ngpus):
        gpu_worker = partial(
            gpu_worker,
            build_cmd=build_cmd,
            run_exps=run_exps,
            job_queue=job_queue,
            top_log_dir=top_log_dir,
            ngpus=ngpus,
        )
        t = Thread(target=gpu_worker, args=(gpu,))
        time.sleep(0.1)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
