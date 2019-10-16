import copy
import itertools
import os
import random
import time
from collections import defaultdict, namedtuple

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from tabulate import tabulate
from tqdm import tqdm

from cogdl import options
from cogdl.tasks import build_task


def main(args):
    if torch.cuda.is_available() and not args.cpu:
        pid = mp.current_process().pid
        torch.cuda.set_device(args.pid_to_cuda[pid])

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    task = build_task(args)
    result = task.train()
    return result


def gen_variants(**items):
    Variant = namedtuple("Variant", items.keys())
    return itertools.starmap(Variant, itertools.product(*items.values()))


def getpid(_):
    # HACK to get different pids
    time.sleep(1)
    return mp.current_process().pid


if __name__ == "__main__":
    # Magic for making multiprocessing work for PyTorch
    mp.set_start_method("spawn")

    parser = options.get_training_parser()
    args, _ = parser.parse_known_args()
    args = options.parse_args_and_arch(parser, args)
    print(args)
    variants = list(
        gen_variants(dataset=args.dataset, model=args.model, seed=args.seed)
    )
    results_dict = defaultdict(list)

    device_ids = args.device_id
    if args.cpu:
        num_workers = len(args.dataset) * len(args.model * len(args.seed))
    else:
        num_workers = len(device_ids)
    print("num_workers", num_workers)

    with mp.Pool(processes=num_workers) as pool:
        # Map process to cuda device
        pids = pool.map(getpid, range(num_workers))
        pid_to_cuda = dict(zip(pids, device_ids))
        # yield all variants
        def variant_args_generator():
            """Form variants as group with size of num_workers"""
            for variant in variants:
                args.pid_to_cuda = pid_to_cuda
                args.dataset, args.model, args.seed = variant
                yield copy.deepcopy(args)

        # Collect results
        results = pool.map(main, variant_args_generator())
        for variant, result in zip(variants, [results[0][1]]):
            results_dict[variant[:-1]].append(result)

    # Average for different seeds

    tab_data = []
    df_rank=results[0][0]
    #accuracy=results[0][1]
    print(tabulate(df_rank, headers=['user_id']+df_rank.columns.values.tolist(), tablefmt="github"))
    
    
    # Average for different seeds
    col_names = ["Variant"] + list(results_dict[variant[:-1]][-1].keys())
    tab_data = []
    for variant in results_dict:
        Results = np.array([list(res.values()) for res in results_dict[variant]])
        tab_data.append(
            [variant]
            + list(
                itertools.starmap(
                    lambda x, y: f"{x:.4f}±{y:.4f}",
                    zip(
                        np.mean(Results, axis=0).tolist(),
                        np.std(Results, axis=0).tolist(),
                    ),
                )
            )
        )
    print(tabulate(tab_data, headers=col_names, tablefmt="github"))