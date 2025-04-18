#python3 -m train_all resnet50_gsnr config/resnet50_sgd.yaml --algorithm GENIE --test_envs 0 --dataset PACS


import argparse
import collections
import random
import sys
from pathlib import Path

import numpy as np
import PIL
import torch
import torchvision
from sconf import Config
from prettytable import PrettyTable

from domainbed.datasets import get_dataset
from domainbed import hparams_registry
from domainbed.lib import misc
from domainbed.lib.writers import get_writer
from domainbed.lib.logger import Logger
from domainbed.trainer import train

def main():
    parser = argparse.ArgumentParser(description="Domain generalization", allow_abbrev=False)
    parser.add_argument("name", type=str)
    parser.add_argument("configs", nargs="*")
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--resume_path", type=str, default="checkpoints/")
    parser.add_argument("--dataset", type=str, default="PACS")
    parser.add_argument("--algorithm", type=str, default="ERM")
    parser.add_argument("--trial_seed", type=int, default=0, help="Trial number (used for seeding split_dataset and random_hparams).")
    parser.add_argument("--seed", type=int, default=0, help="Seed for everything else")
    parser.add_argument("--steps", type=int, default=None, help="Number of steps. Default is dataset-dependent.")
    parser.add_argument("--checkpoint_freq", type=int, default=None, help="Checkpoint every N steps. Default is dataset-dependent.")
    parser.add_argument("--test_envs", type=int, nargs="+", default=None)
    parser.add_argument("--holdout_fraction", type=float, default=0.2)
    parser.add_argument("--model_save", default=None, type=int, help="Model save start step")
    #  parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--tb_freq", default=10)
    parser.add_argument("--debug", action="store_true", help="Run w/ debug mode")
    parser.add_argument("--show", action="store_true", help="Show args and hparams w/o run")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate")
    parser.add_argument("--warmup", action="store_true", help="Warmup")
    parser.add_argument("--full_data", action="store_true", help="Full Data")
    parser.add_argument("--in_domain", action="store_true", help="In Domain")
    parser.add_argument("--dump_scores", action="store_true", help="Dump CLIP scores")
    parser.add_argument("--dump_similarities", action="store_true", help="Dump CLIP scores")
    parser.add_argument("--attn_tune", action="store_true", help="Attention tuning")
    parser.add_argument("--auto_lr", action="store_true", help="Auto LR")
    parser.add_argument("--mpa", action="store_true", help="MPA")
    parser.add_argument("--small_bs", action="store_true", help="MPA")
    parser.add_argument("--evalmode", default="all", help="[fast, all]. if fast, ignore train_in datasets in evaluation time.")
    parser.add_argument("--prebuild_loader", action="store_true", help="Pre-build eval loaders")

    parser.add_argument('--hparams_seed', type=int, default=0)


    args, left_argv = parser.parse_known_args()
    args.deterministic = True

    # setup hparams
    # if args.hparams_seed == 0:
    #     hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    
    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset, misc.seed_hash(args.hparams_seed, args.trial_seed))

    #keys = ["config.yaml"] + args.configs
    keys = args.configs
    keys = [open(key, encoding="utf8") for key in keys]
    hparams = Config(*keys, default=hparams)
    hparams.argv_update(left_argv)

    if args.evaluate:
        hparams['batch_size'] = 1

    if args.small_bs:
       hparams['batch_size'] = 16

    if args.warmup:
        hparams['linear_steps'] = 500

    if args.attn_tune:
        hparams['attn_tune'] = True
    else:
        hparams['attn_tune'] = False

    if args.auto_lr:
        hparams['auto_lr'] = True
    else:
        hparams['auto_lr'] = False


    # setup debug
    if args.debug:
        args.checkpoint_freq = 5
        args.steps = 10
        args.name += "_debug"

    timestamp = misc.timestamp()
    args.unique_name = f"{timestamp}_{args.name}"

    # path setup
    args.work_dir = Path(".")
    args.data_dir = Path(args.data_dir)

    args.out_root = args.work_dir / Path("train_output") / args.dataset / args.algorithm / str(args.test_envs)
    args.out_dir = args.out_root / args.unique_name
    args.out_dir.mkdir(exist_ok=True, parents=True)

    # writer = get_writer(args.out_root / "runs" / args.unique_name)
    writer = get_writer(args.out_dir)
    logger = Logger.get(args.out_dir / "log.txt")
    if args.debug:
        logger.setLevel("DEBUG")
    cmd = " ".join(sys.argv)
    logger.info(f"Command :: {cmd}")

    logger.nofmt("Environment:")
    logger.nofmt("\tPython: {}".format(sys.version.split(" ")[0]))
    logger.nofmt("\tPyTorch: {}".format(torch.__version__))
    logger.nofmt("\tTorchvision: {}".format(torchvision.__version__))
    logger.nofmt("\tCUDA: {}".format(torch.version.cuda))
    logger.nofmt("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    logger.nofmt("\tNumPy: {}".format(np.__version__))
    logger.nofmt("\tPIL: {}".format(PIL.__version__))

    # Different to DomainBed, we support CUDA only.
    assert torch.cuda.is_available(), "CUDA is not available"

    logger.nofmt("Args:")
    for k, v in sorted(vars(args).items()):
        logger.nofmt("\t{}: {}".format(k, v))

    logger.nofmt("HParams:")
    for line in hparams.dumps().split("\n"):
        logger.nofmt("\t" + line)

    if args.show:
        exit()

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.deterministic
    torch.backends.cudnn.benchmark = not args.deterministic

    # Dummy datasets for logging information.
    # Real dataset will be re-assigned in train function.
    # test_envs only decide transforms; simply set to zero.
    dataset, _in_splits, _out_splits = get_dataset([0], args, hparams)

    # print dataset information
    logger.nofmt("Dataset:")
    logger.nofmt(f"\t[{args.dataset}] #envs={len(dataset)}, #classes={dataset.num_classes}")
    for i, env_property in enumerate(dataset.environments):
        logger.nofmt(f"\tenv{i}: {env_property} (#{len(dataset[i])})")
    logger.nofmt("")

    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ
    logger.info(f"n_steps = {n_steps}")
    logger.info(f"checkpoint_freq = {checkpoint_freq}")

    org_n_steps = n_steps
    n_steps = (n_steps // checkpoint_freq) * checkpoint_freq + 1
    logger.info(f"n_steps is updated to {org_n_steps} => {n_steps} for checkpointing")

    if args.dump_scores:
        hparams.test_batchsize = 1

    if args.dump_similarities:
        hparams.test_batchsize = 1


    if not args.test_envs:
        args.test_envs = [[te] for te in range(len(dataset))]
    else:
        if isinstance(args.test_envs, list):
            args.test_envs = [x for x in args.test_envs]

        elif isinstance(args.test_envs, int):
            args.test_envs = [args.test_envs]

    if args.in_domain:
        args.test_envs = ['id']

    test_env = args.test_envs
    train(test_env,
        args=args,
        hparams=hparams,
        n_steps=n_steps,
        checkpoint_freq=checkpoint_freq,
        logger=logger,
        writer=writer,
    )

    # log summary table
    logger.info("=== Summary ===")
    logger.info(f"Command: {' '.join(sys.argv)}")
    logger.info("Unique name: %s" % args.unique_name)
    logger.info("Out path: %s" % args.out_dir)
    logger.info("Algorithm: %s" % args.algorithm)
    logger.info("Dataset: %s" % args.dataset)

if __name__ == "__main__":
    main()