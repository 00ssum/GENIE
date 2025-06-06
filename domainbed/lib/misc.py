# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Things that don't belong anywhere else
"""

import hashlib
import sys
import random
import os
import shutil
import errno
from itertools import chain
from datetime import datetime
from collections import Counter
from typing import List
from contextlib import contextmanager
from subprocess import call

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_weights_for_balanced_classes(dataset):
    counts = Counter()
    classes = []
    for _, y in dataset:
        y = int(y)
        counts[y] += 1
        classes.append(y)

    n_classes = len(counts)

    weight_per_class = {}
    for y in counts:
        weight_per_class[y] = 1 / (counts[y] * n_classes)

    weights = torch.zeros(len(dataset))
    for i, y in enumerate(classes):
        weights[i] = weight_per_class[int(y)]

    return weights


def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2 ** 31)


def to_row(row, colwidth=10, latex=False):
    """Convert value list to row string"""
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.6f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]

    return sep.join([format_val(x) for x in row]) + " " + end_


def random_pairs_of_minibatches(minibatches):
    # n_tr_envs = len(minibatches)
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []

    for i in range(len(minibatches)):
        # j = cyclic(i + 1)
        j = i + 1 if i < (len(minibatches) - 1) else 0

        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]

        min_n = min(len(xi), len(xj))

        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs


###########################################################
# Custom utils
###########################################################


def index_conditional_iterate(skip_condition, iterable, index):
    for i, x in enumerate(iterable):
        if skip_condition(i):
            continue

        if index:
            yield i, x
        else:
            yield x


class SplitIterator:
    def __init__(self, test_envs):
        self.test_envs = test_envs
        if self.test_envs == 'id':
            self.test_envs = []

    def train(self, iterable, index=False):
        return index_conditional_iterate(lambda idx: idx in self.test_envs, iterable, index)

    def test(self, iterable, index=False):
        return index_conditional_iterate(lambda idx: idx not in self.test_envs, iterable, index)


class AverageMeter():
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return "{:.3f} (val={:.3f}, count={})".format(self.avg, self.val, self.count)


class AverageMeters():
    def __init__(self, *keys):
        self.keys = keys
        for k in keys:
            setattr(self, k, AverageMeter())

    def resets(self):
        for k in self.keys:
            getattr(self, k).reset()

    def updates(self, dic, n=1):
        for k, v in dic.items():
            getattr(self, k).update(v, n)

    def __repr__(self):
        return "  ".join(["{}: {}".format(k, str(getattr(self, k))) for k in self.keys])

    def get_averages(self):
        dic = {k: getattr(self, k).avg for k in self.keys}
        return dic


def timestamp(fmt="%y%m%d_%H-%M-%S"):
    return datetime.now().strftime(fmt)


def makedirs(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def rm(path):
    """ remove dir recursively """
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    elif os.path.exists(path):
        os.remove(path)


def cp(src, dst):
    shutil.copy2(src, dst)


def set_seed(seed):
    random.seed(seed)
    #  os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #  torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_lr(optimizer):
    """Assume that the optimizer has single lr"""
    lr = optimizer.param_groups[0]['lr']

    return lr


def entropy(logits):
    ent = F.softmax(logits, -1) * F.log_softmax(logits, -1)
    ent = -ent.sum(1)  # batch-wise
    return ent.mean()


@torch.no_grad()
def hash_bn(module):
    summary = []
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            w = m.weight.detach().mean().item()
            b = m.bias.detach().mean().item()
            rm = m.running_mean.detach().mean().item()
            rv = m.running_var.detach().mean().item()
            summary.append((w, b, rm, rv))

    if not summary:
        return 0., 0.

    w, b, rm, rv = [np.mean(col) for col in zip(*summary)]
    p = np.mean([w, b])
    s = np.mean([rm, rv])

    return p, s


@torch.no_grad()
def hash_params(module):
    return torch.as_tensor([p.mean() for p in module.parameters()]).mean().item()


@torch.no_grad()
def hash_module(module):
    p = hash_params(module)
    _, s = hash_bn(module)

    return p, s


def merge_dictlist(dictlist):
    """Merge list of dicts into dict of lists, by grouping same key.
    """
    ret = {
        k: []
        for k in dictlist[0].keys()
    }
    for dic in dictlist:
        for data_key, v in dic.items():
            ret[data_key].append(v)
    return ret


def zip_strict(*iterables):
    """strict version of zip. The length of iterables should be same.

    NOTE yield looks non-reachable, but they are required.
    """
    # For trivial cases, use pure zip.
    if len(iterables) < 2:
        return zip(*iterables)

    # Tail for the first iterable
    first_stopped = False
    def first_tail():
        nonlocal first_stopped
        first_stopped = True
        return
        yield

    # Tail for the zip
    def zip_tail():
        if not first_stopped:
            raise ValueError('zip_equal: first iterable is longer')
        for _ in chain.from_iterable(rest):
            raise ValueError('zip_equal: first iterable is shorter')
            yield

    # Put the pieces together
    iterables = iter(iterables)
    first = chain(next(iterables), first_tail())
    rest = list(map(iter, iterables))
    return chain(zip(first, *rest), zip_tail())


def freeze_(module):
    for p in module.parameters():
        p.requires_grad_(False)
    module.eval()


def unfreeze_(module):
    for p in module.parameters():
        p.requires_grad_(True)
    module.train()
#Fishr
# class MovingAverage:
#     def __init__(self, ema, oneminusema_correction=True):
#         self.ema = ema
#         self.ema_data = {}
#         self._updates = 0
#         self._oneminusema_correction = oneminusema_correction

#     def update(self, dict_data):
#         ema_dict_data = {}
#         for name, data in dict_data.items():
#             data = data.view(1, -1)
#             if self._updates == 0:
#                 previous_data = torch.zeros_like(data)
#             else:
#                 previous_data = self.ema_data[name]

#             ema_data = self.ema * previous_data + (1 - self.ema) * data
#             if self._oneminusema_correction:
#                 # correction by 1/(1 - self.ema)
#                 # so that the gradients amplitude backpropagated in data is independent of self.ema
#                 ema_dict_data[name] = ema_data / (1 - self.ema)
#             else:
#                 ema_dict_data[name] = ema_data
#             self.ema_data[name] = ema_data.clone().detach()

#         self._updates += 1
#         return ema_dict_data
    
# def l2_between_dicts(dict_1, dict_2):
#     assert len(dict_1) == len(dict_2)
#     dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
#     dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
#     return (
#         torch.cat(tuple([t.view(-1) for t in dict_1_values])) -
#         torch.cat(tuple([t.view(-1) for t in dict_2_values]))
#     ).pow(2).mean()


# class MovingAverage:
#     def __init__(self, ema, oneminusema_correction=True):
#         """
#         Initialize a moving average tracker.
        
#         Args:
#             ema (float): Exponential moving average factor (0 < ema < 1).
#             oneminusema_correction (bool): Whether to apply correction by 1/(1 - ema).
#         """
#         self.ema = ema
#         self.ema_data = {}
#         self._updates = 0
#         self._oneminusema_correction = oneminusema_correction

#     def update(self, dict_data):
#         """
#         Update the moving average with new data.
        
#         Args:
#             dict_data (dict): Dictionary of new data tensors to update the EMA.

#         Returns:
#             dict: Updated EMA values.
#         """
#         if not dict_data:
#             raise ValueError("Input dict_data is empty. Cannot update MovingAverage.")

#         ema_dict_data = {}
#         for name, data in dict_data.items():
#             data = data.view(1, -1)
            
#             # Initialize if the key is missing
#             if name not in self.ema_data:
#                 previous_data = torch.zeros_like(data)
#             else:
#                 previous_data = self.ema_data[name]
            
#             # Compute EMA
#             ema_data = self.ema * previous_data + (1 - self.ema) * data

#             # Apply correction if necessary
#             if self._oneminusema_correction:
#                 ema_dict_data[name] = ema_data / (1 - self.ema ** (self._updates + 1))
#             else:
#                 ema_dict_data[name] = ema_data
            
#             # Update stored EMA data
#             self.ema_data[name] = ema_data.clone().detach()

#         self._updates += 1
#         return ema_dict_data

# def l2_between_dicts(dict_1, dict_2):
#     """
#     Compute L2 distance between tensors in two dictionaries.

#     Args:
#         dict_1 (dict): First dictionary of tensors.
#         dict_2 (dict): Second dictionary of tensors.

#     Returns:
#         Tensor: Scalar L2 distance between tensors in the two dictionaries.
#     """
#     if set(dict_1.keys()) != set(dict_2.keys()):
#         raise ValueError("Keys in dict_1 and dict_2 do not match.")

#     # Sort keys to ensure consistent order
#     keys = sorted(dict_1.keys())

#     # Compute L2 distance
#     l2_distance = 0.0
#     for key in keys:
#         diff = dict_1[key].view(-1) - dict_2[key].view(-1)
#         l2_distance += diff.pow(2).sum()
    
#     return l2_distance / sum(dict_1[key].numel() for key in keys)
