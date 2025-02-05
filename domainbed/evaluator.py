import open_clip
import clip
import os
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from domainbed.lib.fast_data_loader import FastDataLoader
from clip_retrieval.clip_client import ClipClient, Modality

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def accuracy_from_loader(algorithm, loader, weights, debug=False, dump_scores=False, dump_similarities=False, out_dir = None, inout=None, is_test=None):
    correct = 0
    total = 0
    losssum = 0.0
    weights_offset = 0

    algorithm.eval()

    dump_scores = dump_scores and is_test
    dump_similarities = dump_similarities and is_test

    if dump_scores:
        clip_scorer, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion400m_e32')
        clip_scorer = clip_scorer.cuda()
        tokenizer = open_clip.get_tokenizer('ViT-B-16')
        similarity_clip_histogram = []
    if dump_similarities:
        openai_clip, _ = clip.load("ViT-B/32", device=device)
        client = ClipClient(url="http://192.168.17.185:1234/knn-service", indice_name="laion", deduplicate=False,aesthetic_weight=0.0,use_safety_model=False,use_violence_detector=False)
        num_steps = 0
        similarity_img_histogram = []

    for i, batch in enumerate(loader):
        x = batch["x"].to(device)
        y = batch["y"].to(device)


        if dump_scores:
            with torch.no_grad():
                 classname = 'A photo of {}.'.format(loader._infinite_iterator._dataset.underlying_dataset.classes[y])
                 visual_feats = clip_scorer.encode_image(x)
                 txt = tokenizer(classname).cuda()
                 txt_feats = clip_scorer.encode_text(txt)
                 visual_feats = F.normalize(visual_feats, dim=-1)
                 txt_feats = F.normalize(txt_feats, dim=-1)
                 similarity = visual_feats @ txt_feats.T




        with torch.no_grad():
            logits = algorithm.predict(x)
            loss = F.cross_entropy(logits, y).item()

        B = len(x)
        losssum += loss * B

        if weights is None:
            batch_weights = torch.ones(len(x))
        else:
            batch_weights = weights[weights_offset : weights_offset + len(x)]
            weights_offset += len(x)
        batch_weights = batch_weights.to(device)
        if logits.size(1) == 1:
            is_correct = (logits.gt(0).eq(y).float() * batch_weights).sum().item()
            correct += is_correct
        else:
            is_correct = (logits.argmax(1).eq(y).float() * batch_weights).sum().item()
            correct += is_correct
        total += batch_weights.sum().item()
        if debug:
            break

        if dump_scores:
            similarity_clip_histogram.append((similarity.item(),is_correct))

        if dump_similarities:
                num_steps += 1
                if num_steps > 150:
                    break
                with torch.no_grad():
                     image_features = openai_clip.encode_image(x)
                     image_features = F.normalize(image_features, dim=-1)
                     query_results = client.query(embedding_input=image_features.cpu().numpy().tolist()[0])
                     try:
                        similarity = query_results[0]['similarity']
                        similarity_img_histogram.append((similarity,is_correct))
                     except:
                         pass


    algorithm.train()

    acc = correct / total
    loss = losssum / total
    if dump_scores:
        data_domain_path = loader._infinite_iterator._dataset.underlying_dataset.root.split('/')[-1]

        data_path = os.path.join(out_dir, 'similarity_np')
        path = os.path.join(data_path, data_domain_path + '_' + inout)
        os.makedirs(data_path, exist_ok=True)
        np.save(path, similarity_clip_histogram)

    if dump_similarities:
        data_domain_path = loader._infinite_iterator._dataset.underlying_dataset.root.split('/')[-1]

        data_path = os.path.join(out_dir, 'similarity_imgh_np')
        path = os.path.join(data_path, data_domain_path + '_' + inout)
        os.makedirs(data_path, exist_ok=True)
        np.save(path, similarity_img_histogram)


    return acc, loss


def accuracy(algorithm, loader_kwargs, weights, **kwargs):
    if isinstance(loader_kwargs, dict):
        loader = FastDataLoader(**loader_kwargs)
    elif isinstance(loader_kwargs, FastDataLoader):
        loader = loader_kwargs
    else:
        raise ValueError(loader_kwargs)
    return accuracy_from_loader(algorithm, loader, weights, **kwargs)


class Evaluator:
    def __init__(
        self, test_envs, eval_meta, n_envs, logger, evalmode="fast", debug=False, target_env=None
    , dump_scores=False, dump_similarities=False, out_dir = None):
        all_envs = list(range(n_envs))
        train_envs = sorted(set(all_envs) - set(test_envs))
        self.test_envs = test_envs
        if self.test_envs == 'id':
            self.test_envs = []
        self.train_envs = train_envs
        self.eval_meta = eval_meta
        self.n_envs = n_envs
        self.logger = logger
        self.evalmode = evalmode
        self.debug = debug
        self.dump_scores = dump_scores
        self.dump_similarities = dump_similarities
        self.out_dir = out_dir

        if target_env is not None:
            self.set_target_env(target_env)

    def set_target_env(self, target_env):
        """When len(test_envs) == 2, you can specify target env for computing exact test acc."""
        self.test_envs = [target_env]

    def evaluate(self, algorithm, ret_losses=False):
        n_train_envs = len(self.train_envs)
        n_test_envs = len(self.test_envs)
        #assert n_test_envs == 1
        summaries = collections.defaultdict(float)
        # for key order
        summaries["test_in"] = 0.0
        summaries["test_out"] = 0.0
        summaries["train_in"] = 0.0
        summaries["train_out"] = 0.0
        accuracies = {}
        losses = {}

        # order: in_splits + out_splits.
        for name, loader_kwargs, weights in self.eval_meta:
            # env\d_[in|out]
            env_name, inout = name.split("_")
            env_num = int(env_name[3:])
            is_test = env_num in self.test_envs

            skip_eval = self.evalmode == "fast" and inout == "in" and env_num not in self.test_envs
            if skip_eval:
                continue

            is_test = env_num in self.test_envs
            acc, loss = accuracy(algorithm, loader_kwargs, weights, debug=self.debug, dump_scores = self.dump_scores, dump_similarities = self.dump_similarities, out_dir = self.out_dir, inout=inout, is_test=is_test)
            accuracies[name] = acc
            losses[name] = loss

            if env_num in self.train_envs:
                summaries["train_" + inout] += acc / n_train_envs
                if inout == "out":
                    summaries["tr_" + inout + "loss"] += loss / n_train_envs
            elif is_test:
                summaries["test_" + inout] += acc / n_test_envs

        if ret_losses:
            return accuracies, summaries, losses
        else:
            return accuracies, summaries
