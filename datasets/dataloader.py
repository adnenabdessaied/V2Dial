"""
From https://github.com/klauscc/VindLU/blob/main/dataset/dataloader.py
"""

import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torch.distributed as dist
from utils.dist import *
import random
import logging

logger = logging.getLogger(__name__)


class MetaLoader(object):
    """ wraps multiple data loader """
    def __init__(self, name2loader):
        """Iterates over multiple dataloaders, it ensures all processes
        work on data from the same dataloader. This loader will end when
        the shorter dataloader raises StopIteration exception.

        loaders: Dict, {name: dataloader}
        """
        self.name2loader = name2loader
        self.name2iter = {name: iter(l) for name, l in name2loader.items()}
        name2index = {name: idx for idx, (name, l) in enumerate(name2loader.items())}
        index2name = {v: k for k, v in name2index.items()}

        iter_order = []
        for n, l in name2loader.items():
            iter_order.extend([name2index[n]]*len(l))

        random.shuffle(iter_order)
        iter_order = torch.Tensor(iter_order).to(torch.device("cuda")).to(torch.uint8)

        # sync
        if is_dist_avail_and_initialized():
            # make sure all processes have the same order so that
            # each step they will have data from the same loader
            dist.broadcast(iter_order, src=0)
        self.iter_order = [index2name[int(e.item())] for e in iter_order.cpu()]

        logger.info(str(self))

    def __str__(self):
        output = [f"MetaLoader has {len(self.name2loader)} dataloaders, {len(self)} batches in total"]
        for idx, (name, loader) in enumerate(self.name2loader.items()):
            output.append(
                f"dataloader index={idx} name={name}, batch-size={loader.batch_size} length(#batches)={len(loader)} "
            )
        return "\n".join(output)

    def __len__(self):
        return len(self.iter_order)

    def __iter__(self):
        """ this iterator will run indefinitely """
        for name in self.iter_order:
            _iter = self.name2iter[name]
            batch = next(_iter)
            yield name, batch


def load_dataloaders(config, datasets, split, output_dict=False):
    if isinstance(datasets, dict):
        datasets = list(datasets.values())
    shuffles = [True] * len(datasets) if split == 'train' else [False] * len(datasets)
    if config['distributed'] and split != 'test':
        num_tasks = get_world_size()
        global_rank = get_rank()
        samplers = create_samplers(
            datasets, shuffles, num_tasks, global_rank
        )
    else:
        samplers = [None] * len(datasets)

    batch_size = [dataset.datasets[0].batch_size if isinstance(dataset, ConcatDataset) else dataset.batch_size for dataset in datasets]
    collate_fns = []
    for dataset in datasets:
        if isinstance(dataset, ConcatDataset):
            collate_fns.append(getattr(dataset.datasets[0], 'collate_fn', None))
        else:
            collate_fns.append(getattr(dataset, 'collate_fn', None))

    loaders = create_loader(
        datasets,
        samplers,
        batch_size=batch_size,
        num_workers=[config.num_workers] * len(datasets),
        is_trains=shuffles,
        collate_fns=collate_fns,
    )  # [0]
    loaders_dict = {}
    if output_dict:
        for l in loaders:
            if isinstance(l.dataset, ConcatDataset):
                loaders_dict[l.dataset.datasets[0].medium] = l
            else:
                loaders_dict[l.dataset.medium] = l
        return loaders_dict    
    return loaders


def create_samplers(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle
        )
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(
        datasets, samplers, batch_size, num_workers, is_trains, collate_fns
    ):
        if is_train:
            shuffle = sampler is None
            drop_last = True
        else:
            shuffle = False
            drop_last = True
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=False,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
            persistent_workers=True if n_worker > 0 else False,
        )
        loaders.append(loader)
    return loaders