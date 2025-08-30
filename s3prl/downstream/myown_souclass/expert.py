import os
import math
import torch
import random
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence

from .model import Model
from .dataset import SndClassifiDataset


class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(self, upstream_dim, upstream_rate, downstream_expert, expdir, **kwargs):
        """
        Args:
            upstream_dim: int
                Different upstream will give different representation dimension
                You might want to first project them to the same dimension

            upstream_rate: int
                160: for upstream with 10 ms per frame
                320: for upstream with 20 ms per frame

            downstream_expert: dict
                The 'downstream_expert' field specified in your downstream config file
                eg. downstream/example/config.yaml

            expdir: string
                The expdir from command-line argument, you should save all results into
                this directory, like some logging files.

            **kwargs: dict
                All the arguments specified by the argparser in run_downstream.py
                and all the other fields in config.yaml, in case you need it.

                Note1. Feel free to add new argument for __init__ as long as it is
                a command-line argument or a config field. You can check the constructor
                code in downstream/runner.py
        """

        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']

        self.train_dataset = SndClassifiDataset('train', **self.datarc)
        self.dev_dataset = SndClassifiDataset('test', **self.datarc)

        self.model = Model(input_dim=self.upstream_dim,
                           output_class_num=len(self.train_dataset.label_list), **self.modelrc)
        self.objective = nn.CrossEntropyLoss()

        self.logging = os.path.join(expdir, 'log.log')
        self.best = defaultdict(lambda: 0)

    # Interface
    def get_dataloader(self, split, epoch: int = 0):
        """
        Args:
            split: string
                'train'
                    will always be called before the training loop

                'dev', 'test', or more
                    defined by the 'eval_dataloaders' field in your downstream config
                    these will be called before the evaluation loops during the training loop

        Return:
            a torch.utils.data.DataLoader returning each batch in the format of:

            [wav1, wav2, ...], your_other_contents1, your_other_contents2, ...

            where wav1, wav2 ... are in variable length
            each wav is torch.FloatTensor in cpu with:
                1. dim() == 1
                2. sample_rate == 16000
                3. directly loaded by torchaudio
        """

        if split == 'train':
            return self._get_train_dataloader(self.train_dataset, epoch)
        elif split == 'dev':
            return self._get_eval_dataloader(self.dev_dataset)

    def _get_train_dataloader(self, dataset, epoch: int):
        from s3prl.utility.data import get_ddp_sampler
        sampler = get_ddp_sampler(dataset, epoch)
        return DataLoader(
            dataset, batch_size=self.datarc['train_batch_size'],
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.datarc['num_workers'],
            collate_fn=dataset.collate_fn
        )

    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset, batch_size=self.datarc['eval_batch_size'],
            shuffle=False, num_workers=self.datarc['num_workers'],
            collate_fn=dataset.collate_fn
        )

    def _tile_representations(self, reps, factor):
        """ 
        Tile up the representations by `factor`.
        Input - sequence of representations, shape: (batch_size, seq_len, feature_dim)
        Output - sequence of tiled representations, shape: (batch_size, seq_len * factor, feature_dim)
        """
        assert len(reps.shape) == 3, 'Input argument `reps` has invalid shape: {}'.format(reps.shape)
        tiled_reps = reps.repeat(1, 1, factor)
        tiled_reps = tiled_reps.reshape(reps.size(0), reps.size(1)*factor, reps.size(2))
        return tiled_reps

    # Interface
    def forward(self, split, features, labels, records, **kwargs):
        device = features[0].device
        
        labels = torch.LongTensor(labels).to(device=device)

        features_len = torch.IntTensor([len(feat) for feat in features])#.to(device=device)
        mask = torch.arange(features_len.max()).expand(features_len.size(0), features_len.max()) < features_len.unsqueeze(1) 
        att_mask = mask.float().to(device=device)  # [B, T, 1], float for attention use

        features = pad_sequence(features, batch_first=True)
        predicted = self.model(features, att_mask)

        # cause logits are in (batch, seq, class) and labels are in (batch, seq)
        # nn.CrossEntropyLoss expect to have (N, class) and (N,) as input
        # here we flatten logits and labels in order to apply nn.CrossEntropyLoss
        class_num = predicted.size(-1)
        loss = self.objective(predicted.reshape(-1, class_num), labels.reshape(-1))

        predicted_classid = predicted.max(dim=-1).indices
        records['loss'].append(loss.item())
        records['acc'] += (predicted_classid ==
                           labels).view(-1).cpu().float().tolist()

        return loss

    # interface

    def log_records(self, split, records, logger, global_step, **kwargs):
        """
        Args:
            split: string
                'train':
                    records and batchids contain contents for `log_step` batches
                    `log_step` is defined in your downstream config
                    eg. downstream/example/config.yaml

                'dev', 'test' or more:
                    records and batchids contain contents for the entire evaluation dataset

            records:
                defaultdict(list), contents already prepared by self.forward

            logger:
                Tensorboard SummaryWriter
                please use f'{your_task_name}/{split}-{key}' as key name to log your contents,
                preventing conflict with the logging of other tasks

            global_step:
                The global_step when training, which is helpful for Tensorboard logging

            batch_ids:
                The batches contained in records when enumerating over the dataloader

            total_batch_num:
                The total amount of batches in the dataloader

        Return:
            a list of string
                Each string is a filename we wish to use to save the current model
                according to the evaluation result, like the best.ckpt on the dev set
                You can return nothing or an empty list when no need to save the checkpoint
        """
        prefix = f'libri_phone/{split}-'
        average = torch.FloatTensor(records['acc']).mean().item()

        logger.add_scalar(
            f'{prefix}acc',
            average,
            global_step=global_step
        )
        message = f'{prefix}|step:{global_step}|acc:{average}\n'
        save_ckpt = []
        if average > self.best[prefix]:
            self.best[prefix] = average
            message = f'best|{message}'
            name = prefix.split('/')[-1].split('-')[0]
            save_ckpt.append(f'best-states-{name}.ckpt')
        with open(self.logging, 'a') as f:
            f.write(message)
        
        return save_ckpt
