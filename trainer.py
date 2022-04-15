"""Train the model"""

import logging
import wandb
import torch

import utils

import numpy as np

from pathlib2 import Path
from tqdm import tqdm


class trainer:

    def __init__(self, model, optimizer_R, loss_fn_R, loss_fn_D,
                 train_dataloader, val_dataloader, test_dataloader, metrics,
                 params, save_dir):
        self.model = model
        self.optimizer = optimizer_R
        self.loss_fn_R = loss_fn_R
        self.loss_fn_D = loss_fn_D
        self.dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.metrics = metrics
        self.params = params
        self.save_dir = save_dir

    def train_one_epoch(self):
        self.model.train()

        loss_D_avg = utils.RunningAverage()
        loss_R_avg = utils.RunningAverage()

        summary = []

        with tqdm(total=len(self.dataloader)) as t:
            for i, (data, target) in enumerate(self.dataloader):
                if self.params.cuda:
                    data, target = data.to('cuda', non_blocking=True),
                    target.to('cuda', non_blocking=True)
                output = self.model(data)

                # update the parameter of the regressor
                self.optimizer_R.zero_grad()
                loss_R = self.loss_fn_R(output, target)
                loss_R.backward()
                self.optimizer_R.step()

                # update the parameter of distractor
                self.optimizer_D.zero_grad()
                loss_D = self.loss_fn_D(output, target)
                loss_D.backward()
                self.optimizer_D.step()

                if i % self.params.save_summary_steps == 0:
                    # extract data from torch Variable, move to cpu, convert to numpy arrays
                    output = output.data.cpu().numpy()
                    target = target.data.cpu().numpy()

                    # compute all metrics on this batch
                    summary_batch = {
                        metric: self.metrics[metric](output, target)
                        for metric in self.metrics
                    }
                    summary_batch['loss_D'] = loss_D.item()
                    summary_batch['loss_R'] = loss_R.item()
                    if len(self.dataloader
                           ) - i < self.params.save_summary_steps:
                        wandb.log({'train': summary_batch}, commit=False)
                    else:
                        wandb.log({'train': summary_batch})
                    summary.append(summary_batch)

                # update ternimal information
                loss_D_avg.update(loss_D.item())
                loss_R_avg.update(loss_R.item())

                t.set_postfix(loss_D=loss_D_avg(), loss_R=loss_R_avg())
                t.update()

        metrics_mean = {
            metric: np.mean([x[metric] for x in summary_batch])
            for metric in summary[0].keys()
        }

        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                    for k, v in metrics_mean.items())
        logging.info("***** Train metrics: " + metrics_string)

    def validate(self):
        self.model.eval()
        summary = []
        with torch.no_grad():
            for data, target in self.val_dataloader:
                if self.params.cuda:
                    data, target = data.to(
                        'cuda', non_blocking=True), target.to('cuda',
                                                              non_blocking=True)
                output = self.model.regressor(data)
                loss_R = self.loss_fn_R(output, target)
                output = output.data.cpu().numpy()
                target = target.data.cpu().numpy()
                summary_batch = {
                    metric: self.metrics[metric](output, target)
                    for metric in self.metrics.keys()
                }
                summary_batch['loss_R'] = loss_R.item()
                summary.append(summary_batch)

        metrics_mean = {
            metric: np.mean([x[metric] for x in summary_batch])
            for metric in self.metrics.keys()
        }
        wandb.log({'val': metrics_mean})
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                    for k, v in metrics_mean.items())
        logging.info("***** Val metrics: " + metrics_string)
        return metrics_mean

    def test(self):
        self.model.eval()
        summary = []
        with torch.no_grad():
            for data, target in self.test_dataloader:
                if self.params.cuda:
                    data, target = data.to(
                        'cuda', non_blocking=True), target.to('cuda',
                                                              non_blocking=True)
                output = self.model.regressor(data)
                loss_R = self.loss_fn_R(output, target)
                output = output.data.cpu().numpy()
                target = target.data.cpu().numpy()
                summary_batch = {
                    metric: self.metrics[metric](output, target)
                    for metric in self.metrics.keys()
                }
                summary_batch['loss_R'] = loss_R.item()
                summary.append(summary_batch)

        metrics_mean = {
            metric: np.mean([x[metric] for x in summary_batch])
            for metric in self.metrics.keys()
        }
        wandb.log({'test': metrics_mean})
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                    for k, v in metrics_mean.items())
        logging.info("***** Test metrics: " + metrics_string)
        return metrics_mean

    def train(self, restore_path=None):
        if restore_path is not None:
            pass
        best_val_metric = 0.0
        for epoch in range(self.params.epochs):
            self.train_one_epoch()
            metrics = self.validate()
            val_metric = metrics[self.params.eval_metric_name]
            is_best = val_metric > best_val_metric
            if is_best:
                best_val_metric = val_metric
                logging.info("- Found new best accuracy")

                # Save best val metrics in a json file in the model directory
                best_json_path = self.save_dir / "metrics_val_best_weights.json"
                utils.save_dict_to_json(metrics, best_json_path)
            utils.save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optim_dict': self.optimizer.state_dict()
                },
                is_best=is_best,
                checkpoint=self.save_dir)
