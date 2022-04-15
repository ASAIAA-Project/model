"""Train the model"""

import logging

from pathlib2 import Path
import wandb

import utils
import numpy as np
from tqdm import tqdm


class trainer:

    def __init__(self, model, regressor_optimizer, loss_fn_regressor,
                 loss_fn_distractor, train_dataloader, val_dataloader,
                 test_dataloader, metrics, params, save_dir):
        self.model = model
        self.optimizer = regressor_optimizer
        self.loss_fn_regressor = loss_fn_regressor
        self.loss_fn_distractor = loss_fn_distractor
        self.dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.metrics = metrics
        self.params = params
        self.save_dir = save_dir
        self.train_summary = []
        self.val_summary = []
        self.test_summary = []

    def train_one_epoch(self):
        self.model.train()

        loss_avg = utils.RunningAverage()

        with tqdm(total=len(self.dataloader)) as t:
            for i, (data, target) in enumerate(self.dataloader):
                if self.params.cuda:
                    data, target = data.to(
                        'cuda', non_blocking=True), target.to('cuda',
                                                              non_blocking=True)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_fn(output, target)
                loss.backward()

                self.optimizer.step()

                if i % self.params.save_summary_steps == 0:
                    # extract data from torch Variable, move to cpu, convert to numpy arrays
                    output = output.data.cpu().numpy()
                    target = target.data.cpu().numpy()

                    # compute all metrics on this batch
                    summary_batch = {
                        metric: self.metrics[metric](output, target)
                        for metric in self.metrics
                    }
                    summary_batch['loss'] = loss.item()
                    if len(self.dataloader
                           ) - i < self.params.save_summary_steps:
                        wandb.log({'train': summary_batch}, commit=False)
                    else:
                        wandb.log({'train': summary_batch})
                    self.train_summary.append(summary_batch)

                loss_avg.update(loss.item())

                # update ternimal information
                t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
                t.update()

        metrics_mean = {
            metric: np.mean([x[metric] for x in summary_batch])
            for metric in self.train_summary[0].keys()
        }

        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                    for k, v in metrics_mean.items())
        logging.info("***** Train metrics: " + metrics_string)

    def validate(self):
        self.model.eval()
        loss_avg = utils.RunningAverage()
        for data, target in self.val_dataloader:
            if self.params.cuda:
                data, target = data.to('cuda', non_blocking=True), target.to(
                    'cuda', non_blocking=True)
            output = self.model(data)
            loss = self.loss_fn(output, target)
            summary_batch = {
                metric: self.metrics[metric](output, target)
                for metric in self.metrics.keys()
            }
            output = output.data.cpu().numpy()
            target = target.data.cpu().numpy()
            loss_avg.update(loss.item())

        metrics_mean = {
            metric: np.mean([x[metric] for x in summary_batch])
            for metric in self.metrics.keys()
        }
        metrics_mean['loss'] = loss_avg()
        wandb.log({'val': metrics_mean})
        self.val_summary.append(metrics_mean)
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                    for k, v in metrics_mean.items())
        logging.info("***** Val metrics: " + metrics_string)
        return metrics_mean

    def test(self):
        self.model.eval()
        loss_avg = utils.RunningAverage()
        for data, target in self.test_dataloader:
            if self.params.cuda:
                data, target = data.to('cuda', non_blocking=True), target.to(
                    'cuda', non_blocking=True)
            output = self.model(data)
            loss = self.loss_fn(output, target)
            summary_batch = {
                metric: self.metrics[metric](output, target)
                for metric in self.metrics.keys()
            }
            output = output.data.cpu().numpy()
            target = target.data.cpu().numpy()
            loss_avg.update(loss.item())

        metrics_mean = {
            metric: np.mean([x[metric] for x in summary_batch])
            for metric in self.metrics.keys()
        }
        wandb.log({'test': metrics_mean})
        metrics_mean['loss'] = loss_avg()
        self.test_summary.append(metrics_mean)
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
