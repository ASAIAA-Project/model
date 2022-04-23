import wandb
import torch

import utils

import numpy as np

from tqdm import tqdm
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler


class Trainer:
    def __init__(self, model, optimizer_R, optimizer_D, loss_fn_R, loss_fn_D,
                 train_dataloader, val_dataloader, test_dataloader, metrics,
                 save_dir, logger, params):
        self.model = model
        if params['cuda']:
            self.model.cuda()
        self.scaler = GradScaler(params['amp'])
        self.optimizer_R = optimizer_R
        self.optimizer_D = optimizer_D
        self.loss_fn_R = loss_fn_R
        self.loss_fn_D = loss_fn_D
        self.dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.metrics = metrics
        self.params = params
        self.save_dir = save_dir
        self.logger = logger

    def train_one_epoch(self):
        self.model.train()

        loss_D_avg = utils.RunningAverage()
        loss_R_avg = utils.RunningAverage()

        summary = []

        with tqdm(total=len(self.dataloader)) as t:
            for i, (data, target) in enumerate(self.dataloader):
                if self.params['cuda']:
                    data, target = data.to(
                        'cuda', non_blocking=True), target.to('cuda',
                                                              non_blocking=True)

                self.optimizer_R.zero_grad()

                with autocast(enabled=self.params['amp']):
                    output, _ = self.model(data)
                    loss_R = self.loss_fn_R(target, output)

                self.scaler.scale(loss_R).backward()
                self.scaler.unscale_(self.optimizer_R)
                self.scaler.step(self.optimizer_R)
                self.scaler.update()

                # update the parameter of extractor with momentum
                params_backbone_D = self.model.distractor.feature_extracter.state_dict(
                )
                params_backbone_R = self.model.regressor.backbone.state_dict()
                for key in params_backbone_D.keys():
                    params_backbone_D[key] = params_backbone_D[key] * (
                        1 - self.params['momentum_D_backbone']) + self.params[
                            'momentum_D_backbone'] * params_backbone_R[key]
                self.model.distractor.feature_extracter.load_state_dict(
                    params_backbone_D)

                # save memory
                del params_backbone_D
                del params_backbone_R

                self.optimizer_D.zero_grad()

                with autocast(enabled=self.params['amp']):
                    output, mask = self.model(data)
                    loss_D = self.loss_fn_D(target, output, mask)

                self.scaler.scale(loss_D).backward()
                self.scaler.unscale_(self.optimizer_D)
                self.scaler.step(self.optimizer_D)
                self.scaler.update()

                if i % self.params['save_summary_steps'] == 0:
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
                           ) - i < self.params['save_summary_steps']:
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
            metric: np.mean([x[metric] for x in summary])
            for metric in summary[0].keys()
        }

        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                    for k, v in metrics_mean.items())
        if self.logger is not None:
            self.logger.info("***** Train metrics: " + metrics_string)
        else:
            print("***** Train metrics: " + metrics_string)

    def validate(self):
        self.model.eval()
        summary = []
        with torch.no_grad():
            for data, target in tqdm(self.val_dataloader):
                if self.params['cuda']:
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
            metric: np.mean([x[metric] for x in summary])
            for metric in self.metrics.keys()
        }
        wandb.log({'val': metrics_mean})
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                    for k, v in metrics_mean.items())
        if self.logger is not None:
            self.logger.info("***** Val metrics: " + metrics_string)
        else:
            print("***** Val metrics: " + metrics_string)
        return metrics_mean

    def test(self):
        utils.load_best_checkpoint(self.save_dir, self.model)
        self.model.eval()
        summary = []
        with torch.no_grad():
            for data, target in tqdm(self.test_dataloader):
                if self.params['cuda']:
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
            metric: np.mean([x[metric] for x in summary])
            for metric in self.metrics.keys()
        }
        wandb.log({'test': metrics_mean})
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                    for k, v in metrics_mean.items())
        if self.logger is not None:
            self.logger.info("***** Test metrics: " + metrics_string)
        else:
            print("***** Test metrics: " + metrics_string)
        return metrics_mean

    def train(self, restore_path=None):
        start_epoch = 0
        if restore_path is not None:
            start_epoch = utils.load_checkpoint(restore_path, self.model,
                                                self.optimizer_R,
                                                self.optimizer_D)
        best_val_metric = 0.0
        for epoch in range(start_epoch, self.params['epochs']):
            self.train_one_epoch()
            metrics = self.validate()
            val_metric = metrics[self.params['eval_metric_name']]
            is_best = val_metric > best_val_metric
            if is_best:
                best_val_metric = val_metric
                if self.logger is not None:
                    self.logger.info("- Found new best accuracy")
                else:
                    print("- Found new best accuracy")

                # Save best val metrics in a json file in the model directory
                best_json_path = self.save_dir / "metrics_val_best_weights.json"
                utils.save_dict_to_json(metrics, best_json_path)
            utils.save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optim_R_dict': self.optimizer_R.state_dict(),
                    'optim_D_dict': self.optimizer_D.state_dict()
                },
                is_best=is_best,
                checkpoint=self.save_dir)
