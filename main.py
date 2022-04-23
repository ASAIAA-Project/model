import configargparse
import torch
import wandb

from pathlib2 import Path
from torch.utils.data import DataLoader
from torch import optim

from dataset import AVADatasetEmp
from trainer import Trainer
from metrics import accuracy_ten, accuracy_bi
from loss import cjs_loss_10_R, CJSLoss10D
from utils import set_all_random_seed, set_logger
from model import create_ASAIAANet


def set_parse():
    config_file_path = './config/config.yml'
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        default_config_files=[config_file_path])
    parser.add('-c', '--config', is_config_file=True, help='config file path')
    parser.add_argument('--backbone_type',
                        type=str,
                        required=True,
                        help='backbone type')
    parser.add_argument('--feature_target_layer',
                        type=str,
                        required=True,
                        action='append',
                        help='the list of node name of target layers')
    parser.add_argument('--distracting_block',
                        type=str,
                        required=True,
                        help='the node name of layer to be distracted')
    parser.add_argument('--center_bias_weight',
                        type=int,
                        help='The initial weight of the center bias')
    parser.add_argument('--GB_kernel_size',
                        type=int,
                        required=True,
                        help='The kernel size of the gaussian blur')
    parser.add_argument('--GB_sigma',
                        type=float,
                        required=True,
                        help='The sigma of the gaussian blur')
    parser.add_argument('--learning_rate',
                        type=float,
                        required=True,
                        help='The learning rate of the optimizer')
    parser.add_argument('--batch_size',
                        type=int,
                        required=True,
                        help='The batch size of the training')
    parser.add_argument('--epochs',
                        type=int,
                        required=True,
                        help='The epochs of the training')
    parser.add_argument('--pretrained',
                        type=bool,
                        required=True,
                        help='Whether to use pretrained backbone')
    parser.add_argument('--feature_channels_num',
                        type=int,
                        required=True,
                        help='The number of feature channels')
    parser.add_argument('--feature_h',
                        type=int,
                        required=True,
                        help='The height of feature map')
    parser.add_argument('--feature_w',
                        type=int,
                        required=True,
                        help='The width of feature map')
    parser.add_argument('--save_dir',
                        type=str,
                        required=True,
                        help='The directory to save this experiment')
    parser.add_argument('--wrap_size',
                        type=int,
                        required=True,
                        help='The size of the image to be wrapped')
    parser.add_argument('--wandb_project',
                        type=str,
                        required=True,
                        help='The project name of wandb')
    parser.add_argument('--seed',
                        type=int,
                        required=True,
                        help='The seed of the random number generator')
    parser.add_argument('--learning_rate_D',
                        type=float,
                        required=True,
                        help='The learning rate of the distractor')
    parser.add_argument('--learning_rate_R',
                        type=float,
                        required=True,
                        help='The learning rate of the regressor')
    parser.add_argument('--weight_path',
                        type=str,
                        help='The path of saved weights')
    parser.add_argument('--weight_decay_R',
                        type=float,
                        required=True,
                        help='The weight decay factor of the regressor')
    parser.add_argument('--L1_D',
                        type=float,
                        required=True,
                        help='The L1 regularization factor of the distractor')
    parser.add_argument(
        '--momentum_D_backbone',
        type=float,
        required=True,
        help='The momentum update factor of the discriminator backbone')
    parser.add_argument(
        '--save_summary_steps',
        type=int,
        required=True,
        help='The number of gap steps to save summary during training')
    parser.add_argument(
        '--eval_metric_name',
        type=str,
        required=True,
        help='The name of the metric to be evaluated for validation and test')
    parser.add_argument('--data_dir',
                        type=str,
                        required=True,
                        help='The directory of the dataset')
    parser.add_argument('--amp',
                        type=bool,
                        required=True,
                        help='Whether to use automatic mixed precision')
    parser.add_argument(
        '--restore_path',
        type=str,
        help='the path to the saved checkpoint file for restore training')

    return parser


def create_configs(args):
    wandb_config = {
        'backbone_type': args.backbone_type,
        'feature_target_layer': args.feature_target_layer,
        'distracting_block': args.distracting_block,
        'center_bias_weight': args.center_bias_weight,
        'GB_kernel_size': args.GB_kernel_size,
        'GB_sigma': args.GB_sigma,
        'learning_rate_D': args.learning_rate_D,
        'learning_rate_R': args.learning_rate_R,
        'weight_decay_R': args.weight_decay_R,
        'L1_D': args.L1_D,
        'momentum_D_backbone': args.momentum_D_backbone,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'pretrained': args.pretrained,
        'feature_channels_num': args.feature_channels_num,
        'feature_h': args.feature_h,
        'feature_w': args.feature_w,
        'wrap_size': args.wrap_size,
        'seed': args.seed,
        'eval_metric_name': args.eval_metric_name,
        'amp': args.amp and torch.cuda.is_available()
    }

    trainer_config = {
        'cuda': torch.cuda.is_available(),
        'epochs': args.epochs,
        'save_summary_steps': args.save_summary_steps,
        'eval_metric_name': args.eval_metric_name,
        'momentum_D_backbone': args.momentum_D_backbone,
        'amp': args.amp and torch.cuda.is_available()
    }

    return wandb_config, trainer_config


if __name__ == '__main__':
    parser = set_parse()
    args = parser.parse_args()
    wandb_config, trainer_config = create_configs(args)

    set_all_random_seed(args.seed)
    logger = set_logger(Path(args.save_dir) / 'experiment.log')

    wandb_resume = True if args.restore_path is not None else False

    # init wandb for logging
    wandb.init(project=args.wandb_project, resume=wandb_resume)
    wandb.config.update(wandb_config)

    model = create_ASAIAANet(args)
    optimizer_R = optim.Adam(model.regressor.parameters(),
                             weight_decay=args.weight_decay_R,
                             lr=args.learning_rate_R)
    optimizer_D = optim.Adam(model.distractor.readout_net.parameters(),
                             lr=args.learning_rate_D)

    data_dir = Path(args.data_dir)

    train_data = AVADatasetEmp('train.pickle', data_dir, args.wrap_size)
    train_dataloader = DataLoader(train_data,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=2,
                                  pin_memory=True)

    val_data = AVADatasetEmp('val.pickle', data_dir, args.wrap_size)
    val_dataloader = DataLoader(val_data,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=2,
                                pin_memory=True)

    test_data = AVADatasetEmp('test.pickle', data_dir, args.wrap_size)
    test_dataloader = DataLoader(test_data,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=2,
                                 pin_memory=True)

    metrics = {'accuracy_ten': accuracy_ten, 'accuracy_bi': accuracy_bi}

    cjs_loss_10_D = CJSLoss10D(args.L1_D)
    trainer = Trainer(
        model,
        optimizer_R,
        optimizer_D,
        cjs_loss_10_R,
        cjs_loss_10_D,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        metrics,
        Path(args.save_dir),
        logger,
        trainer_config,
    )

    #trainer.train(restore_path=args.restore_path)
    trainer.test()

    wandb.finish()
