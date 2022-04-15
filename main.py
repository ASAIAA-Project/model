import configargparse
import torch
import wandb

from pathlib2 import Path
from utils import set_all_random_seed
from model import create_ASAIAANet


def set_parse():
    config_file_path = '/Users/zhangjunjie/model/config/config.yml'
    # TODO: change the path into relative after testing
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
    parser.add_argument(
        '--save_summary_steps',
        type=int,
        required=True,
        help='The number of gap steps to save summary during training')
    parser.add_argument(
        '--eval_metric_name',
        type=int,
        required=True,
        help='The name of the metric to be evaluated for validation and test')

    return parser


def create_configs(args):
    wandb_config = {
        'backbone_type': args.backbone_type,
        'feature_target_layer': args.feature_target_layer,
        'distracting_block': args.distracting_block,
        'center_bias_weight': args.center_bias_weight,
        'GB_kernel_size': args.GB_kernel_size,
        'GB_sigma': args.GB_sigma,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'pretrained': args.pretrained,
        'feature_channels_num': args.feature_channels_num,
        'feature_h': args.feature_h,
        'feature_w': args.feature_w,
        'wrap_size': args.wrap_size,
        'seed': args.seed,
        'eval_metric_name': args.eval_metric_name
    }

    trainer_config = {
        'cuda': torch.cuda.is_available(),
        'save_dir': args.save_dir,
        'epochs': args.epochs,
        'save_summary_steps': args.save_summary_steps,
        'eval_metric_name': args.eval_metric_name
    }

    return wandb_config, trainer_config


if __name__ == '__main__':
    parser = set_parse()
    args = parser.parse_args()
    wandb_config, trainer_config = create_configs(args)

    # init wandb for logging
    wandb.init(project=args.wandb_project)
    wandb.config.update(wandb_config)

    set_all_random_seed(args.seed)

    model = create_ASAIAANet(args)

    wandb.finish()

    # train
