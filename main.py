import configargparse
import torch

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
    parser.add_argument('--weight_path', type=str, help='weight path')
    parser.add_argument('--device',
                        type=str,
                        required=True,
                        help='device',
                        default='cuda')
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
    return parser


if __name__ == '__main__':
    parser = set_parse()
    args = parser.parse_args()
    model = create_ASAIAANet(args)
    print(model.regressor.parameters())
    print(model.distractor.parameters())
