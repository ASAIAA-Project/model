from pydoc import Helper
import configargparse
from sympy import re

if __name__ == '__main__':
    config_file_path = './config/config.yml'
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
    parser.add_argument('--target_layer',
                        type=list,
                        required=True,
                        help='the node str of target layer')

    args = parser.parse_args()
    print(args.backbone_type)