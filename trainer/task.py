import os
print(os.getcwd())
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Atari_runner import train_agent
import argparse

def get_args():
    args_parser = argparse.ArgumentParser()


    args_parser.add_argument(
        '--config_path',
        help='Config path for training.',
        default=os.path.join("RLAlgorithms", "modeling", "configs", "trainer_config_Breakout_llp5.yaml"),
        type=str)

    args_parser.add_argument(
        '--conv_layers_params',
        help='convulion layers.',
        default=None)

    args_parser.add_argument(
        '--fc_layers',
        help='fc_layers.',
        default=None)

    args_parser.add_argument(
        '--continuous',
        help='continuous.',
        default=None)

    return args_parser.parse_args()


args = get_args()

print(args)

train_agent(args.config_path, args.conv_layers_params, args.fc_layers, continuous=args.continuous)
