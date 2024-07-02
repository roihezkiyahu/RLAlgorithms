from modeling.models import CNNDQNAgent, DQN, ActorCritic
from modeling.trainer import Trainer
from modeling.AtariGameWrapper import AtariGameWrapper
import os
import gym
from modeling.A2C import A2CAgent
from modeling.PPO import PPOAgent
import yaml


def common_training_setup(config, game, game_wrapper, conv_layers_params, fc_layers):
    state, info = game_wrapper.reset()
    input_shape, output_size = state.shape, game.action_space.n
    fix_groups_and_input(input_shape, conv_layers_params)
    actor_model = ActorCritic(input_shape, output_size, conv_layers_params, fc_layers, mode='actor')
    critic_model = ActorCritic(input_shape, output_size, conv_layers_params, fc_layers, mode='critic')
    config["trainer"]["input_shape"] = input_shape
    config['trainer']['n_actions'] = output_size
    return config, actor_model, critic_model


def train_a2c(config, game, game_wrapper, conv_layers_params, fc_layers):
    config, actor_model, critic_model = common_training_setup(config, game, game_wrapper, conv_layers_params, fc_layers)
    A2C = A2CAgent(config["trainer"], critic_model, actor_model)
    A2C.training_batch(config["trainer"]["epochs"], config["trainer"]["batch_size"])


def train_ppo(config, game, game_wrapper, conv_layers_params, fc_layers):
    config, actor_model, critic_model = common_training_setup(config, game, game_wrapper, conv_layers_params, fc_layers)
    ppo = PPOAgent(config["trainer"], critic_model, actor_model)
    ppo.training_batch(config["trainer"]["epochs"], config["trainer"]["batch_size"])


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def initialize_game(config, continuous):
    frameskip = config['atari_game_wrapper']['frame_skip']
    if continuous is not None:
        if frameskip is not None:
            game = gym.make(config['trainer']['game'], render_mode=config['trainer']['render_mode'],
                            continuous=continuous, frameskip=config['atari_game_wrapper']['frame_skip'])
        else:
            game = gym.make(config['trainer']['game'], render_mode=config['trainer']['render_mode'],
                            continuous=continuous)
    else:
        if frameskip is not None:
            game = gym.make(config['trainer']['game'], render_mode=config['trainer']['render_mode'],
                            frameskip=config['atari_game_wrapper']['frame_skip'])
        else:
            game = gym.make(config['trainer']['game'], render_mode=config['trainer']['render_mode'])
    return game


def get_game_wrapper(game, config, game_wrapper):
    if game_wrapper is None:
        game_wrapper = AtariGameWrapper(game, config['atari_game_wrapper'])
    else:
        game_wrapper = game_wrapper(game, config["trainer"]['game_wrapper'])
    return game_wrapper


def initialize_trainer(config, model, clone_model):
    return Trainer(config['trainer'], model, clone_model)


def fix_groups_and_input(input_shape, conv_layers_params):
    num_channels = input_shape[0]
    first_layer = conv_layers_params[0]
    if "in_channels" in first_layer.keys():
        in_channels = first_layer["in_channels"]
        if in_channels != num_channels:
            print(f"wrong number of input channels: expected {in_channels} got {num_channels}, changed number of input channels")
            first_layer["in_channels"] = num_channels
            if "groups" in first_layer.keys():
                first_layer["groups"] = num_channels
                first_layer["out_channels"] = num_channels
                conv_layers_params[1]["in_channels"] = num_channels


def create_models(config, game_wrapper, conv_layers_params, fc_layers, dueling, use_cnn=True):
    state, info = game_wrapper.reset()
    input_shape, output_size = state.shape, game_wrapper.game.action_space.n

    if use_cnn:
        fix_groups_and_input(input_shape, conv_layers_params)
        model = CNNDQNAgent(input_shape, output_size, conv_layers_params, fc_layers, dueling=dueling)
        clone_model = CNNDQNAgent(input_shape, output_size, conv_layers_params, fc_layers, dueling=dueling)
    else:
        n_observations = state.shape[0]
        model = DQN(fc_layers, n_observations, output_size)
        clone_model = DQN(fc_layers, n_observations, output_size)

    config['trainer']['n_actions'] = output_size
    return model, clone_model


def train_agent(config_path, conv_layers_params, fc_layers, continuous=None, a2c=False,
                game_wrapper=None, game=None, ppo=False):
    config = load_config(config_path)
    conv_layers_params = conv_layers_params.copy() if not conv_layers_params is None else None
    fc_layers = fc_layers if not fc_layers is None else config["fc_layers"]
    if game is None:
        game = initialize_game(config, continuous)
    game_wrapper = get_game_wrapper(game, config, game_wrapper)
    config['trainer']['game_wrapper'] = game_wrapper
    dueling = config['trainer'].get("dueling", True)
    use_cnn = config['trainer'].get("use_cnn", True)
    if a2c or config["trainer"].get("a2c", False):
        train_a2c(config, game, game_wrapper, conv_layers_params, fc_layers)
        return
    if ppo or config["trainer"].get("ppo", False):
        train_ppo(config, game, game_wrapper, conv_layers_params, fc_layers)
        return

    model, clone_model = create_models(config, game_wrapper, conv_layers_params, fc_layers, dueling, use_cnn)
    trainer = initialize_trainer(config, model, clone_model)
    try:
        trainer.train()
    finally:
        pass


if __name__ == "__main__":
    #
    # layer_params = [
    #     {'out_features': 256},
    #     {'out_features': 128},
    #     {'out_features': 64},
    # ]

    # game = gymnas.make('MountainCar-v0', render_mode="rgb_array")
    #
    # output_size = game.action_space.n
    # state, info = game.reset()
    # n_observations = state.shape[0]
    #
    # model = DQN(layer_params, n_observations, output_size)
    # clone_model = DQN(layer_params, n_observations, output_size)
    #
    # trainer = Trainer(game, model, clone_model, episodes=10000, learning_rate=1e-3,
    #                   gamma=0.99, validate_every_n_episodes=100, n_memory_episodes=100,
    #                  folder=os.path.join("logging", "MountainCar_1e3_disc05_rand"), save_gif_every_x_epochs=100,
    #                  use_ddqn=True, EPS_END=0, EPS_START=1, batch_size=512, EPS_DECAY=250,
    #                    max_episode_len=1000, replaymemory=5000, n_actions=output_size, gif_fps=10, discount_rate=0.5,
    #                   reset_options={'randomize_position': True})
    #
    # trainer.train()

    # game = gymnas.make('Acrobot-v1', render_mode="rgb_array")
    #
    # output_size = game.action_space.n
    # state, info = game.reset()
    # n_observations = state.shape[0]
    #
    # model = DQN(layer_params, n_observations, output_size)
    # clone_model = DQN(layer_params, n_observations, output_size)
    #
    # trainer = Trainer(game, model, clone_model, episodes=1000, learning_rate=1e-4,
    #                   gamma=0.99, validate_every_n_episodes=100, n_memory_episodes=100,
    #                  folder=os.path.join("logging", "Acrobot_1e4_disc075"), save_gif_every_x_epochs=100,
    #                  use_ddqn=True, EPS_END=0, EPS_START=1, batch_size=512, EPS_DECAY=250,
    #                    max_episode_len=1000, replaymemory=5000, n_actions=output_size, gif_fps=10, discount_rate=0.75,
    #                   reset_options={'randomize_position': False}, update_every_n_steps=10)
    #
    # trainer.train()

    # game = gymnas.make("LunarLander-v2", render_mode="rgb_array")
    #
    # layer_params = [
    #     {'out_features': 64},
    #     {'out_features': 64},
    # ]
    # state, info = game.reset()
    # n_observations, output_size = state.shape[0], game.action_space.n
    #
    # model = DQN(layer_params, n_observations, output_size, dueling=True)
    # clone_model = DQN(layer_params, n_observations, output_size, dueling=True)
    #
    # trainer = Trainer(game, model, clone_model, episodes=10000, learning_rate=5e-4,
    #                   gamma=0.99, validate_every_n_episodes=100, n_memory_episodes=100,
    #                  folder=os.path.join("logging", "LunarLander_5e4_disc099_dueling"), save_gif_every_x_epochs=50,
    #                  use_ddqn=True, EPS_END=0, EPS_START=1, batch_size=64, EPS_DECAY=200,
    #                    max_episode_len=500, replaymemory=5000, n_actions=output_size, gif_fps=10, discount_rate=0.99,
    #                   reset_options={'randomize_position': False}, per_alpha=0, update_every_n_steps=10)
    #
    # trainer.train()

    # SnakeGame

    # conv_layers_params = [
    #     {'in_channels': 11, 'out_channels': 11, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'groups': 11},
    #     {'in_channels': 11, 'out_channels': 16, 'kernel_size': 1, 'stride': 1},
    #     {'in_channels': 16, 'out_channels': 32, 'kernel_size': 3, 'stride': 2, 'padding': 1},
    #     {'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'stride': 2, 'padding': 1},
    #     {'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'stride': 2, 'padding': 1}
    # ]
    # fc_layers = [256]
    #
    # config_path = os.path.join("modeling", "configs", "trainer_config_snake.yaml")
    # dueling = True
    # train_agent(config_path, conv_layers_params, fc_layers, dueling, game=SnakeGame(10, 10, 10, default_start_prob=0.1),
    #             game_wrapper=SnakeGameWrap)

    layer_params = [
        {'out_features': 256},
        {'out_features': 128},
        {'out_features': 64},
    ]

    config_path = os.path.join("modeling", "configs", "cart_pole.yaml")
    train_agent(config_path, None, None)

    config_path = os.path.join("modeling", "configs", "cart_pole_dsp1.yaml")
    train_agent(config_path, None, None)

    config_path = os.path.join("modeling", "configs", "cart_pole_update10.yaml")
    train_agent(config_path, None, None)

    config_path = os.path.join("modeling", "configs", "MountainCar.yaml")
    train_agent(config_path, None, None)

    config_path = os.path.join("modeling", "configs", "Acrobot.yaml")
    train_agent(config_path, None, None)

    config_path = os.path.join("modeling", "configs", "LunarLander.yaml")
    train_agent(config_path, None, None)


    conv_layers_params = [
        {'in_channels': 4, 'out_channels': 8, 'kernel_size': 7, 'stride': 4, 'padding': 1},
        {'in_channels': 8, 'out_channels': 16, 'kernel_size': 7, 'stride': 4, 'padding': 1},
        {'in_channels': 16, 'out_channels': 32, 'kernel_size': 5, 'stride': 2, 'padding': 1},
        {'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'stride': 2, 'padding': 1}]
    fc_layers = [256, 128]

    # MsPacmanNoFrameskip
    # config_path = os.path.join("modeling", "configs", "trainer_config_MsPacman.yaml")
    # train_agent(config_path, conv_layers_params, fc_layers)

    config_path = os.path.join("modeling", "configs", "trainer_config_MsPacman_llp500.yaml")
    train_agent(config_path, conv_layers_params, fc_layers)

    # SpaceInvaders
    config_path = os.path.join("modeling", "configs", "trainer_config_SpaceInvaders.yaml")
    train_agent(config_path, conv_layers_params, fc_layers)

    config_path = os.path.join("modeling", "configs", "trainer_config_SpaceInvaders_llp50.yaml")
    train_agent(config_path, conv_layers_params, fc_layers)

    config_path = os.path.join("modeling", "configs", "trainer_config_SpaceInvaders_llp250.yaml")
    train_agent(config_path, conv_layers_params, fc_layers)

    # config_path = os.path.join("modeling", "configs", "trainer_config_SpaceInvaders_gamma999.yaml")
    # train_agent(config_path, conv_layers_params, fc_layers)

    # Breakout
    config_path = os.path.join("modeling", "configs", "trainer_config_Breakout.yaml")
    train_agent(config_path, conv_layers_params, fc_layers)

    config_path = os.path.join("modeling", "configs", "trainer_config_Breakout_llp5.yaml")
    train_agent(config_path, conv_layers_params, fc_layers)

    config_path = os.path.join("modeling", "configs", "trainer_config_Breakout_llp10.yaml")
    train_agent(config_path, conv_layers_params, fc_layers)

    # SkiingDeterministic
    config_path = os.path.join("modeling", "configs", "trainer_config_Skiing.yaml")
    train_agent(config_path, conv_layers_params, fc_layers)

    config_path = os.path.join("modeling", "configs", "trainer_config_Skiing_gamma999.yaml")
    train_agent(config_path, conv_layers_params, fc_layers)

    # CarRacing
    config_path = os.path.join("modeling", "configs", "trainer_config_CarRacing.yaml")
    train_agent(config_path, conv_layers_params, fc_layers, continuous=False)



