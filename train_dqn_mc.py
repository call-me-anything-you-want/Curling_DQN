import torch
from torch import nn
from torch import optim

from utils import train, save, make_parser
from Curling_Env import curling_env

def calculate_loss_mc(training_net, train_data, gamma):
    loss_func = nn.SmoothL1Loss()

    current_q_value = training_net(train_data["observation"])
    current_q_value = current_q_value[list(range(current_q_value.shape[0])), train_data["action"]]

    target_q_value = train_data["return"]

    return loss_func(current_q_value, target_q_value)

if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    reward_type = args.reward_type
    position_target_difference_state = args.position_target_difference_state

    env_config = {
        "position_target_difference_state": position_target_difference_state,
        "delete_fig": True,
        "reward_type": "distance_difference" if reward_type == "dd" else "negative_distance"
    }
    env = curling_env()
    test_obs, _ = env.reset(options = env_config)
    obs_length = len(test_obs)

    learning_rate = 1e-3
    q_net = nn.Sequential(
        nn.Linear(obs_length, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 4)
    )
    optimizer = optim.Adam(q_net.parameters(), lr = learning_rate)

    q_net, all_losses, all_rewards = train(
        q_net,
        optimizer,
        config = {
            "buffer_size": 1e4,
            "train_round": 100,
            "episodes": 1,
            "batch_size": 256,
            "train_time": 300,
            "epsilon": [5e-1, 5e-1],
            "gamma": 0.9,
            "test_interval": 10,
            "test_episode": 100,
            "visualize": False,
            "loss_func": calculate_loss_mc,
            "env_config": env_config
        }
    )

    save(q_net, all_losses, all_rewards, f"./dqn_mc_{reward_type}")
