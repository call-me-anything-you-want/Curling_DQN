import torch
from torch import nn
from torch import optim

from utils import train, save, make_parser
from Curling_Env import curling_env

def calculate_loss(nets, data, gamma = 1.0):
    all_loss = []
    for i in range(2):
        training_net = nets[i]
        nontraining_net = nets[1 - i]
        train_data = data[i]

        loss_func = nn.SmoothL1Loss()

        current_q_value = training_net(train_data["observation"])
        current_q_value = current_q_value[list(range(current_q_value.shape[0])), train_data["action"]]

        next_q_value = nontraining_net(train_data["new_observation"]).detach()
        next_best_action = torch.argmax(
            training_net(train_data["new_observation"]),
            dim = 1
        ).detach()
        next_q_value = next_q_value[list(range(next_q_value.shape[0])), next_best_action]

        target_q_value = train_data["reward"] + gamma * (1 - torch.logical_or(train_data["terminated"], train_data["truncated"]).float()) * next_q_value

        all_loss.append(loss_func(current_q_value, target_q_value))
    return all_loss

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
    q_net_1 = nn.Sequential(
        nn.Linear(obs_length, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 4)
    )
    q_net_2 = nn.Sequential(
        nn.Linear(obs_length, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 4)
    )
    optimizer_1 = optim.Adam(q_net_1.parameters(), lr = learning_rate)
    optimizer_2 = optim.Adam(q_net_2.parameters(), lr = learning_rate)

    q_net, all_losses, all_rewards = train(
        [q_net_1, q_net_2],
        [optimizer_1, optimizer_2],
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
            "loss_func": calculate_loss,
            "env_config": env_config
        }
    )

    save(q_net, all_losses, all_rewards, f"./double_dqn_{reward_type}")
