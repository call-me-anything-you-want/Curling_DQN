import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv

import random
import copy
import os
import json
import argparse

from Curling_Env import curling_env
from Replay_Buffer import ReplayBuffer

ACTION = {
    0: np.array([5, 0]),
    1: np.array([0, 5]),
    2: np.array([-5, 0]),
    3: np.array([0, -5]),
}

def get_data(
    env: curling_env,
    env_config: dict,
    q_net_1,
    q_net_2,
    replay_buffer: ReplayBuffer,
    episodes,
    epsilon
    ):
    print("start gathering data")
    total_reward = 0.0
    for i in tqdm(range(episodes)):
        # print(f"episode {i + 1}/{episodes}")
        observation, info = env.reset(options = env_config)
        observation = torch.tensor(observation).float()
        while True:
            q_net = random.choice((q_net_1, q_net_2))
            q_values = q_net(observation)
            best_action = torch.argmax(q_values)
            action_num = len(q_values)
            action_probs = [epsilon / action_num if i != best_action else 1 - epsilon + epsilon / action_num for i in range(action_num)]

            action_idx = random.choices(
                list(range(action_num)),
                weights = action_probs,
                k = 1
            )[0]
            action = ACTION[action_idx]

            new_observation, reward, terminated, truncated, info = env.step(action)
            new_observation = torch.tensor(new_observation).float()
            replay_buffer.insert_one({
                "observation": copy.copy(observation),
                "action": copy.copy(action_idx),
                "reward": torch.tensor(reward).float(),
                "new_observation": copy.copy(new_observation),
                "terminated": torch.tensor(terminated).float(),
                "truncated": torch.tensor(truncated).float(),
                "info": info
            })
            observation = new_observation
            total_reward += reward
            if terminated or truncated:
                break
    print(f"average reward: {total_reward / episodes}")
    return replay_buffer

def calculate_return(replay_buffer: ReplayBuffer, gamma):
    buffer_size = replay_buffer.current_size()
    calculated_return = replay_buffer.Data.get("return", torch.tensor([]))
    calculated_return_size = calculated_return.shape[0]
    calculating_return = torch.zeros(buffer_size - calculated_return_size)
    for idx in range(calculating_return.shape[0] - 1, calculated_return_size, -1):
        if replay_buffer.Data["terminated"][idx] or replay_buffer.Data["truncated"][idx]:
            calculating_return[idx] = replay_buffer.Data["reward"][idx]
        else:
            calculating_return[idx] = replay_buffer.Data["reward"][idx] + gamma * calculating_return[idx + 1]
    replay_buffer.Data["return"] = torch.concatenate((calculated_return, calculating_return), dim = 0)
    return replay_buffer

def train(
    q_net,
    optimizer,
    config: dict
    ):
    # config should contains:
    # buffer_size,
    # train_round,
    # episodes,
    # batch_size,
    # train_time,
    # epsilon,
    # gamma,
    # test_interval
    # test_episode
    # loss_func
    # visualize
    # env_config

    train_round = int(config["train_round"])
    train_time = int(config["train_time"])
    test_interval = int(config["test_interval"])
    replay_buffer = ReplayBuffer(config["buffer_size"])
    env = curling_env()
    all_losses = []
    all_rewards = []
    for i in range(train_round):
        print(f"start training round {i + 1}/{train_round}")
        current_round_loss = []
        if isinstance(config["epsilon"], list):
            current_epsilon = config["epsilon"][0] + (config["epsilon"][1] - config["epsilon"][0]) * (i + 1) / train_round
        else:
            current_epsilon = config["epsilon"]
        if isinstance(q_net, list):
            q_net_1 = q_net[0]
            q_net_2 = q_net[1]
        else:
            q_net_1 = q_net
            q_net_2 = q_net
        replay_buffer = get_data(env, config["env_config"], q_net_1, q_net_2, replay_buffer, config["episodes"], current_epsilon)
        replay_buffer = calculate_return(replay_buffer, config["gamma"])
        for j in tqdm(range(train_time)):
            if isinstance(q_net, list):
                network_num = len(q_net)
                train_data = [replay_buffer.get(config["batch_size"]) for _ in range(network_num)]
                loss = config["loss_func"](q_net, train_data, config["gamma"])
                for net_idx in range(len(q_net)):
                    optimizer[net_idx].zero_grad()
                    loss[net_idx].backward()
                    optimizer[net_idx].step()
                current_round_loss.append([l.item() for l in loss])
            else:
                train_data = replay_buffer.get(config["batch_size"])
                loss = config["loss_func"](q_net, train_data, config["gamma"])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                current_round_loss.append(loss.item())
        all_losses.append(current_round_loss)
        if (i + 1) % test_interval == 0:
            current_round_reward = []
            current_test_num = (i + 1) / test_interval
            for test_ep in range(config["test_episode"]):
                reward = test(q_net, config["visualize"], config["env_config"])
                if config["visualize"]:
                    generate_gif("./fig/", f"{current_test_num}_{test_ep}_{reward}.gif")
                current_round_reward.append(reward)
            all_rewards.append(current_round_reward)
    return q_net, all_losses, all_rewards

def test(q_net, visualize: bool = False, fig_save_path: str = "./fig/", env_config: dict = {}):
    if isinstance(q_net, list):
        q_net_num = len(q_net)
        all_reward = []
        for i in range(q_net_num):
            env = curling_env(fig_save_path)
            obs, info = env.reset(options = env_config)
            total_reward = 0.0
            while True:
                q_values = q_net[i](torch.tensor(obs).float())
                best_action = torch.argmax(q_values)
                action = ACTION[int(best_action)]
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if visualize:
                    env.render()
                if terminated or truncated:
                    break
            all_reward.append(total_reward)
    else:
        env = curling_env(fig_save_path)
        obs, info = env.reset(options = env_config)
        total_reward = 0.0
        while True:
            q_values = q_net(torch.tensor(obs).float())
            best_action = torch.argmax(q_values)
            action = ACTION[int(best_action)]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if visualize:
                env.render()
            if terminated or truncated:
                break
        all_reward = total_reward
    return all_reward

def generate_gif(folder_path: str, gif_name: str = "output.gif"):
    files = os.listdir(folder_path)
    png_files = []
    for file in files:
        if file.endswith(".png"):
            try:
                number = float(file[:-4])
                png_files.append((number, file))
            except ValueError:
                pass
    png_files.sort(key=lambda x: x[0])
    images = []
    for _, file in png_files:
        file_path = os.path.join(folder_path, file)
        img = Image.open(file_path)
        images.append(img)
    if images:
        output_path = os.path.join(folder_path, gif_name)
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=200,
        )
        print(f"GIF saved to: {output_path}")

def save(net, loss, reward, path):
    os.makedirs(path, exist_ok=True)
    if isinstance(net, list):
        for i in range(len(net)):
            torch.save(net[i], os.path.join(path, f"network_{i}.pt"))
    else:
        torch.save(net, os.path.join(path, f"network.pt"))
    with open(os.path.join(path, f"loss.json"), "w") as f:
        json.dump(loss, f)
    with open(os.path.join(path, f"reward.json"), "w") as f:
        json.dump(reward, f)

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reward_type", 
        help = "specify the reward type of the environment, can be 'nd' (negative distance) or 'dd' (distance difference)",
        choices = ["dd", "nd"],
        default = "nd"
        )
    parser.add_argument(
        "--position_target_difference_state", 
        help = "specify the state type of the environment, if set to be 'True', current position and current target will be replaced by the difference between them",
        type = bool,
        default = False
        )
    return parser

def plot_history(files_and_intervals: list, output_file: str, label_func, x_label, y_label):
    def get_color_palette(n_groups):
        cmap = plt.get_cmap('tab10')
        colors = []
        for i in range(n_groups):
            base_color = cmap(i % cmap.N)
            
            h, s, v = rgb_to_hsv(base_color[:3])
            lighter_color = hsv_to_rgb([h, s * 0.5, min(v + 0.4, 1.0)])
            colors.append((base_color, lighter_color))
        return colors

    all_data = []
    for idx, file_and_interval in enumerate(files_and_intervals):
        file_name = file_and_interval[0]
        interval = file_and_interval[1]
        with open(file_name, "r") as f:
            d = json.load(f)
        if isinstance(d[0][0], list):
            all_data_num = len(d[0][0])
            all_data.extend(
                [
                    (file_name, interval, [[d[i][j][k] for j in range(len(d[0]))] for i in range(len(d))], k)
                    for k in range(all_data_num)
                ]
            )
        else:
            all_data.append((file_name, interval, d, None))

    colors = get_color_palette(len(all_data))

    for idx, d in enumerate(all_data):
        file_name, interval, data, net_idx = d
        means = [np.mean(sublist) for sublist in data]
        variances = [np.std(sublist) for sublist in data]
        line_color, shade_color = colors[idx]

        x = [i * interval for i in range(len(data))]

        if net_idx is not None:
            plt.plot(x, means, '-', color=line_color, label=f'{label_func(file_name)} {net_idx} Mean')
            plt.fill_between(
                x, 
                np.array(means) - np.array(variances),
                np.array(means) + np.array(variances),
                color=shade_color, 
                alpha=0.2,
                label=f'{label_func(file_name)} {net_idx} Standard Error Range'
            )
        else:
            plt.plot(x, means, '-', color=line_color, label=f'{label_func(file_name)} Mean')
            plt.fill_between(
                x, 
                np.array(means) - np.array(variances),
                np.array(means) + np.array(variances),
                color=shade_color, 
                alpha=0.2,
                label=f'{label_func(file_name)} Standard Error Range'
            )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(
        bbox_to_anchor = (1.05, 0),
        loc = 3
    )
    plt.grid(True)

    plt.savefig(output_file, bbox_inches = "tight")
    plt.close()

if __name__ == "__main__":
    env = curling_env("./fig/")
    obv, info = env.reset()
    for _ in range(300):
        action = np.array([0.0, 5.0])
        o, r, ter, tun, info = env.step(action)
        env.render()
    generate_gif("./fig/")
