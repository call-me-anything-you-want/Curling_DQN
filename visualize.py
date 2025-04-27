import os

import torch

from utils import plot_history, test, generate_gif

if __name__ == "__main__":
    q_net = torch.load("./dqn_nd/network.pt", weights_only=False)
    test(
        q_net,
        visualize = True,
        fig_save_path = "./dqn_nd_fig",
        env_config = {
            "position_target_difference_state": True,
            "delete_fig": True,
        }
    )
    generate_gif("./dqn_nd_fig")

    q_net = torch.load("./double_dqn_nd/network_0.pt", weights_only=False)
    test(
        q_net,
        visualize = True,
        fig_save_path = "./double_dqn_nd_fig",
        env_config = {
            "position_target_difference_state": True,
            "delete_fig": True,
        }
    )
    generate_gif("./double_dqn_nd_fig", "output_0.gif")

    q_net = torch.load("./double_dqn_nd/network_1.pt", weights_only=False)
    test(
        q_net,
        visualize = True,
        fig_save_path = "./double_dqn_nd_fig",
        env_config = {
            "position_target_difference_state": True,
            "delete_fig": True,
        }
    )
    generate_gif("./double_dqn_nd_fig", "output_1.gif")

    q_net = torch.load("./dqn_mc_nd/network.pt", weights_only=False)
    test(
        q_net,
        visualize = True,
        fig_save_path = "./dqn_mc_nd_fig",
        env_config = {
            "position_target_difference_state": True,
            "delete_fig": True,
        }
    )
    generate_gif("./dqn_mc_nd_fig")


    def label_func(file_path):
        directory = os.path.dirname(file_path)
        parent_folder_name = os.path.basename(directory)
        return parent_folder_name[:-3]

    plot_history(
        [
            ("./double_dqn_nd/reward.json", 10),
            ("./dqn_nd/reward.json", 10),
            ("./dqn_mc_nd/reward.json", 10),
        ],
        "./return.png",
        label_func,
        "episode",
        "return"
    )

    plot_history(
        [
            ("./double_dqn_nd/loss.json", 1),
            ("./dqn_nd/loss.json", 1),
            ("./dqn_mc_nd/loss.json", 1),
        ],
        "./loss.png",
        label_func,
        "episode",
        "loss"
    )
