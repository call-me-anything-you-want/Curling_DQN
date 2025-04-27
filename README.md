This folder contains all the code needed to train and visualize DQN networks in a curling environment.

# Installation
Create your conda environment and install the libraries in requirements.txt. Python 3.10 is recommended.
```bash
conda create -n curling_env python=3.10
conda activate curling_env
pip install -r requirements.txt
```

# Run Code
Depending on the way you want your network trained, run `train_dqn.py`, `train_dqn_mc.py`, `train_double_dqn.py` as you want.

These files contains two command-line parameters:
* `--reward_type`: this parameter can be "dd" or "nd".
* `--position_target_difference_state`: this parameter can be "True" or "False"
To see how these parameter works, please use the `--help` flag.

We recommend running the files using the following commands:
```bash
python train_dqn.py --position_target_difference_state True --reward_type nd
python train_double_dqn.py --position_target_difference_state True --reward_type nd
python train_dqn_mc.py --position_target_difference_state True --reward_type nd
```

After running, the result will be saved in `./dqn_nd`, `./double_dqn_nd`, `./dqn_mc_nd`

# Visualize Result
After running what we recommend above, you can run `visualize.py` to see the figures and gifs of the results.
```bash
python visualize.py
```

You can check for gifs in `xxx_fig` folders and check for pngs in the png file in the current folder.