import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrow

import os
import copy

class curling_env(gym.Env):
    def __init__(self, fig_save_path = None):
        super().__init__()
        self.Court_Size = np.array([100, 100])
        self.Radius = 1
        self.Mass = 1
        self.Bounce_Factor = 0.9
        self.Simulation_Time_Interval = 1e-2
        self.Action_Time_Interval = 1e-1
        self.Randomize_Time_Interval = 30 # Randomize both target and speed
        self.Speed_Randomize_Interval = np.array([-10, 10])
        self.Save_Path = fig_save_path
        if self.Save_Path is not None:
            os.makedirs(self.Save_Path, exist_ok=True)

    def Friction(self, speed):
        if sum(speed ** 2) == 0:
            return 0
        else:
            return - 5e-3 * sum(speed ** 2) * self.normalize(speed)

    def normalize(self, speed):
        return speed / sum(speed ** 2)

    def distance(self, pos1, pos2):
        return np.linalg.norm(pos1 - pos2)

    def update_time_position_speed(self, action):
        self.Time += self.Simulation_Time_Interval
        a = action / self.Mass + self.Friction(self.Speed) / self.Mass
        self.Position += self.Speed * self.Simulation_Time_Interval + 1 / 2 * a * self.Simulation_Time_Interval ** 2
        self.Speed += a * self.Simulation_Time_Interval

        while True:
            # bounce against the wall
            bounce = True
            if self.Position[0] - self.Radius < 0:
                # bounce against the left wall
                self.Position[0] = 2 * self.Radius - self.Position[0]
                self.Speed[0] = - self.Speed[0]
            elif self.Position[0] + self.Radius > self.Court_Size[0]:
                # bounce against the right wall
                self.Position[0] = (self.Court_Size[0] - self.Radius) * 2 - self.Position[0]
                self.Speed[0] = - self.Speed[0]
            elif self.Position[1] - self.Radius < 0:
                # bounce against the lower wall
                self.Position[1] = 2 * self.Radius - self.Position[1]
                self.Speed[1] = - self.Speed[1]
            elif self.Position[1] + self.Radius > self.Court_Size[1]:
                # bounce against the upper wall
                self.Position[1] = (self.Court_Size[1] - self.Radius) * 2 - self.Position[1]
                self.Speed[1] = - self.Speed[1]
            else:
                bounce = False

            if not bounce:
                break
            else:
                self.Speed = self.Bounce_Factor * sum(self.Speed ** 2) * self.normalize(self.Speed)

    def randomize_target_speed_position(self):
        self.Position = np.random.rand(2) * self.Court_Size
        self.Speed = np.random.rand(2) * (self.Speed_Randomize_Interval[1] - self.Speed_Randomize_Interval[0]) + self.Speed_Randomize_Interval[0]
        self.Target = np.random.rand(2) * self.Court_Size
        '''
        self.Position = np.array([0., 0])
        self.Speed = np.array([0., 0])
        self.Target = np.array([90., 90])
        '''

    def get_current_observation(self):
        if self.Position_Target_Difference_State:
            return np.concatenate((self.Target - self.Position, self.Speed, np.array([self.Time])), axis = 0)
        else:
            return np.concatenate((self.Position, self.Speed, self.Target, np.array([self.Time])), axis = 0)

    def step(self, action):
        # action should only be:
        # * [5, 0]
        # * [-5, 0]
        # * [0, 5]
        # * [0, -5]

        last_position = copy.copy(self.Position)

        terminated = False
        truncated = False

        for _ in range(int(self.Action_Time_Interval / self.Simulation_Time_Interval)):
            self.update_time_position_speed(action)

        if self.Time >= self.Randomize_Time_Interval:
            terminated = True

        observation = self.get_current_observation()
        info = {
            "position": self.Position,
            "speed": self.Speed,
            "target": self.Target,
            "time": self.Time
        }
        if self.Reward_Type == "negative_distance":
            reward = - self.distance(self.Position, self.Target)
        else:
            reward = self.distance(last_position, self.Target) - self.distance(self.Position, self.Target)

        return observation, reward, terminated, truncated, info

    def reset(self, *, seed = None, options = {}):
        # available options:
        # delete_fig, can be True or False
        # position_target_difference_state, can be True or False
        # reward_type, can be "negative_distance" or "distance_difference"

        self.Time = 0

        self.randomize_target_speed_position()

        self.Position_Target_Difference_State = options.get("position_target_difference_state", False)

        observation = self.get_current_observation()
        info = {
            "position": self.Position,
            "speed": self.Speed,
            "target": self.Target,
            "time": self.Time
        }

        if options.get("delete_fig", False) and self.Save_Path is not None:
            for filename in os.listdir(self.Save_Path):
                file_path = os.path.join(self.Save_Path, filename)
                if filename.endswith(".png"):
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        pass
        
        self.Reward_Type = options.get("reward_type", "negative_distance")

        return observation, info

    def render(self):
        fig, ax = plt.subplots()
        
        court = Rectangle((0, 0), self.Court_Size[0], self.Court_Size[1], linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(court)
        
        circle = Circle((self.Position[0], self.Position[1]), self.Radius, edgecolor='r', facecolor='none', linewidth=2)
        ax.add_patch(circle)
        
        target = plt.scatter(self.Target[0], self.Target[1], color='g', marker='x', s=100, label='Target')
        
        speed_arrow = FancyArrow(self.Position[0], self.Position[1], self.Speed[0], self.Speed[1], 
                                 width=0.1, head_width=1, color='k', label='Speed')
        ax.add_patch(speed_arrow)
        
        ax.set_xlim(0, self.Court_Size[0])
        ax.set_ylim(0, self.Court_Size[1])
        ax.set_aspect('equal', adjustable='box')
        
        if self.Save_Path is not None:
            fig_save_path = f"{self.Save_Path}/{self.Time}.png"
            plt.savefig(fig_save_path)
            plt.close()
        else:
            plt.show()
