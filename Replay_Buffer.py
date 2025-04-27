import torch
import numpy as np
import random

class ReplayBuffer:
    def __init__(self, size):
        self.Max_Size = int(size)
        self.Data = {}

    def current_size(self):
        current_size = 0
        for value in self.Data.values():
            current_size = max(len(value), current_size)
        return current_size


    def insert_one(self, new_data):
        # insert one item to the buffer
        for key, value in new_data.items():
            if self.Data.get(key, None) is None:
                if isinstance(value, torch.Tensor):
                    self.Data[key] = torch.unsqueeze(value, dim = 0)
                else:
                    self.Data[key] = [value]
            else:
                if isinstance(value, torch.Tensor):
                    self.Data[key] = torch.cat((self.Data[key], torch.unsqueeze(value, dim = 0)))
                else:
                    self.Data[key].append(value)

            if len(self.Data[key]) > self.Max_Size:
                self.Data[key] = self.Data[key][1:]


    def get(self, num):
        # get `num` items from buffer
        num = int(num)
        current_size = 0
        for value in self.Data.values():
            current_size = max(current_size, len(value))

        if current_size == 0:
            raise Exception("Can not get item from an empty buffer.")

        randomly_chosen_index = random.choices(range(current_size), k = num)
        return_data = {}
        for key, value in self.Data.items():
            if isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
                return_data[key] = value[randomly_chosen_index]
            else:
                return_data[key] = [value[i] for i in randomly_chosen_index]
        return return_data