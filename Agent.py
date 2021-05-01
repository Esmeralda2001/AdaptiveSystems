import random
import numpy as np

class Agent:
    def __init__(self, start):
        Agent.previous_position = start
        Agent.position = start
        Agent.actions = {
            "^": (-1, 0),
            "V": (1, 0),
            "<": (0, -1),
            ">": (0, 1),
            }
    
    def move(self, action, chosen_prob):
        actions = dict(Agent.actions)
        rand_val = random.random()

        if rand_val <= chosen_prob:
            self.previous_position = self.position
            new_pos = tuple(np.add(self.position, actions[action])) 
            return new_pos, action 
        else:
            del actions[action]
            self.previous_position = self.position
            #print(list(actions.keys()))
            action = random.choice(list(actions.keys()))
            #print("Moving ", action, " instead")
            new_pos = tuple(np.add(self.position, actions[action]))
            return new_pos, action
