import numpy as np 
import random 
from State import State
from Agent import Agent

class Grid:
    def __init__(self, width, height, terminal, start, y=1, prob=.7):
        self.grid = np.zeros((width, height), dtype=int)
        self.values = np.copy(self.grid).astype(float)
        self.policy = np.array(["X" for _ in range(height * width)]).reshape(width, height)
        self.width = width-1
        self.height = height-1
        self.terminal = terminal 
        self.start = start 
        self.prob = prob
        self.y = y
    
    def set(self, rewards):
        """sets rewards of grid"""
        self.rewards = np.array(rewards)
    
    def set_agent(self, agent):
        """places agent on grid"""
        self.grid[self.start[0]][self.start[1]] = 1
    
    def move_agent(self, agent, action):
        """moves agent on grid"""
        new_pos = agent.move(action, self.prob)
        #print(self.height, self.width)
        #print(new_pos)
        if new_pos[0] > self.height or new_pos[0] < 0 or new_pos[1] < 0 or new_pos[1] > self.width:
            agent.position = agent.previous_position
        else:
            self.grid[agent.previous_position[0]][agent.previous_position[1]] = 0
            self.grid[new_pos[0]][new_pos[1]] = 1
            agent.position = new_pos
    
    def reset(self):
        """reset agent's position"""
        self.grid = np.zeros((self.width+1, self.height+1), dtype=int)
    
    def value_iteration(self):
        """calculates value of each state"""
        new_values = np.copy(self.values)
        theta = 0.01
        counter = 0
        for i in range(100):
            print("Iteration ", counter)
            print("----------------------------------------------")
            delta = 0
            for x in range(len(self.grid)):
                for y in range(len(self.grid[x])):
                    if [x, y] in self.terminal:
                        continue 
                    neighbors = [[x+1, y], [x-1, y], [x, y+1], [x, y-1]]
                    actions = ["V", "^", ">", "<"]
                    values_dict = {}
                    for k in range(len(neighbors)):
                        n = neighbors[k]
                        if n[0] > self.height or n[0] < 0 or n[1] < 0 or n[1] > self.width:
                            continue 
                        i, j = n[0], n[1]
                        value = self.prob * (self.rewards[i][j] + (self.y * self.values[i][j]))
                        values_dict[value] = actions[k]
                    values_list = values_dict.keys()
                    max_val = max(values_list)
                    delta = max(delta, np.abs(max_val - self.values[x][y]))
                    new_values[x][y] = max_val
                    self.policy[x][y] = values_dict[max_val]
            self.values = new_values 
            print(self.values)
            print(self.policy)
            counter += 1 
            print(delta)
            if delta < theta:
                break 
        policy_values = np.add(self.values, self.rewards)
        return policy_values 
    
    def traverse_path(self, agent, visualize=False):
        """agent will try to follow the path given by the policy"""
        if visualize:
            for i in range(len(self.grid)):
                print(self.grid[i], self.policy[i])
            print("\n ---------------------")
        while True:
            x, y = agent.position
            move = self.policy[x][y]
            if move == "X":
                return 
            self.move_agent(agent, move)
            for i in range(len(self.grid)):
                print(self.grid[i], self.policy[i])
            print(self.grid)
            print("\n ---------------------")


    def demonstrate_movement(self, agent):
        """demonstrates agent movement. no other use."""
        print("Moving right..")
        for x in range(0, 2):
            self.move_agent(agent, ">")
            print(self.grid)
            print("------")
        
        print("Moving up..")
        for x in range(0, 2):
            self.move_agent(agent, "^")
            print(self.grid)
            print("------")
        
        print("Moving left..")
        for x in range(0, 2):
            self.move_agent(agent, "<")
            print(self.grid)
            print("------")

        print("Moving down..")
        for x in range(0, 2):
            self.move_agent(agent, "V")
            print(self.grid)
            print("------")
    

    def random_policy(self):
        movement = ["^", "V", ">", "<"]

        for x in range(len(self.policy)):
            for y in range(len(self.policy[x])):
                if [x, y] in self.terminal:
                        continue
                self.policy[x][y] = random.choice(movement)
        

    def get_random_state(self):
        x = random.randint(0, self.width)
        y = random.randint(0, self.height)

        return [x, y]

    def mc_policy_evaluation(self, rand=True):
        states = {}

        for x in range(len(self.policy)):
            for y in range(len(self.policy[x])):
                if [x, y] in self.terminal:
                        continue
                new_state = State(x, y)
                states[str(x)+","+str(y)] = new_state
        
        for i in range(100000):
            visited_states = []
            if rand:
                self.random_policy()
            
            start_state = self.get_random_state()
            agent = Agent(start_state)
            G = 0

            while True:
                t = -1
                t += 1 

                if [agent.position[0], agent.position[1]] in self.terminal:
                    break 
                current_state = states[str(agent.position[0])+","+str(agent.position[1])]

                self.move_agent(agent, self.policy[agent.position[0]][agent.position[1]])
                pos = agent.position
                
                G = ((self.y**t) * G) + self.rewards[pos[0]][pos[1]]

                if current_state in visited_states:
                    continue 
                visited_states.append(current_state)

                current_state.returns.append(G)

                self.values[current_state.x][current_state.y] = sum(current_state.returns)/len(current_state.returns)
            print(self.values, "iteration= "+str(i))
                






            





    
