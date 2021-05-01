import numpy as np 
import random 
import operator
from State import State
from Agent import Agent
from QTable import QTable

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
        self.states = {}
        self.Q = {}
        self.Q_max_values = np.zeros((self.width+1, self.height+1), dtype=float)
    
    def set(self, rewards):
        """sets rewards of grid"""
        self.rewards = np.array(rewards)
    
    def set_agent(self, agent):
        """places agent on grid"""
        self.grid[self.start[0]][self.start[1]] = 1
    
    def move_agent(self, agent, action, return_action_taken=False):
        """moves agent on grid"""
        new_pos, act = agent.move(action, self.prob)
        
        if new_pos[0] > self.height or new_pos[0] < 0 or new_pos[1] < 0 or new_pos[1] > self.width:
            agent.position = agent.previous_position
        else:
            self.grid[agent.previous_position[0]][agent.previous_position[1]] = 0
            self.grid[new_pos[0]][new_pos[1]] = 1
            agent.position = new_pos
        
        if return_action_taken:
                return act
    
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
    
    def generate_episode(self, start_state):
        episode = []
        agent = Agent(start_state)

        while True:
            current_state = self.states[str(agent.position[0])+","+str(agent.position[1])]
            episode.append(current_state)
            if [agent.position[0], agent.position[1]] in self.terminal:
                break
            self.move_agent(agent, self.policy[agent.position[0]][agent.position[1]])
        return episode 


    def mc_policy_evaluation(self, rand=True):
        print("starting mc policy")
        self.values = np.zeros((self.width+1, self.height+1), dtype=float)
        if rand:
            self.random_policy()

        for x in range(len(self.policy)):
            for y in range(len(self.policy[x])):
                #if [x, y] in self.terminal:
                        #continue
                new_state = State(x, y, self.rewards[x][y])
                self.states[str(x)+","+str(y)] = new_state
        
        for i in range(10000):
            visited_states = []
            # generate an episode following policy pi 
            start_state = self.get_random_state()
            episode = self.generate_episode(start_state)
            episode.reverse()
            G = 0

            t = -1
            # loop for each step of the episode 
            for step in range(1, len(episode)):
                t += 1 
                current_state = episode[step]

                # update total return 
                reward = episode[step-1].reward
                G = ((self.y**t) * G) + reward

                # check if state is first occurence in episode 
                if current_state in visited_states:
                    continue 
                visited_states.append(current_state)

                # append G to return (St)
                current_state.returns.append(G)

                # update value 
                print(sum(current_state.returns)/len(current_state.returns))
                self.values[current_state.x][current_state.y] = sum(current_state.returns)/len(current_state.returns)
            print(self.values, "iteration= "+str(i))


    def td_learning(self, rand=True, a=.1):
        print("starting td learning")
        self.values = np.zeros((self.width+1, self.height+1), dtype=float)
        if rand:
            self.random_policy()

        for x in range(len(self.policy)):
            for y in range(len(self.policy[x])):
                #if [x, y] in self.terminal:
                    #continue
                new_state = State(x, y, self.rewards[x][y])
                self.states[str(x)+","+str(y)] = new_state


        #loop for each episode 
        for i in range(1000):
            t = -1

            # Initialize state
            state = self.get_random_state()
            agent = Agent(state)
            state = self.states[str(state[0])+","+str(state[1])]
            # Loop for each step of episode
            while True:
                t += 1

                # Until S is terminal
                if [state.x, state.y] in self.terminal:
                    break

                # Initialize A 
                A = self.policy[state.x][state.y]
                # Take action A
                self.move_agent(agent, A)
                next_state = self.states[str(agent.position[0])+","+str(agent.position[1])]

                # Observe R
                reward = next_state.reward
                if reward > 0:
                    print(reward)
                current_state_value = state.value
                next_state_value = next_state.value 

                # Update V(S)
                state.value = current_state_value + a * (reward + (self.y**t * next_state_value) - current_state_value)
                self.values[state.x][state.y] = state.value

                # Initialize next state
                state = next_state

            print(self.values, "iteration= "+str(i))

    def generate_episode_q(self, start_state):
        episode = []
        agent = Agent(start_state)

        while True:
            pos = [agent.position[0], agent.position[1]]
            action = self.policy[pos[0]][pos[1]]
            current_q = self.Q[str(pos[0])+","+str(pos[1])+","+action]
            episode.append(current_q)
            if [pos[0], pos[1]] in self.terminal:
                break
            self.move_agent(agent, self.policy[agent.position[0]][agent.position[1]])
        return episode

    def mc_control(self, rand=True, e=0.05):
        print("starting mc_control")
        self.values = np.zeros((self.width+1, self.height+1), dtype=float)
        Q_max_values = np.zeros((self.width+1, self.height+1), dtype=float)
        actions = ["^", ">", "<", "V"]

        # initialize policy
        if rand:
            self.random_policy()

        # Q(s,a) in R (arbitrarily) for all s in S, a in A(s)
        for x in range(len(self.policy)):
            for y in range(len(self.policy[x])):
                new_state = State(x, y, self.rewards[x][y])
                if [x, y] in self.terminal:
                    new_qtable = QTable(new_state, "X")
                    self.Q[str(x)+","+str(y)+","+"X"] = new_qtable
                    continue 
                for action in actions:
                    new_qtable = QTable(new_state, action)
                    self.Q[str(x)+","+str(y)+","+action] = new_qtable

        # repeat 'forever' 
        for i in range(10000):
            visited_qt = []
            if rand:
                self.random_policy()
            
            start_state = self.get_random_state()

            # Generate episode
            episode = self.generate_episode_q(start_state)
            episode.reverse()

            # Initialize G
            G = 0

            t = -1
            # Loop for each step of episode 
            for step in range(1, len(episode)):
                t += 1 

                # Update G 
                G = ((self.y**t) * G) + episode[step-1].state.reward

                current_qt = episode[step]
                # Check if state is first occurence in episode 
                if current_qt in visited_qt:
                    continue 
                visited_qt.append(current_qt)

                # Append G to Returns(St, At)
                current_qt.returns.append(G)
                # Update Q value 
                current_qt.Q = sum(current_qt.returns)/len(current_qt.returns)

                xpos, ypos = current_qt.state.x, current_qt.state.y
                state = str(xpos)+","+str(ypos)+","
                state_and_actions = {"^":self.Q[state+"^"].Q, "V":self.Q[state+"V"].Q, ">":self.Q[state+">"].Q, "<":self.Q[state+"<"].Q}
                # Argmax 
                A = max(state_and_actions.items(), key=operator.itemgetter(1))[0]
                chance = e/len(state_and_actions)
                rand_val = random.random()
                if rand_val <= chance:
                    del state_and_actions[A]
                    action = random.choice(list(state_and_actions))
                    self.policy[xpos][ypos] = action
                    Q_max_values[xpos][ypos] = state_and_actions[action]      
                else:
                    self.policy[xpos][ypos] = A
                    Q_max_values[xpos][ypos] = state_and_actions[A]
            print(Q_max_values)
            print(self.policy, "iteration= "+str(i))
    
    
    def pick_A(self, state, e):
        xpos, ypos = state.x, state.y
        str_state = str(xpos)+","+str(ypos)+","
        state_and_actions = {"^":self.Q[str_state+"^"].Q, "V":self.Q[str_state+"V"].Q, ">":self.Q[str_state+">"].Q, "<":self.Q[str_state+"<"].Q}

        A = max(state_and_actions.items(), key=operator.itemgetter(1))[0]
        chance = e/len(state_and_actions)
        rand_val = random.random()
        if rand_val <= chance:
            del state_and_actions[A]
            action = random.choice(list(state_and_actions))
            self.policy[xpos][ypos] = action
            self.Q_max_values[xpos][ypos] = state_and_actions[action]      
        else:
            self.policy[xpos][ypos] = A
            self.Q_max_values[xpos][ypos] = state_and_actions[A]
        return A  
    
    def sarsa(self, rand=True, a=0.1, e=0.1):
        actions = ["^", ">", "<", "V"]

        # initialize policy
        if rand:
            self.random_policy()

        # Q(s,a) in R (arbitrarily) for all s in S, a in A(s)
        for x in range(len(self.policy)):
            for y in range(len(self.policy[x])):
                new_state = State(x, y, self.rewards[x][y])
                self.states[str(x)+","+str(y)] = new_state
                if [x, y] in self.terminal:
                    new_qtable = QTable(new_state, "X")
                    self.Q[str(x)+","+str(y)+","+"X"] = new_qtable
                    continue 
                for action in actions:
                    new_qtable = QTable(new_state, action)
                    self.Q[str(x)+","+str(y)+","+action] = new_qtable

        # Loop for each episode 
        for i in range(1000):
            t = -1
            # Initialize S
            start_state = self.get_random_state()
            agent = Agent(start_state)
            
            # Initialize S
            state = str(start_state[0])+","+str(start_state[1])
            # Choose A from S
            if [self.states[state].x, self.states[state].y] in self.terminal:
                continue 
            A = self.pick_A(self.states[state], e)
            
            # Loop for each step of episode
            while True:
                t += 1

                # Take action A
                self.move_agent(agent, A)
                next_state = str(agent.position[0])+","+str(agent.position[1]) 

                # Observe R 
                reward = self.states[next_state].reward

                # Choose A' from S'
                A_next = "X"
                if [self.states[next_state].x, self.states[next_state].y] not in self.terminal:
                    A_next = self.pick_A(self.states[next_state], e)

                # Update Q(S, A)
                state_q = self.Q[state+","+A]
                current_q_value = state_q.Q 
                next_q_value = self.Q[next_state+","+A_next].Q
                state_q.Q = current_q_value + a * (reward + (self.y**t) * next_q_value - current_q_value)

                state = next_state 
                A = A_next
                # Until S is terminal
                if [self.states[state].x, self.states[state].y] in self.terminal:
                    break
            print(self.Q_max_values)
            print(self.policy) 
            #print(Q_vals)
    
    def sarsa_max(self, rand=True, a=0.1, e=0.1):
        print("starting sarsa max")
        actions = ["^", ">", "<", "V"]

        # initialize policy
        if rand:
            self.random_policy()

        print("initialized policy")

        # Q(s,a) in R (arbitrarily) for all s in S, a in A(s)
        for x in range(len(self.policy)):
            for y in range(len(self.policy[x])):
                new_state = State(x, y, self.rewards[x][y])
                self.states[str(x)+","+str(y)] = new_state
                if [x, y] in self.terminal:
                    new_qtable = QTable(new_state, "X")
                    self.Q[str(x)+","+str(y)+","+"X"] = new_qtable
                    continue 
                for action in actions:
                    new_qtable = QTable(new_state, action)
                    self.Q[str(x)+","+str(y)+","+action] = new_qtable

        print("set Q(S,A)")

        # Loop for each episode 
        for i in range(10000):
            t = -1
            print("start loop")
            # Initialize S
            start_state = self.get_random_state()
            agent = Agent(start_state)
            print("Initialized S")
            # Initialize S
            state = str(start_state[0])+","+str(start_state[1])
            
            # Loop for each step of episode
            while True:
                t += 1
                print(self.states[state].x, self.states[state].y)
                # Choose A from S
                if [self.states[state].x, self.states[state].y] in self.terminal:
                    break 
                A = self.pick_A(self.states[state], e)
                print("Chosen A")
                # Take action A
                self.move_agent(agent, A)
                next_state = str(agent.position[0])+","+str(agent.position[1]) 
                print("Took action A")
                # Observe R 
                reward = self.states[next_state].reward

                A_max = "X"
                if [self.states[next_state].x, self.states[next_state].y] not in self.terminal:
                    state_and_actions = {"^":self.Q[next_state+",^"].Q, "V":self.Q[next_state+",V"].Q, ">":self.Q[next_state+",>"].Q, "<":self.Q[next_state+",<"].Q}
                    A_max = max(state_and_actions.items(), key=operator.itemgetter(1))[0]

                # Update Q(S, A)
                state_q = self.Q[state+","+A]
                current_q_value = state_q.Q 
                next_q_value = self.Q[next_state+","+A_max].Q
                state_q.Q = current_q_value + a * (reward + (self.y**t) * next_q_value - current_q_value)

                print(state_q.Q)
                state = next_state 
                # Until S is terminal
                if [self.states[state].x, self.states[state].y] in self.terminal:
                    break
            print(self.Q_max_values)
            print(self.policy, "iteration=", t)
        