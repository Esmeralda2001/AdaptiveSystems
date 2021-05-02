class QAS:
    def __init__(self, state, action):
        self.state = state
        self.action = action
        self.returns = []
        self.Q = 0 
        self.chance = 1