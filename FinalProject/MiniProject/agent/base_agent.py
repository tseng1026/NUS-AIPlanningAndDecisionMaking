try:
    from runner.abstracts import Agent
except:
    class Agent(object): pass

try:
    from models import Lane
    from env import construct_random_lane_env
except: pass

try:
    from .models import Lane
    from .env import construct_random_lane_env
except: pass

import os
import numpy as np
import torch
import torch.nn as nn

MODEL_NAME1 = os.path.join(os.path.dirname(os.path.realpath(__file__)), "position1.pth.tar")
MODEL_NAME2 = os.path.join(os.path.dirname(os.path.realpath(__file__)), "position2.pth.tar")

def get_agent_pos(state):
    for i in range(10):
        for j in range(50):
            if state[1][i][j] > 0:
                return (i, j)

class Base_Agent(Agent):
    """
    An example agent that just output a random action.
    """
    def __init__(self, *args, **kwargs):
        """
        [OPTIONAL]
        Initialize the agent with the `test_case_id` (string), which might be important
        if your agent is test case dependent.
        
        For example, you might want to load the appropriate neural networks weight 
        in this method.
        """
        test_case_id = kwargs.get("test_case_id")
        """
        # Uncomment to help debugging
        print(">>> __INIT__ >>>")
        print("test_case_id:", test_case_id)
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
        self.lane1 = Lane()
        self.lane2 = Lane()
        self.lane1.load_state_dict(torch.load(MODEL_NAME1, map_location=lambda storage, loc: storage))
        self.lane2.load_state_dict(torch.load(MODEL_NAME2, map_location=lambda storage, loc: storage))
        self.lane1.to(self.device)
        self.lane2.to(self.device)
        self.lane1.eval()
        self.lane2.eval()

    def initialize(self, **kwargs):
        """
        [OPTIONAL]
        Initialize the agent.

        Input:
        * `fast_downward_path` (string): the path to the fast downward solver
        * `agent_speed_range` (tuple(float, float)): the range of speed of the agent
        * `gamma` (float): discount factor used for the task

        Output:
        * None

        This function will be called once before the evaluation.
        """
        fast_downward_path  = kwargs.get("fast_downward_path")
        agent_speed_range   = kwargs.get("agent_speed_range")
        gamma               = kwargs.get("gamma")
        """
        # Uncomment to help debugging
        print(">>> INITIALIZE >>>")
        print("fast_downward_path:", fast_downward_path)
        print("agent_speed_range:", agent_speed_range)
        print("gamma:", gamma)
        """
        self.hidden1 = torch.zeros(10, 50).to(self.device)
        self.hidden2 = torch.zeros(10, 50).to(self.device)

    def reset(self, state, *args, **kwargs):
        """ 
        [OPTIONAL]
        Reset function of the agent which is used to reset the agent internal state to prepare for a new environement.
        As its name suggests, it will be called after every `env.reset`.
        
        Input:
        * `state`: tensor of dimension `[channel, height, width]`, with 
                   `channel=[cars, agent, finish_position, occupancy_trails]`

        Output:
        * None
        """
        """
        # Uncomment to help debugging
        print(">>> RESET >>>")
        print("state:", state)
        """
        pass

    def step(self, state, *args, **kwargs):
        """ 
        [REQUIRED]
        Step function of the agent which computes the mapping from state to action.
        As its name suggests, it will be called at every step.
        
        Input:
        * `state`: tensor of dimension `[channel, height, width]`, with 
                   `channel=[cars, agent, finish_position, occupancy_trails]`

        Output:
        * `action`: `int` representing the index of an action or instance of class `Action`.
                    In this example, we only return a random action
        """
        """
        # Uncomment to help debugging
        print(">>> STEP >>>")
        print("state:", state)
        """
        (posi, posj) = get_agent_pos(state)

        # special rules for initial situation
        if posi == 9 and posj == 49:
            return 4

        # predict distribution
        with torch.no_grad():
            data =  [torch.Tensor(state[0]).to(self.device), torch.Tensor(state[3]).to(self.device)]
            (outputU, self.hidden1) = self.lane1(torch.cat(data + [self.hidden1], dim = 1))
            (outputF, self.hidden2) = self.lane2(torch.cat(data + [self.hidden2], dim = 1))

        # find max_speed
        max_speed = -1
        if posj - 1 >= 0 and state[0][posi][posj - 1] == 0: max_speed = -2
        if posj - 2 >= 0 and state[0][posi][posj - 1] == 0 \
                         and state[0][posi][posj - 2] == 0: max_speed = -3

        # special rules for in/out square
        if posi == 0: return 5 + max_speed
        if posi >= posj: return 0
        if posi == posj - 1 and max_speed < -1: max_speed = -1
        if posi == posj - 2 and max_speed < -2: max_speed = -2

        # compute probablity
        probablity = np.zeros((-max_speed + 1, ))
        for speed in range(0, max_speed - 1, -1):
            output = None
            if speed == 0: output = outputU
            if speed != 0: output = outputF

            if posi - 1 + speed >= 0: 
                probablity[-speed] = 1 - output[posi - 1][posj - 1 + speed].item()

        # determine action
        for speed in range(0, max_speed - 1, -1):
            if probablity[-speed] >= 0.95: 
                if speed == 0: return 0
                if speed != 0: return 5 + speed
        return 4

    def update(self, *args, **kwargs):
        """
        [OPTIONAL]
        Update function of the agent. This will be called every step after `env.step` is called.
        
        Input:
        * `state`: tensor of dimension `[channel, height, width]`, with 
                   `channel=[cars, agent, finish_position, occupancy_trails]`
        * `action` (`int` or `Action`): the executed action (given by the agent through `step` function)
        * `reward` (float): the reward for the `state`
        * `next_state` (same type as `state`): the next state after applying `action` to the `state`
        * `done` (`int`): whether the `action` induce terminal state `next_state`
        * `info` (dict): additional information (can mostly be disregarded)

        Output:
        * None

        This function might be useful if you want to have policy that is dependent to its past.
        """
        state       = kwargs.get("state")
        action      = kwargs.get("action")
        reward      = kwargs.get("reward")
        next_state  = kwargs.get("next_state")
        done        = kwargs.get("done")
        info        = kwargs.get("info")
        """
        # Uncomment to help debugging
        print(">>> UPDATE >>>")
        print("state:", state)
        print("action:", action)
        print("reward:", reward)
        print("next_state:", next_state)
        print("done:", done)
        print("info:", info)
        """
        pass