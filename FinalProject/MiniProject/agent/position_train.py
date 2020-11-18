try:
    from models import Lane
    from env import construct_random_lane_env
except: pass

try:
    from .models import Lane
    from .env import construct_random_lane_env
except: pass

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

OFFSET = int(sys.argv[1])   # UP(1) or FORWARD(2)
SAVE_PATH = "position{}.pth.tar".format(OFFSET)
SAVE_LOOP = 100
EPOCH = 20000

def get_cars(state):
    return torch.Tensor(state[0][0:9])

def get_lane(state):
    return torch.Tensor(state[3][0:9])

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model  = Lane().to(device)
    if os.path.exists(SAVE_PATH):
        print("[DONE] Resuming training from previous training!")
        model .load_state_dict(torch.load(SAVE_PATH, map_location=lambda storage, loc: storage))
    else:
        print("[DONE] Starting new training!")
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

    iteration = 0
    history   = []
    env = construct_random_lane_env() 
    env.reset()
    print ("[Done] Initializing!")

    while iteration != EPOCH:
        # initialize environment
        next_state, reward, done, info = env.step(4)

        # repeat rendering until done
        if not done:
            history.append((
                get_cars(next_state).to(device), 
                get_lane(next_state).to(device)))
            continue

        # compute the loss value
        item = 0
        loss = 0.0
        hidden = torch.zeros((9, 50)).to(device)
        for i in range(0, len(history) - OFFSET):
            (output, hidden) = model(torch.cat([history[i][0], history[i][1], hidden], dim=1))

            item += 1
            loss += F.binary_cross_entropy(output, history[i+OFFSET][1])

        if item == 0: continue
        optimizer.zero_grad()
        loss = loss / item
        loss.backward()
        print("[DONE] epoch: [{:3d}/{:4d}] loss: {:.6f}".format(iteration, EPOCH, loss))
        
        optimizer.step()
    
        iteration += 1
        history    = []
        env.reset()
        
        if iteration % SAVE_LOOP == 0:
            torch.save(model.state_dict(), SAVE_PATH)
            print ("[DONE] Saving model!")

