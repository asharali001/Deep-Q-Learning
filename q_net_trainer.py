import torch
import torch.nn as nn
import torch.optim as optim 
import numpy as np

class QNetTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.model = model
        self.gamma = gamma
        self.lossFunction = nn.functional.smooth_l1_loss
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)

    def train(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        pred = self.model(state)

        target = pred.clone()

        for idx in range(len(done)):
            Q_new = reward[idx]
            Q_new = reward[idx] + self.gamma * (1 - done[idx]) * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
        
        self.optimizer.zero_grad()
        loss = self.lossFunction(target, pred)
        loss.backward()

        self.optimizer.step()