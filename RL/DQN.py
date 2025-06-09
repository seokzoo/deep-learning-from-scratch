import numpy as np
import random
import torch
import copy
from collections import deque

class Tictaktoe:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)

    def render(self):
        print("O: 1, X: -1")
        print("Current board:")
        for row in self.board:
            print(' '.join(['O' if x == 1 else 'X' if x == -1 else '.' for x in row]))

    def check_winner(self):
        for i in range(3):
            if np.all(self.board[i, :] == 1) or np.all(self.board[:, i] == 1):
                return True
        if np.all(np.diag(self.board) == 1) or np.all(np.diag(np.fliplr(self.board)) == 1):
            return True
        return False

    def step(self, action):
        reward, done = 0, False
        (row, col) = divmod(action, 3)
        if self.board[row, col] == 0:
            self.board[row, col] = 1

            if self.check_winner():
                reward, done = 1, True
            elif np.all(self.board != 0):
                reward, done = 0.5, True

            self.board = -self.board
        else:
            reward, done = -2, True

        return self.board.flatten(), reward, done

class Model(torch.nn.Module):
    def __init__(self, state_size=9, hidden_size=100, action_size=9):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(state_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, action_size)
        self.state_size = state_size
        self.action_size = action_size

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def get_action(self, state):
        state = torch.FloatTensor(state)
        q_values = self.forward(state)
        action = torch.argmax(q_values).item()
        return action

    def epsilon_greedy_action(self, state, epsilon=0.1):
        with torch.no_grad():
            if np.random.rand() < epsilon:
                return np.random.choice(self.action_size)
            else:
                return self.get_action(state)

    def get_td_target(self, state):
        state = torch.FloatTensor(state)
        with torch.no_grad():
            q_values = self.forward(state)
            maxq = torch.max(q_values, dim=1)[0]
        return maxq

if __name__ == '__main__':
    env = Tictaktoe()
    env.reset()

    buffer_size = 100
    batch_size = 32

    model = Model()
    train = True
    test = True

    if train:
        target_model = copy.deepcopy(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        history = deque(maxlen=buffer_size)

        for episode in range(1000):
            env.reset()
            done = False
            total_loss, cnt = 0, 0

            if episode % 100 == 0:
                target_model = copy.deepcopy(model)

            while not done:
                state = env.board.flatten()
                action = model.epsilon_greedy_action(state)
                next_state, reward, done = env.step(action)

                history.append({
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'next_state': next_state,
                    'done': done
                })

                if history[-1]['reward'] == 1:
                    history[-2]['reward'] = -1

                if history[-1]['reward'] == 0.5:
                    history[-2]['reward'] = 0.5

                if len(history) >= buffer_size:
                    cnt += 1
                    data = random.sample(history, batch_size)
                    batch_state = np.stack([x['state'] for x in data])
                    batch_action = np.array([x['action'] for x in data])
                    batch_reward = torch.FloatTensor(np.array([x['reward'] for x in data]))
                    batch_next_state = np.stack([x['next_state'] for x in data])
                    batch_done = torch.FloatTensor(np.stack([x['done'] for x in data]))

                    optimizer.zero_grad()
                    maxq = target_model.get_td_target(torch.FloatTensor(batch_next_state))
                    q = model(torch.FloatTensor(batch_state))[torch.arange(batch_size),batch_action]

                    loss = ((batch_reward + (1 - batch_done) * 0.98 * maxq - q) ** 2).mean()
                    total_loss += loss.item()
                    loss.backward()
                    optimizer.step()

            if cnt and episode % 100 == 0:
                print(f"{episode}: {total_loss / cnt}")

        torch.save(model.state_dict(), "tictactoe_model.pth")
        print("Model saved successfully!")

    if test:
        model = Model()
        model.load_state_dict(torch.load("tictactoe_model.pth"))
        model.eval()
        env.reset()

        env.render()
        done = False
        cnt = 0
        while not done:
            if cnt % 2 == 0:
                state = env.board.flatten()
                action = model.get_action(state)
                next_state, reward, done = env.step(action)
            else:
                print("Your turn!")
                action = int(input("Enter your move (0-8): "))
                next_state, reward, done = env.step(action)
            env.render()

            if done:
                print("finished")
            cnt += 1
