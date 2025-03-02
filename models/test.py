import torch
from models.dqn import DQN
from env.game_controls import move_left, move_right, jump, attack

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DQN((4, 84, 84), num_actions=4).to(device)
model.load_state_dict(torch.load("dqn_model.pth"))

def select_action(state):
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(state).argmax(dim=1).item()

# Implement the test loop here
