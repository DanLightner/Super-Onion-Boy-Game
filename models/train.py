import os
import random
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from Environment.game_capture import capture_screen, detect_game_over
from Environment.game_controls import perform_action, reset_game
from models.dqn import DQN
from utils.memory import ReplayMemory
from utils.preprocess import preprocess_frame


def train(episodes=1000,
          batch_size=32,
          gamma=0.99,
          epsilon_start=1.0,
          epsilon_final=0.01,
          epsilon_decay=10000,
          target_update=1000,
          checkpoint_dir="checkpoints"):
    """
    Train the DQN model to play Super Onion Boy.

    Args:
        episodes (int): Number of training episodes
        batch_size (int): Batch size for training
        gamma (float): Discount factor for future rewards
        epsilon_start (float): Starting exploration rate
        epsilon_final (float): Final exploration rate
        epsilon_decay (int): Decay rate for exploration (in steps)
        target_update (int): Number of steps between target network updates
        checkpoint_dir (str): Directory to save model checkpoints
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Set device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define input shape and number of actions
    input_shape = (4, 84, 84)  # 4 stacked frames, 84x84 pixels
    num_actions = 5  # do_nothing, left, right, jump, attack

    # Initialize DQN model (policy network)
    policy_net = DQN(input_shape, num_actions).to(device)

    # Initialize target network with same weights
    target_net = DQN(input_shape, num_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()  # Target network stays in evaluation mode

    # Initialize optimizer and loss function
    optimizer = optim.Adam(policy_net.parameters(), lr=0.0001)
    criterion = nn.SmoothL1Loss()  # Huber loss is more stable than MSE

    # Initialize replay memory
    memory = ReplayMemory(capacity=100000)

    # Tracking variables
    steps_done = 0
    best_avg_reward = float('-inf')
    rewards_history = []

    # Training loop
    for episode in range(episodes):
        print(f"Starting episode {episode + 1}/{episodes}")

        # Reset game to starting state
        reset_game()
        time.sleep(2)  # Wait for game to fully reset

        # Initialize frame stack with the first frame repeated
        frame = preprocess_frame(capture_screen())
        frame_stack = deque([frame for _ in range(4)], maxlen=4)

        # Track episode statistics
        episode_reward = 0
        episode_steps = 0
        done = False

        # Start episode loop
        while not done:
            # Calculate exploration rate (epsilon)
            epsilon = epsilon_final + (epsilon_start - epsilon_final) * \
                      np.exp(-1. * steps_done / epsilon_decay)
            steps_done += 1
            episode_steps += 1

            # Stack frames and prepare input for model
            stacked_frames = np.array(frame_stack)
            state = torch.FloatTensor(stacked_frames).unsqueeze(0).to(device)

            # Select action: epsilon-greedy policy
            if random.random() < epsilon:
                # Random action
                action = random.randint(0, num_actions - 1)
            else:
                # Greedy action
                with torch.no_grad():
                    q_values = policy_net(state)
                    action = q_values.max(1)[1].item()

            # Execute selected action
            perform_action(action)

            # Capture new frame
            next_frame = preprocess_frame(capture_screen())
            frame_stack.append(next_frame)
            next_state = np.array(frame_stack)

            # Implement game-specific reward function
            reward = calculate_reward(next_frame)
            episode_reward += reward

            # Check if episode is done
            done = detect_game_over() or episode_steps >= 10000  # Add a step limit

            # Store transition in replay memory
            memory.push(
                stacked_frames,  # state
                action,
                reward,
                next_state,
                done
            )

            # Only train if we have enough samples in memory
            if len(memory) >= batch_size:
                optimize_model(policy_net, target_net, memory, optimizer, criterion, batch_size, gamma, device)

            # Update target network every target_update steps
            if steps_done % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

            # Optional: display current state
            if episode_steps % 100 == 0:
                print(f"Step {episode_steps}, Action: {action}, Reward: {reward:.2f}, Epsilon: {epsilon:.2f}")

        # Log episode statistics
        rewards_history.append(episode_reward)
        avg_reward = np.mean(rewards_history[-100:])  # Average of last 100 episodes

        print(f"Episode {episode + 1} finished with reward: {episode_reward:.2f}, "
              f"Avg Reward (100 ep): {avg_reward:.2f}, Epsilon: {epsilon:.2f}")

        # Save model checkpoint if performance improved
        if avg_reward > best_avg_reward and episode >= 100:
            best_avg_reward = avg_reward
            torch.save(policy_net.state_dict(), f"{checkpoint_dir}/dqn_model_best.pth")
            print(f"New best model saved with average reward: {best_avg_reward:.2f}")

        # Save periodic checkpoints
        if (episode + 1) % 100 == 0:
            torch.save(policy_net.state_dict(), f"{checkpoint_dir}/dqn_model_ep{episode + 1}.pth")

    # Save final model
    torch.save(policy_net.state_dict(), f"{checkpoint_dir}/dqn_model_final.pth")

    # Plot rewards history
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history)
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig(f"{checkpoint_dir}/training_rewards.png")
    plt.close()

    return policy_net


def optimize_model(policy_net, target_net, memory, optimizer, criterion, batch_size, gamma, device):
    """
    Perform a single step of optimization on the DQN model.

    Args:
        policy_net: The DQN model being trained
        target_net: The target network for stable Q-targets
        memory: Replay memory buffer
        optimizer: Optimizer for model parameters
        criterion: Loss function
        batch_size: Number of samples in a batch
        gamma: Discount factor for future rewards
        device: Device to use for computation (CPU/GPU)
    """
    # Sample batch from replay memory
    transitions = memory.sample(batch_size)

    # Prepare batch data
    state_batch = torch.FloatTensor(np.array([t[0] for t in transitions])).to(device)
    action_batch = torch.LongTensor([[t[1]] for t in transitions]).to(device)
    reward_batch = torch.FloatTensor([t[2] for t in transitions]).to(device)
    next_state_batch = torch.FloatTensor(np.array([t[3] for t in transitions])).to(device)
    done_batch = torch.FloatTensor([t[4] for t in transitions]).to(device)

    # Compute Q-values for current state-action pairs
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states using the target network
    with torch.no_grad():
        next_state_values = target_net(next_state_batch).max(1)[0]

    # Compute expected Q-values
    expected_state_action_values = reward_batch + gamma * next_state_values * (1 - done_batch)
    expected_state_action_values = expected_state_action_values.unsqueeze(1)

    # Compute loss
    loss = criterion(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # Clip gradients to stabilize training
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def calculate_reward(frame):
    """
    Calculate the reward based on the current game frame.
    This is a placeholder - implement game-specific rewards.

    Args:
        frame (numpy.ndarray): Current preprocessed frame

    Returns:
        float: Calculated reward
    """
    # TODO: Implement game-specific reward function
    # Examples:
    # - Distance traveled to the right
    # - Enemies defeated
    # - Coins collected
    # - Penalties for taking damage

    # Placeholder reward
    return 0.1  # Small reward for staying alive


if __name__ == "__main__":
    train()