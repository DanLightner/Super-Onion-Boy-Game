import torch
import time
import numpy as np
from PIL import Image
import cv2
from collections import deque

from models.dqn import DQN
from Environment.game_capture import capture_screen, detect_game_over
from Environment.game_controls import perform_action, reset_game
from utils.preprocess import preprocess_frame


def test(episodes=10, model_path="checkpoints/dqn_model_best.pth"):
    """
    Test the trained DQN model on the game.

    Args:
        episodes (int): Number of test episodes to run
        model_path (str): Path to the saved model
    """
    # Set device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define input shape and number of actions
    input_shape = (4, 84, 84)  # 4 stacked frames, 84x84 pixels
    num_actions = 5  # do_nothing, left, right, jump, attack

    # Initialize the DQN model
    model = DQN(input_shape, num_actions).to(device)

    # Load the trained model weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded successfully from {model_path}")
    except FileNotFoundError:
        print(f"Model file {model_path} not found. Please train the model first.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Set model to evaluation mode
    model.eval()

    # Statistics tracking
    total_rewards = []

    # Run test episodes
    for episode in range(episodes):
        print(f"Starting test episode {episode + 1}/{episodes}")

        # Reset game to starting state
        reset_game()
        time.sleep(2)  # Wait for game to fully reset

        # Initialize frame stack with the first frame repeated
        frame = preprocess_frame(capture_screen())
        frame_stack = deque([frame for _ in range(4)], maxlen=4)

        # Track episode statistics
        episode_reward = 0
        steps = 0
        done = False

        # Start episode loop
        while not done:
            # Stack frames and prepare input for model
            stacked_frames = np.array(frame_stack)
            state = torch.FloatTensor(stacked_frames).unsqueeze(0).to(device)

            # Select action with highest Q-value
            with torch.no_grad():
                q_values = model(state)
                action = q_values.max(1)[1].item()

            # Execute selected action
            perform_action(action)

            # Capture new frame
            next_frame = preprocess_frame(capture_screen())
            frame_stack.append(next_frame)

            # Implement game-specific reward function
            reward = calculate_reward(next_frame)
            episode_reward += reward

            # Check if episode is done
            done = detect_game_over() or steps >= 10000  # Add a step limit

            steps += 1

            # Optional: add delay for visualization
            time.sleep(0.01)

            # Optional: display current state
            if steps % 10 == 0:
                print(f"Step {steps}, Action: {action}, Reward: {reward:.2f}, Total Reward: {episode_reward:.2f}")

        # Log episode statistics
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1} finished with total reward: {episode_reward:.2f} in {steps} steps")

    # Print overall performance
    print(f"Average reward over {episodes} episodes: {np.mean(total_rewards):.2f}")


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
    test()