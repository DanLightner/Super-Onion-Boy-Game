import os
import time
import argparse
from models.train import train
from models.test import test
from Environment.game_capture import detect_game_window, capture_screen
from Environment.game_controls import reset_game


def setup_environment():
    """
    Setup the environment for running the agent.

    Returns:
        dict: Dictionary containing environment configuration
    """
    print("Setting up environment...")

    # Detect game window
    game_region = detect_game_window()
    if game_region is None:
        print("Game window not detected. Using default monitor.")
    else:
        print(f"Game window detected at: {game_region}")

    # Verify capture is working
    try:
        frame = capture_screen(game_region)
        print(f"Screen capture working. Frame shape: {frame.shape}")
    except Exception as e:
        print(f"Error capturing screen: {e}")
        return None

    # Create checkpoints directory if it doesn't exist
    os.makedirs("checkpoints", exist_ok=True)

    return {
        "game_region": game_region
    }


def main():
    """
    Main entry point for the Super Onion Boy AI agent.
    """
    parser = argparse.ArgumentParser(description="Super Onion Boy Reinforcement Learning Agent")
    parser.add_argument("--mode", choices=["train", "test", "interactive"], default="interactive",
                        help="Mode to run the agent in (train, test, or interactive)")
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Number of episodes to train or test")
    parser.add_argument("--model_path", type=str, default="checkpoints/dqn_model_best.pth",
                        help="Path to model file for testing or continuing training")

    args = parser.parse_args()

    # Setup environment
    env_config = setup_environment()
    if env_config is None:
        print("Failed to setup environment. Exiting.")
        return

    # Run in selected mode
    if args.mode == "train":
        print(f"Starting training for {args.episodes} episodes...")
        train(episodes=args.episodes)
    elif args.mode == "test":
        print(f"Testing model from {args.model_path} for {args.episodes} episodes...")
        test(episodes=args.episodes, model_path=args.model_path)
    else:
        # Interactive mode
        while True:
            choice = input("\nSelect an option:\n1. Train\n2. Test\n3. Exit\n> ")

            if choice == "1":
                episodes = int(input("Enter number of episodes to train: "))
                train(episodes=episodes)
            elif choice == "2":
                episodes = int(input("Enter number of episodes to test: "))
                model_path = input(f"Enter model path (default: {args.model_path}): ")
                if not model_path:
                    model_path = args.model_path
                test(episodes=episodes, model_path=model_path)
            elif choice == "3":
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()