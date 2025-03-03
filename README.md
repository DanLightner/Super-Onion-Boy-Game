# Super Onion Boy - Reinforcement Learning Agent

## Project Structure

```
super_onion_boy/         # Root directory
│── env/                 # Environment setup
│   ├── game_capture.py   # Captures game frames
│   ├── game_controls.py  # Sends keyboard inputs
│── models/              # RL models
│   ├── dqn.py           # Deep Q-Network implementation
│   ├── train.py         # Training script for the agent
│   ├── test.py          # Testing script for evaluation
│── utils/               # Helper functions
│   ├── memory.py        # Experience replay buffer
│   ├── preprocess.py    # Frame preprocessing (resizing, grayscale, stacking)
│── main.py              # Main entry point for running the agent
│── requirements.txt     # Dependencies (PyTorch, OpenCV, mss, etc.)
│── README.md            # Project documentation
```

## Features

- **Game Capture**: Extracts frames from *Super Onion Boy* using screen capture.
- **Automated Controls**: Sends keyboard inputs to play the game.
- **Deep Q-Network (DQN)**: Uses reinforcement learning to improve gameplay.
- **Experience Replay**: Implements a memory buffer for efficient training.
- **Preprocessing**: Converts game frames to grayscale, resizes, and stacks them for neural network input.

## Dependencies

- Python 3.8+
- PyTorch
- OpenCV
- MSS (for screen capture)
- NumPy

