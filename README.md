super onion boy game      # Root directory
│── env/                  # Environment setup
│   ├── game_capture.py   # Captures game frames
│   ├── game_controls.py  # Sends keyboard inputs
│── models/               # RL models
│   ├── dqn.py            # Deep Q-Network model
│   ├── train.py          # Training script
│   ├── test.py           # Testing script
│── utils/                # Helper functions
│   ├── memory.py         # Experience replay buffer
│   ├── preprocess.py     # Frame preprocessing (resizing, grayscale, stacking)
│── main.py               # Main entry point
│── requirements.txt      # Dependencies (PyTorch, OpenCV, mss, etc.)
│── README.md             # Project documentation
