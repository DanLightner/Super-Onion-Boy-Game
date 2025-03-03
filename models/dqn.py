import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DQN(nn.Module):
    """
    Deep Q-Network model architecture.
    Processes game frames and outputs Q-values for each possible action.
    """

    def __init__(self, input_shape, num_actions):
        """
        Initialize the DQN model.

        Args:
            input_shape (tuple): Shape of input tensor (channels, height, width)
            num_actions (int): Number of possible actions
        """
        super(DQN, self).__init__()

        # Convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Calculate the size of the flattened convolutional output
        # This formula might need adjustment based on your input size
        conv_out_size = self._get_conv_output(input_shape)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def _get_conv_output(self, shape):
        """
        Calculate the output size of the convolutional layers.

        Args:
            shape (tuple): Input shape (channels, height, width)

        Returns:
            int: Size of flattened convolutional output
        """
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output = self.conv(input)
        return int(np.prod(output.data.shape))

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)

        Returns:
            torch.Tensor: Q-values for each action
        """
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc(x)


class DuelingDQN(nn.Module):
    """
    Dueling DQN architecture that separates state value and advantage streams.
    Often provides better performance than standard DQN.
    """

    def __init__(self, input_shape, num_actions):
        """
        Initialize the Dueling DQN model.

        Args:
            input_shape (tuple): Shape of input tensor (channels, height, width)
            num_actions (int): Number of possible actions
        """
        super(DuelingDQN, self).__init__()

        # Convolutional layers (shared)
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Calculate the size of the flattened convolutional output
        conv_out_size = self._get_conv_output(input_shape)

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)  # Single value for state
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)  # One advantage value per action
        )

    def _get_conv_output(self, shape):
        """
        Calculate the output size of the convolutional layers.

        Args:
            shape (tuple): Input shape (channels, height, width)

        Returns:
            int: Size of flattened convolutional output
        """
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output = self.conv(input)
        return int(np.prod(output.data.shape))

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)

        Returns:
            torch.Tensor: Q-values for each action
        """
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten

        value = self.value_stream(x)
        advantages = self.advantage_stream(x)

        # Combine value and advantages using the dueling network formula
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        return value + (advantages - advantages.mean(dim=1, keepdim=True))