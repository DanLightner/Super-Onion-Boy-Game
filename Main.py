from models.train import train
from models.test import test

if __name__ == "__main__":
    choice = input("Enter 'train' to train or 'test' to test the AI: ").strip().lower()
    if choice == "train":
        train()
    elif choice == "test":
        test()
