import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.datasets import fashion_mnist
import wandb

from utils.neural_network import NeuralNetwork
from utils.helper_functions import get_optimizer, load_data, parse_args
import wandb



def main():
    args = parse_args()

    # Initialize Weights & Biases with the project, entity, and configuration
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=vars(args)
    )

    # Print the configuration (or call your actual training function here)
    print("Starting training with the following configuration:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    # === PLACEHOLDER FOR YOUR TRAINING CODE ===
    # Here you would load your data, build your model, train, and log metrics.
    # For example:
    #   train_data, val_data = load_data(args.dataset)
    #   model = build_model(args)
    #   train(model, train_data, val_data, args)
    # ============================================

    # Mark the end of the wandb run
    wandb.finish()

if __name__ == '__main__':
    main()
    