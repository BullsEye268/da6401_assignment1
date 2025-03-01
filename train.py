import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.datasets import fashion_mnist
import wandb

from utils.neural_network import NeuralNetwork
from utils.wandb_classes import WandbCallback
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
    
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(dataset_name=args.dataset)
    
    
    nn = NeuralNetwork(layer_sizes=[784] + [args.hidden_size]*args.num_layers + [10],
                       activation_functions=[args.activation]*args.num_layers + ['softmax'],
                       weight_init=args.weight_init,
                       weight_decay=args.weight_decay,
                       LOG_EACH=True)
    
    optimizer = get_optimizer(args.optimizer, args.learning_rate)
    nn.set_optimizer(optimizer)
    
    wandb_callback = WandbCallback()
    
    history = nn.train(
        X_train, 
        y_train, 
        X_val, 
        y_val, 
        batch_size=args.batch_size, 
        num_epochs=args.epochs, 
        loss_type=args.loss,
        log_every=1000,
        callback=wandb_callback  # Assuming your NeuralNetwork class supports callbacks
    )
    

    # Mark the end of the wandb run
    wandb.finish()

if __name__ == '__main__':
    main()



    
    # wandb_entity = 'bullseye2608-indian-institute-of-technology-madras'
    # wandb_project = 'fashion_mnist_hp_search'
    # dataset = 'fashion_mnist'
    # epochs = 1
    # batch_size = 4
    # loss = 'cross_entropy'
    # optimizer = 'sgd'
    # learning_rate = 0.1
    # momentum = 0.9
    # beta = 0.9
    # beta1 = 0.9
    # beta2 = 0.999
    # epsilon = 1e-8
    # weight_decay = 0
    # weight_init = 'Xavier'
    # num_layers = 2
    # hidden_size = 128
    # activation = 'ReLU'
    