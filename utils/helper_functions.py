import numpy as np
import argparse


def get_optimizer(name, learning_rate):
    # This should match how your NeuralNetwork class handles optimizers
    if name == 'sgd':
        return {'name': 'sgd','learning_rate': learning_rate}
    elif name == 'momentum':
        return {'name': 'momentum','learning_rate': learning_rate, 'momentum': 0.9}
    elif name == 'nesterov':
        return {'name': 'nesterov','learning_rate': learning_rate, 'momentum': 0.9}
    elif name == 'rmsprop':
        return {'name': 'rmsprop','learning_rate': learning_rate, 'beta': 0.9, 'epsilon': 1e-8}
    elif name == 'adam':
        return {'name': 'adam','learning_rate': learning_rate, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8}
    elif name == 'nadam':
        return {'name': 'nadam','learning_rate': learning_rate, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8}
    else:
        raise ValueError(f"Optimizer {name} not recognized")
    
def create_validation_set(X, Y, val_ratio=0.2, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    split_index = int(n_samples * (1 - val_ratio))
    train_indices = indices[:split_index]
    val_indices = indices[split_index:]
    
    X_train = X[train_indices]
    Y_train = Y[train_indices]
    X_val = X[val_indices]
    Y_val = Y[val_indices]
    
    return X_train, X_val, Y_train, Y_val

def load_data(dataset_name='fashion_mnist'):
    if dataset_name == 'mnist':
        from keras.datasets import mnist
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    elif dataset_name == 'fashion_mnist':
        from keras.datasets import fashion_mnist
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    (X_train, X_val, y_train, y_val) = create_validation_set(X_train, y_train, val_ratio=0.1, seed=42)

    # class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
    #             'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    X_train_flat = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_val_flat = X_val.reshape(X_val.shape[0], -1) / 255.0
    X_test_flat = X_test.reshape(X_test.shape[0], -1) / 255.0
    
    return X_train_flat, y_train, X_val_flat, y_val, X_test_flat, y_test

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a neural network and log experiments to Weights & Biases"
    )
    parser.add_argument(
        "-wp", "--wandb_project",
        default="fashion_mnist_hp_search",
        help="Project name used to track experiments in Weights & Biases dashboard"
    )
    parser.add_argument(
        "-we", "--wandb_entity",
        default="bullseye2608-indian-institute-of-technology-madras",
        help="Wandb Entity used to track experiments in the Weights & Biases dashboard"
    )
    parser.add_argument(
        "-d", "--dataset",
        default="fashion_mnist",
        choices=["mnist", "fashion_mnist"],
        help='Dataset to use. Choices: ["mnist", "fashion_mnist"]'
    )
    parser.add_argument(
        "-e", "--epochs",
        default=1, type=int,
        help="Number of epochs to train neural network."
    )
    parser.add_argument(
        "-b", "--batch_size",
        default=4, type=int,
        help="Batch size used to train neural network."
    )
    parser.add_argument(
        "-l", "--loss",
        default="cross_entropy",
        choices=["mean_squared_error", "cross_entropy"],
        help='Loss function to use. Choices: ["mean_squared_error", "cross_entropy"]'
    )
    parser.add_argument(
        "-o", "--optimizer",
        default="sgd",
        choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"],
        help='Optimizer to use. Choices: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]'
    )
    parser.add_argument(
        "-lr", "--learning_rate",
        default=0.1, type=float,
        help="Learning rate used to optimize model parameters."
    )
    parser.add_argument(
        "-m", "--momentum",
        default=0.5, type=float,
        help="Momentum used by momentum and nag optimizers."
    )
    parser.add_argument(
        "-beta", "--beta",
        default=0.5, type=float,
        help="Beta used by rmsprop optimizer."
    )
    parser.add_argument(
        "-beta1", "--beta1",
        default=0.5, type=float,
        help="Beta1 used by adam and nadam optimizers."
    )
    parser.add_argument(
        "-beta2", "--beta2",
        default=0.5, type=float,
        help="Beta2 used by adam and nadam optimizers."
    )
    parser.add_argument(
        "-eps", "--epsilon",
        default=0.000001, type=float,
        help="Epsilon used by optimizers."
    )
    parser.add_argument(
        "-w_d", "--weight_decay",
        default=0.0, type=float,
        help="Weight decay used by optimizers."
    )
    parser.add_argument(
        "-w_i", "--weight_init",
        default="random",
        choices=["random", "Xavier"],
        help='Weight initialization method. Choices: ["random", "Xavier"]'
    )
    parser.add_argument(
        "-nhl", "--num_layers",
        default=1, type=int,
        help="Number of hidden layers used in feedforward neural network."
    )
    parser.add_argument(
        "-sz", "--hidden_size",
        default=4, type=int,
        help="Number of hidden neurons in a feedforward layer."
    )
    parser.add_argument(
        "-a", "--activation",
        default="sigmoid",
        choices=["identity", "sigmoid", "tanh", "ReLU"],
        help='Activation function. Choices: ["identity", "sigmoid", "tanh", "ReLU"]'
    )
    return parser.parse_args()