import numpy as np
import argparse

class OptimalConfig:
    def __init__(self, epochs=10, batch_size=64, loss="cross_entropy", optimizer="adam",
                 learning_rate=0.001, momentum=0.9, beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 weight_decay=0, weight_init="xavier", num_layers=3, hidden_size=128, activation="relu"):
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss = loss
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.weight_init = weight_init
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.activation = activation
    
    def print_config(self):
        print(f"epochs: {self.epochs}")
        print(f"batch_size: {self.batch_size}")
        print(f"loss: {self.loss}")
        print(f"optimizer: {self.optimizer}")
        print(f"learning_rate: {self.learning_rate}")
        print(f"momentum: {self.momentum}")
        print(f"beta: {self.beta}")
        print(f"beta1: {self.beta1}")
        print(f"beta2: {self.beta2}")
        print(f"epsilon: {self.epsilon}")
        print(f"weight_decay: {self.weight_decay}")
        print(f"weight_init: {self.weight_init}")
        print(f"num_layers: {self.num_layers}")
        print(f"hidden_size: {self.hidden_size}")
        print(f"activation: {self.activation}")
        
        
        
        

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
        default=10, type=int,
        help="Number of epochs to train neural network."
    )
    parser.add_argument(
        "-b", "--batch_size",
        default=64, type=int,
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
        default="adam",
        choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"],
        help='Optimizer to use. Choices: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]'
    )
    parser.add_argument(
        "-lr", "--learning_rate",
        default=0.001, type=float,
        help="Learning rate used to optimize model parameters."
    )
    parser.add_argument(
        "-m", "--momentum",
        default=0.9, type=float,
        help="Momentum used by momentum and nag optimizers."
    )
    parser.add_argument(
        "-beta", "--beta",
        default=0.9, type=float,
        help="Beta used by rmsprop optimizer."
    )
    parser.add_argument(
        "-beta1", "--beta1",
        default=0.9, type=float,
        help="Beta1 used by adam and nadam optimizers."
    )
    parser.add_argument(
        "-beta2", "--beta2",
        default=0.999, type=float,
        help="Beta2 used by adam and nadam optimizers."
    )
    parser.add_argument(
        "-eps", "--epsilon",
        default=1e-8, type=float,
        help="Epsilon used by optimizers."
    )
    parser.add_argument(
        "-w_d", "--weight_decay",
        default=0.0, type=float,
        help="Weight decay used by optimizers."
    )
    parser.add_argument(
        "-w_i", "--weight_init",
        default="Xavier",
        choices=["random", "Xavier"],
        help='Weight initialization method. Choices: ["random", "Xavier"]'
    )
    parser.add_argument(
        "-nhl", "--num_layers",
        default=3, type=int,
        help="Number of hidden layers used in feedforward neural network."
    )
    parser.add_argument(
        "-sz", "--hidden_size",
        default=128, type=int,
        help="Number of hidden neurons in a feedforward layer."
    )
    parser.add_argument(
        "-a", "--activation",
        default="ReLU",
        choices=["identity", "sigmoid", "tanh", "ReLU"],
        help='Activation function. Choices: ["identity", "sigmoid", "tanh", "ReLU"]'
    )
    return parser.parse_args()

def get_optimizer(name, learning_rate, momentum=0.9, beta = 0.9, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
    # This should match how your NeuralNetwork class handles optimizers
    if name == 'sgd':
        return {'name': 'sgd','learning_rate': learning_rate}
    elif name == 'momentum':
        return {'name': 'momentum','learning_rate': learning_rate, 'momentum': momentum}
    elif name == 'nesterov':
        return {'name': 'nesterov','learning_rate': learning_rate, 'momentum': momentum}
    elif name == 'rmsprop':
        return {'name': 'rmsprop','learning_rate': learning_rate, 'beta': beta, 'epsilon': epsilon}
    elif name == 'adam':
        return {'name': 'adam','learning_rate': learning_rate, 'beta1': beta1, 'beta2': beta2, 'epsilon': epsilon}
    elif name == 'nadam':
        return {'name': 'nadam','learning_rate': learning_rate, 'beta1': beta1, 'beta2': beta2, 'epsilon': epsilon}
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
    else:
        raise ValueError(f"Dataset {dataset_name} not recognized")
    (X_train, X_val, y_train, y_val) = create_validation_set(X_train, y_train, val_ratio=0.1, seed=42)

    # class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
    #             'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    X_train_flat = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_val_flat = X_val.reshape(X_val.shape[0], -1) / 255.0
    X_test_flat = X_test.reshape(X_test.shape[0], -1) / 255.0
    
    return X_train_flat, y_train, X_val_flat, y_val, X_test_flat, y_test