import numpy as np
import wandb
from keras.datasets import fashion_mnist


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

def load_data():
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    (X_train, X_val, y_train, y_val) = create_validation_set(X_train, y_train, val_ratio=0.1, seed=42)

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    X_train_flat = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_val_flat = X_val.reshape(X_val.shape[0], -1) / 255.0
    X_test_flat = X_test.reshape(X_test.shape[0], -1) / 255.0
    
    return X_train_flat, y_train, X_val_flat, y_val, X_test_flat, y_test