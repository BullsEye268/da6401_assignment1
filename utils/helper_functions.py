import numpy as np
import argparse
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

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
        



def _confusion_matrix(y_true, y_pred, num_classes=None):
    """
    Compute confusion matrix to evaluate the accuracy of a classification.
    
    Parameters:
    -----------
    y_true : array-like, shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like, shape (n_samples,)
        Estimated targets as returned by a classifier.
    num_classes : int, optional
        Number of classes. If None, it will be determined from y_true and y_pred.
        
    Returns:
    --------
    confusion_matrix : ndarray, shape (n_classes, n_classes)
        Confusion matrix where rows represent true classes and columns represent predicted classes.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # If inputs are one-hot encoded, convert to class indices
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    # Determine number of classes if not provided
    if num_classes is None:
        num_classes = max(np.max(y_true) + 1, np.max(y_pred) + 1)
    
    # Initialize confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    
    # Populate confusion matrix
    for i in range(len(y_true)):
        cm[y_true[i], y_pred[i]] += 1
    
    return cm

def plot_confusion_matrix(y_true, y_pred, run_id):
    cm = _confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create a DataFrame for better visualization
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_norm_df = pd.DataFrame(cm_normalized, index=class_names, columns=class_names)
    
    # Create figure with normalized confusion matrix
    plt.figure(figsize=(10, 8))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(cm_norm_df, annot=True, fmt='.2f', cmap=cmap, 
                linewidths=.5, cbar_kws={"shrink": .8})
    
    plt.title(f'Normalized Confusion Matrix - Run {run_id+1}', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Calculate metrics
    precision = np.diag(cm) / np.sum(cm, axis=0)
    recall = np.diag(cm) / np.sum(cm, axis=1)
    f1 = 2 * precision * recall / (precision + recall)
    
    # Save and return the figure
    cm_filename = f"./confusion_matrices/confusion_matrix_run_{run_id+1}.png"
    plt.savefig(cm_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return cm_filename, cm, precision, recall, f1

def create_plotly_confusion_matrix(cm, class_names, run_id=0):
    """
    Create an interactive confusion matrix using Plotly.js
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth (correct) target values
    y_pred : array-like
        Estimated targets as returned by a classifier
    class_names : list
        List of class names (e.g., ['T-shirt', 'Trouser', ...])
    run_id : int
        Run identifier for multiple runs
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Plotly figure object of the confusion matrix
    cm : numpy.ndarray
        Raw confusion matrix
    """
    num_classes = len(class_names)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap using Plotly
    fig = go.Figure(data=go.Heatmap(
        z=cm_normalized,
        x=class_names,
        y=class_names,
        colorscale='Blues',
        text=cm,  # Show raw counts on hover
        texttemplate="%{text}",
        hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{text}<br>Rate: %{z:.2f}<extra></extra>",
    ))
    
    # Calculate accuracy
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    
    # Add title and labels
    fig.update_layout(
        title={
            'text': f'Confusion Matrix - Run {run_id+1}<br><sup>Accuracy: {accuracy:.4f}</sup>',
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        xaxis={'side': 'bottom'},
        width=800,
        height=800,
        font=dict(size=12)
    )
    
    # Add diagonal line effect
    diagonal_effect = []
    for i in range(len(class_names)):
        diagonal_effect.append(
            go.Scatter(
                x=[i-0.5, i+0.5],
                y=[i-0.5, i+0.5],
                mode='lines',
                line=dict(color='rgba(0,0,0,0.3)', width=1.5),
                showlegend=False,
                hoverinfo='none'
            )
        )
    
    for effect in diagonal_effect:
        fig.add_trace(effect)
    
    # fig.update_layout(annotations=annotations)
    # zoom in
    fig.update_layout(
        xaxis=dict(range=[-0.5, num_classes-0.5]),
        yaxis=dict(range=[-0.5, num_classes-0.5]),
    )
    
    
    return fig


        

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