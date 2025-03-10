import numpy as np
import wandb
import plotly.io as pio
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
        """
        Description: Initializes an OptimalConfig object with hyperparameters for a neural network
        Input format: 
          - epochs: int (number of training epochs, default=10)
          - batch_size: int (size of training batches, default=64)
          - loss: str (loss function name, default="cross_entropy")
          - optimizer: str (optimizer name, default="adam")
          - learning_rate: float (learning rate, default=0.001)
          - momentum: float (momentum for optimizers, default=0.9)
          - beta: float (beta for RMSprop, default=0.9)
          - beta1: float (beta1 for Adam/Nadam, default=0.9)
          - beta2: float (beta2 for Adam/Nadam, default=0.999)
          - epsilon: float (epsilon for numerical stability, default=1e-8)
          - weight_decay: float (weight decay coefficient, default=0)
          - weight_init: str (weight initialization method, default="xavier")
          - num_layers: int (number of hidden layers, default=3)
          - hidden_size: int (size of hidden layers, default=128)
          - activation: str (activation function name, default="relu")
        Output format: None (creates instance with specified attributes)
        """
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
        """
        Description: Prints all configuration parameters of the OptimalConfig instance
        Input format: None (uses instance attributes)
        Output format: None (prints to console)
        """
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
       

def load_data(dataset_name='fashion_mnist'):
    """
    Description: Loads and preprocesses MNIST or Fashion MNIST dataset
    Input format:
      - dataset_name: str ('mnist' or 'fashion_mnist', default='fashion_mnist')
    Output format:
      - X_train_flat: numpy.ndarray (flattened training features)
      - y_train: numpy.ndarray (training labels)
      - X_val_flat: numpy.ndarray (flattened validation features)
      - y_val: numpy.ndarray (validation labels)
      - X_test_flat: numpy.ndarray (flattened test features)
      - y_test: numpy.ndarray (test labels)
    """
    if dataset_name == 'mnist':
        from keras.datasets import mnist
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    elif dataset_name == 'fashion_mnist':
        from keras.datasets import fashion_mnist
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError(f"Dataset {dataset_name} not recognized")
    (X_train, X_val, y_train, y_val) = _create_validation_set(X_train, y_train, val_ratio=0.1, seed=42)

    X_train_flat = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_val_flat = X_val.reshape(X_val.shape[0], -1) / 255.0
    X_test_flat = X_test.reshape(X_test.shape[0], -1) / 255.0
    
    return X_train_flat, y_train, X_val_flat, y_val, X_test_flat, y_test

def _create_validation_set(X, Y, val_ratio=0.2, seed=None):
    """
    Description: Splits dataset into training and validation sets
    Input format:
      - X: numpy.ndarray (feature matrix of shape (n_samples, n_features))
      - Y: numpy.ndarray (label array of shape (n_samples,))
      - val_ratio: float (validation set proportion, default=0.2)
      - seed: int or None (random seed for reproducibility, default=None)
    Output format:
      - X_train: numpy.ndarray (training features)
      - X_val: numpy.ndarray (validation features)
      - Y_train: numpy.ndarray (training labels)
      - Y_val: numpy.ndarray (validation labels)
    """
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

def load_df(X, y):
    """
    Description: Creates a pandas DataFrame from input features and labels
    Input format:
      - X: numpy.ndarray (feature matrix of shape (n_samples, n_features))
      - y: numpy.ndarray (label array of shape (n_samples,))
    Output format:
      - df: pandas.DataFrame (DataFrame with features, numeric labels, and class names)
    """
    df = pd.DataFrame(X)
    df['label'] = y
    df['label_name'] = [class_names[label] for label in y]
    return df

def run_name_generator(config):
    """
    Description: Generates a run name string based on configuration parameters
    Input format:
      - config: OptimalConfig (configuration object with attributes)
    Output format:
      - run_name: str (formatted string with configuration details)
    """
    run_name = f"hl:{config.num_layers}_hs:{config.hidden_size}_bs:{config.batch_size}_act:{config.activation}"
    return run_name
    
def get_optimizer(name, learning_rate, momentum=0.9, beta = 0.9, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
    """
    Description: Returns optimizer configuration dictionary based on name and parameters
    Input format:
      - name: str (optimizer name: 'sgd', 'momentum', 'nesterov', 'rmsprop', 'adam', 'nadam')
      - learning_rate: float (learning rate)
      - momentum: float (momentum parameter, default=0.9)
      - beta: float (beta for RMSprop, default=0.9)
      - beta1: float (beta1 for Adam/Nadam, default=0.9)
      - beta2: float (beta2 for Adam/Nadam, default=0.999)
      - epsilon: float (epsilon for numerical stability, default=1e-8)
    Output format:
      - dict: dictionary containing optimizer name and parameters
    """
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
    
def log_plotly_confusion_matrix_to_wandb(fig, run_id=0):
    """
    Description: Logs a Plotly confusion matrix figure to Weights & Biases
    Input format:
      - fig: plotly.graph_objects.Figure (Plotly figure object containing confusion matrix)
      - run_id: int (run identifier, default=0)
    Output format:
      - html_path: str (path to saved HTML file)
    """
    print(f"Starting Logging confusion matrix for run {run_id+1}")
    # Save the figure as HTML and PNG
    html_path = f"./plots/aggregate_confusion_matrix.html"
    
    # Save as interactive HTML
    pio.write_html(fig, file=html_path, auto_open=False)
    
    # Log both versions to wandb
    wandb.log({
        "confusion_matrix_interactive": wandb.Html(html_path),
        # "confusion_matrix_image": wandb.Image(png_path)
    })
    return html_path

def _confusion_matrix(y_true, y_pred, num_classes=None):
    """
    Description: Computes a confusion matrix to evaluate classification accuracy
    Input format:
      - y_true: array-like (ground truth labels, shape (n_samples,) or one-hot (n_samples, n_classes))
      - y_pred: array-like (predicted labels, shape (n_samples,) or one-hot (n_samples, n_classes))
      - num_classes: int or None (number of classes, if None inferred from inputs)
    Output format:
      - confusion_matrix: numpy.ndarray (confusion matrix of shape (n_classes, n_classes))
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
    """
    Description: Plots and saves a normalized confusion matrix with metrics
    Input format:
      - y_true: array-like (ground truth labels, shape (n_samples,))
      - y_pred: array-like (predicted labels, shape (n_samples,))
      - run_id: int (run identifier)
    Output format:
      - cm_filename: str (path to saved plot)
      - cm: numpy.ndarray (confusion matrix)
      - precision: numpy.ndarray (precision per class)
      - recall: numpy.ndarray (recall per class)
      - f1: numpy.ndarray (F1 score per class)
    """
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
    cm_filename = f"./plots/confusion_matrix_run_{run_id+1}.png"
    plt.savefig(cm_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return cm_filename, cm, precision, recall, f1

def create_plotly_confusion_matrix(cm, class_names, num_runs):
    """
    Description: Creates an interactive Plotly confusion matrix
    Input format:
      - cm: numpy.ndarray (confusion matrix of shape (n_classes, n_classes))
      - class_names: list (list of class names)
      - num_runs: int (number of runs averaged)
    Output format:
      - fig: plotly.graph_objects.Figure (interactive Plotly figure)
    """
    num_classes = len(class_names)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap using Plotly
    fig = go.Figure(data=go.Heatmap(
        z=cm_normalized,
        x=class_names,
        y=class_names,
        colorscale='Blues',
        text=np.round(cm, decimals=1),  # Show raw counts on hover
        texttemplate="%{text}",
        hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{text}<br>Rate: %{z:.2f}<extra></extra>",
    ))
    
    # Calculate accuracy
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    
    # Add title and labels
    fig.update_layout(
        title={
            'text': f'Confusion Matrix - Avg. of {num_runs}<br><sup>Accuracy: {accuracy:.4f}</sup>',
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
    
    # Zoom in
    fig.update_layout(
        xaxis=dict(range=[-0.5, num_classes-0.5]),
        yaxis=dict(range=[-0.5, num_classes-0.5]),
    )
    
    return fig

def plot_loss_comparison(ce_histories, ce_models, mse_histories, mse_models, epochs, num_runs=5):
    """
    Description: Plots comparison of cross entropy vs MSE loss functions
    Input format:
      - ce_histories: list (list of dictionaries with training history for cross entropy)
      - ce_models: list (list of trained models using cross entropy)
      - mse_histories: list (list of dictionaries with training history for MSE)
      - mse_models: list (list of trained models using MSE)
      - epochs: int (number of epochs)
      - num_runs: int (number of runs averaged, default=5)
    Output format:
      - results: dict (dictionary containing models and metrics)
    """
    from matplotlib.gridspec import GridSpec
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.color'] = '#333333'
    plt.rcParams['ytick.color'] = '#333333'
    
    def average_metrics(histories):
        """
        Description: Computes average metrics from multiple training histories
        Input format:
          - histories: list (list of dictionaries with 'train_loss', 'train_acc', 'val_loss', 'val_acc')
        Output format:
          - train_loss: numpy.ndarray (average training loss)
          - train_acc: numpy.ndarray (average training accuracy)
          - val_loss: numpy.ndarray (average validation loss)
          - val_acc: numpy.ndarray (average validation accuracy)
        """
        train_loss = np.mean([h['train_loss'] for h in histories], axis=0)
        train_acc = np.mean([h['train_acc'] for h in histories], axis=0)
        val_loss = np.mean([h['val_loss'] for h in histories], axis=0)
        val_acc = np.mean([h['val_acc'] for h in histories], axis=0)
        return train_loss, train_acc, val_loss, val_acc
    
    train_loss_ce, train_acc_ce, val_loss_ce, val_acc_ce = average_metrics(ce_histories)
    train_loss_mse, train_acc_mse, val_loss_mse, val_acc_mse = average_metrics(mse_histories)
    
    # Define color palettes
    ce_colors = {"train": "#1f77b4", "val": "#7fbfff"}
    mse_colors = {"train": "#d62728", "val": "#ff9896"}
    
    # Second figure:   
    plt.figure(figsize=(16, 10), dpi=150)
    gs = GridSpec(2, 1, figure=plt.gcf(), height_ratios=[1, 1])
    
    early_epochs = min(10, epochs)
    
    ax3 = plt.subplot(gs[0, :])
    ax3.plot(train_loss_ce[:early_epochs], label='Cross Entropy - Training', color=ce_colors["train"], linewidth=2, marker='o', markersize=5)
    ax3.plot(val_loss_ce[:early_epochs], label='Cross Entropy - Validation', color=ce_colors["val"], linewidth=2, linestyle='--', marker='o', markersize=5)
    ax3.plot(train_loss_mse[:early_epochs], label='MSE - Training', color=mse_colors["train"], linewidth=2, marker='s', markersize=5)
    ax3.plot(val_loss_mse[:early_epochs], label='MSE - Validation', color=mse_colors["val"], linewidth=2, linestyle='--', marker='s', markersize=5)
    
    ax3.set_xlabel('Epochs', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Loss Value', fontsize=12, fontweight='bold')
    ax3.set_title(f'Loss (Avg of {num_runs} runs): Cross Entropy vs. Mean Squared Error', 
                  fontsize=18, fontweight='bold', pad=15)
    ax3.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=10)
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    ax4 = plt.subplot(gs[1, :])
    ax4.plot(train_acc_ce[:early_epochs], label='Cross Entropy - Training', color=ce_colors["train"], linewidth=2, marker='o', markersize=5)
    ax4.plot(val_acc_ce[:early_epochs], label='Cross Entropy - Validation', color=ce_colors["val"], linewidth=2, linestyle='--', marker='o', markersize=5)
    ax4.plot(train_acc_mse[:early_epochs], label='MSE - Training', color=mse_colors["train"], linewidth=2, marker='s', markersize=5)
    ax4.plot(val_acc_mse[:early_epochs], label='MSE - Validation', color=mse_colors["val"], linewidth=2, linestyle='--', marker='s', markersize=5)
    
    ax4.set_xlabel('Epochs', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax4.set_title(f'Accuracy (Avg of {num_runs} runs): Cross Entropy vs. Mean Squared Error', 
                  fontsize=18, fontweight='bold', pad=15)
    ax4.legend(loc='lower right', frameon=True, framealpha=0.9, fontsize=10)
    ax4.grid(True, linestyle='--', alpha=0.7)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    textstr = '\n'.join((
        'Key Observations:',
        f'Cross Entropy @ 10 epochs: {val_acc_ce[min(10, len(val_acc_ce)-1)]:.4f}',
        f'MSE @ 10 epochs: {val_acc_mse[min(10, len(val_acc_mse)-1)]:.4f}',
        f'CE improvement rate: {(val_acc_ce[early_epochs-1] - val_acc_ce[0])/early_epochs:.4f}/epoch',
        f'MSE improvement rate: {(val_acc_mse[early_epochs-1] - val_acc_mse[0])/early_epochs:.4f}/epoch'
    ))
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax4.text(0.02, 0.95, textstr, transform=ax4.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig('./plots/early_convergence_comparison_avg.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    results = {
        'cross_entropy': ce_models,
        'mse': mse_models,
        'metrics': {
            'ce': {
                'train_loss': train_loss_ce,
                'val_loss': val_loss_ce,
                'train_acc': train_acc_ce,
                'val_acc': val_acc_ce,
                'all_histories': ce_histories  # Store all runs for potential further analysis
            },
            'mse': {
                'train_loss': train_loss_mse,
                'val_loss': val_loss_mse,
                'train_acc': train_acc_mse,
                'val_acc': val_acc_mse,
                'all_histories': mse_histories
            }
        }
    }
    
    __analyze_results(results=results)
    
    return results

def __analyze_results(results):
    """
    Description: Analyzes and visualizes performance comparison between cross entropy and MSE
    Input format:
      - results: dict (dictionary containing metrics from plot_loss_comparison)
    Output format:
      - dict: dictionary with analysis metrics (convergence, overfit, final accuracy)
    """
    ce_metrics = results['metrics']['ce']
    mse_metrics = results['metrics']['mse']
    
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    metrics_names = ['Final Val Accuracy', 'Convergence Speed (epochs)', 'Overfit Gap']
    
    # Calculate the metrics
    ce_final_acc = ce_metrics['val_acc'][-1]
    mse_final_acc = mse_metrics['val_acc'][-1]
    
    ce_threshold = 0.9 * ce_final_acc
    mse_threshold = 0.9 * mse_final_acc
    
    ce_convergence = next((i for i, acc in enumerate(ce_metrics['val_acc']) if acc >= ce_threshold), len(ce_metrics['val_acc']))
    mse_convergence = next((i for i, acc in enumerate(mse_metrics['val_acc']) if acc >= mse_threshold), len(mse_metrics['val_acc']))
    
    ce_overfit = ce_metrics['train_acc'][-1] - ce_metrics['val_acc'][-1]
    mse_overfit = mse_metrics['train_acc'][-1] - mse_metrics['val_acc'][-1]
    
    # Create values for visualization with epsilon to avoid division by zero
    EPSILON = 1e-10  # Small value to prevent division by zero
    ce_values = [ce_final_acc, 1.0/(ce_convergence + EPSILON), 1.0-ce_overfit]
    mse_values = [mse_final_acc, 1.0/(mse_convergence + EPSILON), 1.0-mse_overfit]
    
    # Normalize values safely
    ce_values_norm = []
    mse_values_norm = []
    for i, (ce_v, mse_v) in enumerate(zip(ce_values, mse_values)):
        max_val = max(abs(ce_v), abs(mse_v), EPSILON)  # Use abs to handle negative values and ensure non-zero
        ce_values_norm.append(ce_v / max_val)
        mse_values_norm.append(mse_v / max_val)
    
    # Close the circle for radar chart
    ce_values_norm += ce_values_norm[:1]
    mse_values_norm += mse_values_norm[:1]
    
    # Create radar chart
    categories = ['Accuracy', 'Convergence\nSpeed', 'Resistance to\nOverfitting']
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    ax = plt.subplot(111, polar=True)
    
    ax.plot(angles, ce_values_norm, 'o-', linewidth=2, label='Cross Entropy', color='#1f77b4')
    ax.fill(angles, ce_values_norm, alpha=0.25, color='#1f77b4')
    
    ax.plot(angles, mse_values_norm, 'o-', linewidth=2, label='MSE', color='#d62728')
    ax.fill(angles, mse_values_norm, alpha=0.25, color='#d62728')
    
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 1.1))
    plt.title('Cross Entropy vs MSE Performance Comparison', size=15, fontweight='bold', y=1.1)
    
    plt.figtext(0.15, 0.55, f"CE Final Accuracy: {ce_final_acc:.4f}\nMSE Final Accuracy: {mse_final_acc:.4f}", 
                ha="left", bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "alpha": 0.8})
    
    plt.figtext(0.15, 0.45, f"CE Convergence: {ce_convergence} epochs\nMSE Convergence: {mse_convergence} epochs", 
                ha="left", bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "alpha": 0.8})
    
    plt.figtext(0.15, 0.35, f"CE Overfit Gap: {ce_overfit:.4f}\nMSE Overfit Gap: {mse_overfit:.4f}", 
                ha="left", bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "alpha": 0.8})
    
    plt.tight_layout()
    plt.savefig('./plots/performance_radar.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("=== CROSS ENTROPY VS MSE COMPARISON ===")
    print(f"Final Cross Entropy Validation Accuracy: {ce_final_acc:.4f}")
    print(f"Final MSE Validation Accuracy: {mse_final_acc:.4f}")
    
    print(f"\nConvergence Speed (epochs to reach 90% of final accuracy):")
    print(f"Cross Entropy: {ce_convergence}")
    print(f"MSE: {mse_convergence}")
    
    print(f"\nOverfitting Assessment (train_acc - val_acc):")
    print(f"Cross Entropy: {ce_overfit:.4f}")
    print(f"MSE: {mse_overfit:.4f}")
    
    if ce_final_acc > mse_final_acc:
        acc_winner = "Cross Entropy"
    else:
        acc_winner = "MSE"
        
    if ce_convergence < mse_convergence:
        conv_winner = "Cross Entropy"
    else:
        conv_winner = "MSE"
        
    if ce_overfit < mse_overfit:
        overfit_winner = "Cross Entropy"
    else:
        overfit_winner = "MSE"
    
    print("\n=== CONCLUSION ===")
    print(f"Best for Accuracy: {acc_winner}")
    print(f"Best for Convergence Speed: {conv_winner}")
    print(f"Best for Avoiding Overfitting: {overfit_winner}")
    
    return {
        'ce_convergence': ce_convergence,
        'mse_convergence': mse_convergence,
        'ce_overfit': ce_overfit,
        'mse_overfit': mse_overfit,
        'ce_final_acc': ce_final_acc,
        'mse_final_acc': mse_final_acc
    }
    
def main():
    pass

if __name__ == "__main__":
    main()