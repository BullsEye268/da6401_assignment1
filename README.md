# Feedforward Neural Network for Fashion-MNIST Classification

### DA6401 Assignment 1 - Achyutha Munimakula PH21B004

## Description

This project implements a feedforward neural network from scratch using NumPy to classify images from the Fashion-MNIST dataset. It includes backpropagation with various optimization algorithms and utilizes Weights & Biases [(Wandb)](https://api.wandb.ai/links/bullseye2608-indian-institute-of-technology-madras/rj51csft) for experiment tracking and hyperparameter tuning.

## Features

- **Flexible Neural Network:** Implementation of a feedforward neural network with customizable number of hidden layers and neurons.
- **Backpropagation:** Implemented backpropagation algorithm with support for multiple optimization functions.
- **Optimization Algorithms:** Includes SGD, Momentum, Nesterov Accelerated Gradient, RMSprop, Adam, and Nadam optimizers. Adding other optimizers has been simplified to dictating `__init__()` and `update()` functions in `class NewOptimizer(Optimizer):`, in `optimizer.py`. Steps on this have been included at the bottom of `optimizer.py`.
- **Hyperparameter Tuning:** Uses Wandb sweeps to efficiently search for optimal hyperparameters.
- **Experiment Tracking:** Leverages Wandb for detailed experiment tracking, visualization, and analysis.
- **[Wandb Report:](https://api.wandb.ai/links/bullseye2608-indian-institute-of-technology-madras/rj51csft)** Contains detailed logs, visualizations, and analysis of the experiments, including hyperparameter tuning results and performance metrics, as well as question-wise responses to the assignment.
- **Evaluation Metrics:** Computes loss and accuracy, and generates confusion matrices for performance evaluation.
- **Loss Function Comparison:** Compares Cross Entropy loss and Mean Squared Error loss.
- **Command-Line Interface:** `train.py` supports command-line arguments for configurable training.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/BullsEye268/da6401_assignment1.git
    cd da6401_assignment1
    ```
2.  **Install dependencies:**
    ```bash
    pip install numpy pandas matplotlib seaborn plotly keras wandb
    ```
3.  **Set up Wandb:**
    - Create a Wandb account and log in using `wandb login`.
    - Or, use the python functionality `wandb.login()`

## Usage

1.  **Run the training script:**
    - To train the neural network with default or specified parameters, run the `train.py` script.
    - To run the hyperparameter sweep, use the `sweep_config.yml` configuration file with Wandb.
2.  **Run the final ipynb file:**
    - `final.ipynb` contains the final execution of the code, with cells segregated by question numbers from the assignment.
3.  **Use Wandb:**
    - View experiment results, visualizations, and hyperparameter tuning insights on the Wandb dashboard.
    - Use the link of the [report generated on wandb.ai](https://api.wandb.ai/links/bullseye2608-indian-institute-of-technology-madras/rj51csft) to check the results.
4.  **Plotting images:**
    - To plot the images, use the `utils/wandb_helper.py` file, and the `log_images` function.
5.  **Using train.py with command line arguments:**
    - Run the train.py file with the following command line arguments:
      ```bash
      python train.py --wandb_entity <your_wandb_entity> --wandb_project <your_wandb_project>
      ```
    - You can also specify other arguments to customize the training process. For example:
      ```bash
        python train.py --wandb_entity <your_wandb_entity> --wandb_project <your_wandb_project> --epochs 10 --batch_size 32 --learning_rate 0.001 --optimizer adam
      ```

## Code Specifications

- `train.py`: Accepts command-line arguments to configure and run the training process.
- `neural_network.py`: Implements the `NeuralNetwork` class with forward and backward propagation, loss computation, and training methods.
- `optimizer.py`: Contains implementations of various optimization algorithms (SGD, Momentum, Nesterov, RMSprop, Adam, Nadam).
- `helper_functions.py`: Includes utility functions for data loading, confusion matrix plotting, and other helper tasks.
- `wandb_helper.py`: Manages Wandb integration for experiment tracking and logging.

### `train.py` Command-Line Arguments

- `--wandb_project` / `-wp`: Project name used in Wandb (default: `myprojectname`).
- `--wandb_entity` / `-we`: Wandb entity (default: `myname`).
- `--dataset` / `-d`: Dataset to use (`mnist` or `fashion_mnist`, default: `fashion_mnist`).
- `--epochs` / `-e`: Number of epochs (default: `1`).
- `--batch_size` / `-b`: Batch size (default: `4`).
- `--loss` / `-l`: Loss function (`mean_squared_error` or `cross_entropy`, default: `cross_entropy`).
- `--optimizer` / `-o`: Optimizer (`sgd`, `momentum`, `nag`, `rmsprop`, `adam`, `nadam`, default: `sgd`).
- `--learning_rate` / `-lr`: Learning rate (default: `0.1`).
- `--momentum` / `-m`: Momentum (default: `0.5`).
- `--beta` / `--beta`: Beta for RMSprop (default: `0.5`).
- `--beta1` / `--beta1`: Beta1 for Adam and Nadam (default: `0.5`).
- `--beta2` / `--beta2`: Beta2 for Adam and Nadam (default: `0.5`).
- `--epsilon` / `-eps`: Epsilon for optimizers (default: `0.000001`).
- `--weight_decay` / `-w_d`: Weight decay (default: `0.0`).
- `--weight_init` / `-w_i`: Weight initialization (`random` or `Xavier`, default: `random`).
- `--num_layers` / `-nhl`: Number of hidden layers (default: `1`).
- `--hidden_size` / `-sz`: Number of hidden neurons (default: `4`).
- `--activation` / `-a`: Activation function (`identity`, `sigmoid`, `tanh`, `ReLU`, default: `sigmoid`).

## Folder Structure

- `utils/`: Contains helper functions, neural network implementation, optimizers, and Wandb utilities.
- `final.ipynb`: Jupyter Notebook containing the final implementation and results.
- `README.md`: Project documentation.
- `sweep_config.yml`: Configuration file for Wandb sweeps.
- `train.py`: Python script for training the neural network.
- `trials.ipynb`: Jupyter notebook used for initial experimentation and debugging.

## Author

- Achyutha Munimakula
