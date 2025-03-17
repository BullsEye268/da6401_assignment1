import wandb
import numpy as np
import datetime
from IPython.display import clear_output

from .neural_network import NeuralNetwork
from .helper_functions import load_data, get_optimizer

class WandbCallback:
    def __init__(self):
        """Initialize the WandbCallback class
        
        Input format: None
        
        Output format: None (sets up callback instance with epoch counter)"""
        self.epoch = 0
    
    def on_epoch_end(self, loss, accuracy, val_loss, val_accuracy):
        """Log training metrics to Weights & Biases at the end of each epoch
        
        Input format:
        - loss: float (training loss)
        - accuracy: float (training accuracy)
        - val_loss: float (validation loss)
        - val_accuracy: float (validation accuracy)
        
        Output format: None (logs metrics to W&B)"""
        if wandb.run is not None:
            wandb.log({
                "epoch": self.epoch,
                "train_loss": loss,
                "train_accuracy": accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            })
            self.epoch += 1
            # wandb.log(data)
        else:
            raise ValueError("Warning: Attempted to log to wandb before initialization")

class WandbTrainer:
    def __init__(self, dataset_name='fashion_mnist'):
        """Initialize the WandbTrainer class with dataset and callback
        
        Input format:
        - dataset_name: str (name of dataset to load, default='fashion_mnist')
        
        Output format: None (sets up trainer with data and callback)"""
        self.callback = WandbCallback()
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = load_data(dataset_name=dataset_name)
        self.group = f"sweep-{datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S')}"
    
    def train(self):
        """Train a neural network with W&B logging
        
        Input format: None (uses instance attributes and W&B config)
        
        Output format: None (trains model and logs results to W&B)"""
        with wandb.init(group=self.group, tags=["sweep"]) as run:
            run_name = f"hl:{wandb.config.hidden_layers}_hs:{wandb.config.hidden_size}_bs:{wandb.config.batch_size}_act:{wandb.config.activation}"
            # print('run name is supposed to be ', run_name, run.name)
            run.name = run_name
            # project='PH21B004_DA6401-Assignment-1',
            config = wandb.config
            
            
            layer_sizes = [784] + [config.hidden_size]*config.hidden_layers + [10]
            activation_functions = [config.activation]*config.hidden_layers + ['softmax']
            
            nn = NeuralNetwork(layer_sizes=layer_sizes, 
                            activation_functions=activation_functions,
                            weight_init=config.weight_init, 
                            weight_decay=config.weight_decay)
            
            wandb_callback = WandbCallback()
            
            optimizer = get_optimizer(config.optimizer, config.learning_rate)
            nn.set_optimizer(optimizer)
            
            nn.train(
                self.X_train, 
                self.y_train, 
                self.X_val, 
                self.y_val, 
                batch_size=config.batch_size, 
                num_epochs=config.epochs, 
                loss_type='cross_entropy', 
                log_every=1000,
                callback=wandb_callback
            )
            
            test_accuracy = nn.compute_accuracy(self.X_test, self.y_test)
            wandb.log({"test_accuracy": test_accuracy})
            clear_output(wait=True)
        
        return

def log_images(X, y, entity, project):
    """Log sample images from dataset to Weights & Biases
    
    Input format:
    - X: numpy.ndarray (image data, shape (n_samples, n_features))
    - y: numpy.ndarray (labels, shape (n_samples,))
    - entity: str (W&B entity name)
    - project: str (W&B project name)
    
    Output format: None (logs images to W&B)"""
    # Initialize a W&B run
    wandb.init(
        entity=entity,
        project=project, 
        name="Images_"+datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        tags=['image_examples']
    )
    
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # Log images to W&B
    wandb.log({
        "fashion_mnist_samples": [
            wandb.Image(X[np.where(y == i)[0][0]].reshape(28,28), caption=class_names[i])
            for i in range(10)
        ]
    })

    wandb.finish()

def log_plots(plots_to_be_added, entity, project):
    run = wandb.init(entity=entity, 
                     project=project, 
                     name="Plots_"+datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                     tags=['Plot Upload'])
    
    for title, path_to_plot in plots_to_be_added:
        wandb.log({title: wandb.Image(path_to_plot)})
    
    wandb.finish()
    clear_output(wait=True)
    return True
    