import wandb
import numpy as np
import datetime

from .neural_network import NeuralNetwork
from .helper_functions import load_data, get_optimizer

class WandbCallback:
    def __init__(self):
        self.epoch = 0
    
    def on_epoch_end(self, loss, accuracy, val_loss, val_accuracy):
        wandb.log({
            "epoch": self.epoch,
            "train_loss": loss,
            "train_accuracy": accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        })
        self.epoch += 1

class WandbTrainer:
    def __init__(self, dataset_name='fashion_mnist'):
        self.callback = WandbCallback()
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = load_data(dataset_name=dataset_name)
    
    def train(self):
        with wandb.init() as run:
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
        
        return

def log_images(X, y):
    # Initialize a W&B run
    wandb.init(
        entity="bullseye2608-indian-institute-of-technology-madras",
        project="my-awesome-project", 
        name="Images_"+datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
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