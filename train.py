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
        log_every=900,
        callback=wandb_callback  # Assuming your NeuralNetwork class supports callbacks
    )
    
    wandb.log({'test_accuracy': nn.compute_accuracy(X_test, y_test)})
    

    # Mark the end of the wandb run
    wandb.finish()

if __name__ == '__main__':
    main()

    