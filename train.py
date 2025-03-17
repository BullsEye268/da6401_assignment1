import wandb
import argparse

from utils import neural_network, wandb_helper, helper_functions


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a neural network and log experiments to Weights & Biases"
    )
    parser.add_argument(
        "-wp", "--wandb_project",
        default="PH21B004_DA6401-Assignment-1",
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

def main():
    args = parse_args()

    if args.dataset == 'mnist':
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f'MNIST {helper_functions.run_name_generator(args)}',
            tags=['individual_runs', 'MNIST'],
            config=vars(args)
        )
    else:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f'{helper_functions.run_name_generator(args)}',
            tags=['individual_runs'],
            config=vars(args)
        )
        
    
    X_train, y_train, X_val, y_val, X_test, y_test = helper_functions.load_data(dataset_name=args.dataset)
    
    
    nn = neural_network.NeuralNetwork(layer_sizes=[784] + [args.hidden_size]*args.num_layers + [10],
                       activation_functions=[args.activation]*args.num_layers + ['softmax'],
                       weight_init=args.weight_init,
                       weight_decay=args.weight_decay,
                       LOG_EACH=True)
    
    optimizer = helper_functions.get_optimizer(args.optimizer, args.learning_rate)
    nn.set_optimizer(optimizer)
    
    wandb_callback = wandb_helper.WandbCallback()
    
    history = nn.train(
        X_train, 
        y_train, 
        X_val, 
        y_val, 
        batch_size=args.batch_size, 
        num_epochs=args.epochs, 
        loss_type=args.loss,
        log_every=900,
        callback=wandb_callback 
    )
    
    wandb.log({'test_accuracy': nn.compute_accuracy(X_test, y_test)})
    
    wandb.finish()

if __name__ == '__main__':
    main()

    