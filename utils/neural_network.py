import numpy as np
import matplotlib.pyplot as plt

from .optimizer import SGDOptimizer, MomentumOptimizer, NesterovOptimizer, RMSpropOptimizer, AdamOptimizer, NadamOptimizer
from .helper_functions import get_optimizer


class NeuralNetwork:
    def __init__(self, layer_sizes=[784, 17, 19, 10], 
                 activation_functions=['sigmoid', 'sigmoid', 'softmax'], 
                 weight_decay=0.0, weight_init='random', LOG_EACH=False):
        assert len(layer_sizes) == len(activation_functions) + 1, "Number of layers (excluding input layer) and activations must match"
        self.layer_sizes = layer_sizes
        self.L = len(layer_sizes) - 1
        self.activation_functions = activation_functions
        self.LOG_EACH = LOG_EACH
        
        if weight_init.lower() == 'random':
            self.W = [np.random.uniform(-0.5, 0.5, (self.layer_sizes[i], self.layer_sizes[i - 1])) for i in range(1, len(self.layer_sizes))]
            self.B = [np.zeros(self.layer_sizes[i]).reshape(1,-1) for i in range(1, len(self.layer_sizes))]
        elif weight_init.lower() == 'xavier':
            # Xavier/Glorot initialization for weights
            self.W = [np.random.randn(self.layer_sizes[i], self.layer_sizes[i - 1]) * 
                    np.sqrt(2.0 / (self.layer_sizes[i] + self.layer_sizes[i - 1])) 
                    for i in range(1, len(self.layer_sizes))]
            self.B = [np.zeros(self.layer_sizes[i]).reshape(1,-1) for i in range(1, len(self.layer_sizes))]
        else:
            raise ValueError(f"Unsupported weight initialization: {weight_init}")
        
        self.weight_decay = weight_decay
        
        self.optimizer = None
    
    def set_optimizer(self, optimizer_dict):
        optimizer_map = {
            'sgd': SGDOptimizer,
            'momentum': MomentumOptimizer,
            'nesterov': NesterovOptimizer,
            'rmsprop': RMSpropOptimizer,
            'adam': AdamOptimizer,
            'nadam': NadamOptimizer
        }

        if optimizer_dict['name'].lower() not in optimizer_map:
            raise ValueError(f"Unsupported optimizer: {optimizer_dict['name']}")
        
        optimizer_dict.update({'LOG_EACH': self.LOG_EACH})
        
        self.optimizer = optimizer_map[optimizer_dict['name'].lower()](self.W, self.B, **optimizer_dict)
    
    def activate(self, A, activation):
        if activation.lower() == 'sigmoid':
            return 1 / (1 + np.exp(-A))
        elif activation.lower() == 'relu':
            return np.maximum(0, A)
        elif activation.lower() == 'tanh':
            return np.tanh(A)
        elif activation.lower() == 'softmax':
            # For numerical stability, subtract max value
            exps = np.exp(A - np.max(A, axis=-1, keepdims=True))
            return exps / np.sum(exps, axis=-1, keepdims=True)
        elif activation.lower() == 'identity':
            return A
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
    
    def _activate_derivative(self, A, H, activation):
        """Calculate derivative of activation function"""
        if activation.lower() == 'sigmoid':
            return H * (1 - H)
        elif activation.lower() == 'relu':
            return (A > 0).astype(float)
        elif activation.lower() == 'tanh':
            return 1 - H**2
        elif activation.lower() == 'identity':
            return 1
        elif activation.lower() == 'softmax':            
            return 1
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    
    def forward_propagation(self, X):
        H = [X]
        A = []
        
        for i in range(self.L):
            A.append(np.dot(H[i], self.W[i].T) + self.B[i])
            H.append(self.activate(A[i], self.activation_functions[i]))
        
        return H, A
    
    def compute_loss(self, H_final, y, loss_type='cross_entropy'):
        if y.ndim == 1:
            y = self.one_hot(y)
        m = y.shape[0]  # Number of examples
        
        if loss_type.lower() == 'cross_entropy':
            # Add small epsilon to avoid log(0)
            epsilon = 1e-15
            loss = -np.sum(y * np.log(H_final + epsilon)) / m
        elif loss_type.lower() == 'mse' or loss_type.lower() == 'mean_squared_error':
            loss = np.sum((H_final - y)**2) / (2 * m)
        else:
            raise ValueError(f"Unsupported loss function: {loss_type}")
            
        return loss
    
    def _loss_derivative(self, y_pred, y, loss_type):
        if loss_type.lower() == 'cross_entropy':
            epsilon = 1e-15
            return -y / (y_pred + epsilon)
        elif loss_type.lower() == 'mse' or loss_type.lower() == 'mean_squared_error':
            if y.ndim == 1:
                y = self.one_hot(y)
            m = y.shape[0]  # Number of examples
            return (y_pred - y) / m  # Divide by m for proper scaling
        else:
            raise ValueError(f"Unsupported loss function: {loss_type}")
    
    def back_propagation(self, X, y, H, A, loss_type='cross_entropy'):
        assert len(H) == self.L + 1 and len(A) == self.L
        N = X.shape[0]
        assert N==y.size and self.L==len(A) and self.L + 1==len(H)
        
        dW, dB = [None] * self.L, [None] * self.L
        
        if loss_type.lower() == 'cross_entropy' and self.activation_functions[-1].lower() == 'softmax':
            # Gradient simplifies when using softmax + cross entropy
            dA = H[-1] - self.one_hot(y)
        else:
            dH = self._loss_derivative(H[-1], y, loss_type)
            dA = dH * self._activate_derivative(A[-1], H[-1], self.activation_functions[-1])
            
        dW[-1] = np.dot(dA.T, H[-2]) / N
        dB[-1] = np.sum(dA, axis=0, keepdims=True) / N
        
        for k in range(self.L-2, -1, -1):
            
            dA = np.dot(dA, self.W[k+1]) * self._activate_derivative(A[k], H[k+1], self.activation_functions[k])
            
            dWk = (np.dot(dA.T, H[k])) / N
            dBk = np.sum(dA, axis=0, keepdims=True) / N
            dW[k] = dWk
            dB[k] = dBk
        
        if self.weight_decay > 0:
            for i in range(len(self.W)):
                dW[i] += self.weight_decay * self.W[i]
               
        return dW, dB
    
    def one_hot(self, y):
        one_hot_y = np.zeros((y.size, self.layer_sizes[-1]))
        one_hot_y[np.arange(y.size), y] = 1
        return one_hot_y
    
    def plot_history(self, history):
        plt.figure(figsize=(12, 6))
        plt.plot(history['train_loss'], label='Training Loss')
        if 'val_loss' in history:
            plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
    
    def predict(self, X):
        H, _ = self.forward_propagation(X)
        return H[-1]
    
    def compute_accuracy_from_predictions(self, y_pred, y):
        if y_pred.ndim != 1:
            y_pred = np.argmax(y_pred, axis=1)
        return np.mean(y_pred == y)
    
    def compute_accuracy(self, X, y):
        y_pred = np.argmax(self.predict(X), axis=1)
        return np.mean(y_pred == y)
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              batch_size=64, num_epochs=10, loss_type='cross_entropy', 
              log_every=100, callback=None):
        
        if self.optimizer is None:
            self.set_optimizer({'name':'sgd', 'learning_rate':0.01})
        
        num_datapoints = X_train.shape[0]
        num_batches = int(np.ceil(num_datapoints / batch_size))
        
        spacer_1 = int(np.log10(num_epochs)+1)
        spacer_2 = int(np.log10(num_batches)+1)
        
        history = {
            'train_loss' : [],
            'train_acc' : [],
            'val_loss' : [] if X_val is not None else None,
            'val_acc' : [] if X_val is not None else None
        }
        
        iteration = 0
        
        for epoch in range(num_epochs):
            
            permutation = np.random.permutation(num_datapoints)
            X_train = X_train[permutation]
            y_train = y_train[permutation]
            
            for batch in range(num_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, num_datapoints)
                X_batch = X_train[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]
                
                H, A = self.forward_propagation(X_batch)
                dW, dB = self.back_propagation(X_batch, y_batch, H, A, loss_type)
                
                self.W, self.B = self.optimizer.update(self.W, self.B, dW, dB, iteration)
                
                if iteration % log_every == 0:
                    
                    train_loss = self.compute_loss(H[-1], y_batch, loss_type)
                    if X_val is not None and y_val is not None:
                        val_loss = self.compute_loss(self.predict(X_val), y_val, loss_type)
                        if self.LOG_EACH:    print(f"Epoch {epoch+1 :>{spacer_1}}/{num_epochs}, Iteration {iteration%num_batches :>{spacer_2}}/{num_batches} --> Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}")
                    else:
                        if self.LOG_EACH:    print(f"Epoch {epoch+1 :>{spacer_1}}/{num_epochs}, Iteration {iteration%num_batches :>{spacer_2}}/{num_batches} --> Train Loss: {train_loss:.5f}")
                
                iteration += 1
            
            train_loss = self.compute_loss(H[-1], y_batch, loss_type)
            train_acc = self.compute_accuracy(X_batch, y_batch)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            val_loss = self.compute_loss(self.predict(X_val), y_val, loss_type)
            val_acc = self.compute_accuracy(X_val, y_val)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            if callback is not None:
                callback.on_epoch_end(train_loss, self.compute_accuracy(X_train, y_train), 
                                      self.compute_loss(self.predict(X_val), y_val, loss_type), 
                                      self.compute_accuracy(X_val, y_val))
                
        return history
    
    
    

def nn_from_config(config, wandb_callback, X_train, y_train, X_val, y_val):
    layer_sizes = [784] + [config.hidden_size]*config.num_layers + [10]
    activation_functions = [config.activation]*config.num_layers + ['softmax']
    
    nn = NeuralNetwork(layer_sizes=layer_sizes, 
                    activation_functions=activation_functions,
                    weight_init=config.weight_init, 
                    weight_decay=config.weight_decay)
    
    optimizer = get_optimizer(config.optimizer, config.learning_rate)
    nn.set_optimizer(optimizer)
    
    history = nn.train(
        X_train,
        y_train,
        X_val,
        y_val,
        batch_size=config.batch_size,
        num_epochs=config.epochs,
        loss_type=config.loss,
        log_every=1000,
        callback=wandb_callback
    )
    
    return nn, history