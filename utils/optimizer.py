import numpy as np

class Optimizer:
    def __init__(self, W, B, LOG_EACH=False, **kwargs):
        """
        Initialize optimizer with weights and biases shapes.
        
        Parameters:
        W (list): List of weight matrices
        B (list): List of bias vectors
        **kwargs: Optimizer-specific parameters
        """
        self.L = len(W)  # Number of layers
        self.LOG_EACH = LOG_EACH
        self.params = kwargs
    
    def update(self, W, B, dW, dB, iteration):
        """
        Update weights and biases based on gradients.
        
        Parameters:
        W (list): Current weights
        B (list): Current biases
        dW (list): Weight gradients
        dB (list): Bias gradients
        iteration (int): Current iteration number
        
        Returns:
        tuple: (new_W, new_B) updated weights and biases
        """
        raise NotImplementedError("Each optimizer must implement this method")


class SGDOptimizer(Optimizer):
    def __init__(self, W, B, learning_rate=0.01, **kwargs):
        super().__init__(W, B, **kwargs)
        self.learning_rate = learning_rate
    
    def update(self, W, B, dW, dB, iteration):
        if self.LOG_EACH and iteration==0:
            print(f'Running SGDOptimizer {self.learning_rate = }')
        for i in range(self.L):
            W[i] -= self.learning_rate * dW[i]
            B[i] -= self.learning_rate * dB[i]
        return W, B


class MomentumOptimizer(Optimizer):
    def __init__(self, W, B, learning_rate=0.01, momentum=0.9, **kwargs):
        super().__init__(W, B, **kwargs)
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        # Initialize velocity vectors
        self.v_W = [np.zeros_like(W[i]) for i in range(self.L)]
        self.v_B = [np.zeros_like(B[i]) for i in range(self.L)]
    
    def update(self, W, B, dW, dB, iteration):
        if self.LOG_EACH and iteration==0:
            print(f'Running MomentumOptimizer {self.learning_rate = } {self.momentum = }')
        for i in range(self.L):
            # Update velocity
            self.v_W[i] = self.momentum * self.v_W[i] - self.learning_rate * dW[i]
            self.v_B[i] = self.momentum * self.v_B[i] - self.learning_rate * dB[i]
            
            # Update parameters
            W[i] += self.v_W[i]
            B[i] += self.v_B[i]
        
        return W, B


class NesterovOptimizer(Optimizer):
    def __init__(self, W, B, learning_rate=0.01, momentum=0.9, **kwargs):
        super().__init__(W, B, **kwargs)
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        # Initialize velocity vectors
        self.v_W = [np.zeros_like(W[i]) for i in range(self.L)]
        self.v_B = [np.zeros_like(B[i]) for i in range(self.L)]
    
    def update(self, W, B, dW, dB, iteration):
        if self.LOG_EACH and iteration==0:
            print(f'Running NesterovOptimizer {self.learning_rate = } {self.momentum = }')
        W_lookahead = [None] * self.L
        B_lookahead = [None] * self.L
        
        # Calculate lookahead position
        for i in range(self.L):
            W_lookahead[i] = W[i] + self.momentum * self.v_W[i]
            B_lookahead[i] = B[i] + self.momentum * self.v_B[i]
        
        # Update velocity
        for i in range(self.L):
            self.v_W[i] = self.momentum * self.v_W[i] - self.learning_rate * dW[i]
            self.v_B[i] = self.momentum * self.v_B[i] - self.learning_rate * dB[i]
            
            # Update parameters
            W[i] += self.v_W[i]
            B[i] += self.v_B[i]
        
        return W, B


class RMSpropOptimizer(Optimizer):
    def __init__(self, W, B, learning_rate=0.001, decay_rate=0.9, epsilon=1e-8, **kwargs):
        super().__init__(W, B, **kwargs)
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        
        # Initialize cache vectors
        self.cache_W = [np.zeros_like(W[i]) for i in range(self.L)]
        self.cache_B = [np.zeros_like(B[i]) for i in range(self.L)]
    
    def update(self, W, B, dW, dB, iteration):
        if self.LOG_EACH and iteration==0:
            print(f'Running RMSpropOptimizer {self.learning_rate = } {self.decay_rate = } {self.epsilon = }')
        for i in range(self.L):
            # Update cache with squared gradients
            self.cache_W[i] = self.decay_rate * self.cache_W[i] + (1 - self.decay_rate) * np.square(dW[i])
            self.cache_B[i] = self.decay_rate * self.cache_B[i] + (1 - self.decay_rate) * np.square(dB[i])
            
            # Update parameters
            W[i] -= self.learning_rate * dW[i] / (np.sqrt(self.cache_W[i]) + self.epsilon)
            B[i] -= self.learning_rate * dB[i] / (np.sqrt(self.cache_B[i]) + self.epsilon)
        
        return W, B


class AdamOptimizer(Optimizer):
    def __init__(self, W, B, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, **kwargs):
        super().__init__(W, B, **kwargs)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # Initialize moment vectors
        self.m_W = [np.zeros_like(W[i]) for i in range(self.L)]
        self.m_B = [np.zeros_like(B[i]) for i in range(self.L)]
        
        # Initialize velocity vectors
        self.v_W = [np.zeros_like(W[i]) for i in range(self.L)]
        self.v_B = [np.zeros_like(B[i]) for i in range(self.L)]
    
    def update(self, W, B, dW, dB, iteration):
        if self.LOG_EACH and iteration==0:
            print(f'Running AdamOptimizer {self.learning_rate = } {self.beta1 = } {self.beta2 = } {self.epsilon = }')
        t = iteration + 1  # Timestep starts at 1
        
        for i in range(self.L):
            # Update biased first and second moment estimates
            self.m_W[i] = self.beta1 * self.m_W[i] + (1 - self.beta1) * dW[i]
            self.m_B[i] = self.beta1 * self.m_B[i] + (1 - self.beta1) * dB[i]
            
            self.v_W[i] = self.beta2 * self.v_W[i] + (1 - self.beta2) * np.square(dW[i])
            self.v_B[i] = self.beta2 * self.v_B[i] + (1 - self.beta2) * np.square(dB[i])
            
            # Compute bias-corrected first and second moment estimates
            m_W_corrected = self.m_W[i] / (1 - self.beta1**t)
            m_B_corrected = self.m_B[i] / (1 - self.beta1**t)
            
            v_W_corrected = self.v_W[i] / (1 - self.beta2**t)
            v_B_corrected = self.v_B[i] / (1 - self.beta2**t)
            
            # Update parameters
            W[i] -= self.learning_rate * m_W_corrected / (np.sqrt(v_W_corrected) + self.epsilon)
            B[i] -= self.learning_rate * m_B_corrected / (np.sqrt(v_B_corrected) + self.epsilon)
        
        return W, B


class NadamOptimizer(Optimizer):
    def __init__(self, W, B, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, **kwargs):
        super().__init__(W, B, **kwargs)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # Initialize moment vectors
        self.m_W = [np.zeros_like(W[i]) for i in range(self.L)]
        self.m_B = [np.zeros_like(B[i]) for i in range(self.L)]
        
        # Initialize velocity vectors
        self.v_W = [np.zeros_like(W[i]) for i in range(self.L)]
        self.v_B = [np.zeros_like(B[i]) for i in range(self.L)]
    
    def update(self, W, B, dW, dB, iteration):
        if self.LOG_EACH and iteration==0:
            print(f'Running NadamOptimizer {self.learning_rate = } {self.beta1 = } {self.beta2 = } {self.epsilon = }')
        t = iteration + 1  # Timestep starts at 1
        
        for i in range(self.L):
            # Update biased first and second moment estimates
            self.m_W[i] = self.beta1 * self.m_W[i] + (1 - self.beta1) * dW[i]
            self.m_B[i] = self.beta1 * self.m_B[i] + (1 - self.beta1) * dB[i]
            
            self.v_W[i] = self.beta2 * self.v_W[i] + (1 - self.beta2) * np.square(dW[i])
            self.v_B[i] = self.beta2 * self.v_B[i] + (1 - self.beta2) * np.square(dB[i])
            
            # Compute bias-corrected first and second moment estimates
            m_W_corrected = self.m_W[i] / (1 - self.beta1**t)
            m_B_corrected = self.m_B[i] / (1 - self.beta1**t)
            
            v_W_corrected = self.v_W[i] / (1 - self.beta2**t)
            v_B_corrected = self.v_B[i] / (1 - self.beta2**t)
            
            # Apply Nesterov momentum to first moment estimate
            m_W_nesterov = self.beta1 * m_W_corrected + (1 - self.beta1) * dW[i] / (1 - self.beta1**t)
            m_B_nesterov = self.beta1 * m_B_corrected + (1 - self.beta1) * dB[i] / (1 - self.beta1**t)
            
            # Update parameters
            W[i] -= self.learning_rate * m_W_nesterov / (np.sqrt(v_W_corrected) + self.epsilon)
            B[i] -= self.learning_rate * m_B_nesterov / (np.sqrt(v_B_corrected) + self.epsilon)
        
        return W, B
    

# A new optimizer can be added by following these steps:
# 1. Create a new class that inherits from the Optimizer class
# 2. Implement the update method to update the parameters
# 3. Add the new optimizer to the optimizer_map dictionary in `utils.neural_network.set_optimiser`