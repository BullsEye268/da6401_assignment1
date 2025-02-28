import numpy as np

def back_propagation(h, a, W, y, y_hat):
    """
    Perform backpropagation through a neural network exactly as shown in the image.
    
    Args:
        h: List of hidden layer activations (h_1, h_2, ..., h_{L-1})
        a: List of pre-activations (a_1, a_2, ..., a_L)
        W: List of weight matrices
        y: True output
        y_hat: Predicted output
        
    Returns:
        gradients: Dictionary containing gradients for weights and biases
    """
    L = len(h)  # Number of layers (including input layer)
    gradients = {}
    
    # Initialize lists to store gradients
    grad_W = [None] * (L - 1)
    grad_b = [None] * (L - 1)
    grad_h = [None] * L
    grad_a = [None] * L
    
    # Compute output gradient: ∇_{a_L}ℒ(θ) = -(e(y) - ŷ);
    # Here e(y) is the one-hot encoding of y, but simplified as just y for regression
    grad_a[L-1] = -(y - y_hat)
    
    # Ensure gradients are properly shaped as column vectors if needed
    if grad_a[L-1].ndim == 1:
        grad_a[L-1] = grad_a[L-1].reshape(-1, 1)
    
    # Loop from L to 1 (backward)
    for k in range(L-1, 0, -1):
        # Ensure proper shapes for matrix operations
        if h[k-1].ndim == 1:
            h_reshaped = h[k-1].reshape(-1, 1)
        else:
            h_reshaped = h[k-1]
            
        if grad_a[k].ndim == 1:
            grad_a_reshaped = grad_a[k].reshape(-1, 1)
        else:
            grad_a_reshaped = grad_a[k]
        
        # Compute gradients w.r.t. parameters
        # ∇_{W_k}ℒ(θ) = ∇_{a_k}ℒ(θ)h^T_{k-1};
        grad_W[k-1] = np.dot(grad_a_reshaped, h_reshaped.T)
        
        # ∇_{b_k}ℒ(θ) = ∇_{a_k}ℒ(θ);
        grad_b[k-1] = grad_a_reshaped
        
        # Compute gradients w.r.t. layer below
        # ∇_{h_{k-1}}ℒ(θ) = W^T_k∇_{a_k}ℒ(θ);
        grad_h[k-1] = np.dot(W[k-1].T, grad_a_reshaped)
        
        # Compute gradients w.r.t. layer below (pre-activation)
        if k > 1:
            # ∇_{a_{k-1}}ℒ(θ) = ∇_{h_{k-1}}ℒ(θ) ⊙ [..., g'(a_{k-1,j}), ...];
            # Element-wise multiplication with the derivative of the activation function
            if a[k-1].ndim == 1:
                a_reshaped = a[k-1].reshape(-1, 1)
            else:
                a_reshaped = a[k-1]
                
            # For simplicity, assuming derivative of activation function can be computed from a
            activation_deriv = activation_derivative(a_reshaped)
            grad_a[k-1] = grad_h[k-1] * activation_deriv
    
    # Store gradients in dictionary
    gradients["W"] = grad_W
    gradients["b"] = grad_b
    
    return gradients

def activation_derivative(x, activation_type='sigmoid'):
    """
    Compute the derivative of the activation function.
    
    Args:
        x: Input value
        activation_type: Type of activation function (sigmoid, relu, tanh)
        
    Returns:
        Derivative of the activation function at x
    """
    if activation_type == 'sigmoid':
        sigmoid_x = 1 / (1 + np.exp(-x))
        return sigmoid_x * (1 - sigmoid_x)
    elif activation_type == 'relu':
        return (x > 0).astype(float)
    elif activation_type == 'tanh':
        return 1 - np.tanh(x)**2
    else:
        raise ValueError(f"Activation function {activation_type} not recognized")

# Example usage
if __name__ == "__main__":
    # Define a simple network
    np.random.seed(42)
    
    # Network with 3 layers: input(size 2) -> hidden(size 3) -> output(size 1)
    h = [np.array([0.5, 0.8]), np.array([0.2, 0.3, 0.4])]  # h_0 (input), h_1 (hidden)
    a = [None, np.array([0.1, 0.2, 0.3]), np.array([0.6])]  # a_1 (hidden), a_2 (output)
    W = [np.random.randn(3, 2), np.random.randn(1, 3)]  # W_1, W_2
    
    y = np.array([1.0])  # True output
    y_hat = np.array([0.6])  # Predicted output
    
    # Run backpropagation
    gradients = back_propagation(h, a, W, y, y_hat)
    
    # Print results
    print("Gradient for output layer weights (W_2):", gradients["W"][1])
    print("Gradient for output layer bias (b_2):", gradients["b"][1])
    print("Gradient for hidden layer weights (W_1):", gradients["W"][0])
    print("Gradient for hidden layer bias (b_1):", gradients["b"][0])