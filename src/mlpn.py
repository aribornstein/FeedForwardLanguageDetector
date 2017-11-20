"""
Written by Ari Bornstein
"""
import numpy as np
import loglinear as ll
import mlp1

def params_to_layers(params):
    """
    Generates a list of layers (wieght bias pairs) from the input function
    """
    return [tuple(params[i: i + 2]) for i in xrange(0, len(params), 2)]

def evaluate_layer(x, layers, index):
    """
    Evalutes layer at given index
    """
    probs = np.array(x)
    for W, b in layers[:index]:
        probs = np.tanh(np.dot(probs, W) + b)
    return probs

def classifier_output(x, params):
    # create pairs of parameters (W, b)
    layers = params_to_layers(params)
    # pass x through all  hidden layers (all but the last layer)
    x = evaluate_layer(x, layers, len(layers)-1)
    # last layer is the loglinear layer
    W, b = layers[-1]  
    probs = ll.classifier_output(x, [W,b])
    return probs

def predict(x, params):
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params):
    # Forward prop
    y_pred = classifier_output(x, params)

    # y_hot encode our y values
    y_hot = np.zeros(y_pred.size)
    y_hot[y] = 1
    
    # calculate loss
    loss = -np.sum(y_hot * np.log(y_pred))
    
    # calculate the gradients
    layers = params_to_layers(params)
    num_layers = len(layers)
    layer_out = evaluate_layer(x, layers, num_layers-1)
    
    # calculate gradients
    grads = []
    
    # gradient for the last layer
    W, b = layers[-1]
    gb  = y_pred - y_hot
    gW = np.outer(layer_out, y_pred) - np.outer(layer_out, y_hot)
    grads.append(gb) 
    grads.append(gW) 
    
    # We calculate our gradients in reverse order
    reversed_layers = layers[:-1]
    reversed_layers.reverse() # this reverse the layers

    curr_dt = -W[:, y] + np.dot(W, y_pred) # last layer used for chain rule
    layer_index = len(layers) - 2
    for W, b in reversed_layers:
        # Evaluate layer
        layer_in = evaluate_layer(x, layers, layer_index)
        # Calculate layer gradients using chain rule
        gb = 1 - (np.tanh(np.dot(layer_in, W) + b)) ** 2
        gW = np.dot(layer_in.reshape(len(layer_in), 1), gb.reshape(1, len(gb)))
        grads.append(curr_dt * gb) 
        grads.append(curr_dt * gW)
        dt_dprevt = gb * W
        curr_dt = np.dot(dt_dprevt, curr_dt) # stored dt for next layer chain rule
        layer_index -= 1

    grads.reverse()
    return loss, grads


def create_classifier(dims):
    """
    returns the parameters for a multi-layer perceptron with an arbitrary number
    of hidden layers.
    dims is a list of length at least 2, where the first item is the input
    dimension, the last item is the output dimension, and the ones in between
    are the hidden layers.
    For example, for:
        dims = [300, 20, 30, 40, 5]
    We will have input of 300 dimension, a hidden layer of 20 dimension, passed
    to a layer of 30 dimensions, passed to learn of 40 dimensions, and finally
    an output of 5 dimensions.
    
    Assume a tanh activation function between all the layers.
    return:
    a list of parameters where the first two elements are the W and b from input
    to first layer, then the second two are the matrix and vector from first to
    second layer, and so on.
    """
    params = []
    params_ari = []
    for i in range(len(dims) - 1):
        # create and append weights for the input and output dimensions
        W,b = ll.create_classifier(dims[i], dims[i + 1])
        params.append(W)
        params.append(b)

    return params


if __name__ == '__main__':
    # Sanity checks. If these fail, your gradient calculation is definitely wrong.
    # If they pass, it is likely, but not certainly, correct.
    from grad_check import gradient_check

    dims = [30, 2, 3, 4, 5]
    params = create_classifier(dims)

    def _loss_and_p_grad(p):
        """
        General function - return loss and the gradients with respect to parameter p
        """
        params_to_send = np.copy(params)
        par_num = 0
        for i in range(len(params)):
            if p.shape == params[i].shape:
                params_to_send[i] = p
                par_num = i
        loss, grads = loss_and_gradients(range(dims[0]), 0, params_to_send)
        return loss, grads[par_num]

    for _ in xrange(10):
        my_params = create_classifier(dims)
        for p in my_params:
            gradient_check(_loss_and_p_grad, p)
