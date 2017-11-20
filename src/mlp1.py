"""
Written by Ari Bornstein
"""

import numpy as np
import loglinear as ll
           
def classifier_output(x, params):
    U, W, b, b_prime = params # Extract paramaters
    tan_h = np.tanh(np.dot(x, W) + b) # Hidden layer and activiation
    probs = ll.classifier_output(tan_h, [U, b_prime]) # Softmax layer 
    return probs

def predict(x, params):
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params):
    """
        Compute the loss and the gradients at point x with given parameters.
        y is a scalar indicating the correct label.
        returns:
            loss,[gU, gW, gb, gb_prime]
        loss: scalar
        gU: matrix, gradients of U
        gW: matrix, gradients of W
        gb: vector, gradients of b
        gb_prime: vector, gradients of b_prime
    """
    U, W, b, b_prime = params

    # forward prop
    y_pred = classifier_output(x, params)

    # y_hot encode our y values
    y_hot = np.zeros(y_pred.size)
    y_hot[y] = 1

    # loss
    loss = -np.sum(y_hot * np.log(y_pred))
    
    # hidden layer
    tan_h = np.tanh(np.dot(x, W) + b)
    h_probs = ll.classifier_output(tan_h, [U, b_prime]) 
    
    # gradient of U and b_prime
    gb_prime = h_probs - y_hot    
    gU = np.outer(tan_h, h_probs) - np.outer(tan_h, y_hot)

    # Use chain rule to find gradients of W and b
    dloss_dtanh = -U[:, y] + np.dot(U, h_probs)
    dtanh_db = 1 - tan_h ** 2
    gb = dloss_dtanh * dtanh_db 
    gW = np.outer(x, dtanh_db * dloss_dtanh)

    return loss, [gU, gW, gb, gb_prime]

def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.
    """
    W,b = ll.create_classifier(in_dim, hid_dim)     # Hidden layer    
    U,b_prime = ll.create_classifier(hid_dim, out_dim)    # Output layer
    params = [U, W, b, b_prime]
    return params

if __name__ == '__main__':
    # Sanity checks. If these fail, your gradient calculation is definitely wrong.
    # If they pass, it is likely, but not certainly, correct.
    from grad_check import gradient_check

    U, W, b, b_prime = create_classifier(3, 2, 4)

    def _loss_and_U_grad(U):
        loss, grads = loss_and_gradients([1, 2, 3], 0, [U, W, b, b_prime])
        return loss, grads[0]

    def _loss_and_W_grad(W):
        global b
        loss, grads = loss_and_gradients([1, 2, 3], 0, [U, W, b, b_prime])
        return loss, grads[1]

    def _loss_and_b_grad(b):
        global W
        loss, grads = loss_and_gradients([1, 2, 3], 0, [U, W, b, b_prime])
        return loss, grads[2]

    def _loss_and_bprime_grad(b_prime):
        loss, grads = loss_and_gradients([1, 2, 3], 0, [U, W, b, b_prime])
        return loss, grads[3]

    for _ in xrange(10):
        W = np.random.randn(W.shape[0], W.shape[1])
        b = np.random.randn(b.shape[0])
        U = np.random.randn(U.shape[0], U.shape[1])
        b_prime = np.random.randn(b_prime.shape[0])
        loss, grads = loss_and_gradients([1, 2, 3], 0, [U, W, b, b_prime])
        gradient_check(_loss_and_U_grad, U)
        gradient_check(_loss_and_W_grad, W)
        gradient_check(_loss_and_b_grad, b)
        gradient_check(_loss_and_bprime_grad, b_prime)