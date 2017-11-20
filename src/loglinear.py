"""
Written by Ari Bornstein
"""

import numpy as np

def softmax(x):
    """
    Compute the softmax vector.
    x: a n-dim vector (numpy array)
    returns: an n-dim vector (numpy array) of softmax values
    """
    # Your code should be fast, so use a vectorized implementation using numpy,
    # don't use any loops.
    # With a vectorized implementation, the code should be no more than 2 lines.
    #
    # For numeric stability, use the identify you proved in Ex 2 Q1.
    
    # Substracting np.max(x) from the softmax so that our float doesn't overflow 
    # on large values of x use the axis=0 to ensure we can mutiply correctly against matricies 

    e_x = np.exp(x - np.max(x)) 
    return e_x / e_x.sum(axis=0)
        

def classifier_output(x, params):
    """
    Return the output layer (class probabilities) 
    of a log-linear classifier with given params on input x.
    """
    W,b = params
    probs = softmax(np.dot(x, W) + b)
    return probs

def predict(x, params):
    """
    Returnss the prediction (highest scoring class id) of a
    a log-linear classifier with given parameters on input x.
    """
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params):
    """
    Compute the loss and the gradients at point x with given parameters.
    y is a scalar indicating the correct label.

    returns:
        loss,[gW,gb]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    """
    W,b = params
    # one hot encode y 
    np_x = np.array(x)
    #forward prop
    y_pred = classifier_output(np_x, params)
    #hot enode y 
    y_hot = np.zeros(y_pred.size)
    y_hot[y] = 1
    #calculate loss
    loss = -np.sum(y_hot * np.log(y_pred))
    gb = y_pred - y_hot
    gW = np.outer(np_x, gb)
    
    return loss,[gW,gb]

def glorot(n, m):
    """
    Xavier Glorot distribution from the nlp book for initializing weights and biases   
    """
    return np.random.uniform(-np.sqrt(6.0 / (n + m)),
                             np.sqrt(6.0 / (n + m)),
                             (n, m) if (n != 1 and m != 1) else n * m)

def create_classifier(in_dim, out_dim):
    """
    returns the parameters (W,b) for a log-linear classifier
    with input dimension in_dim and output dimension out_dim.
    """
    W = glorot(in_dim, out_dim)
    b = glorot(1, out_dim)

    return [W,b]

if __name__ == '__main__':
    # Sanity checks for softmax. If these fail, your softmax is definitely wrong.
    # If these pass, it may or may not be correct.
    test1 = softmax(np.array([1,2]))
    print test1
    assert np.amax(np.fabs(test1 - np.array([0.26894142,  0.73105858]))) <= 1e-6

    test2 = softmax(np.array([1001,1002]))
    print test2
    assert np.amax(np.fabs(test2 - np.array( [0.26894142, 0.73105858]))) <= 1e-6

    test3 = softmax(np.array([-1001,-1002])) 
    print test3 
    assert np.amax(np.fabs(test3 - np.array([0.73105858, 0.26894142]))) <= 1e-6

    # Sanity checks. If these fail, your gradient calculation is definitely wrong.
    # If they pass, it is likely, but not certainly, correct.
    from grad_check import gradient_check

    W,b = create_classifier(3,4)

    def _loss_and_W_grad(W):
        global b
        loss,grads = loss_and_gradients([1,2,3],0,[W,b])
        return loss,grads[0]

    def _loss_and_b_grad(b):
        global W
        loss,grads = loss_and_gradients([1,2,3],0,[W,b])
        return loss,grads[1]

    for _ in xrange(10):
        W = np.random.randn(W.shape[0],W.shape[1])
        b = np.random.randn(b.shape[0])
        gradient_check(_loss_and_b_grad, b)
        gradient_check(_loss_and_W_grad, W)
