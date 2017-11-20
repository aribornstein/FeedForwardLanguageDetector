"""
Written by Ari Bornstein
"""

import mlpn
import random
import utils
import pickle

def accuracy_on_dataset(dataset, params):
    # in case of no data set, like xor
    if not dataset:
        return 0

    good = bad = 0.0
    for label, features in dataset:
        y_prediction = mlpn.predict(features, params)
        if y_prediction == label:
            good += 1
        else:
            bad += 1
    return good / (good + bad)


def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.
    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    for I in xrange(num_iterations):
        cum_loss = 0.0  # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = features  # numpy vector.
            y = label  # a number.
            loss, grads = mlpn.loss_and_gradients(x, y, params)
            cum_loss += loss

            # SGD update parameters
            for i in range(0, len(params)):
                params[i] -= learning_rate * grads[i]

        # notify progress
        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print I, train_loss, train_accuracy, dev_accuracy
    return params


if __name__ == '__main__':
    # training parameters
    num_iterations = 100
    learning_rate = 1e-2

    # get params
    vocab_size, num_langs, train_data, dev_data = utils.get_bigram_data()
    dims = [vocab_size, 144, 30, num_langs]
 
    params = mlpn.create_classifier(dims)
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)
    pickle.dump(trained_params, open( "model.p", "wb" ) )

