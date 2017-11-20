"""
Written by Ari Bornstein
"""

import loglinear as ll
import utils
import random
from collections import Counter
import numpy as np
 
def accuracy_on_dataset(dataset, params):
    """
    Measures accuracy on a given dataset
    """
    good = bad = 0.0
    for label, features in dataset:
        # Compute the accuracy (a scalar) of the current parameters
        # on the dataset.
        # accuracy is (correct_predictions / all_predictions)
        predict = ll.predict(features, params)
        if predict == label:
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
        cum_loss = 0.0 # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = features # convert features to a vector.
            y = label       # convert the label to number if needed.
            loss, grads = ll.loss_and_gradients(x,y,params)
            cum_loss += loss
            # update the parameters according to the gradients
            # and the learning rate.
            gW, gb = grads
            params[0] -= gW * learning_rate
            params[1] -= gb * learning_rate
                      
        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print I, train_loss, train_accuracy, dev_accuracy
    return params

if __name__ == '__main__':
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.

    # training parameters
    learning_rate = 1e-2
    num_iterations = 100

    vocab_size, num_langs, train_data, dev_data = utils.get_bigram_data()

    params = ll.create_classifier(vocab_size, num_langs)
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)

