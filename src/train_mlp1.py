"""
Written by Ari Bornstein
"""

import mlp1
import random
import utils

def accuracy_on_dataset(dataset, params):
    if not dataset:
        return 0

    good = bad = 0.0
    for label, features in dataset:
        y_prediction = mlp1.predict(features, params)
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
            y = label       
            loss, grads = mlp1.loss_and_gradients(features, y, params)
            cum_loss += loss

            # SGD update parameters
            U, W, b, b_tag = params
            params[0] = U - learning_rate * grads[0]
            params[1] = W - learning_rate * grads[1]
            params[2] = b - learning_rate * grads[2]
            params[3] = b_tag - learning_rate * grads[3]

        # notify progress
        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print I, train_loss, train_accuracy, dev_accuracy
        
    return params

if __name__ == '__main__':
    # optional models - xor, unigrams, bigrams (use the utils.<function> associated with the model)

    model = 'unigram'
    if model == 'xor':
        in_dim, out_dim, train_data, dev_data = utils.get_xor_data()
        hid_dim = 4
        num_iterations = 30
        learning_rate = 0.25
        params = mlp1.create_classifier(in_dim, hid_dim, out_dim)
        train_classifier(train_data, '', num_iterations, learning_rate, params)

    if model == 'unigram':
        hid_dim = 144
        learning_rate = 1e-2
        num_iterations = 100
        vocab_size, num_langs, train_data, dev_data = utils.get_unigram_data()
        params = mlp1.create_classifier(vocab_size, hid_dim, num_langs)
        trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)

    if model == 'bigram':
        hid_dim = 144
        learning_rate = 1e-2
        num_iterations = 100
        vocab_size, num_langs, train_data, dev_data = utils.get_bigram_data()
        params = mlp1.create_classifier(vocab_size, hid_dim, num_langs)
        trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)