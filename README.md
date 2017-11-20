# Feed Forward Language Detector
A feed forward "multi-layer perceptron" neural network that detects what language a tweet is written in.

The code is written in numpy using the included datasets, tan-h as an activation function and log loss for it's loss function.

Supports english, dutch, italian, french, spanish and german.

Written for course in Deep Learning for Text and Sequences

## Usage

To train the model navigate to the src directory and run

```python
python train_mlpn.py
```

To evalutate the model run
```python
python model_evaluator.py
```
