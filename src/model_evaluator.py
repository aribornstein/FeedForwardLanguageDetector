"""
Written by Ari Bornstein
"""

import pickle
import utils
import mlpn as mlp

params = pickle.load( open( "model.p", "rb" ) )
test = utils.TEST_BIGRAMS
out = open(r'..\data\test.pred.', 'w')
for x in test:
     pred = utils.I2L[mlp.predict(x, params)]
     out.write("{}\n".format(pred))
out.close()

