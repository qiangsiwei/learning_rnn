# -*- coding: utf-8 -*-  

import numpy as np
from lstm import LstmParam, LstmNetwork

class ToyLossLayer:
    """
    Computes square loss with first element of hidden layer array.
    """
    @classmethod
    def loss(self, pred, label):
        return (pred[0] - label) ** 2

    @classmethod
    def bottom_diff(self, pred, label):
        diff = np.zeros_like(pred)
        diff[0] = 2 * (pred[0] - label)
        return diff


if __name__ == "__main__":
    # learns to repeat simple sequence from random inputs
    np.random.seed(0)
    # parameters for input data dimension and lstm cell count 
    x_dim = 50
    mem_cell_ct = 100
    lstm_param = LstmParam(mem_cell_ct, x_dim) 
    lstm_net = LstmNetwork(lstm_param)
    y_list = [-0.5,0.2,0.1,-0.5]
    input_val_arr = [np.random.random(x_dim) for _ in y_list]
    
    for cur_iter in range(100):
        print "cur iter: ", cur_iter
        for ind in range(len(y_list)):
            lstm_net.x_list_add(input_val_arr[ind])
            print "y_pred[%d] : %f" % (ind, lstm_net.lstm_node_list[ind].state.h[0])

        loss = lstm_net.y_list_is(y_list, ToyLossLayer)
        print "loss: ", loss
        lstm_param.apply_diff(lr=0.1)
        lstm_net.x_list_clear()

# cur iter:  0
# y_pred[0] : 0.041349
# y_pred[1] : 0.069304
# y_pred[2] : 0.116993
# y_pred[3] : 0.165624
# loss:  0.753483886253
# ...
# cur iter:  99
# y_pred[0] : -0.500331
# y_pred[1] : 0.201063
# y_pred[2] : 0.099122
# y_pred[3] : -0.499226
# loss:  2.61076357962e-06
