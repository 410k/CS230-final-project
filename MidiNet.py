from keras.models import Sequential
from keras.layers import Bidirectional
from keras.layers import Dense
from keras.layers import LSTM, GRU
from keras.layers import TimeDistributed
from keras.layers import Lambda
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
from iso226 import iso226
from iso226 import weight_loss
import numpy as np


def MidiNet(input_shape, output_shape, loss_domain, elc = [], 
            num_hidden_units=128, num_layers=2, unidirectional_flag=False,
            cell_type='GRU', batch_norm_flag=False, dropout=0, layer_lock_out_mask=[]):
    
    # create RNN
    print('[*] Creating network', flush=True)
    model = Sequential()
    for l in range(num_layers):
        if unidirectional_flag:
            layer = create_unidirectional(cell_type, input_shape, num_hidden_units, layer_lock_out_mask[l])
        else:
            layer = create_bidirectional(cell_type, input_shape, num_hidden_units, layer_lock_out_mask[l])
        model.add(layer)
        if batch_norm_flag:
            model.add(BatchNormalization())
        if dropout > 0:
            model.add(Dropout(dropout))
    # fully connected layer
    model.add(TimeDistributed(Dense(output_shape[1], activation=None)))
    # add frequency domain weighting function
    if loss_domain == 'frequency' and elc.size != 0 :
        model.add(Lambda(lambda x: elc*x))
    return model


def create_LSTM_cell(num_hidden_units):
    return LSTM(num_hidden_units, return_sequences=True)


def create_GRU_cell(num_hidden_units):
    return GRU(num_hidden_units, return_sequences=True)


def create_bidirectional(cell_type, input_shape, num_hidden_units, layer_lock_out_flag):
    if cell_type == 'GRU':
        print('trainable flag = ' + str(layer_lock_out_flag))
        return Bidirectional(create_GRU_cell(num_hidden_units), input_shape=input_shape, trainable=layer_lock_out_flag)
    elif cell_type == 'LSTM':
        print('trainable flag = ' + str(layer_lock_out_flag))
        return Bidirectional(create_LSTM_cell(num_hidden_units), input_shape=input_shape, trainable=layer_lock_out_flag)
    else:
        print('Incorrect cell type specified!')
        
#def weight_loss(x, sampling_frequency, output_shape):
        # apply equal loudness contour weighting 
#        elc,_ = iso226(30, sampling_frequency, output_shape[1]/2) 
#        elc = (10**(-np.concatenate((elc,elc),axis = 0))/20) # convert from dB and invert
#        elc = elc/np.max(elc) # normalize so maximum is 1
#        x = x*elc 
#        return x
