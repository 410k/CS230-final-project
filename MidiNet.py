from keras.models import Sequential
from keras.layers import Bidirectional
from keras.layers import Dense
from keras.layers import LSTM, GRU
from keras.layers import TimeDistributed


def MidiNet(input_shape, output_shape, 
            num_hidden_units=128, num_layers=2, unidirectional_flag=False,
            cell_type='GRU'):

    # create RNN
    print('[*] Creating network', flush=True)
    model = Sequential()
    for l in range(num_layers):
        if unidirectional_flag:
            layer = create_unidirectional(cell_type, input_shape, num_hidden_units)
        else:
            layer = create_bidirectional(cell_type, input_shape, num_hidden_units)
        model.add(layer)
    # fully connected layer
    model.add(TimeDistributed(Dense(output_shape[1], activation=None)))
    return model


def create_LSTM_cell(num_hidden_units):
    return LSTM(num_hidden_units, return_sequences=True)


def create_GRU_cell(num_hidden_units):
    return GRU(num_hidden_units, return_sequences=True)


def create_bidirectional(cell_type, input_shape, num_hidden_units):
    if cell_type == 'GRU':
        return Bidirectional(create_GRU_cell(num_hidden_units), input_shape=input_shape)
    elif cell_type == 'LSTM':
        return Bidirectional(create_LSTM_cell(num_hidden_units), input_shape=input_shape)
    else:
        print('Incorrect cell type specified!')