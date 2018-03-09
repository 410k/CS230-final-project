import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split

from MidiNet import MidiNet
from util import load_data
import os


example_duration = 4.0
time_window_duration = 0.05
sampling_frequency = 8000

num_hidden_units = 128
num_layers = 2
unidirectional_flag = False
cell_type = 'GRU'
batch_size = 8

num_epochs = 100
epoch_save_interval = 10



path = 'data/test'

main_path = 'runs'
data_path = 'data'
current_run = 'test_keras'

current_run = os.path.join(main_path, current_run)
# data
train_path = os.path.join(data_path, 'train')
train_dev_path = os.path.join(data_path, 'train_dev')
# model
model_path = os.path.join(current_run, 'model')
logs_path = os.path.join(current_run, 'logs')
png_path = os.path.join(current_run, 'png')
pred_path = os.path.join(current_run, 'predictions')
dirs = {'main_path': main_path,
        'current_run': current_run,
        'train_path': train_path,
        'train_dev_path': train_dev_path,
        'model_path': model_path,
        'logs_path': logs_path,
        'png_path': png_path,
        'pred_path': pred_path}
# create the directories if they don't already exist
for dir_name, dir_path in dirs.items():
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


# # load data 
X_data, Y_data, filenames = load_data(path, example_duration, time_window_duration, sampling_frequency)
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data)
input_shape = (X_train.shape[1], X_train.shape[2])
output_shape = (Y_train.shape[1], Y_train.shape[2])
# input_shape = (20, 176)
# output_shape = (20, 2205)
# input_shape = (X_data.shape[1], X_data.shape[2])
# output_shape = (Y_data.shape[1], Y_data.shape[2])


# compile model
model = MidiNet(input_shape, output_shape, num_hidden_units, num_layers, 
    unidirectional_flag, cell_type)
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())

# train the network
# num_epoch_batches = int(np.ceil(num_epochs/epoch_save_interval))
# for epoch_batch_num in range(num_epoch_batches):
#     history_callback = model.fit(X_train, Y_train, epochs=epoch_save_interval, batch_size=batch_size)
#     loss_history = history_callback.history["loss"]
#     # save the model every n epochs
#     filename = 'model-e' + str(epoch_batch_num*epoch_save_interval)
#     filepath = os.path.join(dirs['current_run'], filename)
#     model.save(filepath)

from keras.callbacks import ModelCheckpoint
# import pdb
# pdb.set_trace()

filename = 'weights-epoch{epoch:03d}-loss{val_loss:.4f}.hdf5'
filepath = os.path.join(dirs['current_run'], filename)
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, period=epoch_save_interval)
callbacks_list = [checkpoint]

model.fit(X_train, Y_train, validation_split=0.05, epochs=num_epochs, batch_size=batch_size, callbacks=callbacks_list)

# save the final model
filename = 'model-e' + str(num_epochs)
filepath = os.path.join(dirs['current_run'], filename)
model.save(filepath)


# evaluate model on training and test data
Y_test_pred = model.predict(X_test, batch_size=batch_size)
Y_train_pred = model.predict(X_train, batch_size=batch_size)

# save predictions
save_dict = {'Y_test_pred': Y_test_pred, 
             'X_test': X_test,
             'Y_test': Y_test}
filepath = os.path.join(dirs['pred_path'], "test_pred.mat")
scipy.io.savemat(filepath, save_dict)

save_dict = {'Y_train_pred': Y_train_pred, 
             'X_train': X_train,
             'Y_train': Y_train}
filepath = os.path.join(dirs['pred_path'], "train_pred.mat")
scipy.io.savemat(filepath, save_dict)
