import numpy as np
import os
import scipy.io
import scipy.io.wavfile
import glob

def load_data(dirpath, example_duration, time_window_duration, sampling_frequency):
    X_data, Y_data, filenames = [], [], []

    num_pts_per_example = int(example_duration * sampling_frequency)
    num_windows_per_example = int(example_duration / time_window_duration)

    print('[*] Loading data...', flush=True)
    listing = glob.glob(os.path.join(dirpath, '*.mat'))
    for i, filename in enumerate(listing):
        filename = filename.split('/')[-1].split('.')[0]
        # load the MIDI file
        midi_filepath = os.path.join(dirpath, filename + '.mat')
        data = scipy.io.loadmat(midi_filepath)
        X = data['X']
        num_examples_in_midi = np.ceil(X.shape[0] / num_windows_per_example)
        # load the wav file
        wav_filepath = os.path.join(dirpath, filename + '.wav')
        fs, Y = scipy.io.wavfile.read(wav_filepath)
        Y = Y.astype(float)
        Y = Y / 32768   # scipy.io.wavfile outputs values that are int16
        assert(fs == sampling_frequency)
        num_examples_in_wav = np.ceil(len(Y) / sampling_frequency / example_duration)
        # make sure there will be the same number of examples from each file
        assert(num_examples_in_midi == num_examples_in_wav)
        # pad both arrays
        pad_amount_X = int(num_examples_in_midi * num_windows_per_example) - X.shape[0]
        X = np.pad(X, (pad_amount_X,0), 'constant')
        pad_amount_Y = int(num_examples_in_wav * num_pts_per_example) - Y.shape[0]
        Y = np.pad(Y, (pad_amount_Y,0), 'constant')
        # create the examples
        for example_num in range(int(num_examples_in_midi)):
            example_x = X[example_num*num_windows_per_example:(example_num+1)*num_windows_per_example, :]
            example_y = Y[example_num*num_pts_per_example:(example_num+1)*num_pts_per_example]
            example_y = np.reshape(example_y, (num_windows_per_example,-1))
            X_data.append(example_x)
            Y_data.append(example_y)
            filenames.append(filename)

        X_data = np.stack(X_data)
        Y_data = np.stack(Y_data)
        assert(X_data.shape[0] == Y_data.shape[0])
        assert(X_data.shape[1] == Y_data.shape[1])

    return X_data, Y_data, filenames

