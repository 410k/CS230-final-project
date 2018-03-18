import numpy as np
import os
import scipy.io
import scipy.io.wavfile
import glob
from iso226 import iso226
import csv
import random
import math
import pandas as pd
from scipy.io.wavfile import write

def setup_dirs(args):
    print('[*] Setting up directory...', flush=True)
    main_path = args.runs_dir
    current_run = os.path.join(main_path, args.current_run)
    # data
    data_path = args.data_dir
    #train_path = os.path.join(data_path, 'train')
    #train_dev_path = os.path.join(data_path, 'train_dev')
    #test_dev_path = os.path.join(data_path, 'test_dev')
    #test_path = os.path.join(data_path, 'test')
    
    # model
    model_path = os.path.join(current_run, 'model')
    logs_path = os.path.join(current_run, 'logs')
    png_path = os.path.join(current_run, 'png')
    pred_path = os.path.join(current_run, 'predictions')
    weight_path = os.path.join(current_run,'weights')
    # dictionary of directory paths
    dirs = {'main_path': main_path,
            'current_run': current_run,
            'data_path': data_path,
            # 'train_path': train_path,
            # 'train_dev_path': train_dev_path,
            # 'test_dev_path': test_dev_path,
            # 'test_path': test_path,
            'model_path': model_path,
            'logs_path': logs_path,
            'png_path': png_path,
            'pred_path': pred_path,
            'weight_path': weight_path}
    # create the directories if they don't already exist
    for dir_name, dir_path in dirs.items():
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    return dirs

def split_data(allfilenames_path,num_songs):
    # calculate number of songs in train/train_dev/test
    if (num_songs - math.floor(num_songs*0.8)) % 2 == 0:
        ntrain = math.floor(num_songs*0.8)
        ntrain_dev = int((num_songs - ntrain) / 2)
        ntest = ntrain_dev
    else:
        ntrain = math.floor(num_songs*0.8) + 1
        ntrain_dev = int((num_songs - ntrain) / 2)
        ntest = ntrain_dev
    
    # get list of all filenames
    allfiles = os.listdir(allfilenames_path)
    allfiles.sort()
    allfiles.remove('.DS_Store')
    allfilenames = [file.split('_sf11025')[0] for file in allfiles]
    indices = list(range(len(allfiles)))
    random.shuffle(indices)
    # create one-hot arrays
    train_OH = np.zeros(len(allfiles))
    train_dev_OH = np.zeros(len(allfiles))
    test_OH = np.zeros(len(allfiles))
    train_OH[indices[0:ntrain]] = 1
    train_dev_OH[indices[ntrain:ntrain+ntrain_dev]] = 1
    test_OH[indices[ntrain+ntrain_dev:ntrain+ntrain_dev+ntest]] = 1

    datasplit_dict = {'filename' : allfilenames,
                      'train' : train_OH,
                      'train_dev' : train_dev_OH,
                      'test' : test_OH
                     }

    # create pandas df
    df = pd.DataFrame(datasplit_dict)
    # write to csv
    df.to_csv('datasplit_'+ str(num_songs) + '.csv')
    return datasplit_dict


def load_data(data_path, datasplit_dict, mode, example_duration, time_window_duration, sampling_frequency, loss_domain, equal_loudness):

    if sampling_frequency == 44100:
        wav_path = os.path.join(data_path, 'TPD 44kHz')
    elif sampling_frequency == 22050:
        wav_path = os.path.join(data_path, 'TPD 22kHz')
    elif sampling_frequency == 11025:
        wav_path = os.path.join(data_path, 'TPD 11kHz')
    else:
        raise ValueError('sampling frequency not recognized!')

    if time_window_duration == 0.05:
        twd_suffix = '_dt50'
    elif time_window_duration == 0.025:
        twd_suffix = '_dt25'
    else:
        raise ValueError('time window duration not recognized!')
    midi_path = os.path.join(data_path, twd_suffix)

    X_data, Y_data, filenames = [], [], []

    # need to do some rounding because 50 ms at 11025 is 551.25 samples! (which
    # will not work)
    num_pts_per_window = int(np.round(time_window_duration * sampling_frequency))
    num_windows_per_example = int(example_duration / time_window_duration)
    num_pts_per_example = int(num_pts_per_window * num_windows_per_example)

    print('[*] Loading data...', flush=True)
    # wav_listing = glob.glob(os.path.join(wav_path, '*.wav'))
    # midi_listing = glob.glob(os.path.join(midi_path, '*.mat'))

    # mask filenames for specific mode
    all_filenames = np.array(datasplit_dict['filename'])
    if mode == 'train':
        idx = np.array(np.array(datasplit_dict['train']) > 0)
    elif mode == 'train_dev':
        idx = np.array(np.array(datasplit_dict['train_dev']) > 0)
    elif mode == 'test':
        idx = np.array(np.array(datasplit_dict['test']) > 0)
    else:
        raise ValueError('arg.mode not recognized!')
    
    all_filenames = all_filenames[idx]
    wav_filenames = [file + '_sf' + str(sampling_frequency) + '.wav' for file in all_filenames]
    midi_filenames = [file + twd_suffix + '.mat' for file in all_filenames]
    wav_listing = [os.path.join(wav_path, file) for file in wav_filenames]
    midi_listing = [os.path.join(midi_path, file) for file in midi_filenames]
    
    print('number of files = ' + str(len(wav_filenames)) + '=' + str(len(midi_filenames)))

    target_wav_files = []
    target_midi_files = []
    for midi_file in midi_listing:
        filename = midi_file.split('/')[-1].split(twd_suffix+'.mat')[0]
        corresponding_wav_file = os.path.join(wav_path, filename+'_sf'+str(sampling_frequency)+'.wav')
        if corresponding_wav_file in wav_listing:
            target_wav_files.append(corresponding_wav_file)
            target_midi_files.append(midi_file)

    #
    for i, midi_filepath in enumerate(target_midi_files):
        filename = midi_filepath.split('/')[-1].split(twd_suffix+'.mat')[0]
        print('   loading ' + filename, flush=True)
        # load the MIDI file
        data = scipy.io.loadmat(midi_filepath)
        X = data['Xin']
        num_examples_in_midi = np.ceil(X.shape[0] / num_windows_per_example)
        # load the wav file
        wav_filepath = target_wav_files[i]
        fs, Y = scipy.io.wavfile.read(wav_filepath)
        Y = Y.astype(float)
        Y = Y / 32768   # scipy.io.wavfile outputs values that are int16
        assert(fs == sampling_frequency)
        num_examples_in_wav = np.ceil(len(Y) / num_pts_per_example)
        # # make sure there will be the same number of examples from each file
        # assert(num_examples_in_midi == num_examples_in_wav)
        # pad both arrays
        pad_amount_X = int(num_examples_in_midi * num_windows_per_example) - X.shape[0]
        X = np.pad(X, ((0,pad_amount_X), (0,0)), 'constant')
        if num_examples_in_midi >= num_examples_in_wav:
            pad_amount_Y = int(num_examples_in_midi * num_pts_per_example) - Y.shape[0]
            Y = np.pad(Y, ((0,pad_amount_Y)), 'constant')
        else:
            Y = Y[0:int(num_examples_in_midi*num_pts_per_example)]
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
    # train on the frequency domain loss function
    if loss_domain == 'frequency':
        #import pdb
        #pdb.set_trace()
        Y_data = np.fft.rfft(Y_data,axis=2)
        Y_data = np.concatenate((np.real(Y_data),np.imag(Y_data)),axis=2)
        if equal_loudness:
            # apply equal loudness contour weighting 
            elc,_ = iso226(30, sampling_frequency, Y_data.shape[2]/2) 
            elc = (10**(-np.concatenate((elc,elc),axis = 0))/20) # convert from dB and invert
            elc = elc/np.max(elc)
            Y_data = Y_data*elc 

            
            
    assert(X_data.shape[0] == Y_data.shape[0])
    assert(X_data.shape[1] == Y_data.shape[1])

    return X_data, Y_data, filenames


def save_predictions(save_path, pred_type, X, Y, Y_pred):
    save_dict = {'Y_'+pred_type+'_pred': Y_pred, 
                 'X_'+pred_type: X,
                 'Y_'+pred_type: Y}
    save_name = pred_type + '_pred.mat'
    filepath = os.path.join(save_path, save_name)
    scipy.io.savemat(filepath, save_dict)
    
def unweight(Y, sampling_frequency):
    # undo equal loudness contour weighting 
    elc,_ = iso226(30, sampling_frequency, Y.shape[2]/2) 
    elc = (10**(-np.concatenate((elc,elc),axis = 0))/20) # convert from dB and invert
    elc = elc/np.max(elc)
    Y = Y/elc
    return Y

def inverse_rfft(Y):
    # reshape and perform inverse rfft from [real imaginary]
    midpoint = int(Y.shape[2]/2)
    y = Y[:,:,0:midpoint] + 1j*Y[:,:,midpoint:]
    y = np.fft.irfft(y,axis=2)
    return y

def reshape_audio(y):
    # reshape audio
    y_flattened = y.reshape(y.size,1)
    return y_flattened

def scale_audio(y):
    # scale audio to fit into 16 bit integers
    scaled = np.int16(y/np.max(np.abs(y))*32767).flatten()
    return scaled

def process_audio(Y, sampling_frequency, loss_domain, use_equal_loudness):
    if loss_domain == 'frequency':
        if use_equal_loudness:
            Y = unweight(Y,sampling_frequency)
        Y = inverse_rfft(Y)
    Y = reshape_audio(Y)
    Y = scale_audio(Y)
    return Y

def save_audio(save_path, pred_type, Y, Y_pred, sampling_frequency, loss_domain, use_equal_loudness):
    Y = process_audio(Y, sampling_frequency, loss_domain, use_equal_loudness)
    Y_pred = process_audio(Y_pred, sampling_frequency, loss_domain, use_equal_loudness)
    print('saving audio')
    # save original audio
    save_name = pred_type + '.mp3'
    filepath = os.path.join(save_path, save_name)
    write(filepath,sampling_frequency,Y)
    # save predicted audio
    save_name = pred_type + '_pred.mp3'
    filepath = os.path.join(save_path, save_name)
    write(filepath,sampling_frequency,Y_pred)
