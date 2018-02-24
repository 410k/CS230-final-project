import numpy as np
import os
import scipy.io

class BatchGenerator(object):
    '''Generator for returning shuffled batches.

    data_x -- list of input matrices
    data_y -- list of output matrices
    batch_size -- size of batch
    input_size -- input width
    output_size -- output width
    mini -- create subsequences for truncating backprop
    mini_len -- truncated backprop window'''

    def __init__(self, data_x, data_y, batch_size, input_size, output_size, mini=True, mini_len=200):
        self.input_size = input_size
        self.output_size = output_size
        self.data_x = data_x
        self.data_y = data_y
        self.batch_size = batch_size
        self.batch_count = len(range(0, len(self.data_x), self.batch_size))
        self.batch_length = None
        self.mini = mini
        self.mini_len = mini_len


    def batch(self):
        while True:                 # prevents a StopIteration error from being thrown
            # shuffle the data
            inds = np.arange(0, len(self.data_x))
            np.random.shuffle(inds)
            shuffled_x = []
            shuffled_y = []
            for i in inds:
                shuffled_x.append(self.data_x[i])
                shuffled_y.append(self.data_y[i])

            for batch_ind in range(0, len(self.data_x), self.batch_size):
                # create a batch list of the selected examples
                input_batch_list = []
                output_batch_list = []
                for example_ind in range(batch_ind, min(batch_ind+self.batch_size, len(self.data_x))):
                    input_batch_list.append(shuffled_x[example_ind])
                    output_batch_list.append(shuffled_y[example_ind])
                    
                # add any necessary padding and reformat to the list to a matrix
                input_batch, output_batch = self._pad(input_batch_list, output_batch_list)
                if self.mini:
                    input_batch, output_batch = self._rebatch(input_batch, output_batch, self.mini_len)
                seq_len = input_batch.shape[1]

                yield input_batch, output_batch, seq_len


    def _pad(self, input_batch_list, output_batch_list):
        current_batch_size = len(input_batch_list)

        example_lens = [input_batch_list[i].shape[0] for i in range(0, current_batch_size)]
        example_max_len = max(example_lens)

        mod_input_batch = []
        mod_output_batch = []

        # pad all examples so that they have the same length
        for i, example_len in enumerate(example_lens):
            pad_amount = example_max_len - example_len
            x = np.pad(input_batch_list[i], ((pad_amount,0), (0,0)), 'constant')
            y = np.pad(output_batch_list[i], ((pad_amount,0), (0,0)), 'constant')
            mod_input_batch.append(x)
            mod_output_batch.append(y)

        new_input_batch, new_output_batch = self._list_to_batch(mod_input_batch, mod_output_batch)
        return new_input_batch, new_output_batch


    def _rebatch(self, input_batch, output_batch, rebatch_example_len):
        current_batch_size = input_batch.shape[0]
        example_len = input_batch.shape[1]
        pad_amount = example_len % rebatch_example_len

        mod_input_batch = []
        mod_output_batch = []

        for ind in range(0, current_batch_size):
            # pad each example
            example_x = input_batch[ind,:,:]
            example_y = output_batch[ind,:,:]
            x = np.pad(example_x, ((pad_amount,0), (0,0)), 'constant')
            y = np.pad(example_y, ((pad_amount,0), (0,0)), 'constant')
            # generate multiple new examples from each original example
            num_new_examples = x.shape[0] // rebatch_example_len
            for i in range(0, num_new_examples):
                start_ind = i * rebatch_example_len
                end_ind = (i+1) * rebatch_example_len
                tmp_x = x[start_ind:end_ind, :]
                tmp_y = y[start_ind:end_ind, :]
                mod_input_batch.append(tmp_x)
                mod_output_batch.append(tmp_y)

        new_input_batch, new_output_batch = self._list_to_batch(mod_input_batch, mod_output_batch)
        return new_input_batch, new_output_batch


    def _list_to_batch(self, input_batch, output_batch):
        # format data from list to [example #, # time points, pitch encoding / # samples]
        new_input_batch = np.stack(input_batch)
        new_output_batch = np.stack(output_batch)
        return new_input_batch, new_output_batch



# def load_data(dirpath):
#     X_data = []
#     Y_data = []
#     filenames = []
#     print('[*] Loading data...', flush=True)
#     listing = os.listdir(dirpath)
#     for i, filename in enumerate(listing):
#         if filename.split('.')[-1] == 'mat':
#             filepath = os.path.join(dirpath, filename)
#             data = scipy.io.loadmat(filepath)
#             loaded_x = data['Xin']
#             loaded_y = data['Yout']
#             assert(loaded_x.shape[0] == loaded_y.shape[0])
#             X_data.append(loaded_x)
#             Y_data.append(loaded_y)
#             filenames.append(filename)

#     return X_data, Y_data, filenames


def load_data(dirpath):
    X_data = []
    Y_data = []
    filenames = []
    print('[*] Loading data...', flush=True)

    x_path = os.path.join('data/test/inputs/jazz')
    y_path = os.path.join('data/test/velocities/jazz')

    for i, filename in enumerate(os.listdir(x_path)):
        if filename.split('.')[-1] == 'npy':
            filenames.append(filename)

    for i, filename in enumerate(filenames):
        abs_x_path = os.path.join(x_path, filename)
        abs_y_path = os.path.join(y_path, filename)
        loaded_x = np.load(abs_x_path)

        X_data.append(loaded_x)

        loaded_y = np.load(abs_y_path)
        loaded_y = loaded_y/127
        Y_data.append(loaded_y)
        assert X_data[i].shape[0] == Y_data[i].shape[0]

    return X_data, Y_data, filenames