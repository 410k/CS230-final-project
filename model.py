import tensorflow as tf
from data_util import BatchGenerator, load_data
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.pyplot.ioff()

import scipy.io

class GenreLSTM(object):
    def __init__(self, dirs, input_size, output_size, mini=False, bi=False, 
        one_hot=True, num_layers=1, batch_size=8):
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.num_layers = int(num_layers)
        self.batch_size = int(batch_size)
        self.dirs = dirs
        self.bi = bi
        self.mini = mini
        self.one_hot = one_hot

        self.sess = tf.Session()


    def prepare_model(self):
        # placeholders
        self.inputs = tf.placeholder(tf.float32, [None, None, self.input_size], name='inputs')
        self.true_classical_outputs = tf.placeholder(tf.float32, [None, None, self.output_size], name='true_outputs')
        self.seq_len = tf.placeholder(tf.int32, [None], name='seq_len')

        self.input_keep_prob = tf.placeholder(tf.float32, None, name='input_keep_prob')
        self.output_keep_prob = tf.placeholder(tf.float32, None, name='output_keep_prob')

        # network architecture
        # need this for shape of weights to match
        # only throws error when num_layers > 1
        # with 100 hidden units, shapes of some variables (error message
        # doesn't say) are (200,400) and (276,400)
        # with 10 hidden units, shapes are (20,40) and (186,40)
        self.num_hidden_units = 256
        # self.prepare_unidirectional()
        self.prepare_bidirectional()
        with tf.variable_scope('fc') as scope:
            self.classical_linear_out = tf.contrib.layers.fully_connected(self.classical_outputs, 
                                                                          self.output_size, 
                                                                          activation_fn=None, 
                                                                          scope='layer1')

        self.classical_difference = tf.subtract(self.true_classical_outputs, self.classical_linear_out)
        self.classical_loss = tf.losses.mean_squared_error(self.true_classical_outputs,
                                                           self.classical_linear_out)

        # variables_names =[v.name for v in tf.trainable_variables()]
        # values = self.sess.run(variables_names)
        # for k, v in zip(variables_names, values):
        #     print(k)
        # import pdb
        # pdb.set_trace()

        # tensorboard summaries
        tf.summary.scalar("Loss", self.classical_loss)
        tf.summary.histogram("1. FC outputs", self.classical_linear_out)
        tf.summary.histogram("1. RNN outputs", self.classical_outputs)
        tf.summary.histogram("Output difference", self.classical_difference, family='Other')
        tf.summary.histogram("True outputs", self.true_classical_outputs, family='Other')
        # tf.summary.histogram("FC layer weights", tf.get_variable('fc/layer1/weights'))
        self.summary_op = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(os.path.join(self.dirs['logs_path'], 'train'), graph=self.sess.graph_def)
        self.test_writer = tf.summary.FileWriter(os.path.join(self.dirs['logs_path'], 'test'), graph=self.sess.graph_def)


    def prepare_unidirectional(self):
        print("[*] Preparing unidirectional dynamic RNN...", flush=True)
        with tf.variable_scope("encode") as scope:
            self.rnn_cell = tf.contrib.rnn.LSTMBlockCell(self.num_hidden_units, forget_bias=1.0)
            # self.rnn_cell = tf.contrib.rnn.DropoutWrapper(self.rnn_cell, 
            #                                             input_keep_prob=self.input_keep_prob, 
            #                                             output_keep_prob=self.output_keep_prob)
            if self.num_layers > 1:
                self.rnn_cell  = tf.contrib.rnn.MultiRNNCell([self.rnn_cell]*self.num_layers)

            self.classical_outputs, last_state = tf.nn.dynamic_rnn(
                                                        self.rnn_cell,
                                                        self.inputs,
                                                        sequence_length=self.seq_len,
                                                        dtype=tf.float32)


    def prepare_bidirectional(self):
        print("[*] Preparing bidirectional dynamic RNN...", flush=True)
        with tf.variable_scope("encode") as scope:
            self.rnn_cell_fw = tf.contrib.rnn.LSTMBlockCell(self.num_hidden_units, forget_bias=1.0)
            # self.rnn_cell_fw = tf.contrib.rnn.DropoutWrapper(self.rnn_cell_fw, 
            #                                                input_keep_prob=self.input_keep_prob, 
            #                                                output_keep_prob=self.output_keep_prob)
            self.rnn_cell_bw = tf.contrib.rnn.LSTMBlockCell(self.num_hidden_units, forget_bias=1.0)
            # self.rnn_cell_bw = tf.contrib.rnn.DropoutWrapper(self.rnn_cell_bw, 
            #                                                input_keep_prob=self.input_keep_prob, 
            #                                                output_keep_prob=self.output_keep_prob)
            if self.num_layers > 1:
                self.rnn_cell_fw  = tf.contrib.rnn.MultiRNNCell([self.rnn_cell_fw]*self.num_layers)
                self.rnn_cell_bw  = tf.contrib.rnn.MultiRNNCell([self.rnn_cell_bw]*self.num_layers)

            (self.rnn_fw, self.rnn_bw), last_state = tf.nn.bidirectional_dynamic_rnn(
                                                        self.rnn_cell_fw,
                                                        self.rnn_cell_bw,
                                                        self.inputs,
                                                        sequence_length=self.seq_len,
                                                        dtype=tf.float32)
            self.classical_outputs  = tf.concat([self.rnn_fw, self.rnn_bw], axis=2)


    def train(self, data, model=None, starting_epoch=0, clip_grad=False, 
        epochs=1001, input_keep_prob=1.0, output_keep_prob=1.0, 
        learning_rate=0.001 , eval_epoch=5, val_epoch=20, save_epoch=5):

        self.data = data

        # specify the optimizer
        if clip_grad:
            classical_optimizer = self.clip_optimizer(learning_rate, self.classical_loss)
        else:
            classical_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.classical_loss)

        # initialize the model
        if model:
            self.load(model)
        else:
            self.sess.run(tf.global_variables_initializer())

        classical_batcher = BatchGenerator(self.data["classical"]["X"], 
                                           self.data["classical"]["Y"], 
                                           self.batch_size,
                                           rebatch_flag=self.mini)
        classical_generator = classical_batcher.batch()

        self.validation_batcher, self.validation_files = self.setup_validation("classical")


        print("[*] Initiating training...", flush=True)
        for epoch in range(starting_epoch, epochs):
            print("[*] Epoch %d" % epoch, flush=True)
            classical_epoch_avg = 0
            for batch_num in range(classical_batcher.num_batches):
                batch_X, batch_Y, batch_len = next(classical_generator)
                batch_len = [batch_len] * len(batch_X)
                _, epoch_error, classical_summary  =  self.sess.run([classical_optimizer,
                                                                     self.classical_loss,
                                                                     self.summary_op,], 
                                                                     feed_dict={self.inputs: batch_X,
                                                                                self.true_classical_outputs: batch_Y,
                                                                                self.seq_len: batch_len,
                                                                                self.input_keep_prob: input_keep_prob,
                                                                                self.output_keep_prob: output_keep_prob})
                classical_epoch_avg += epoch_error
                print("\tBatch %d/%d, Training MSE for Classical batch: %.9f" % (batch_num+1, classical_batcher.num_batches, epoch_error), flush=True)
                self.train_writer.add_summary(classical_summary, epoch*classical_batcher.num_batches + batch_num)
            print("[*] Average Training MSE for Classical epoch %d: %.9f" % (epoch, classical_epoch_avg/classical_batcher.num_batches), flush=True)

            # if epoch % val_epoch == 0 and epoch != 0:
            #     print("[*] Validating model...", flush=True)
            #     self.validation(epoch)

            if epoch % save_epoch == 0 :
                self.save(epoch)

            if epoch % eval_epoch == 0 :
                print("[*] Evaluating model...", flush=True)
                self.evaluate(epoch)

        print("[*] Training complete.", flush=True)


    def clip_optimizer(self, learning_rate, loss):
        opt = tf.train.AdamOptimizer(learning_rate)
        gradients = opt.compute_gradients(loss)

        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                gradients[i] = (tf.clip_by_norm(grad, 10), var)

        return opt.apply_gradients(gradients)
        

    def load(self, model_name, path=None) :
        print(" [*] Loading checkpoint...", flush=True)
        self.saver = tf.train.Saver(max_to_keep=0)
        if not path:
            self.saver.restore(self.sess, os.path.join(self.dirs['model_path'], model_name))
        else:
            self.sess = tf.Session()
            self.saver.restore(self.sess, path)


    def save(self, epoch):
        print("[*] Saving checkpoint...", flush=True)
        model_name =  "model-e%d.ckpt" % (epoch)
        self.saver = tf.train.Saver(max_to_keep=0)
        save_path = self.saver.save(self.sess, os.path.join(self.dirs['model_path'], model_name))
        print("[*] Model saved in file: %s" % save_path, flush=True)


    def predict(self, input_path, output_path):
        in_list = []
        out_list = []
        filenames = []
        input_lens = []

        loaded = np.load(input_path)
        true_vel = np.load(output_path)/127

        in_list.append(loaded)
        out_list.append(true_vel)

        input_len = [len(loaded)]

        c_loss, c_output = self.sess.run([self.classical_loss, self.classical_linear_out],
                                         feed_dict={self.inputs: in_list,
                                                    self.seq_len: input_len,
                                                    self.input_keep_prob: 1.0,
                                                    self.output_keep_prob: 1.0,
                                                    self.true_classical_outputs: out_list})
        return c_loss, c_output


    def setup_validation(self, genreType):
        '''Handles validation set data'''
        input_folder = self.dirs['eval_path']
        X_data, Y_data, filenames = load_data(input_folder)

        validation_generator = BatchGenerator(X_data, 
                                              Y_data, 
                                              self.batch_size, 
                                              rebatch_flag=False)
        return validation_generator, filenames


    def validation(self, epoch, pred_save=False):
        '''Computes and logs loss of validation set'''
        validation_batch = self.validation_batcher.batch()
        validation_X, validation_Y, input_len = next(validation_batch)
        input_len = [input_len] * len(validation_X)
        c_loss, c_output, c_summary = self.sess.run([self.classical_loss,
                                                     self.classical_linear_out,
                                                     self.summary_op],
                                                     feed_dict={self.inputs: validation_X,
                                                                self.seq_len: input_len,
                                                                self.input_keep_prob: 1.0,
                                                                self.output_keep_prob: 1.0,
                                                                self.true_classical_outputs: validation_Y})

        print("[*] Average Test MSE for Classical epoch %d: %.9f" % (epoch, c_loss), flush=True)
        self.test_writer.add_summary(c_summary, epoch)


    def eval_set(self, genreType):
        '''Loads validation set.'''
        input_folder = self.dirs['eval_path']

        in_list = []
        out_list = []
        filenames = []
        input_lens = []

        for i, filename in enumerate(os.listdir(input_folder)):
            # if filename.split('.')[-1] == 'npy':
            if filename.split('.')[-1] == 'mat':
                filepath = os.path.join(input_folder, filename)
                data = scipy.io.loadmat(filepath)
                loaded_x = data['Xin']
                loaded_y = data['Yout']
                assert(loaded_x.shape[0] == loaded_y.shape[0])
                in_list.append([loaded_x])
                out_list.append([loaded_y])
                filenames.append(filename)
                input_len = [len(loaded_x)]
                input_lens.append(input_len)

        return in_list, out_list, input_lens, filenames


    def evaluate(self, epoch, pred_save=False):
        '''Performs prediction and plots results on validation set.'''
        for i, filename in enumerate(self.validation_files):
            single_input = np.expand_dims(self.validation_batcher.data_x[i], axis=0)
            single_output = np.expand_dims(self.validation_batcher.data_y[i], axis=0)
            seq_len = [single_input.shape[1]]
            c_loss, c_output, c_summary = self.sess.run([self.classical_loss,
                                                         self.classical_linear_out,
                                                         self.summary_op],
                                                         feed_dict={self.inputs: single_input,
                                                                    self.seq_len: seq_len,
                                                                    self.input_keep_prob: 1.0,
                                                                    self.output_keep_prob: 1.0,
                                                                    self.true_classical_outputs: single_output})
            self.plot_evaluation(epoch, filename, c_output, c_output, c_output, single_output)
            
            save_dict = {'model_output': c_output, 'true_output': single_output}
            filepath = os.path.join(self.dirs['pred_path'], filename.split('.')[0] + "-e%d" % (epoch)+".mat")
            scipy.io.savemat(filepath, save_dict)


    def plot_evaluation(self, epoch, filename, c_out, j_out, e_out, out_list, path=None):
        '''Plotting/Saving training session graphs
        epoch -- epoch number
        c_out -- classical output
        j_out -- jazz output
        e_out -- interpretation layer output
        out_list -- actual output
        output_size -- output width
        path -- Save path'''

        fig = plt.figure(figsize=(14,11), dpi=120)
        fig.suptitle(filename, fontsize=10, fontweight='bold')

        graph_items = [out_list[-1], c_out[-1]]
        plots = len(graph_items)
        cmap = ['jet', 'jet']
        vmin = [-1, -1]
        vmax = [1, 1]
        names = ["Actual", "Classical"]

        for i in range(0, plots):
            fig.add_subplot(1,plots,i+1)
            plt.imshow(graph_items[i], vmin=vmin[i], vmax=vmax[i], cmap=cmap[i], aspect='auto')

            a = plt.colorbar(aspect=80)
            a.ax.tick_params(labelsize=7)
            ax = plt.gca()
            ax.xaxis.tick_top()

            ax.set_xlabel('Sample #')
            if i == 0:
                ax.set_ylabel('Time window #')
            ax.xaxis.set_label_position('top')
            ax.tick_params(axis='both', labelsize=7)
            fig.subplots_adjust(top=0.85)
            ax.set_title(names[i], y=1.09)
            # plt.tight_layout()

        # don't show the figure and save it
        if not path:
            out_png = os.path.join(self.dirs['png_path'], filename.split('.')[0] + "-e%d" % (epoch)+".png")
            plt.savefig(out_png, bbox_inches='tight')
            plt.close(fig)
        else:
            # out_png = os.path.join(self.dirs['png_path'], filename.split('.')[0] + "-e%d" % (epoch)+".png")
            # plt.savefig(out_png, bbox_inches='tight')
            # plt.close(fig)
            plt.show()
            plt.close(fig)
