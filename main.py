import argparse
import tensorflow as tf
import os
import numpy as np
from model import GenreLSTM

import scipy.io
from data_util import load_data


parser = argparse.ArgumentParser(description='How to run this')

parser.add_argument(
    "-current_run",
    type=str,
    help="The name of the model which will also be the name of the session's folder."
)

parser.add_argument(
    "-data_dir",
    type=str,
    default="./data",
    help="Directory of datasets"
)

parser.add_argument(
    "-data_set",
    type=str,
    default="train",
    help="The name of training dataset"
)

parser.add_argument(
    "-runs_dir",
    type=str,
    default="./runs",
    help="The name of the model which will also be the name of the session folder"
)

parser.add_argument(
    "-bi",
    help="True for bidirectional",
    action='store_true'
)

parser.add_argument(
    "-forward_only",
    action='store_true',
    help="True for forward only, False for training [False]"
)

parser.add_argument(
    "-load_model",
    type=str,
    default=None,
    help="Folder name of model to load"
)

parser.add_argument(
    "-load_last",
    action='store_true',
    help="Start from last epoch"
)

args = parser.parse_args()

def setup_dir():

    print('[*] Setting up directory...', flush=True)

    main_path = args.runs_dir
    current_run = os.path.join(main_path, args.current_run)

    data_path = args.data_dir
    train_path = os.path.join(data_path, 'train')
    train_dev_path = os.path.join(data_path, 'train_dev')

    model_path = os.path.join(current_run, 'model')
    logs_path = os.path.join(current_run, 'tmp')
    png_path = os.path.join(current_run, 'png')
    pred_path = os.path.join(current_run, 'predictions')

    if not os.path.exists(current_run):
        os.makedirs(current_run)
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(png_path):
        os.makedirs(png_path)
    if not os.path.exists(pred_path):
        os.makedirs(pred_path)

    dirs = {
            'main_path': main_path,
            'current_run': current_run,
            'model_path': model_path,
            'logs_path': logs_path,
            'png_path': png_path,
            'train_path': train_path,
            'train_dev_path': train_dev_path,
            'pred_path': pred_path
        }
    return dirs


def main():
    tf.logging.set_verbosity(tf.logging.ERROR)

    dirs = setup_dir()
    data = {}
    data['classical'] = {}
    data['classical']['X'], data['classical']['Y'], _ = load_data(dirs['train_path'])
    input_size = data['classical']['X'][0].shape[1]
    output_size = data['classical']['Y'][0].shape[1]

    network  = GenreLSTM(dirs, input_size=input_size, output_size=output_size, 
                         mini=True, bi=args.bi, batch_size=256)
    network.prepare_model()

    if not args.forward_only:
        if args.load_model:
            # assumes that model name is [name]-[e][epoch_number]
            loaded_epoch = args.load_model.split('.')[0]
            loaded_epoch = loaded_epoch.split('-')[-1]
            loaded_epoch = loaded_epoch[1:]
            print("[*] Loading " + args.load_model + " and continuing from " + loaded_epoch, flush=True)
            loaded_epoch = int(loaded_epoch)
            model = args.load_model
            starting_epoch = loaded_epoch+1
        elif args.load_last:
            # list all the .ckpt files in a tuple (epoch, model_name)
            tree = os.listdir(dirs["model_path"])
            tree.remove('checkpoint')
            files = [(int(file.split('.')[0].split('-')[-1][1:]), file.split('.')[0]) for file in tree]
            # find the properties of the last checkpoint
            files.sort(key = lambda t: t[0])
            target_file = files[-1]
            loaded_epoch = target_file[0]
            model_name = target_file[1]
            model = model_name + ".ckpt"
            print("[*] Loading " + model + " and continuing from epoch " + str(loaded_epoch), flush=True)
            starting_epoch = loaded_epoch+1
        else:
            model = None
            starting_epoch = 0
        network.train(data, model=model, starting_epoch=starting_epoch, 
                      epochs=1001, eval_epoch=10, save_epoch=10)
    else:
        network.load(args.load_model)

if __name__ == '__main__':
    main()
