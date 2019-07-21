import os
import time
import pprint
import argparse
import torch
import numpy as np
import pickle
import utils
import csv

from model.hidden import Hidden
from noise_layers.noiser import Noiser
from average_meter import AverageMeter

def write_validation_loss(file_name, losses_accu, experiment_name, epoch, write_header=False):
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            row_to_write = ['experiment_name', 'epoch'] + [loss_name.strip() for loss_name in losses_accu.keys()]
            writer.writerow(row_to_write)
        row_to_write = [experiment_name, epoch] + ['{:.4f}'.format(loss_avg.avg) for loss_avg in losses_accu.values()]
        writer.writerow(row_to_write)

def main():
    device = torch.device('cuda')

    parser = argparse.ArgumentParser(description='Output audios using trained model')
    parser.add_argument('--data-dir', '-d', default=os.path.join('.', 'data'), type=str, help='The directory where the data is stored.')
    parser.add_argument('--runs_root', '-r', default=os.path.join('.', 'runs'), type=str,
                        help='The root folder where data about experiments are stored.')
    parser.add_argument('--batch-size', '-b', default=100, type=int, help='Validation batch size.')

    args = parser.parse_args()
    print_each = 25

    completed_runs = [o for o in os.listdir(args.runs_root)
                      if os.path.isdir(os.path.join(args.runs_root, o)) and o != 'no-noise-defaults']

    print(completed_runs)
    write_csv_header = True
    current_run = completed_runs[0]
    print(f'Run folder: {current_run}')
    current_run = os.path.join(args.runs_root, current_run)
    options_file = os.path.join(current_run, 'options-and-config.pickle')
    train_options, hidden_config, noise_config = utils.load_options(options_file)
    train_options.train_folder = os.path.join(args.data_dir, 'val')
    train_options.validation_folder = os.path.join(args.data_dir, 'val')
    train_options.batch_size = args.batch_size
    checkpoint, chpt_file_name = utils.load_last_checkpoint(os.path.join(current_run, 'checkpoints'))
    print(f'Loaded checkpoint from file {chpt_file_name}')

    noiser = Noiser(noise_config, device)
    model = Hidden(hidden_config, device, noiser, tb_logger=None)
    utils.model_from_checkpoint(model, checkpoint)

    print('Model loaded successfully. Starting validation run...')
    _, val_data = utils.get_data_loaders(hidden_config, train_options)

    file_count = len(val_data.dataset)
    if file_count % train_options.batch_size == 0:
        steps_in_epoch = file_count // train_options.batch_size
    else:
        steps_in_epoch = file_count // train_options.batch_size + 1

    losses_accu = {}
    step = 0
    os.makedirs(os.path.join(current_run, 'decoded_audio'), exist_ok=True)
    count = 0
    for image, ekeys, dkeys, audio, length in val_data:
        step += 1
        image = image.to(device)
        message = audio
        image, audio, ekeys, dkeys = image.to(device), audio.to(device), ekeys.to(device), dkeys.to(device)
        losses, (_, _, decoded_message) = model.validate_on_batch([image, ekeys, dkeys, audio, length])

        if not losses_accu:  # dict is empty, initialize
            for name in losses:
                losses_accu[name] = AverageMeter()
        for name, loss in losses.items():
            losses_accu[name].update(loss)
        if step % print_each == 0 or step == steps_in_epoch:
            print(f'Step {step}/{steps_in_epoch}')
            utils.print_progress(losses_accu)
            print('-' * 40)

        write_validation_loss(os.path.join(args.runs_root, 'validation_run.csv'), losses_accu, step,
                              checkpoint['epoch'],
                              write_header=write_csv_header)
        write_csv_header = False
        utils.save_audio(message, decoded_message, os.path.join(current_run, 'decoded_audio'), step)
        count += 1
        if count == 3:
            break

if __name__ == '__main__':
    main()