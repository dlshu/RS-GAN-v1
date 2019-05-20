import os
import time
import torch
import numpy as np
import utils
import logging
from collections import defaultdict

from options import *
from model.hidden import Hidden
from average_meter import AverageMeter

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def train(model: Hidden,
          device: torch.device,
          hidden_config: HiDDenConfiguration,
          train_options: TrainingOptions,
          this_run_folder: str,
          tb_logger, vocab):
    """
    Trains the HiDDeN model
    :param model: The model
    :param device: torch.device object, usually this is GPU (if avaliable), otherwise CPU.
    :param hidden_config: The network configuration
    :param train_options: The training settings
    :param this_run_folder: The parent folder for the current training run to store training artifacts/results/logs.
    :param tb_logger: TensorBoardLogger object which is a thin wrapper for TensorboardX logger.
                Pass None to disable TensorboardX logging
    :return:
    """

    train_data, val_data = utils.get_data_loaders(hidden_config, train_options, vocab)
    file_count = len(train_data.dataset)
    if file_count % train_options.batch_size == 0:
        steps_in_epoch = file_count // train_options.batch_size
    else:
        steps_in_epoch = file_count // train_options.batch_size + 1

    print_each = 10
    images_to_save = 8
    saved_images_size = (512, 512)

    for epoch in range(train_options.start_epoch, train_options.number_of_epochs + 1):
        logging.info('\nStarting epoch {}/{}'.format(epoch,
                                                     train_options.number_of_epochs))
        logging.info('Batch size = {}\nSteps in epoch = {}'.format(
            train_options.batch_size, steps_in_epoch))
        training_losses = defaultdict(AverageMeter)
        epoch_start = time.time()
        step = 1
        for image, ekeys, dkeys, caption, length in train_data:
            image, caption, ekeys, dkeys = image.to(device), caption.to(device), ekeys.to(device), dkeys.to(device)

            losses, _ = model.train_on_batch([image, ekeys, dkeys, caption, length])

            for name, loss in losses.items():
                training_losses[name].update(loss)
            if step % print_each == 0 or step == steps_in_epoch:
                logging.info(
                    'Epoch: {}/{} Step: {}/{}'.format(epoch, train_options.number_of_epochs, step, steps_in_epoch))
                utils.log_progress(training_losses)
                logging.info('-' * 40)
            step += 1

        train_duration = time.time() - epoch_start
        logging.info('Epoch {} training duration {:.2f} sec'.format(
            epoch, train_duration))
        logging.info('-' * 40)
        utils.write_losses(os.path.join(
            this_run_folder, 'train.csv'), training_losses, epoch, train_duration)
        if tb_logger is not None:
            tb_logger.save_losses(training_losses, epoch)
            tb_logger.save_grads(epoch)
            tb_logger.save_tensors(epoch)

        first_iteration = True
        validation_losses = defaultdict(AverageMeter)
        logging.info('Running validation for epoch {}/{}'.format(epoch,
                                                                 train_options.number_of_epochs))
        for image, ekeys, dkeys, caption, length in val_data:
            image, caption, ekeys, dkeys = image.to(device), caption.to(device), ekeys.to(device), dkeys.to(device)

            losses, (encoded_images, noised_images, decoded_messages, predicted_sents) = \
                model.validate_on_batch([image, ekeys, dkeys, caption, length])

            #print(predicted)
            #exit()
            predicted_sents = predicted_sents.cpu().numpy()
            for i in range(train_options.batch_size):
                try:
                    #print(''.join([vocab.idx2word[int(w)] + ' ' for w in predicted.cpu().numpy()[i::train_options.batch_size]][1:length[i]-1]))
                    print("".join([vocab.idx2word[int(idx)]+' ' for idx in predicted_sents[i]]))
                    break
                except IndexError:
                    print(f'{i}th batch does not have enough length.')

            for name, loss in losses.items():
                validation_losses[name].update(loss)
            if first_iteration:
                if hidden_config.enable_fp16:
                    image = image.float()
                    encoded_images = encoded_images.float()
                utils.save_images(image.cpu()[:images_to_save, :, :, :],
                                  encoded_images[:images_to_save,
                                                 :, :, :].cpu(),
                                  epoch,
                                  os.path.join(this_run_folder, 'images'), resize_to=saved_images_size)
                first_iteration = False

        utils.log_progress(validation_losses)
        logging.info('-' * 40)
        utils.save_checkpoint(model, train_options.experiment_name, epoch, os.path.join(
            this_run_folder, 'checkpoints'))
        utils.write_losses(os.path.join(this_run_folder, 'validation.csv'), validation_losses, epoch,
                           time.time() - epoch_start)
