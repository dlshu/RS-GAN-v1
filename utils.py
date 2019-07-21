import os
import re
import csv
import time
import pickle
import logging

import torch
from torchvision import datasets, transforms
import torchaudio
import torchvision.utils
from torch.utils import data
import torch.nn.functional as F

from options import HiDDenConfiguration, TrainingOptions
from model.hidden import Hidden

from PIL import Image
from pycocotools.coco import COCO
import random
import numpy as np
def image_to_tensor(image):
    """
    Transforms a numpy-image into torch tensor
    :param image: (batch_size x height x width x channels) uint8 array
    :return: (batch_size x channels x height x width) torch tensor in range [-1.0, 1.0]
    """
    image_tensor = torch.Tensor(image)
    image_tensor.unsqueeze_(0)
    image_tensor = image_tensor.permute(0, 3, 1, 2)
    image_tensor = image_tensor / 127.5 - 1
    return image_tensor


def tensor_to_image(tensor):
    """
    Transforms a torch tensor into numpy uint8 array (image)
    :param tensor: (batch_size x channels x height x width) torch tensor in range [-1.0, 1.0]
    :return: (batch_size x height x width x channels) uint8 array
    """
    image = tensor.permute(0, 2, 3, 1).cpu().numpy()
    image = (image + 1) * 127.5
    return np.clip(image, 0, 255).astype(np.uint8)


def save_images(original_images, watermarked_images, epoch, folder, resize_to=None):
    images = original_images[:original_images.shape[0], :, :, :].cpu()
    watermarked_images = watermarked_images[:watermarked_images.shape[0], :, :, :].cpu()

    # scale values to range [0, 1] from original range of [-1, 1]
    images = (images + 1) / 2
    watermarked_images = (watermarked_images + 1) / 2

    if resize_to is not None:
        images = F.interpolate(images, size=resize_to)
        watermarked_images = F.interpolate(watermarked_images, size=resize_to)

    stacked_images = torch.cat([images, watermarked_images], dim=0)
    filename = os.path.join(folder, 'epoch-{}.png'.format(epoch))
    torchvision.utils.save_image(stacked_images, filename, original_images.shape[0], normalize=False)

# images saving with noise
def save_images_with_noise(original_images, watermarked_images, noise_images, epoch, folder, resize_to=None):
    images = original_images[:original_images.shape[0], :, :, :].cpu()
    watermarked_images = watermarked_images[:watermarked_images.shape[0], :, :, :].cpu()
    noise_images = noise_images[:watermarked_images.shape[0], :, :, :].cpu()

    images = (images + 1) / 2
    watermarked_images = (watermarked_images + 1) / 2
    noise_images = (noise_images + 1) / 2

    if resize_to is not None:
        images = F.interpolate(images, size=resize_to)
        watermarked_images = F.interpolate(watermarked_images, size=resize_to)
        noise_images = F.interpolate(noise_images, size=resize_to)

    stacked_images = torch.cat([images, watermarked_images, noise_images], dim=0)
    filename = os.path.join(folder, 'epoch-{}.png'.format(epoch))
    torchvision.utils.save_image(stacked_images, filename, original_images.shape[0], normalize=False)


def save_audio(caption, decoded_messages, folder, epoch, sample_rate=16000):
    # caption is the original audio
    caption = caption.contiguous().view(caption.size(0), -1).cpu()
    decoded_messages = decoded_messages.contiguous().view(decoded_messages.size(0), -1).cpu()
    for (i, audio) in enumerate(decoded_messages):
        filename = os.path.join(folder, f'epoch-{epoch}-{i}.wav')
        torchaudio.save(filename, audio, sample_rate)
    for (i, audio) in enumerate(caption):
        filename = os.path.join(folder, f'epoch-{epoch}-{i}-original.wav')
        torchaudio.save(filename, audio, sample_rate)


def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def last_checkpoint_from_folder(folder: str):
    last_file = sorted_nicely(os.listdir(folder))[-1]
    last_file = os.path.join(folder, last_file)
    return last_file


def save_checkpoint(model: Hidden, experiment_name: str, epoch: int, checkpoint_folder: str):
    """ Saves a checkpoint at the end of an epoch. """
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    checkpoint_filename = f'{experiment_name}--epoch-{epoch}.pyt'
    checkpoint_filename = os.path.join(checkpoint_folder, checkpoint_filename)
    logging.info('Saving checkpoint to {}'.format(checkpoint_filename))
    checkpoint = {
        'enc-dec-model': model.encoder_decoder.state_dict(),
        'enc-dec-optim': model.optimizer_enc_dec.state_dict(),
        'discrim-model': model.discriminator.state_dict(),
        'discrim-optim': model.optimizer_discrim.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, checkpoint_filename)
    logging.info('Saving checkpoint done.')


# def load_checkpoint(hidden_net: Hidden, options: Options, this_run_folder: str):
def load_last_checkpoint(checkpoint_folder):
    """ Load the last checkpoint from the given folder """
    last_checkpoint_file = last_checkpoint_from_folder(checkpoint_folder)
    checkpoint = torch.load(last_checkpoint_file)

    return checkpoint, last_checkpoint_file


def model_from_checkpoint(hidden_net, checkpoint):
    """ Restores the hidden_net object from a checkpoint object """
    hidden_net.encoder_decoder.load_state_dict(checkpoint['enc-dec-model'])
    hidden_net.optimizer_enc_dec.load_state_dict(checkpoint['enc-dec-optim'])
    hidden_net.discriminator.load_state_dict(checkpoint['discrim-model'])
    hidden_net.optimizer_discrim.load_state_dict(checkpoint['discrim-optim'])


def load_options(options_file_name) -> (TrainingOptions, HiDDenConfiguration, dict):
    """ Loads the training, model, and noise configurations from the given folder """
    with open(os.path.join(options_file_name), 'rb') as f:
        train_options = pickle.load(f)
        noise_config = pickle.load(f)
        hidden_config = pickle.load(f)
        # for backward-capability. Some models were trained and saved before .enable_fp16 was added
        if not hasattr(hidden_config, 'enable_fp16'):
            setattr(hidden_config, 'enable_fp16', False)

    return train_options, hidden_config, noise_config


def get_data_loaders(hidden_config: HiDDenConfiguration, train_options: TrainingOptions, vocab=None):
    """ Get torch data loaders for training and validation. The data loaders take a crop of the image,
    transform it into tensor, and normalize it."""
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomCrop((hidden_config.H, hidden_config.W), pad_if_needed=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'test': transforms.Compose([
            transforms.CenterCrop((hidden_config.H, hidden_config.W)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    audio_transforms = {
        'train': torchaudio.transforms.Compose([
            torchaudio.transforms.PadTrim(16384)
        ]),
        'valid': torchaudio.transforms.Compose([
            torchaudio.transforms.PadTrim(16384)
        ])
    }

    #audio_data = torchaudio.datasets.VCTK('./audio_data', transform=audio_transforms['train'])
    #audio_data_size = len(audio_data)
    #train_size = int(0.8*audio_data_size)
    #train_images = datasets.ImageFolder(train_options.train_folder, data_transforms['train'])
    #train_audios, val_audios = torch.utils.data.random_split(audio_data, [train_size, audio_data_size-train_size])

    train_audios = AudioDataset('./audio_data/speech/train', hidden_config.embed_size, normalization=True,
                                transform=audio_transforms['train'])
    val_audios = AudioDataset('./audio_data/speech/valid', hidden_config.embed_size, normalization=True,
                              transform=audio_transforms['valid'])

    train_data = CocoDataset(root=train_options.train_folder, audio=train_audios, json=train_options.ann_train, vocab=vocab, sample=10000, transform=data_transforms['train'])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_options.batch_size, shuffle=True,
                                                num_workers=4, collate_fn=collate_fn)

    #validation_images = datasets.ImageFolder(train_options.validation_folder, data_transforms['test'])
    val_data = CocoDataset(root=train_options.validation_folder, audio=val_audios, json=train_options.ann_val, vocab=vocab, sample=1000, transform=data_transforms['test'])
    validation_loader = torch.utils.data.DataLoader(val_data, batch_size=train_options.batch_size,
                                                     shuffle=False, num_workers=4, collate_fn=collate_fn)

    return train_loader, validation_loader


def log_progress(losses_accu):
    log_print_helper(losses_accu, logging.info)


def print_progress(losses_accu):
    log_print_helper(losses_accu, print)


def log_print_helper(losses_accu, log_or_print_func):
    max_len = max([len(loss_name) for loss_name in losses_accu])
    for loss_name, loss_value in losses_accu.items():
        log_or_print_func(loss_name.ljust(max_len + 4) + '{:.4f}'.format(loss_value.avg))


def create_folder_for_run(runs_folder, experiment_name):
    if not os.path.exists(runs_folder):
        os.makedirs(runs_folder)

    this_run_folder = os.path.join(runs_folder, f'{experiment_name} {time.strftime("%Y.%m.%d--%H-%M-%S")}')

    os.makedirs(this_run_folder)
    os.makedirs(os.path.join(this_run_folder, 'checkpoints'))
    os.makedirs(os.path.join(this_run_folder, 'images'))
    os.makedirs(os.path.join(this_run_folder, 'audio'))

    return this_run_folder


def write_losses(file_name, losses_accu, epoch, duration):
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if epoch == 1:
            row_to_write = ['epoch'] + [loss_name.strip() for loss_name in losses_accu.keys()] + ['duration']
            writer.writerow(row_to_write)
        row_to_write = [epoch] + ['{:.4f}'.format(loss_avg.avg) for loss_avg in losses_accu.values()] + [
            '{:.0f}'.format(duration)]
        writer.writerow(row_to_write)


class AudioDataset(data.Dataset):

    def __init__(self, root, embed_size, normalization=True, transform=None):
        self.root = root
        self.normalization = normalization
        self.audio_list = os.listdir(root)
        self.len = len(self.audio_list)
        self.embed_size = embed_size
        self.transform = transform


    def __getitem__(self, item):

        audio = torchaudio.load(os.path.join(self.root, self.audio_list[item % self.len]),
                                normalization=True)
        audio = self.transform(audio[0])
        return audio.reshape(-1, self.embed_size)

    def __len__(self):
        return self.len


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, json, vocab, audio, sample=None, transform=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        if sample is not None:
            self.ids = random.sample(self.ids, sample)
        self.vocab = vocab
        self.transform = transform
        self.audio = audio

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        audios = self.audio
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # # Convert caption (string) to word ids.
        # tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        # caption = []
        # caption.append(vocab('<start>'))
        # caption.extend([vocab(token) for token in tokens])
        # caption.append(vocab('<end>'))
        # target = torch.Tensor(caption)

        # generate encrypt keys using
        keys = np.random.permutation(512)
        ekeys = np.eye(512)[keys]
        dkeys = np.transpose(ekeys)

        # audio tuple (data, class)
        audio = audios[index]
        return image, audio, torch.Tensor(ekeys), torch.Tensor(dkeys)

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ekeys, dkeys = zip(*data)

    lengths = [len(cap) for cap in captions]

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images, ekeys, dkeys, targets = torch.stack(images, 0), torch.stack(ekeys, 0), torch.stack(dkeys, 0),  torch.stack(captions, 0)

    # # Merge captions (from tuple of 1D tensor to 2D tensor).

    # targets = torch.zeros(len(captions), max(lengths)).long()
    # for i, cap in enumerate(captions):
    #     end = lengths[i]
    #     targets[i, :end] = cap[:end]
    # targets = targets[torch.randperm(targets.size(0))]
    return images, ekeys, dkeys, targets, lengths
