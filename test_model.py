import torch
import torch.nn
import argparse
import os
import numpy as np
from options import HiDDenConfiguration

import utils
from model.hidden import *
from noise_layers.noiser import Noiser
from PIL import Image
import torchvision.transforms.functional as TF
from build_vocab import Vocabulary
import nltk
import pickle
import random


def randomCrop(img, height, width):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    x = np.random.randint(0, img.shape[1] - width)
    y = np.random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width]
    return img


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    parser = argparse.ArgumentParser(description='Test trained models')
    parser.add_argument('--options-file', '-o', default='options-and-config.pickle', type=str,
                        help='The file where the simulation options are stored.')
    parser.add_argument('--checkpoint-file', '-c', required=True,
                        type=str, help='Model checkpoint file')
    parser.add_argument('--batch-size', '-b', default=12,
                        type=int, help='The batch size.')
    parser.add_argument('--source-image', '-s', required=True, type=str,
                        help='The image to watermark')
    parser.add_argument('--source-text', '-t', required=True, type=str,
                        help='The text to watermark', default='data/val_captions.txt')
    parser.add_argument('--vocab-path', '-v', required=True, type=str,
                        help='The path of vocabulary', default='data/vocab.pkl')

    # parser.add_argument('--times', '-t', default=10, type=int,
    #                     help='Number iterations (insert watermark->extract).')

    args = parser.parse_args()

    train_options, hidden_config, noise_config = utils.load_options(
        args.options_file)
    noiser = Noiser(noise_config, device)

    checkpoint = torch.load(args.checkpoint_file)
    hidden_net = Hidden(hidden_config, device, noiser, None)
    utils.model_from_checkpoint(hidden_net, checkpoint)

    image_pil = Image.open(args.source_image)
    image = randomCrop(np.array(image_pil), hidden_config.H, hidden_config.W)
    image_tensor = TF.to_tensor(image).to(device)
    image_tensor = image_tensor * 2 - 1  # transform from [0, 1] to [-1, 1]
    images = torch.stack([image_tensor for _ in range(args.batch_size)], 0)

    # for t in range(args.times):
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    with open(args.source_text) as f:
        captions = f.readlines()
        captions = random.sample(captions, args.batch_size)

    targets = []
    for i, caption in enumerate(captions):
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        targets.append(target)
    targets.sort(key=lambda x: len(x), reverse=True)

    lengths = [len(cap) for cap in targets]
    captions = torch.zeros(len(targets), max(lengths)).long()

    for i, target in enumerate(targets):
        end = lengths[i]
        captions[i, :end] = target[:end]
    captions = captions.to(device)

    losses, (encoded_images, noised_images, decoded_messages, predicted_sents) = \
        hidden_net.validate_on_batch([images, captions, lengths])

    predicted_sents = predicted_sents.cpu().numpy()
    for i in range(args.batch_size):
        try:
            print("predict     : "+"".join([vocab.idx2word[int(idx)] +
                           ' ' for idx in predicted_sents[i]]))
            print("ground truth: "+"".join([vocab.idx2word[int(idx)] +
                           ' ' for idx in captions[i]]))
        except IndexError:
            print(f'{i}th batch does not have enough length.')

    utils.save_images(images.cpu(), encoded_images.cpu(),
                      'test_%d' % i, '.', resize_to=(256, 256))


if __name__ == '__main__':
    main()
