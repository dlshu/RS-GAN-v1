import torch
import torch.nn as nn
from options import HiDDenConfiguration
from model.conv_bn_relu import ConvBNRelu
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Decoder(nn.Module):
    """
    Decoder module. Receives a watermarked image and extracts the watermark.
    The input image may have various kinds of noise applied to it,
    such as Crop, JpegCompression, and so on. See Noise layers for more.
    """

    def __init__(self, config: HiDDenConfiguration):

        super(Decoder, self).__init__()
        self.channels = config.decoder_channels

        layers = [ConvBNRelu(3, self.channels)]
        for _ in range(config.decoder_blocks - 1):
            layers.append(ConvBNRelu(self.channels, self.channels))

        # layers.append(block_builder(self.channels, config.message_length))
        layers.append(ConvBNRelu(self.channels, 2*config.message_length))

        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.layers = nn.Sequential(*layers)

        self.linear = nn.Linear(2*config.message_length, 2*config.message_length)

    def forward(self, image_with_wm):
        x = self.layers(image_with_wm)
        # the output is of shape b x c x 1 x 1, and we want to squeeze out the last two dummy dimensions and make
        # the tensor of shape b x c. If we just call squeeze_() it will also squeeze the batch dimension when b=1.
        x.squeeze_(3).squeeze_(2)
        x = self.linear(x)
        return x


class DecoderRNN(nn.Module):
    def __init__(self, config: HiDDenConfiguration):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()

        embed_size = config.embed_size
        hidden_size = config.message_length
        num_layers = config.num_layers
        max_seg_length = config.max_seg_length
        vocab_size = config.vocab_size

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(2*hidden_size, vocab_size)
        self.max_seg_length = max_seg_length

    def forward(self, features, captions, lengths):
        """Decode feature vectors and generates captions."""

        features = features.reshape(1, features.size(0), -1)

        #packed = pack_padded_sequence(embeddings, lengths, batch_first=True)

        outputs, _ = self.gru(captions, features)
        output = self.linear(outputs[0])
        with torch.no_grad():
            output_detach = pad_packed_sequence(outputs, batch_first=True)[0]
            output_detach = output_detach.contiguous()
            batch_size, max_length, dims = output_detach.size()
            output_detach = self.linear(output_detach.view(batch_size * max_length, dims))
            _, predicted_sents = torch.max(output_detach.data, 1)
            predicted_sents = predicted_sents.view(batch_size, max_length)
        return output, predicted_sents
