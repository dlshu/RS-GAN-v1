import torch.nn as nn
import torch
from model.encoder import Encoder, EncoderRNN
from model.decoder import Decoder, DecoderRNN
from options import HiDDenConfiguration
from noise_layers.noiser import Noiser

class EncoderDecoder(nn.Module):
    """
    Combines Encoder->Noiser->Decoder into single pipeline.
    The input is the cover image and the watermark message. The module inserts the watermark into the image
    (obtaining encoded_image), then applies Noise layers (obtaining noised_image), then passes the noised_image
    to the Decoder which tries to recover the watermark (called decoded_message). The module outputs
    a three-tuple: (encoded_image, noised_image, decoded_message)
    """
    def __init__(self, config: HiDDenConfiguration, noiser: Noiser):

        super(EncoderDecoder, self).__init__()

        self.encoder = Encoder(config)
        self.encoder = nn.DataParallel(self.encoder)

        self.encode_rnn = EncoderRNN(config)
        #self.encode_rnn = nn.DataParallel(self.encode_rnn)

        self.noiser = noiser

        self.decoder = Decoder(config)
        self.decoder = nn.DataParallel(self.decoder)

        self.decode_rnn = DecoderRNN(config)
        self.adversarial_decode_rnn = DecoderRNN(config)
        #self.decode_rnn = nn.DataParallel(self.decode_rnn)


    def forward(self, image, ekeys, dkeys, caption, length):
        encoded_message = self.encode_rnn(caption, length)
        #print(encoded_message.shape, ekeys.shape)
        #encoded_message = torch.bmm(encoded_message.unsqueeze(1), ekeys).squeeze()
        encoded_image = self.encoder(image, encoded_message)
        noised_and_cover = self.noiser([encoded_image, image])
        noised_image = noised_and_cover[0]
        decoded_message = self.decoder(noised_image)
        #decoded_message = torch.bmm(decoded_message.unsqueeze(1), dkeys).squeeze()
        decode_sentence, predicted_sent = self.decode_rnn(decoded_message, caption, length)

        #adversarial_decode_sentence, adversarial_predicted_sent = \
        #    self.decode_rnn(encoded_message, caption, length)

        return encoded_image, noised_image, decode_sentence, \
            encoded_message, decoded_message, predicted_sent
