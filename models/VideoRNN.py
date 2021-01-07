import torch
import torch.nn as nn
from torchvision import models

# https://discuss.pytorch.org/t/how-to-use-pack-sequence-if-we-are-going-to-use-word-embedding-and-bilstm/28184/4

# Inspiration : https://arxiv.org/abs/1603.09025
class VideoRNN(nn.Module):
    def __init__(self, hidden_size, n_classes, batch_size, device, num_layer=1):
        super(VideoRNN, self).__init__()

        self.hidden_size = hidden_size
        self.batch = batch_size
        self.num_layer = num_layer
        self.device = device

        # Loading a VGG16
        vgg = models.vgg16(pretrained=True)

        # Removing the vgg-16 classification layer
        embed = nn.Sequential(*list(vgg.classifier.children())[:-1])
        vgg.classifier = embed

        # Freezing all except 3 last layers
        for param in vgg.parameters():
            param.requires_grad = False

        vgg.classifier[0].requires_grad = True
        vgg.classifier[1].requires_grad = True
        vgg.classifier[2].requires_grad = True
        vgg.classifier[3].requires_grad = True
        vgg.classifier[4].requires_grad = True
        vgg.classifier[5].requires_grad = True

        self.embedding = vgg
        self.gru = nn.LSTM(4096, hidden_size, num_layer, bidirectional=True)

        # Classification layer
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * self.num_layer * 2, 1024),
            nn.ReLU(),
            nn.Linear(1024, n_classes),
        )

    def forward(self, input):
        hidden = torch.zeros(self.num_layer * 2, self.batch, self.hidden_size).to(
            self.device
        )

        c_0 = torch.zeros(self.num_layer * 2, self.batch, self.hidden_size).to(
            self.device
        )

        embedded = self.simple_elementwise_apply(self.embedding, input)
        output, hidden = self.gru(embedded, (hidden, c_0))
        hidden = hidden[0].view(-1, self.hidden_size * self.num_layer * 2)

        output = self.classifier(hidden)

        return output

    def simple_elementwise_apply(self, fn, packed_sequence):
        return torch.nn.utils.rnn.PackedSequence(
            fn(packed_sequence.data), packed_sequence.batch_sizes
        )
