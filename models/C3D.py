# Deepmind I3D ? : https://github.com/deepmind/kinetics-i3d
# Original inspiration : https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Tran_Learning_Spatiotemporal_Features_ICCV_2015_paper.pdf
# tuto : https://towardsdatascience.com/pytorch-step-by-step-implementation-3d-convolution-neural-network-8bf38c70e8b3
# Get from : https://github.com/csuhuihui/pytorch-c3d

import torch
import torch.nn as nn


class C3D(nn.Module):
    """
    The C3D network.
    """

    def __init__(self, num_classes):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(
            kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1)
        )

        self.conv6a = nn.Conv3d(512, 512, kernel_size=(2, 2, 2), padding=(1, 1, 1))
        self.pool6 = nn.MaxPool3d(
            kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1)
        )

        self.fc7 = nn.Linear(25600, 4096)
        self.fc8 = nn.Linear(4096, 4096)
        self.fc9 = nn.Linear(4096, num_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)

        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)

        x = self.relu(self.conv6a(x))
        x = self.pool6(x)

        x = torch.flatten(x, start_dim=1)

        x = self.relu(self.fc7(x))
        x = self.dropout(x)
        x = self.relu(self.fc8(x))
        x = self.dropout(x)

        logits = self.fc9(x)

        return logits
