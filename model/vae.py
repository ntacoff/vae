#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)


class Decoder(nn.Module):
    def __init__(self, z_dim):
        super(Decoder, self).__init__()

        self.layer1 = nn.Sequential(nn.Linear(z_dim, 128), nn.ReLU(),)
        self.layer2 = nn.Sequential(nn.Linear(128, 256), nn.ReLU(),)
        self.layer3 = nn.Sequential(nn.Linear(256, 28 * 28), nn.Sigmoid())
        self.layer_reshape = Reshape(1, 28, 28)

        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(
            self.layer1[0].weight.data, mode="fan_in", nonlinearity="relu"
        )
        nn.init.zeros_(self.layer1[0].bias.data)
        nn.init.kaiming_normal_(
            self.layer2[0].weight.data, mode="fan_in", nonlinearity="relu"
        )
        nn.init.zeros_(self.layer2[0].bias.data)
        nn.init.xavier_normal_(
            self.layer3[0].weight.data,
            gain=nn.init.calculate_gain(nonlinearity="sigmoid"),
        )
        nn.init.zeros_(self.layer3[0].bias.data)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer_reshape(out)

        out = torch.max(out, torch.ones_like(out) * 1e-6)
        out = torch.min(out, torch.ones_like(out) * (1 - 1e-6))

        return out


class Encoder(nn.Module):
    def __init__(self, z_dim):
        super(Encoder, self).__init__()

        self.layer_reshape = Reshape(28 * 28)
        self.layer1 = nn.Sequential(nn.Linear(28 * 28, 256), nn.ReLU(),)
        self.layer2 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(),)
        self.layer3 = nn.Sequential(
            nn.Linear(128, z_dim * 2), nn.Sigmoid(), Reshape(z_dim, 2)
        )

        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(
            self.layer1[0].weight.data, mode="fan_in", nonlinearity="relu"
        )
        nn.init.zeros_(self.layer1[0].bias.data)
        nn.init.kaiming_normal_(
            self.layer2[0].weight.data, mode="fan_in", nonlinearity="relu"
        )
        nn.init.zeros_(self.layer2[0].bias.data)
        nn.init.xavier_normal_(
            self.layer3[0].weight.data,
            gain=nn.init.calculate_gain(nonlinearity="sigmoid"),
        )
        nn.init.zeros_(self.layer3[0].bias.data)

    def forward(self, x):
        out = self.layer_reshape(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        return out[:, :, 0], out[:, :, 1]
