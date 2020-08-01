#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import model.vae as model
import util

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_default = {"dtype": torch.float32, "device": device}

# config
learning_rate = 0.001
z_dim = 20
mini_batch_size = 64


def main():

    (x_train, _), (_, _) = util.load_data("mnist")
    data_size = x_train.shape[0]

    encoder = model.Encoder(z_dim)
    decoder = model.Decoder(z_dim)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), learning_rate
    )

    encoder.train()
    decoder.train()

    for i in range(100000):

        img_idx = random.sample(range(data_size), mini_batch_size)
        real_img = x_train[img_idx]

        z_mean, z_var = encoder(real_img)
        kl_divergence = -0.5 * torch.mean(
            torch.sum(1 + torch.log(z_var) - z_mean ** 2 - z_var)
        )

        epsilon = torch.empty_like(z_mean).normal_(0, 1).to(**torch_default)
        z = z_mean + epsilon * z_var ** 0.5
        fake_img = decoder(z)

        x = real_img.view(real_img.shape[0], -1)
        y = fake_img.view(fake_img.shape[0], -1)
        reconstruction = torch.mean(
            torch.sum(x * torch.log(y) + (1 - x) * torch.log(1 - y))
        )
        loss = kl_divergence - reconstruction

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===== visualization =====
        if i % 1000 == 0:
            print("loss: " + str(loss))

            (col, row) = (5, 4)

            real_img_for_plot = x_train[range(col * 2)]

            z_mean, z_std = encoder(real_img_for_plot)
            epsilon = torch.empty_like(z_mean).normal_(0, 1).to(**torch_default)
            z = z_mean + epsilon * z_std
            fake_img_for_plot = decoder(z)

            img_for_plot = torch.cat((real_img_for_plot, fake_img_for_plot))
            util.plot_images(img_for_plot, col=col, row=row)
            plt.savefig("img/figure_" + str(i) + ".png")


if __name__ == "__main__":
    main()
