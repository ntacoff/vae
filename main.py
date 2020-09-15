#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

import model.vae as model
from util import plot_images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
torch_default = {"dtype": torch.float32, "device": device}

# config
learning_rate = 0.001
z_dim = 20
mini_batch_size = 60
(col_plot, row_plot) = (5, 4)
n_test_plot = col_plot * 2


def main():

    transform = transforms.Compose([transforms.ToTensor()])

    trainloader = DataLoader(
        datasets.MNIST(
            root="./datasets/", train=True, download=True, transform=transform
        ),
        batch_size=mini_batch_size,
        shuffle=True,
    )

    testloader = DataLoader(
        datasets.MNIST(
            root="./datasets/", train=False, download=True, transform=transform
        ),
        batch_size=n_test_plot,
        shuffle=False,
    )

    encoder = model.Encoder(z_dim).to(device)
    decoder = model.Decoder(z_dim).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), learning_rate
    )

    for i in range(10):

        with tqdm(total=len(trainloader.dataset)) as progress_bar:
            for real_img, _ in trainloader:

                real_img = real_img.to(device)

                encoder.train()
                decoder.train()

                z_mean, z_var = encoder(real_img)
                kl_divergence = -0.5 * torch.mean(
                    torch.sum(1 + torch.log(z_var) - z_mean ** 2 - z_var, dim=-1)
                )

                epsilon = torch.empty_like(z_mean).normal_(0, 1).to(**torch_default)
                z = z_mean + epsilon * z_var ** 0.5
                fake_img = decoder(z)

                x = real_img.view(real_img.shape[0], -1)
                y = fake_img.view(fake_img.shape[0], -1)
                reconstruction = torch.mean(
                    torch.sum(x * torch.log(y) + (1 - x) * torch.log(1 - y), dim=-1)
                )
                loss = kl_divergence - reconstruction

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                progress_bar.set_postfix(loss=loss.item())
                progress_bar.update(real_img.shape[0])

        # ===== visualization =====

        real_img_for_plot = iter(testloader).next()[0].to(device)

        encoder.eval()
        decoder.eval()

        z_mean, _ = encoder(real_img_for_plot)
        z = z_mean
        fake_img_for_plot = decoder(z)

        img_for_plot = torch.cat((real_img_for_plot, fake_img_for_plot))
        plot_images(img_for_plot, col=col_plot, row=row_plot)
        plt.savefig("img/figure_" + str(i) + ".png")


if __name__ == "__main__":
    main()
