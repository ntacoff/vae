#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import torch
from keras.datasets import cifar10, mnist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_default = {"dtype": torch.float32, "device": device}


def load_data(data_type):

    if data_type == "mnist":
        # x.shape: (-1, 28, 28)     => (-1, 1, 28, 28)
        # y.shape: (-1)             => (-1)
        (x_train_np, y_train_np), (x_test_np, y_test_np) = mnist.load_data()

        x_train = torch.from_numpy(x_train_np).to(**torch_default).view(-1, 1, 28, 28)
        x_train = torch.min(
            torch.max(x_train / 255, torch.ones_like(x_train) * 1e-6),
            torch.ones_like(x_train) * (1 - 1e-6),
        )

        y_train = torch.from_numpy(y_train_np).to(**torch_default).view(-1)

        x_test = torch.from_numpy(x_test_np).to(**torch_default).view(-1, 1, 28, 28)
        x_test = torch.min(
            torch.max(x_test / 255, torch.ones_like(x_test) * 1e-6),
            torch.ones_like(x_test) * (1 - 1e-6),
        )

        y_test = torch.from_numpy(y_test_np).to(**torch_default).view(-1)

    elif data_type == "cifer10":
        # x.shape: (-1, 32, 32, 3)  => (-1, 3, 32, 32)
        # y.shape: (-1, 1)          => (-1)
        (x_train_np, y_train_np), (x_test_np, y_test_np) = cifar10.load_data()

        x_train = torch.from_numpy(x_train_np).to(**torch_default).permute(0, 3, 1, 2)
        x_train = x_train / 255

        y_train = torch.from_numpy(y_train_np).to(**torch_default).view(-1)

        x_test = torch.from_numpy(x_test_np).to(**torch_default).permute(0, 3, 1, 2)
        x_test = x_test / 255

        y_test = torch.from_numpy(y_test_np).to(**torch_default).view(-1)

    else:
        raise NotImplementedError

    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":

    (x, _), (_, _) = load_data("mnist")

    num = 0
    (row, col) = (5, 5)
    for i in range(row * col):
        plt.subplot(row, col, num + 1)
        image = x[num].view(28, 28).detach().cpu().numpy() / 2 + 0.5
        plt.imshow(image, "gray")
        plt.axis("off")
        num += 1
    plt.show()

    (x, _), (_, _) = load_data("cifer10")

    num = 0
    (row, col) = (5, 5)
    for i in range(row * col):
        plt.subplot(row, col, num + 1)
        img = x[num].permute(1, 2, 0).detach().cpu().numpy() / 2 + 0.5
        plt.imshow(img)
        plt.axis("off")
        num += 1
    plt.show()
