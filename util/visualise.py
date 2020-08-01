#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt


def plot_images(img_in, row=1, col=1):

    if img_in.shape[1] == 1:
        is_gray = True
    else:
        is_gray = False

    num = 0
    for i in range(row * col):
        plt.subplot(row, col, num + 1)
        if is_gray:
            img = img_in[num].view(28, 28).detach().cpu().numpy()
            plt.imshow(img, "gray")
        else:
            img = img_in[num].permute(1, 2, 0).detach().cpu().numpy()
            plt.imshow(img)
        plt.axis("off")
        num += 1
