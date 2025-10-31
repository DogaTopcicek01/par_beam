#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 14:36:15 2025

@author: 1dh
"""

import torch


def eval_btilde(delta, bsr, p, q, sigma_x, T):
    delta = torch.abs(delta)
    delta_scl = delta / (T * sigma_x)
    btilde = torch.zeros(delta.shape)
    btilde = bsr * delta ** (p - 2) / (2 * sigma_x ** p) * delta_scl ** (q - p) * (q / p + delta_scl ** (q - p)) / (
                1 + delta_scl ** (q - p))
    return btilde


def qggmrf_loss(
        img3d,
        p=1.2,
        q=2.0,
        T=1.0,
        SigmaX=16.0,
        SigmaY=1.0,
        b_nearest=1.0,
        b_diag=0.707,
        b_interslice=1.0,
):
    delta_nn = img3d - torch.roll(img3d, shifts=(1,), dims=(1,))
    delta_ne = img3d - torch.roll(img3d, shifts=(1, -1), dims=(1, 2))
    delta_ee = img3d - torch.roll(img3d, shifts=(1,), dims=(2))
    delta_se = img3d - torch.roll(img3d, shifts=(-1, -1), dims=(1, 2))
    delta_zz = img3d - torch.roll(img3d, shifts=(1,), dims=(0))

    with torch.no_grad():
        # Shift along multiple dimensions
        abs_delta_nn = torch.abs(delta_nn)
        abs_delta_ne = torch.abs(delta_ne)
        abs_delta_ee = torch.abs(delta_ee)
        abs_delta_se = torch.abs(delta_se)
        abs_delta_zz = torch.abs(delta_zz)

        # def get_btilde(delta, bsr, p, q, sigma_x, T):
        btilde_nn = torch.ones(abs_delta_nn.shape) * b_nearest / (p * SigmaX ** p)
        btilde_nn = torch.where(abs_delta_nn > 0, eval_btilde(delta_nn, b_nearest, p, q, SigmaX, T), btilde_nn)

        btilde_ne = torch.ones(abs_delta_ne.shape) * b_diag / (p * SigmaX ** p)
        btilde_ne = torch.where(abs_delta_ne > 0, eval_btilde(delta_ne, b_diag, p, q, SigmaX, T), btilde_ne)

        btilde_ee = torch.ones(abs_delta_ee.shape) * b_nearest / (p * SigmaX ** p)
        btilde_ee = torch.where(abs_delta_ee > 0, eval_btilde(delta_ee, b_nearest, p, q, SigmaX, T), btilde_ee)

        btilde_se = torch.ones(abs_delta_se.shape) * b_diag / (p * SigmaX ** p)
        btilde_se = torch.where(abs_delta_se > 0, eval_btilde(delta_se, b_diag, p, q, SigmaX, T), btilde_se)

        btilde_zz = torch.ones(abs_delta_zz.shape) * b_interslice / (p * SigmaX ** p)
        btilde_zz = torch.where(abs_delta_zz > 0, eval_btilde(delta_zz, b_interslice, p, q, SigmaX, T), btilde_nn)

    loss = torch.sum(btilde_nn * delta_nn ** 2 + btilde_ne * delta_ne ** 2 +
                     btilde_ee * delta_ee ** 2 + btilde_se * delta_se ** 2 +
                     btilde_zz * delta_zz ** 2)

    return loss






