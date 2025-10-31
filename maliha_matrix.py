#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 20:44:30 2025

@author: 1dh
"""

import torch
from torch import nn
import numpy as np
import tqdm


# %%
class SparseAMat_2D(nn.Module):
    def __init__(
            self,
            Nz,
            Ny,
            Nx,
            angles,
            N_det=None,  # num detector channels in sino
            c_rot=0.0,  # center of rotation offset in detectors
            del_p=1.00,  # pixel width
            del_t=None,  # detector width
            device=None,
    ):

        super().__init__()
        # user defined parameters
        self.Nz = Nz
        self.Ny = Ny
        self.Nx = Nx
        self.angles = angles
        if N_det is None:
            self.N_det = self.Nx
        else:
            self.N_det = N_det
        self.c_rot = c_rot
        self.del_p = del_p
        if del_t is None:
            self.del_t = del_p
        else:
            self.del_t = del_t
        if device is None:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = device

        # internal parameters
        self.LEN_PIX = 511  # spatial resolution for Detector-Pixel computation. Higher LEN_PIX, higher resolution
        self.LEN_DET = 101  # each detector is "split" into LEN_DET elements to account for sensitivity variation across its aperture
        self.Nviews = len(angles)

        # constants for A matrix calculation
        self.t0 = -(
                    self.N_det - 1) * self.del_t / 2 - self.c_rot * self.del_t  # t axis coord of center of 0th detector channel
        self.x0 = -(self.Nx - 1) * self.del_p / 2  # x coordinate of bottom left pixel center
        self.y0 = -(self.Ny - 1) * self.del_p / 2  # y coordinate of bottom left pixel center
        self.pix_prof = self._computePixelProfilesParallelBeam()
        self.det_prof = self._square_sino_profile()

        self.A_mat = self._compAmatrix()

    def forward(self, x_dense):
        # Perform sparse-dense matrix multiplication
        B, R, C = x_dense.shape

        output = x_dense.flatten(start_dim=1)
        output = torch.sparse.mm(self.A_mat, output.t()).t()  # Adjust for correct dimensions
        output = torch.reshape(output, (B, self.Nviews, self.N_det))
        return output

    def _computePixelProfilesParallelBeam(self, ):
        pix_prof = torch.zeros((self.Nviews, self.LEN_PIX))
        rc = np.sin(np.pi / 4.0)

        for ind in range(self.Nviews):
            ang = self.angles[ind].item()
            while ang >= np.pi / 2: ang -= np.pi / 2.0
            while ang < 0.0: ang += np.pi / 2.0

            if (ang <= np.pi / 4.0):
                maxval = self.del_p / np.cos(ang)
            else:
                maxval = self.del_p / np.cos(np.pi / 2.0 - ang)

            d1 = rc * np.cos(np.pi / 4.0 - ang)
            d2 = rc * np.abs(np.sin(np.pi / 4.0 - ang))

            t_1 = 1.0 - d1
            t_2 = 1.0 - d2
            t_3 = 1.0 + d2
            t_4 = 1.0 + d1

            for jnd in range(self.LEN_PIX):
                t = 2.0 * jnd / self.LEN_PIX;
                if (t <= t_1) or (t > t_4):
                    pix_prof[ind, jnd] = 0.0;
                elif t <= t_2:
                    pix_prof[ind, jnd] = maxval * (t - t_1) / (t_2 - t_1)
                elif t <= t_3:
                    pix_prof[ind, jnd] = maxval
                else:
                    pix_prof[ind, jnd] = maxval * (t_4 - t) / (t_4 - t_3)

        return pix_prof

    def _square_sino_profile(self, ):
        return torch.ones(self.LEN_DET) / self.LEN_DET

    def _compAcol(self, col_ind):
        # t axis coord of center of 0th detector channel
        t0 = self.t0

        im_row = col_ind // self.Nx
        im_col = col_ind % self.Nx

        x = self.x0 + im_col * self.del_p  # x coordinate of pixel j's center
        y = self.y0 + im_row * self.del_p  # y coordinate of pixel j's center

        # proj_count = 0
        A_col_val = []
        A_col_ind = []
        kvec = torch.arange(0, self.LEN_DET)

        for pr in range(self.Nviews):
            pnd = pr * self.N_det
            theta = (self.angles[pr]).to(self.device)
            t_min = (y * torch.cos(theta) - x * torch.sin(theta) - self.del_p).to(self.device)
            t_max = (t_min + 2 * self.del_p).to(self.device)

            ind_min = torch.ceil((t_min - t0) / self.del_t - 0.5);
            ind_min = int(np.max([ind_min, 0]))
            ind_max = (t_max - t0) / self.del_t + 0.5;
            ind_max = int(np.min([ind_max, self.N_det - 1]))

            const1 = t0 - self.del_t / 2.0
            const2 = self.del_t / (self.LEN_DET - 1)
            const3 = (self.del_p - (y * np.cos(theta) - x * np.sin(theta))).to(self.device)
            const4 = (self.LEN_PIX - 1) / (2 * self.del_p)

            if ind_max >= ind_min:
                ivals = torch.arange(ind_min, ind_max + 1)  # slow variable
                karr, iarr = torch.meshgrid(kvec, ivals, indexing='xy', )
                tarr = (const1 + iarr * self.del_t + karr * const2).to(self.device).to(self.device)
                pix_prof_indarr = torch.round((tarr + const3) * const4, ).type(torch.int64).to(self.device)

                for i in range(ind_min, ind_max + 1):
                    ## for loop implementation equivalent to C code
                    # Aval = 0
                    # for k in range(self.LEN_DET):
                    #     t = const1 + i*self.del_t + k*const2
                    #     pix_prof_ind = torch.round((t+const3)*const4,).type(torch.int64)
                    #     if 0 <= pix_prof_ind < self.LEN_PIX:
                    #         Aval = Aval + self.det_prof[k]*self.pix_prof[pr,pix_prof_ind]

                    # vectorized implementation
                    # print('col = {}, i = {}'.format(col_ind,i))
                    # tvec = const1 + i*self.del_t + kvec*const2
                    # pix_prof_indvec = torch.round((tvec+const3)*const4,).type(torch.int64)
                    pix_prof_indvec = (pix_prof_indarr[i - ind_min]).to(self.device)
                    selected_inds = ((pix_prof_indvec >= 0) & (pix_prof_indvec < self.LEN_PIX)).to(self.device)
                    b = (selected_inds.nonzero()).to(self.device)
                    Aval = torch.sum(self.det_prof[b] * self.pix_prof[pr, pix_prof_indvec[b]])

                    if Aval > 0:
                        A_col_val.append(Aval)
                        A_col_ind.append(pnd + i)
                        # proj_count = proj_count + 1
        return (A_col_val, A_col_ind)

    def _compAmatrix(self, ):
        M = self.N_det * self.Nviews  # num rows
        N = self.Ny * self.Nx  # num cols
        # initialize empty sparse matrix
        A_mat = torch.sparse_coo_tensor([[], []], [], (M, N), requires_grad=False)
        # pbar = tqdm.tqdm(total=N, desc="Computing columns")

        with tqdm.tqdm(total=N, position=0, leave=True) as pbar:
            # add each column
            for col in range(N):
                A_col_val, A_col_ind = self._compAcol(col)
                col_ind = torch.ones(len(A_col_ind)) * col
                indices = torch.tensor(np.array([A_col_ind, col_ind]), dtype=torch.long)
                A_mat = A_mat + torch.sparse_coo_tensor(indices, A_col_val, (M, N))
                pbar.update(1)  # Update the progress bar by 1%

        return A_mat



