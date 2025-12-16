import numpy as np

def cfar_2d_ca(rd_mag: np.ndarray, guard=(2, 2), train=(8, 8), p_fa: float = 1e-4):
    gd_r, gd_d = guard
    tr_r, tr_d = train

    n_d, n_r = rd_mag.shape
    det = np.zeros_like(rd_mag, dtype=bool)

    n_train = (2*(tr_d+gd_d)+1)*(2*(tr_r+gd_r)+1) - (2*gd_d+1)*(2*gd_r+1)
    alpha = n_train * (p_fa ** (-1.0 / max(n_train, 1)) - 1.0)

    for i in range(tr_d + gd_d, n_d - (tr_d + gd_d)):
        for j in range(tr_r + gd_r, n_r - (tr_r + gd_r)):
            d0, d1 = i-(tr_d+gd_d), i+(tr_d+gd_d)+1
            r0, r1 = j-(tr_r+gd_r), j+(tr_r+gd_r)+1

            cut_d0, cut_d1 = i-gd_d, i+gd_d+1
            cut_r0, cut_r1 = j-gd_r, j+gd_r+1

            window = rd_mag[d0:d1, r0:r1]
            guard_win = rd_mag[cut_d0:cut_d1, cut_r0:cut_r1]

            noise_sum = window.sum() - guard_win.sum()
            noise_mu = noise_sum / max(n_train, 1)
            thr = alpha * noise_mu

            det[i, j] = rd_mag[i, j] > thr

    return det
