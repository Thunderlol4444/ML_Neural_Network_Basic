import numpy as np
from scipy import signal


def forward_conv(self, X):
    """
        Performs a forward convolution.

        Parameters:
        - X : Last conv layer of shape (m, n_C_prev, n_H_prev, n_W_prev).

        Returns:
        - out: output of convolution.
    """
    self.cache = X
    m, n_C_prev, n_H_prev, n_W_prev = X.shape

    # Define output size.
    n_C = self.n_F
    n_H = int((n_H_prev + 2 * self.p - self.f) / self.s) + 1
    n_W = int((n_W_prev + 2 * self.p - self.f) / self.s) + 1

    out = np.zeros((m, n_C, n_H, n_W))

    for i in range(m):  # For each image.

        for c in range(n_C):  # For each channel.

            for h in range(n_H):  # Slide the filter vertically.
                h_start = h * self.s
                h_end = h_start + self.f

                for w in range(n_W):  # Slide the filter horizontally.
                    w_start = w * self.s
                    w_end = w_start + self.f

                    # Element wise multiplication + sum.
                    out[i, c, h, w] = np.sum(X[i, :, h_start:h_end, w_start:w_end]
                                             * self.W['val'][c, ...]) + self.b['val'][c]
    return out