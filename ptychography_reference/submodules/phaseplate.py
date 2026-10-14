import numpy as np
from imageio.v2 import imread
import cv2

from submodules.function import *

def norm_wave(wave):
    return wave / np.sqrt((wave**2).sum())

class PP:
    def __init__(self, image):

        self.raw = norm_wave(image)
        self.n_px, self.px_img, self.df_img, self.px_idx = self._decompose()

    def _decompose(self):
        img_cv2 = normalize(np.copy(self.raw), (0,255)).astype('uint8')
        ret, labels = cv2.connectedComponents(img_cv2)
        n_px = ret-1
        px_img = np.zeros((n_px, self.raw.shape[0], self.raw.shape[1]))
        df_img = np.zeros((n_px, self.raw.shape[0], self.raw.shape[1]), 'complex128')
        px_idx = []
        for i in range(1,ret):
            px_img[i-1][labels==i] = 1
            df_img[i-1] = fft2d(px_img[i-1])
            px_idx.append( np.argwhere(labels==i) )
        return n_px, px_img, df_img, px_idx
    
    def reorder(self, order):
        self.px_img = self.px_img[order]
        self.df_img = self.df_img[order]
        self.px_idx = self.px_idx[order]

    def set_phase(self, phase_vector):
        for (phase, img) in zip(phase_vector, self.df_img):
            img *= np.exp(1j*phase)


# pp = PP(imread('phaseplates/8.png')[...,0])

# fig, ax = plt.subplots(1)
# ax.imshow(pp.raw)
# for px in range(pp.n_px):
#     x,y = com(pp.px_img[px])
#     plt.text(x,y,px,color='black', fontsize=10)
# ax.set_axis_off()
# plt.show()
# plt.savefig('x',dpi=300)
