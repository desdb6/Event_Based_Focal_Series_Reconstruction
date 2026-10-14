import numpy as np
import matplotlib.pyplot as plt
from colorsys import hls_to_rgb

def fft2d(arr):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(arr)))
def ifft2d(arr):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(arr)))

def show(arr, name = 'x'):
    plt.figure()
    plt.imshow(arr)
    plt.savefig(name, dpi=300)

def plot(arr, name = 'x'):
    plt.figure()
    plt.plot(arr)
    plt.savefig(name, dpi=300)

def safe_div(a,b):
    if b.shape:
        b[b==0] = 1
    return a/b

def com(data):
    ramp = np.arange(data.shape[-1])
    dose = data.sum((-2,-1))
    comx = safe_div((data.sum(-1)*ramp).sum(-1),dose)
    comy = safe_div((data.sum(-2)*ramp).sum(-1),dose)
    com = [comx, comy]
    if len(data.shape)==4:
        com[0][dose==0] = 0
        com[1][dose==0] = 0    
    return com

def normalize(img, minmax):
    img = img.astype('float64')
    img -= img.min()
    img /= img.max()
    img *= (minmax[1]-minmax[0])
    img += minmax[0]
    return img

def colorize(z):
    ### Very handy function thanks to used nadapez on Stack OVerflow, source: https://stackoverflow.com/a/20958684
    z = z/(np.abs(z).max())
    r = np.abs(z)**2
    arg = np.angle(z) 

    h = (arg + np.pi)  / (2 * np.pi) + 0.5
    l = 1.0 - 1.0/(1.0 + r**0.6)
    s = 0.8

    c = np.vectorize(hls_to_rgb) (h,l,s) # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.swapaxes(0,2) 
    return c

def compleximage(cimg, show=True):
   
    extent = [0,cimg.shape[0],0,cimg.shape[1]]
    img = colorize(cimg)
    
    return img, extent

