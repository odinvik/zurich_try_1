import numpy as np
import matplotlib.pyplot as plt

def fft(f):
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(f)))
    #return np.fft.fft2(f)
def ifft(f):
    return np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(f)))
    #return np.fft.ifft2(f)


def smaller(fil,M):
    ffil = np.mean(fil/255,2)
    nx,ny = ffil.shape
    nx -= nx % 2
    ny -= ny % 2
    N = 2
    while N < nx and N < ny:
        N *= 2
    gr = np.zeros((N,N))
    dx = (N-nx)//2
    dy = (N-ny)//2
    gr[dx:nx+dx,dy:ny+dy] = ffil[:nx,:ny]
    fr = fft(gr)
    lo = N//2 - M//2
    hi = N//2 + M//2
    fr = fr[lo:hi,lo:hi]
    ngr = ifft(fr).real
    return ngr,fr

img = plt.imread('galaxy_020.jpg')
#img = plt.imread('bird.jpg')

gray,fou = smaller(img,64)
plt.imshow(gray,cmap='gray')
#plt.imshow(abs(fou)**(1/4),cmap='gray')
plt.show()
