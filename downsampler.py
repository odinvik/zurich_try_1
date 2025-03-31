import numpy as np
import matplotlib.pyplot as plt

def fft(f):
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(f)))
    #return np.fft.fft2(f)
def ifft(f):
    return np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(f)))
    #return np.fft.ifft2(f)

fil = plt.imread('galaxy_020.jpg')

fil = plt.imread('bird.jpg')

ffil = np.mean(fil/255,2)
nx,ny = ffil.shape
nx -= nx % 2
ny -= ny % 2
N = 2
while N < nx and N < ny:
    N *= 2
print(N)
gr = np.zeros((N,N))
dx = (N-nx)//2
dy = (N-ny)//2
gr[dx:nx+dx,dy:ny+dy] = ffil[:nx,:ny]

fr = fft(gr)
M = 512
lo = N//2 - M//2
hi = N//2 + M//2
fr = fr[lo:hi,lo:hi]
ngr = ifft(fr).real

plt.imshow(ngr,cmap='gray')
plt.gca().set_aspect(1)
plt.show()