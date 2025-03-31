import numpy as np
import matplotlib.pyplot as plt

def fft(f):
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(f)))
    #return np.fft.fft2(f)
def ifft(f):
    return np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(f)))
    #return np.fft.ifft2(f)

fil = plt.imread('galaxy_020.jpg')
print(fil.shape)
fil = fil[:554,:,:]


#fil = plt.imread('bird.jpg')

ffil = np.mean(fil/255,2)
nx,ny = ffil.shape
gr = np.zeros((1024,1024))
dx = 250
dy = 100
gr[dx:nx+dx,dy:ny+dy] = ffil

fr = fft(gr)
N = 64
lo0 = fr.shape[0]//2 - N//2
hi0 = fr.shape[0]//2 + N//2
lo1 = fr.shape[1]//2 - N//2
hi1 = fr.shape[1]//2 + N//2
fr = fr[lo0:hi0,lo1:hi1]
ngr = ifft(fr).real



plt.imshow(np.angle(fr))
plt.gca().set_aspect(1)
plt.show()