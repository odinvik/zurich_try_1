import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

def fft(f):
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(f)))
    #return np.fft.fft2(f)
def ifft(f):
    return np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(f)))
    #return np.fft.ifft2(f)

def insert(a,dx,dy,b):
    if dx == dy == 0:
        a[:,:] = b[:,:]
    elif dx == 0:
        a[:,dy:-dy] = b[:,:]
    elif dy == 0:
        a[dx:-dx,:] = b[:,:]
    else:
        a[dx:-dx,dy:-dy] = b[:,:]


def smaller(fil,M,ang):
    ffil = np.mean(fil/255,2)
    ffil = ndimage.rotate(ffil,ang,reshape=True)
    nx,ny = ffil.shape
    print(nx,ny)
    if np.sum(ffil[:nx//2,:]) < np.sum(ffil[nx//2:,:]):
        ffil = ndimage.rotate(ffil,180,reshape=True)
    nx,ny = ffil.shape
    dx,dy = ndimage.center_of_mass(ffil)
    dx,dy = dx-nx/2,dy-ny/2
    print(dx,dy)
    dx,dy = int(abs(dx)),int(abs(dy))
    gray = np.zeros(shape=(nx+2*dx,ny+2*dy))
    print(gray.shape)
    print(dx,dy)
    '''
    if dx == dy == 0:
        gray[:,:] = ffil[:,:]
    elif dx == 0:
        gray[:,dy:-dy] = ffil[:,:]
    elif dy == 0:
        gray[dx:-dx,:] = ffil[:,:]
    else:
        gray[dx:-dx,dy:-dy] = ffil[:,:]
    '''
    insert(gray,dx,dy,ffil)
    nx,ny = gray.shape
    dx,dy = ndimage.center_of_mass(gray)
    dx,dy = dx-nx/2,dy-ny/2
    print(dx,dy)
    gray = ndimage.shift(gray,(-dx,-dy))
    nx,ny = gray.shape
    dx,dy = ndimage.center_of_mass(gray)
    dx,dy = dx-nx/2,dy-ny/2
    print(dx,dy)
    '''
    return gray,gray
    ffil = ndimage.rotate(ffil,ang,reshape=True)
    cm = ndimage.center_of_mass(ffil)
    nx,ny = ffil.shape
    print(nx,ny,cm)
    if cm[0] < nx/2:
        ffil = ndimage.rotate(ffil,180,reshape=False)
    '''
    print(nx,ny)
    nx -= nx % 2
    ny -= ny % 2
    N = 2
    while N < nx or N < ny:
        N *= 2
    gr = np.zeros((N,N))
    dx = N//2-nx//2
    dy = N//2-ny//2
    print(N,nx,ny,dx,dy)
    #gr[dx:-dx,dy:-dy] = gray[:nx,:ny]
    insert(gr,dx,dy,gray[:nx,:ny])
    fr = fft(gr)
    lo = N//2 - M//2
    hi = N//2 + M//2
    fr = fr[lo:hi,lo:hi]
    ngr = ifft(fr).real
    return ngr,fr


M = 32
for n in range(1,214):
    if n in (61,73,82,104,105,121,123,144,170,186,205,208,211):
        continue
    img = plt.imread('galaxies/galaxy_'+str(n).zfill(3)+'.jpg')
    gray,fou = smaller(img,M,140)
    plt.clf()
    plt.title(str(n))
    #plt.imshow(gray,cmap='gray')
    plt.imshow(abs(fou)**(1/4),cmap='gray')
    #plt.show()
    plt.pause(0.5)

