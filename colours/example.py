import numpy as np
import matplotlib.pyplot as plt

orig = plt.imread('Haeckel_Trochilidae.jpeg')
orig = orig/256

img = 1*orig
img[:,:,2] = 0

cimg = img-1/2
mag = (cimg[:,:,0]**2 + cimg[:,:,1]**2)**(1/2)
phase = np.arctan2(cimg[:,:,1],cimg[:,:,0])

res = 0*cimg
res[:,:,0] = mag*np.cos(phase)
res[:,:,1] = mag*np.sin(phase)
res = res+1/2
res[:,:,2] = 0

def disp(im,til,cm=None):
    plt.clf()
    im = im.clip(0,1)
    if not cm:
        plt.imshow(im)
    else:
        plt.imshow(im,cmap=cm)
    plt.title(til)
    plt.show()
    #plt.pause(3)


disp(orig,'original')
disp(img,'2 channels')
disp(mag,'magnitude','gray')
disp(phase,'phase','coolwarm')
disp(res,'2 channels again')
