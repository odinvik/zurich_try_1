import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import os
from PIL import Image

def fft(f):
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(f)))

def ifft(f):
    return np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(f)))

def insert(a, dx, dy, b):
    if dx == dy == 0:
        a[:, :] = b[:, :]
    elif dx == 0:
        a[:, dy:-dy] = b[:, :]
    elif dy == 0:
        a[dx:-dx, :] = b[:, :]
    else:
        a[dx:-dx, dy:-dy] = b[:, :]

# Load mask only once
mask_img = Image.open("mask.png").convert("L")
mask_img = mask_img.resize((64, 64), Image.NEAREST)
mask = np.array(mask_img) / 255.0
mask = (mask > 0.5).astype(float)

def smaller(fil, M, ang):
    ffil = np.mean(fil / 255, 2)
    ffil = ndimage.rotate(ffil, ang, reshape=True)
    nx, ny = ffil.shape
    dx, dy = ndimage.center_of_mass(ffil)
    dx, dy = dx - nx / 2, dy - ny / 2
    dx, dy = int(abs(dx)), int(abs(dy))
    gray = np.zeros(shape=(nx + 2 * dx, ny + 2 * dy))
    insert(gray, dx, dy, ffil)
    nx, ny = gray.shape
    dx, dy = ndimage.center_of_mass(gray)
    dx, dy = dx - nx / 2, dy - ny / 2
    gray = ndimage.shift(gray, (-dx, -dy))
    nx, ny = gray.shape
    dx, dy = ndimage.center_of_mass(gray)
    dx, dy = dx - nx / 2, dy - ny / 2
    print(f'Centering shift: dx={dx:.2f}, dy={dy:.2f}')
    nx -= nx % 2
    ny -= ny % 2
    N = 2
    while N < nx or N < ny:
        N *= 2
    gr = np.zeros((N, N))
    dx = N // 2 - nx // 2
    dy = N // 2 - ny // 2
    insert(gr, dx, dy, gray[:nx, :ny])
    fr = fft(gr)
    lo = N // 2 - M // 2
    hi = N // 2 + M // 2
    fr = fr[lo:hi, lo:hi]
    
    # Apply mask here
    fr_masked = fr * mask

    ngr = ifft(fr_masked).real
    return ngr, fr_masked, fr

# Create output directories if they don't exist
os.makedirs("target", exist_ok=True)
os.makedirs("absolutes", exist_ok=True)
os.makedirs("fouriers", exist_ok=True)

M = 64
rotation_angles = [45, 90, 135, 180]
for n in range(1, 5):
    if n in (61, 73, 82, 104, 105, 121, 123, 144, 170, 186, 205, 208, 211):
        continue
    try:
        img = plt.imread('galaxies/galaxy_' + str(n).zfill(3) + '.jpg')
    except FileNotFoundError:
        print(f"Image {n} not found. Skipping.")
        continue

    for ang in rotation_angles:
        gray, fou, fou_2 = smaller(img, M, ang)
        suffix = f"{str(n).zfill(3)}_{ang}"

        # Save individual figures
        fig, ax = plt.subplots()
        ax.imshow(gray, cmap='gray')
        ax.set_title(f'{n}_{ang}')
        ax.axis('off')
        fig.savefig(f'target/gal_{suffix}.png', bbox_inches='tight')
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.imshow(abs(fou) ** (1 / 4), cmap='gray')
        ax.axis('off')
        fig.savefig(f'absolutes/abs_gal_{suffix}.png', bbox_inches='tight')
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.imshow(fou_2.real, cmap='coolwarm')
        ax.axis('off')
        fig.savefig(f'fouriers/real_gal_{suffix}.png', bbox_inches='tight')
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.imshow(fou_2.imag, cmap='coolwarm')
        ax.axis('off')
        fig.savefig(f'fouriers/imag_gal_{suffix}.png', bbox_inches='tight')
        plt.close(fig)
