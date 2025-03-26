import numpy as np

def tochars(v):
    s = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_@'
    v //= 4
    return ''.join([s[k] for k in v])

if __name__ == '__main__':

    r = np.random.randint(0,256,size=64)
    print(r)
    print(tochars(r))

