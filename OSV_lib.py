"""Knjiznica funkcij, ki jih napisemo tekom vaj OSV.

Sem prenesemo funkcije iz posameznih vaj, da jih lahko uvozimo v naslednjih vajah:
    from OSV_lib import loadImage, displayImage, saveImage
"""
import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------
## VAJA 1

def loadImage(iPath, iSize, iType):
    # najprej preberemo iz diska v spomin
    fid = open(iPath, "rb")

    # shranimo vsebino pomnilnika v buffer spremenljivko
    buffer = fid.read()

    # ugotovimo stevilo prebranih slikovnih elementov
    buffer_len = len(np.frombuffer(dtype=iType, buffer=buffer))

    # ce je elementov vec kot pikslov, je slika barvna -> 3 kanali (R G B)
    if buffer_len != np.prod(iSize):
        oImage_shape = (iSize[1], iSize[0], 3)
    else:
        oImage_shape = (iSize[1], iSize[0])

    # order F (fortran style) -> dimenzije so invertirane, legacy format
    oImage = np.ndarray(oImage_shape, dtype=iType, buffer=buffer, order="F")

    fid.close()

    return oImage


def displayImage(iImage, iTitle):
    # spremenljivka, zato da lahko sliko kasneje referenciramo
    fig = plt.figure()

    plt.title(iTitle)

    plt.imshow(
        iImage,
        cmap=plt.cm.gray,
        vmin=0,
        vmax=255,
        aspect="equal",
        extent=(
            0 - 0.5,
            iImage.shape[1] - 0.5,
            iImage.shape[0] - 0.5,
            0 - 0.5
        )
    )

    return fig


def saveImage(iImage, iPath, iType):
    # "wb" = write binary
    fid = open(iPath, "wb")

    # zapisemo v enakem (fortran) vrstnem redu, kot ga bere loadImage
    fid.write(iImage.astype(iType).tobytes(order="F"))

    fid.close()
