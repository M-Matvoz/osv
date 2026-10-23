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

def computeHistogram(iImage):
    """
    Poipščemo najmanjše število bitov za zapis območja.
    Dinamično območje dobimo iz max vrednosti svivinske slike, ne pa iz podatkovnega tipa.
    """
    nBits = int(np.log2(iImage.max())) + 1

    ## Določimo vektor sivinskih vrednosti oz. dinamičnega območja
    oLevels = np.arange(0, 2**nBits, 1)

    ## Image mora biti tipa int, zato ga castamo v int
    iImage = iImage.astype(np.int64)
    
    ## Inicializiramo histogram z ničlami
    oHist = np.zeros(len(oLevels))

    ## Izračunamo histogram po posameznih sivinskih vrednostih
    for i in range(iImage.shape[0]):
        for j in range(iImage.shape[1]):
            oHist[iImage[i, j]] += 1

    ## noramliziramo histogram
    oProb = oHist / iImage.size

    ## Inštanciramo CDF
    oCDF = np.zeros_like(oProb)

    ## Seštejemo vrednosti do te točke, da dobimo kumulativno porazdelitev
    for i in range(len(oHist)):
        oCDF[i] = oProb[: i + 1].sum()

    return oHist, oProb, oCDF, oLevels

def displayHistogram(iHist, iLevels, iTitle="Histogram"):
    plt.figure()
    plt.bar(iLevels, iHist, width=1.0, edgecolor="black")
    plt.title(iTitle)
    plt.xlim((iLevels.min(), iLevels.max()))
    plt.ylim((0, 1.05 * iHist.max()))
    plt.xlabel("Sivinske vrednosti")
    plt.ylabel("Frekvenca")
    plt.show()

def equalizeHistogram(iImage):

    ## _ pomeni, da vrednosti ne bomo uporabili
    _, _, oCDF, _ = computeHistogram(iImage)

    nBits = int(np.log2(iImage.max())) + 1

    ## hočemo dobiti največjo možno sivinsko vrednost z nBits. -1, ker python šteje od 0
    max_intensity = 2**nBits - 1

    ## Določimo izravnan histogram in sliko
    oImage = np.zeros_like(iImage)

    ## Gremo čez vse piksle slike in izravnamo njihove vrednosti
    for y in range(iImage.shape[0]):
        for x in range(iImage.shape[1]):
            old_intensity = iImage[y, x]

            ## Nova sivinska vrednost določenega poskla = verjetnost stare vrednsti CDF * maks možna sivinska vrednost
            new_intensity = np.floor(oCDF[old_intensity] * max_intensity)
            oImage[y, x] = new_intensity

    return oImage