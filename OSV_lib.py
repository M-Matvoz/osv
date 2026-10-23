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


def displayImage(iImage, iTitle="Slika"):
    # spremenljivka, zato da lahko sliko kasneje referenciramo
    fig = plt.figure()

    plt.title(iTitle)

    plt.imshow(
        iImage,
        cmap=plt.cm.gray,
        # vmin=0,
        # vmax=255,
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

def interpolateImage(iImage, iSize=None, scale=None, iOrder:int=0):
    if iSize is None and scale is not None:
        Y, X = iImage.shape
        iSize = (int(X * scale), int(Y * scale))
    elif iSize is None and scale is None:
        raise ValueError("Either iSize or scale must be provided")
    # vhod:  iImage - 2D sivinska slika (numpy array)
    #        iSize  - zelena velikost izhodne slike kot (sirina, visina), enako kot pri loadImage
    #        iOrder - red interpolacije (0 ali 1)
    # izhod: interpolirana slika velikosti iSize

    # izbira metode glede na red interpolacije
    if iOrder == 0:
        return interpolate_zero_order(iImage, iSize, scale)
    elif iOrder == 1:
        return interpolate_first_order(iImage, iSize, scale)
    else:
        raise ValueError("Unsupported interpolation order")

def interpolate_zero_order(iImage, iSize=None, scale=None):
    # interpolacija nictega reda = metoda najblizjega soseda

    # POZOR na vrstni red osi:
    #   iImage.shape = (vrstice, stolpci) = (visina Y, sirina X)
    #   iSize        = (sirina, visina)   - enako kot pri loadImage
    Y, X = iImage.shape
    if iSize is None and scale is not None:
        iSize = (int(X * scale), int(Y * scale))
    if iSize is None and scale is None:
        raise ValueError("Either iSize or scale must be provided")

    # vzorcne tocke v koordinatah VHODNE slike:
    # iSize[0] enakomerno razporejenih tock od 0 do X-1 (in enako za y).
    # Prva in zadnja tocka sovpadata s prvim in zadnjim pikslom vhodne slike,
    # zato se robovi slike ohranijo ("poravnani koti").
    # Korak med tockami je (X-1)/(iSize[0]-1): < 1 pri povecavi, > 1 pri pomanjsavi.
    # Isto, kot bi naredil dx = (X-1)/(iSize[0]-1) in dy = (Y-1)/(iSize[1]-1)
    x = np.linspace(0, X-1, iSize[0])
    y = np.linspace(0, Y-1, iSize[1])

    # mreza koordinat: xv[r, c], yv[r, c] je tocka v vhodni sliki,
    # ki pripada pikslu (vrstica r, stolpec c) izhodne slike.
    # privzeti indexing='xy' vrne matriki oblike (len(y), len(x)) = (vrstice, stolpci),
    # kar se ujema z obliko slike
    xv, yv = np.meshgrid(x, y)

    # prazna izhodna slika oblike (visina, sirina), enak podatkovni tip kot vhodna (npr. uint8)
    oImage = np.zeros((iSize[1], iSize[0]), dtype=iImage.dtype)

    # gremo cez vse piksle izhodne slike: r po vrsticah (y), c po stolpcih (x)
    for r in range(iSize[1]):
        for c in range(iSize[0]):
            # zaokrozimo decimalno koordinato na najblizji celostevilski piksel
            # (np.round zaokrozuje 0.5 na najblizje sodo stevilo, npr. 2.5 -> 2)
            xi = np.round(xv[r, c]).astype(np.int64)
            yi = np.round(yv[r, c]).astype(np.int64)
            # prepisemo vrednost najblizjega piksla -> "kockast" videz pri povecavi
            # (slika se indeksira kot [vrstica, stolpec] = [y, x])
            oImage[r, c] = iImage[yi, xi]
    return oImage

def interpolate_first_order(iImage, iSize=None, scale=None):
    # interpolacija prvega reda = bilinearna interpolacija

    # priprava vzorcnih tock je enaka kot pri interpolaciji nictega reda
    # (shape = (visina, sirina), iSize = (sirina, visina))
    Y, X = iImage.shape
    if iSize is None and scale is not None:
        iSize = (int(X * scale), int(Y * scale))
    if iSize is None and scale is None:
        raise ValueError("Either iSize or scale must be provided")

    x = np.linspace(0, X-1, iSize[0])
    y = np.linspace(0, Y-1, iSize[1])
    xv, yv = np.meshgrid(x, y)
    oImage = np.zeros((iSize[1], iSize[0]), dtype=iImage.dtype)


    # r - row (vrstica, isto kot i), c - column (stolpec, isto kot j)
    hasRun = False
    firstPass = False
    for r in range(iSize[1]):
        for c in range(iSize[0]):
            if not hasRun and r != 0 and c != 0:
                firstPass = True
                hasRun = True
            # decimalna koordinata tocke v vhodni sliki
            xi = xv[r, c]
            yi = yv[r, c]
            if firstPass:
                print(f"Interpolating point ({r}, {c}) with decimal coordinates ({xi}, {yi})")
                

            # stirje sosednji piksli, ki obkrozajo tocko (xi, yi):
            #
            #   (x0,y0) ---- (x1,y0)
            #      |    *       |      * = (xi, yi)
            #   (x0,y1) ---- (x1,y1)
            #
            # x0 = levi, x1 = desni stolpec; y0 = zgornja, y1 = spodnja vrstica.
            # min(..., X-1) prepreci izhod iz slike na zadnjem robu,
            # kjer je xi == X-1 (takrat je dx = 0, zato x1 ne vpliva na rezultat)
            x0 = int(np.floor(xi))
            x1 = min(x0 + 1, X - 1)
            y0 = int(np.floor(yi))
            y1 = min(y0 + 1, Y - 1)
            if firstPass:
                print(f"Surrounding pixels: x0={x0}, x1={x1}, y0={y0}, y1={y1}")

            # relativna oddaljenost tocke od x0 oz. y0, v intervalu [0, 1)
            dx = xi - x0
            dy = yi - y0
            if firstPass:
                print(f"Relative distances: dx={dx}, dy={dy}")
                firstPass = False

            # bilinearna interpolacija: vsak sosed dobi utez, ki je enaka
            # ploscini nasprotnega pravokotnika - blizji kot je sosed tocki,
            # vecjo utez ima. Vsota vseh stirih utezi je 1.
            # (ekvivalentno: najprej linearno interpoliramo po x med y0 in y1,
            # nato se linearno po y med dobljenima vrednostma)
            #
            # OPOMBA: rezultat je decimalno stevilo; ob zapisu v oImage
            # (uint8) se decimalke odrezejo (ne zaokrozijo), npr. 12.9 -> 12
            # slika se indeksira kot [vrstica, stolpec] = [y, x]
            oImage[r, c] = (iImage[y0, x0] * (1 - dx) * (1 - dy) +
                            iImage[y0, x1] * dx * (1 - dy) +
                            iImage[y1, x0] * (1 - dx) * dy +
                            iImage[y1, x1] * dx * dy)
    return oImage

def subtractImages(I1, I2):
    return I1.astype(np.float32) - I2.astype(np.float32)

if __name__ == "__main__":
    print("OSV_lib.py executed directly. Ka te kurac delas da klices knjiznico direktno???")