#%%
# Vaja 3: interpolacija slik (sprememba velikosti slike)
#
# Ideja: novo sliko velikosti iSize "polozimo" cez staro sliko. Vsakemu
# pikslu nove slike priredimo (decimalno) koordinato v stari sliki in na tej
# koordinati ocenimo sivinsko vrednost:
#   - red 0 (najblizji sosed): vzamemo vrednost najblizjega piksla
#   - red 1 (bilinearna):      utezeno povprecje 4 sosednjih pikslov
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# OSV_lib.py je v nadrejeni mapi (lab/), zato jo dodamo v sys.path,
# da jo lahko uvozimo ne glede na to, od kod skripto pozenemo
vaja_folder = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.dirname(vaja_folder)
sys.path.append(parent_folder)

from OSV_lib import loadImage, displayImage
# %%
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


#%%
def subtractImages(I1, I2):
    return I1.astype(np.float32) - I2.astype(np.float32)


#%%
if __name__ == "__main__":
    I = loadImage(
        os.path.join(vaja_folder, "pumpkin-200x152-08bit.raw"),
        (200, 152),
        np.uint8
        )
    displayImage(I, "Vhodna slika")

    I0 = interpolateImage(I, scale=2, iOrder=0)
    displayImage(I0, "Interpolacija nictega reda");

    I1 = interpolateImage(I, scale=2, iOrder=1)
    displayImage(I1, "Interpolacija prvega reda");
    
    I_diff = subtractImages(I0, I1)
    displayImage(I_diff, "Razlika med I0 in I1");

# %%
