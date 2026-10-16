# %%
import numpy as np
import matplotlib.pyplot as plt
import os, sys

vaja_folder = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.dirname(vaja_folder)
sys.path.append(parent_folder)

from OSV_lib import loadImage, displayImage


if __name__ == "__main__":
    I = loadImage(
        os.path.join(vaja_folder, "valley-1024x683-08bit.raw"),
        (1024, 683),
        np.uint8
        )
    displayImage(I, "Vhodna slika")
# %%
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

if __name__ == "__main__":
    oHist, oProb, oCDF, oLevels = computeHistogram(I)
    displayHistogram(oHist, oLevels, "Histogram vhodne slike")
    displayHistogram(oProb, oLevels, "Normaliziran histogram vhodne slike")
    displayHistogram(oCDF, oLevels, "Kumulativni histogram vhodne slike")

#%%

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

if __name__ == "__main__":
    I_eq = equalizeHistogram(I)
    displayImage(I_eq, "Izravnana slika")
    oHist_eq, oProb_eq, oCDF_eq, oLevels_eq = computeHistogram(I_eq)
    displayHistogram(oHist_eq, oLevels_eq, "Histogram izravnane slike")
    displayHistogram(oProb_eq, oLevels_eq, "Normaliziran histogram izravnane slike")
    displayHistogram(oCDF_eq, oLevels_eq, "Kumulativni histogram izravnane slike")


# %%
