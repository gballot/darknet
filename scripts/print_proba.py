import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

base = "/home/gballot/NTU/FSPT Yolo/darknet/tmp/"
files = (
        "proba.data",
        )

for i in range(len(files)):
    data = np.loadtxt(base + files[i])
    plt.figure(figsize=(20,10))
    tt = -1
    s = []
    t = []
    p = []
    for ligne in data:
        if (ligne[0] != tt):
            s.append([])
            p.append([])
            tt = ligne[0]
            t.append(tt)
        else:
            s[-1].append(ligne[1])
            p[-1].append(ligne[2])
    for j in range(len(t)):
        plt.plot(s[j], p[j], label="t = "+str(t[j]))
        plt.title(files[i])
        plt.xlabel("seuil");
        plt.ylabel("proba");
        plt.legend()
    plt.savefig(base + files[i] + ".png")
