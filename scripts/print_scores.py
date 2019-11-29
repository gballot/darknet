import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

base = "/home/gballot/NTU/FSPT Yolo/darknet/results/"
files = (
        "job-print-fspt-no-limit_fspt_3_vehicle",
        "job-print-fspt-no-limit_fspt_3_cyclist",
        "job-print-fspt-no-limit_fspt_3_pedestrian",
        "job-print-fspt-no-limit_fspt_2_vehicle",
        "job-print-fspt-no-limit_fspt_2_cyclist",
        "job-print-fspt-no-limit_fspt_2_pedestrian",
        "job-print-fspt-no-limit_fspt_1_vehicle",
        "job-print-fspt-no-limit_fspt_1_cyclist",
        "job-print-fspt-no-limit_fspt_1_pedestrian"
        )

for i in range(len(files)):
    data = np.loadtxt(base + files[i] + ".txt")
    plt.figure(figsize=(20,10))
    plt.plot(data.T[0], data.T[1], '.')
    plt.title(files[i])
    plt.savefig(base + files[i] + ".png")
