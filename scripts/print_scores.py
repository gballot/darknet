import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

base = "/home/gballot/NTU/FSPT Yolo/darknet/results/"
files = (
        "job-print-fspt-full_fspt_3_vehicle",
        "job-print-fspt-full_fspt_3_cyclist",
        "job-print-fspt-full_fspt_3_pedestrian",
        "job-print-fspt-full_fspt_3_sign",
        "job-print-fspt-full_fspt_2_vehicle",
        "job-print-fspt-full_fspt_2_cyclist",
        "job-print-fspt-full_fspt_2_pedestrian",
        "job-print-fspt-full_fspt_2_sign",
        "job-print-fspt-full_fspt_1_vehicle",
        "job-print-fspt-full_fspt_1_cyclist",
        "job-print-fspt-full_fspt_1_pedestrian",
        "job-print-fspt-full_fspt_1_sign"
        )

for i in range(len(files)):
    data = np.loadtxt(base + files[i] + ".txt")
    plt.figure(figsize=(20,10))
    j_zero = len(data.T[1])
    for j in range(len(data.T[1])):
        if data.T[1][j] == 0:
            j_zero = j;
            break
    y = data.T[1][:j_zero]
    n_samples = data.T[4][:j_zero]
    total_samples = np.sum(data.T[4])
    print(total_samples)
    x = data.T[0][:j_zero]
    #dy = [y[i+1] - y[i] for i in range(j_zero - 1)] + [y[j_zero - 1] - y[j_zero - 2]]
    yy = np.power(y/y[0], 0.3)
    plt.plot(x, yy, '.', label="yy")
    plt.plot(x, n_samples/total_samples, '.', label="n_samples")
    zz = (yy + (n_samples/total_samples))/2
    plt.plot(x, zz, '-', label="zz")
    #plt.plot(x, dy, '.', label="dy")
    plt.title(files[i])
    plt.xlabel("ordered nodes (only non zero scores)");
    plt.ylabel("yy");
    plt.legend()
    #plt.yscale("log");
    plt.savefig(base + files[i] + ".png")
