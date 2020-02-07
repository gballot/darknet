import numpy as np
import json
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

base = "/home/gballot/NTU/FSPT Yolo/darknet/results/"
files = (
            ("multiple_val-clip-layer-1-5percent/valid_0to20__final.json",
            "multiple_val-no-clip-layer-1-5percent/valid_0to20__final.json"),
            ("multiple_val-clip-layer-2-5percent/valid_0to22__final.json",
            "multiple_val-no-clip-layer-2-5percent/valid_0to22__final.json"),
            ("multiple_val-clip-variation-gini-gain-5percent/valid_0to15__final.json",
            "multiple_val-no-clip-variation-gini-gain-5percent/valid_0to15__final.json"),
            ("multiple_val-clip-gini0.03-5percent/valid_0to10__final.json",
            "multiple_val-no-clip-gini0.03-5percent/valid_0to10__final.json"),
            ("multiple_val-clip-gini0.15-5percent/valid_0to11__final.json",
            "multiple_val-no-clip-gini0.15-5percent/valid_0to11__final.json"),
            ("multiple_val-clip-best-layers-1-and-2-high-gini-10percent/valid_0to7__final.json",
            "multiple_val-no-clip-best-layers-1-and-2-high-gini-10percent/valid_0to7__final.json"),
            ("multiple_val-clip-gini-gain-100percent/valid_0to15__final.json",
            "multiple_val-no-clip-gini-gain-100percent/valid_0to15__final.json"),
        )
save = "/home/gballot/NTU/FSPT Yolo/darknet/results/figures/"

for i in range(len(files)):
    # Clip
    with open(base + files[i][0]) as f:
        data_clip = json.load(f)
    data_one_fspt_thresh_clip = [d for d in data_clip if d["fspt_thresh"] == 0.7]
    score_clip = [d["score"] for d in data_one_fspt_thresh_clip]
    layer1_clip = [d["fspt_layers"][2]["input_layers"][0] for d in data_one_fspt_thresh_clip]
    layer2_clip = [d["fspt_layers"][1]["input_layers"][0] for d in data_one_fspt_thresh_clip]
    layer3_clip = [d["fspt_layers"][0]["input_layers"][0] for d in data_one_fspt_thresh_clip]
    gini_thresh_clip = [d["fspt_layers"][0]["criterion_args"]["gini_gain_thresh"] for d in data_one_fspt_thresh_clip]
    # No clip
    with open(base + files[i][1]) as f:
        data_no_clip = json.load(f)
    data_one_fspt_thresh_no_clip = [d for d in data_no_clip if d["fspt_thresh"] == 0.7]
    score_no_clip = [d["score"] for d in data_one_fspt_thresh_no_clip]
    layer1_no_clip = [d["fspt_layers"][2]["input_layers"][0] for d in data_one_fspt_thresh_no_clip]
    layer2_no_clip = [d["fspt_layers"][1]["input_layers"][0] for d in data_one_fspt_thresh_no_clip]
    layer3_no_clip = [d["fspt_layers"][0]["input_layers"][0] for d in data_one_fspt_thresh_no_clip]
    gini_thresh_no_clip = [d["fspt_layers"][0]["criterion_args"]["gini_gain_thresh"] for d in data_one_fspt_thresh_no_clip]
    # Figures
    plt.figure(figsize=(20,10))
    if i == 0:
        plt.plot(layer1_clip, score_clip, '.', label="score clip")
        plt.plot(layer1_no_clip, score_no_clip, '.', label="score no clip")
        plt.xlabel("feature layer 1");
    if i == 1:
        plt.plot(layer2_clip, score_clip, '.', label="score clip")
        plt.plot(layer2_no_clip, score_no_clip, '.', label="score no clip")
        plt.xlabel("feature layer 2");
    if i >= 2:
        plt.plot(gini_thresh_clip, score_clip, '.', label="score clip")
        plt.plot(gini_thresh_no_clip, score_no_clip, '.', label="score no clip")
        plt.xlabel("gini gain threshold");
    plt.title(files[i][0])
    plt.ylabel("score");
    plt.legend()
    #plt.yscale("log");
    plt.savefig(save + str(i) + ".png")
