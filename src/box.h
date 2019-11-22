#ifndef BOX_H
#define BOX_H
#include "darknet.h"

typedef struct{
    float dx, dy, dw, dh;
} dbox;

extern float box_rmse(box a, box b);
extern dbox diou(box a, box b);
extern box decode_box(box b, box anchor);
extern box encode_box(box b, box anchor);
extern void do_nms_suppression(detection *dets, int *tot_ptr, int classes,
        float thresh);

#endif
