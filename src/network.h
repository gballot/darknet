// Oh boy, why am I about to do this....
#ifndef NETWORK_H
#define NETWORK_H
#include "darknet.h"

#include "image.h"
#include "layer.h"
#include "data.h"
#include "tree.h"


#ifdef GPU
extern void pull_network_output(network *net);
#endif

extern list *get_network_layers_by_type(network *net, LAYER_TYPE type);
extern void compare_networks(network *n1, network *n2, data d);
extern char *get_layer_string(LAYER_TYPE a);
extern network *make_network(int n);
extern float network_accuracy_multi(network *net, data d, int n);
extern int get_predicted_class_network(network *net);
extern void print_network(network *net);
extern int resize_network(network *net, int w, int h);
extern void calc_network_cost(network *net);
extern void fit_fspts(network *net, int classes, int refit, int one_thread,
        int merge);
extern void fspt_layers_set_samples(network *net, int refit, int merge);

#endif

