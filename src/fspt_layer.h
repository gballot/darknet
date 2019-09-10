#include "darknet.h"

void update_fspt_layer(layer l, update_args a);
void forward_fspt_layer(layer l, network net);
void backward_fspt_layer(layer l, network net);
void denormalize_fspt_layer(layer l);
void statistics_fspt_layer(layer l);
