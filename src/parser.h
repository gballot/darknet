#ifndef PARSER_H
#define PARSER_H
#include "darknet.h"
#include "network.h"
#include "list.h"

list *labels;

void save_network(network net, char *filename);
void save_weights_double(network net, char *filename);

#endif
