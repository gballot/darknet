#ifndef LIST_H
#define LIST_H
#include "darknet.h"

extern list *make_list();

extern int list_find(list *l, void *val);
extern void *list_pop(list *l);
extern void list_insert(list *, void *);
extern void free_list_contents(list *l);

#endif
