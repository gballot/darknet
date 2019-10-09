#ifndef LIST_H
#define LIST_H

typedef struct node{
    void *val;
    struct node *next;
    struct node *prev;
} node;

typedef struct list{
    int size;
    node *front;
    node *back;
} list;

#ifdef __cplusplus
extern "C" {
#endif
extern list *make_list();
extern int list_find(list *l, void *val);
extern void list_insert(list *, void *);
extern void **list_to_array(list *l);
extern void free_list(list *l);
extern void list_insert_front(list *l, void *val);
extern void free_list_contents(list *l);
extern void free_list_contents_kvp(list *l);

#ifdef __cplusplus
}
#endif
#endif
