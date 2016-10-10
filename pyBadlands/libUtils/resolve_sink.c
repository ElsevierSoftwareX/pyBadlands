//~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~//
//                                                                                   //
//  This file forms part of the Badlands surface processes modelling application.    //
//                                                                                   //
//  For full license and copyright information, please refer to the LICENSE.md file  //
//  located at the project root, or contact the authors.                             //
//                                                                                   //
//~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~//

#include <stdio.h>
#include <stdlib.h>

typedef struct _stack {
    int size; // number of allocated elements in stack
    int len; // number of items in stack
    int *items;
} stack;

void stack_init(stack *s) {
    s->size = 256;
    s->len = 0;
    s->items = malloc(s->size * sizeof(int));
}

void stack_push(stack *s, int item) {
    if (s->len >= s->size) {
        // double the size of the stack so we have room for another item
        s->size *= 2;
        s->items = realloc(s->items, s->size * sizeof(int));
    }

    s->items[s->len] = item;
    s->len++;
}

void stack_pop(stack *s) {
    s->len--;
}

int stack_peek(stack *s) {
    return s->items[s->len - 1];
}

void stack_del(stack *s) {
    free(s->items);
}

void resolve_sink(int receivers[], int sinks[], int nid, double sea, double elev[], int *sink_id)
{
    stack stack_obj;
    stack *s = &stack_obj;
    stack_init(s);

    // Not found, set up a proper search
    // We build the stack from the end
    stack_push(s, nid);

    while (s->len) {
        int cid = stack_peek(s);
        int recvr = receivers[cid];

        if (cid == recvr || elev[cid] < sea) {
            // EITHER this is a sink node OR we're under the sea, so deposit directly onto this node
            sinks[cid] = cid;
            // cid is now resolved, pop it
            stack_pop(s);
        } else {
            // find recvr's sink
            if (sinks[recvr] >= 0) {
                sinks[cid] = sinks[recvr];
                stack_pop(s);
                continue;
            } else {
                stack_push(s, recvr);
            }
        }
    }

    stack_del(s);

    *sink_id = sinks[nid];
}
