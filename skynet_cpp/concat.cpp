#include "CNN.h"

void concat(float *ifm1, float *ifm2, float *ofm, layer l1, layer l2)
{
    memcpy(ofm, ifm1, l1.oc*l1.ow*l1.oh*sizeof(DT));
    memcpy(&ofm[l1.oc*l1.ow*l1.oh], ifm2, l2.oc*l2.ow*l2.oh*sizeof(DT));
}