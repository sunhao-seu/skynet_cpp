#include "CNN.h"

void maxpool(float *ifm, float *ofm, layer l)
{
    for (int oc = 0; oc < l.oc; oc++)
    {
        for (int oh = 0; oh < l.oh; oh++)
        {
            for (int ow = 0; ow < l.ow; ow++)
            {
                float odata = -34000000;
                for(int kh = 0; kh < l.k; kh++)
                {
                    for(int kw = 0; kw < l.k; kw++)
                    {
                        float ret = 0;
                        int fw = ow*l.s - l.p + kw;
                        int fh = oh*l.s - l.p + kh;
                        int fm_index = oc * l.ih * l.iw + fh * l.iw + fw;

                        if ((fw < 0) || (fw >(l.iw - 1)) || (fh < 0) || (fh >(l.ih - 1)))
                            ret = 34000000;
                        else
                            ret = ifm[fm_index];

                        if (ret > odata) odata = ret;
                    }
                }
                ofm[oc * l.oh * l.ow + oh * l.ow + ow] = odata;
            }
        }
    }
}


void avgpool(float *ifm, float *ofm, layer l)
{
    for (int oc = 0; oc < l.oc; oc++)
    {
        for (int oh = 0; oh < l.oh; oh++)
        {
            for (int ow = 0; ow < l.ow; ow++)
            {
                float odata = 0;
                for(int kh=0;kh<l.k;kh++)
                {
                    for(int kw=0;kw<l.k;kw++)
                    {
                        float ret = 0;
                        int fw = ow*l.s - l.p + kw;
                        int fh = oh*l.s - l.p + kh;
                        int fm_index = oc * l.ih * l.iw + fh * l.iw + fw;

                        if ((fw < 0) || (fw > (l.iw - 1)) || (fh < 0) || (fh > (l.ih - 1)))
                            ret = 0;
                        else
                            ret = ifm[fm_index];

                        odata = odata + ret;
                    }
                }
                ofm[oc * l.oh * l.ow + oh * l.ow + ow] = odata / (l.k*l.k);
            }
        }
    }
}

DT MAX(DT a, DT b, DT c, DT d)
{
#pragma HLS INLINE
	DT t1 = a > b ? a : b;
	DT t2 = c > d ? c : d;
	return t1 > t2 ? t1 : t2;
}

void POOL(DT IFM[32][43][83], DT OFM[32][43][83])
{
	for(int h=1; h<=20; h++){
		for(int w=1; w<=40; w++){
			for (int c=0; c<32; c++){
                OFM[c][h][w] = MAX(IFM[c][2*h-1][2*w-1],IFM[c][2*h-1][2*w],IFM[c][2*h][2*w-1],IFM[c][2*h][2*w]);
			}
		}
	}
}
