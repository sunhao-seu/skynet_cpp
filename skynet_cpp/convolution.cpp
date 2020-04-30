#include "CNN.h"

void pwconv1x1(float *ifm, float *ofm, float *weight, float *bias, int relu, layer l)
{
    for (int oh = 0; oh < l.oh; oh++)
    {
        for (int ow = 0; ow < l.ow; ow++)
        {
            for (int oc = 0; oc < l.oc; oc++)
            {
                float odata = 0;
                int ofm_index = oc * l.oh*l.ow + oh * l.ow + ow;
                for (int ic = 0; ic < l.ic; ic++)
                {
                    for(int kh = 0; kh < l.k; kh++)
                    {
                        for (int kw = 0; kw < l.k; kw++)
                        {
                            float ret;
                            int fw = ow*l.s - l.p + kw;
                            int fh = oh*l.s - l.p + kh;
                            int ifm_index = ic*l.ih*l.iw + fh*l.iw + fw;
                            int wgt_index = oc * l.ic*l.k*l.k + ic * l.k* l.k + kh * l.k + kw;

							if( (fw<0) || (fh<0) || (fw>(l.iw-1)) || (fh>(l.ih-1)))
                                ret=0;
                            else
                                ret = ifm[ifm_index];
                            ret *= weight[wgt_index];
                            odata += ret;
                        }
                    }
                }
                odata += bias[oc];
                if(relu)
                {
                    if(odata>0)                
                        ofm[ofm_index] = odata;
                    else
                        ofm[ofm_index] = 0;
                }
                else
                {
                    ofm[ofm_index] = odata;
                }
            }
        }
    }
}

void dwconv3x3(float *ifm, float *ofm, float *weight, float *bias, int relu, layer l)
{
    for (int oh = 0; oh < l.oh; oh++)
    {
        for (int ow = 0; ow < l.ow; ow++)
        {
            for (int oc = 0; oc < l.oc; oc++)
            {
                float odata = 0;
                int ofm_index = oc*l.oh*l.ow + oh*l.ow + ow;
                for(int kh = 0; kh < l.k; kh++)
                {
                    for (int kw = 0; kw < l.k; kw++)
                    {   
                        float ret;
                        int fw = ow*l.s - l.p + kw;
                        int fh = oh*l.s - l.p + kh;
                        int ifm_index = oc*l.ih*l.iw + fh*l.iw + fw;
                        int wgt_index = oc*l.k*l.k + kh*l.k + kw;
						if( (fw<0) || (fh<0) || (fw>(l.iw-1)) || (fh>(l.ih-1)))
                            ret=0;
                        else
                            ret = ifm[ifm_index];
                        odata += ret*weight[wgt_index];
                    }                    
                }
                odata += bias[oc];
                if(relu)
                {
                    if(odata>0)                
                        ofm[ofm_index] = odata;
                    else
                        ofm[ofm_index] = 0;
                }
                else
                {
                    ofm[ofm_index] = odata;
                }
            }
        }
    }
}

void DWCONV3X3(DT IFM[32][43][83], DT OFM[32][43][83], DT WBUF3x3[32][3][3])
{
    for(int i=0; i<3; i++){
		for(int j=0; j<3; j++){
			for(int h=1; h<42; h++){
				for(int w=1; w<82; w++){
					for(int c=0; c<32; c++){
						OFM[c][h][w] += WBUF3x3[c][i][j] * IFM[c][h+i-1][w+j-1];
					}
				}
			}
		}
	}
}

void PWCONV1X1(DT IFM[32][43][83], DT OFM[32][43][83], DT WBUF1x1[32][32])
{
	for(int h=1; h<42; h++){
		for(int w=1; w<=82; w++){
			for (int tm=0; tm<32; tm++){
				DT odatatmp = OFM[tm][h][w];
				DT odata = 0;
				for (int tn=0; tn<32; tn++){
					odata += WBUF1x1[tm][tn]*IFM[tn][h][w];
				}
                OFM[tm][h][w] = odata + odatatmp;
			}
		}
	}
}
