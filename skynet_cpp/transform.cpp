#include "CNN.h"

void stitch(DT* ifm[4], DT* ofm, layer l)
{   
    int offset_h[4];
    offset_h[0] = offset_h[1] = 1;
    offset_h[2] = offset_h[3] = l.oh + 2;
    int offset_w[4];
    offset_w[0] = offset_w[2] = 1;
    offset_w[1] = offset_w[3] = l.ow + 2;

    for(int p=0; p<4; p++)
    {
        for(int c=0; c<l.oc; c++)
        {
            for(int h=0; h<l.oh; h++)
            {
                for(int w=0; w<l.ow; w++)
                {
                    int ifm_index = c*l.oh*l.ow + h*l.ow + w;
                    int ofm_index = c*(l.oh*2+3)*(l.ow*2+3) + (h+offset_h[p])*(l.ow*2+3) + (w+offset_w[p]);
                    ofm[ofm_index] = ifm[p][ifm_index];
                }
            }
        }
    }
}

void distitch(DT* ifm, DT* ofm[4], layer l)
{   
    int offset_h[4];
    offset_h[0] = offset_h[1] = 1;
    offset_h[2] = offset_h[3] = l.oh + 2;
    int offset_w[4];
    offset_w[0] = offset_w[2] = 1;
    offset_w[1] = offset_w[3] = l.ow + 2;

    for(int p=0; p<4; p++)
    {
        for(int c=0; c<l.oc; c++)
        {
            for(int h=0; h<l.oh; h++)
            {
                for(int w=0; w<l.ow; w++)
                {
                    int ifm_index = c*(l.oh*2+3)*(l.ow*2+3) + (h+offset_h[p])*(l.ow*2+3) + (w+offset_w[p]);
                    int ofm_index = c*l.oh*l.ow + h*l.ow + w;
                    ofm[p][ofm_index] = ifm[ifm_index];
                }
            }
        }
    }
}


void fm_DT_2_DT32(DT* in, DT32* out, layer l)
{
	for (int Mx = 0; Mx < l.oc / 32; Mx++)
	{
		for (int i = 0; i < (2*l.oh+3)*(2*l.ow+3); i++)
		{
			for (int tm = 0; tm < 32; tm++)
			{
				out[Mx*(2*l.oh+3)*(2*l.ow+3)+i].data[tm]=in[(tm+Mx*32)*(2*l.oh+3)*(2*l.ow+3)+i];
			}
		}
	}
}

void fm_DT32_2_DT(DT32* in, DT* out, layer l)
{
	for (int Mx = 0; Mx < l.oc / 32; Mx++)
	{
		for (int i = 0; i < (2*l.oh+3)*(2*l.ow+3); i++)
		{
			for (int tm = 0; tm < 32; tm++)
			{
				out[(tm+Mx*32)*(2*l.oh+3)*(2*l.ow+3) + i] = in[Mx*(2*l.oh+3)*(2*l.ow+3)+i].data[tm];
			}
		}
	}
}

void dwconv_w_DT_2_DT32(DT* in, DT32* out, layer l)
{
        for(int Nx = 0; Nx < l.ic/32; Nx++)
        {
            for (int k = 0; k < l.k*l.k; k++)
            {
                for (int tn = 0; tn < 32; tn++)
                {
                    out[Nx*l.k*l.k + k].data[tn] = in[(Nx*32+tn)*l.k*l.k + k];
                }
            }
        }
}

void w_DT_2_DT32(DT* in, DT32* out, layer l)
{
    for(int oc = 0; oc < l.oc; oc++)
    {
        for(int Nx = 0; Nx < l.ic/32; Nx++)
        {
            for (int k = 0; k < l.k*l.k; k++)
            {
                for (int tn = 0; tn < 32; tn++)
                {
                    out[(oc*l.ic/32 + Nx)*l.k*l.k + k].data[tn] = in[(oc*l.ic+Nx*32+tn)*l.k*l.k + k];
                }
            }
        }
    }
}

void pwconv_w_DT_2_DT32(DT* in, DT32* out, layer l)
{
    for(int Mx=0; Mx<l.oc/32; Mx++)
    {
        for(int Nx=0; Nx<l.ic/32; Nx++)
        {
            for(int m=0; m<32; m++)
            {
                for (int n = 0; n<32; n++)
                {
                    out[(Mx*l.ic/32+Nx)*32+m].data[n] = in[(Mx*32+m)*l.ic+(Nx*32+n)];
                }
            }
        }
    }
}

void b_DT_2_DT32(DT* in, DT32* out, layer l)
{
	for (int Mx = 0; Mx < l.oc / 32; Mx++)
	{
		for (int tm = 0; tm < 32; tm++)
		{
			out[Mx].data[tm]=in[tm+Mx*32];
		}
	}
}