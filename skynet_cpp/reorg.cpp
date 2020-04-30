#include "CNN.h"

void reorg(float *ifm, float *ofm, layer l)
{
    for(int c=0; c<l.ic; c++)
    {
        for(int h=0; h<l.oh; h++)
        {
            for(int w=0; w<l.ow; w++)
            {
                int ofm_index = c*l.oh*l.ow + h*l.ow + w;
                int ifm_index = c*l.ih*l.iw + (2*h)*l.iw + (2*w);
                ofm[ofm_index] = ifm[ifm_index];
            }
        }
    }
    for(int c=l.ic; c<2*l.ic; c++)
    {
        for(int h=0; h<l.oh; h++)
        {
            for(int w=0; w<l.ow; w++)
            {
                int ofm_index = c*l.oh*l.ow + h*l.ow + w;
                int ifm_index = (c-l.ic)*l.ih*l.iw + (2*h)*l.iw + (2*w+1);
                ofm[ofm_index] = ifm[ifm_index];
            }
        }
    }
    for(int c=2*l.ic; c<3*l.ic; c++)
    {
        for(int h=0; h<l.oh; h++)
        {
            for(int w=0; w<l.ow; w++)
            {
                int ofm_index = c*l.oh*l.ow + h*l.ow + w;
                int ifm_index = (c-2*l.ic)*l.ih*l.iw + (2*h+1)*l.iw + (2*w);
                ofm[ofm_index] = ifm[ifm_index];
            }
        }
    }
    for(int c=3*l.ic; c<4*l.ic; c++)
    {
        for(int h=0; h<l.oh; h++)
        {
            for(int w=0; w<l.ow; w++)
            {
                int ofm_index = c*l.oh*l.ow + h*l.ow + w;
                int ifm_index = (c-3*l.ic)*l.ih*l.iw + (2*h+1)*l.iw + (2*w+1);
                ofm[ofm_index] = ifm[ifm_index];
            }
        }
    }
}

void Reorg1(DT32* ifm,  DT IBUF[32][43][83], int Cx)
{
    int h_, w_, ofm_index;
    for (int h=1; h<=20; h++)
    {
        for (int w=1; w<=40; w++)
        {
            h_ = 2*h-1;
            w_ = 2*w-1;
            ofm_index = Cx*83*163 + h_*163 + w_;
            for (int c=0; c<32; c++)
            {
                IBUF[c][h][w] = ifm[ofm_index].data[c];
            }
        }
        for (int w=42; w<=81; w++)
        {
            h_ = 2*h-1;
            w_ = 2*w-2;
            ofm_index = Cx*83*163 + h_*163 + w_;
            for (int c=0; c<32; c++)
            {
                IBUF[c][h][w] = ifm[ofm_index].data[c];
            }
        }
    }
    for (int h=22; h<=41; h++)
    {
        for (int w=1; w<=40; w++)
        {
            h_ = 2*h-2;
            w_ = 2*w-1;
            ofm_index = Cx*83*163 + h_*163 + w_;
            for (int c=0; c<32; c++)
            {
                IBUF[c][h][w] = ifm[ofm_index].data[c];
            }
        }
        for (int w=42; w<=81; w++)
        {
            h_ = 2*h-2;
            w_ = 2*w-2;
            ofm_index = Cx*83*163 + h_*163 + w_;
            for (int c=0; c<32; c++)
            {
                IBUF[c][h][w] = ifm[ofm_index].data[c];
            }
        }
    }
}

void Reorg2(DT32* ifm,  DT IBUF[32][43][83], int Cx)
{
    int h_, w_, ofm_index;
    for (int h=1; h<=20; h++)
    {
        for (int w=1; w<=40; w++)
        {
            h_ = 2*h-1;
            w_ = 2*w;
            ofm_index = Cx*83*163 + h_*163 + w_;
            for (int c=0; c<32; c++)
            {
                IBUF[c][h][w] = ifm[ofm_index].data[c];
            }
        }
        for (int w=42; w<=81; w++)
        {
            h_ = 2*h-1;
            w_ = 2*w-1;
            ofm_index = Cx*83*163 + h_*163 + w_;
            for (int c=0; c<32; c++)
            {
                IBUF[c][h][w] = ifm[ofm_index].data[c];
            }
        }
    }
    for (int h=22; h<=41; h++)
    {
        for (int w=1; w<=40; w++)
        {
            h_ = 2*h-2;
            w_ = 2*w;
            ofm_index = Cx*83*163 + h_*163 + w_;
            for (int c=0; c<32; c++)
            {
                IBUF[c][h][w] = ifm[ofm_index].data[c];
            }
        }
        for (int w=42; w<=81; w++)
        {
            h_ = 2*h-2;
            w_ = 2*w-1;
            ofm_index = Cx*83*163 + h_*163 + w_;
            for (int c=0; c<32; c++)
            {
                IBUF[c][h][w] = ifm[ofm_index].data[c];
            }
        }
    }
}

void Reorg3(DT32* ifm,  DT IBUF[32][43][83], int Cx)
{
    int h_, w_, ofm_index;
    for (int h=1; h<=20; h++)
    {
        for (int w=1; w<=40; w++)
        {
            h_ = 2*h;
            w_ = 2*w-1;
            ofm_index = Cx*83*163 + h_*163 + w_;
            for (int c=0; c<32; c++)
            {
                IBUF[c][h][w] = ifm[ofm_index].data[c];
            }
        }
        for (int w=42; w<=81; w++)
        {
            h_ = 2*h;
            w_ = 2*w-2;
            ofm_index = Cx*83*163 + h_*163 + w_;
            for (int c=0; c<32; c++)
            {
                IBUF[c][h][w] = ifm[ofm_index].data[c];
            }
        }
    }
    for (int h=22; h<=41; h++)
    {
        for (int w=1; w<=40; w++)
        {
            h_ = 2*h-1;
            w_ = 2*w-1;
            ofm_index = Cx*83*163 + h_*163 + w_;
            for (int c=0; c<32; c++)
            {
                IBUF[c][h][w] = ifm[ofm_index].data[c];
            }
        }
        for (int w=42; w<=81; w++)
        {
            h_ = 2*h-1;
            w_ = 2*w-2;
            ofm_index = Cx*83*163 + h_*163 + w_;
            for (int c=0; c<32; c++)
            {
                IBUF[c][h][w] = ifm[ofm_index].data[c];
            }
        }
    }
}

void Reorg4(DT32* ifm,  DT IBUF[32][43][83], int Cx)
{
    int h_, w_, ofm_index;
    for (int h=1; h<=20; h++)
    {
        for (int w=1; w<=40; w++)
        {
            h_ = 2*h;
            w_ = 2*w;
            ofm_index = Cx*83*163 + h_*163 + w_;
            for (int c=0; c<32; c++)
            {
                IBUF[c][h][w] = ifm[ofm_index].data[c];
            }
        }
        for (int w=42; w<=81; w++)
        {
            h_ = 2*h;
            w_ = 2*w-1;
            ofm_index = Cx*83*163 + h_*163 + w_;
            for (int c=0; c<32; c++)
            {
                IBUF[c][h][w] = ifm[ofm_index].data[c];
            }
        }
    }
    for (int h=22; h<=41; h++)
    {
        for (int w=1; w<=40; w++)
        {
            h_ = 2*h-1;
            w_ = 2*w;
            ofm_index = Cx*83*163 + h_*163 + w_;
            for (int c=0; c<32; c++)
            {
                IBUF[c][h][w] = ifm[ofm_index].data[c];
            }
        }
        for (int w=42; w<=81; w++)
        {
            h_ = 2*h-1;
            w_ = 2*w-1;
            ofm_index = Cx*83*163 + h_*163 + w_;
            for (int c=0; c<32; c++)
            {
                IBUF[c][h][w] = ifm[ofm_index].data[c];
            }
        }
    }
}