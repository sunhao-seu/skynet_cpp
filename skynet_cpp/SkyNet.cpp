#include "CNN.h"

layer config[layer_count] = {
{ "data",    320,160,3,  320,160,32, 0,0,0},    //data    0
{ "dwconv1", 320,160,32, 320,160,32, 3,1,1},    //dwconv1 1
{ "pwconv1", 320,160,32, 320,160,64, 1,1,0},    //pwconv1 2
{ "pool1",   320,160,64, 160,80,64,  2,2,0},    //pool1   3
{ "dwconv2", 160,80,64,  160,80,64,  3,1,1},    //dwconv2 4
{ "pwconv2", 160,80,64,  160,80,96,  1,1,0},    //pwconv2 5
{ "pool2",   160,80,96,  80,40,96,   2,2,0},    //pool2   6
{ "dwconv3", 80,40,96,   80,40,96,   3,1,1},    //dwconv3 7
{ "pwconv3", 80,40,96,   80,40,192,  1,1,0},    //pwconv3 8
{ "reorg",   80,40,192,  40,20,768,  2,2,0},    //reorg   9
{ "pool3",   80,40,192,  40,20,192,  2,2,0},    //pool3   10
{ "dwconv4", 40,20,192,  40,20,192,  3,1,1},    //dwconv4 11
{ "pwconv4", 40,20,192,  40,20,384,  1,1,0},    //pwconv4 12
{ "dwconv5", 40,20,384,  40,20,384,  3,1,1},    //dwconv5 13
{ "pwconv5", 40,20,384,  40,20,512,  1,1,0},    //pwconv5 14
{ "cat",     40,20,192,  40,20,1280, 0,0,0},    //concat  15
{ "dwconv6", 40,20,1280, 40,20,1280, 3,1,1},    //dwconv6 16
{ "pwconv6", 40,20,1280, 40,20,96,   1,1,0},    //pwconv6 17
{ "conv7e",   40,20,96,   40,20,32,   1,1,0},    //conv7   18
};

void Load_WBUF3x3(DT32* weight, DT WBUF3x3[32][3][3], int Mx, layer l)
{
    for(int m=0; m<3; m++)
    {
        for(int n=0; n<3; n++)
        {
            for(int c=0; c<32; c++)
            {
                WBUF3x3[c][m][n] = weight[Mx*9 + m*3 + n].data[c];
            }
        }
    }
}

void Load_WBUF1x1(DT32* weight, DT WBUF1x1[32][32], int Mx, int Nx, layer l)
{
    for(int m=0; m<32; m++)
    {
        for(int n=0; n<32; n++)
        {
            WBUF1x1[m][n] = weight[Mx*l.ic+Nx*32+m].data[n];
        }
    }
}

void Load_BBUF(DT32* bias, DT BBUF[32], int Mx)
{
    for(int c=0; c<32; c++)
    {
        BBUF[c] = bias[Mx].data[c];
    }
}

void Load_FM(DT32* ifm, DT IBUF[32][43][83], int Hx, int Wx, int Cx, layer l)
{
    int tile = l.iw/80;
    int h_offset, w_offset;
    if(tile)
    {
        h_offset = Hx*40 + Hx/tile;
        w_offset = Wx*80 + Wx/tile;
    }
    else
    {
        h_offset = 0;
        w_offset = 0;
    }
        
    for (int h=0; h<42; h++)
    {
        for (int w=0; w<82; w++)
        {
            for (int c=0; c<32; c++)
            {
                int ifm_index = Cx*(l.oh*2+3)*(l.ow*2+3) + (h+h_offset)*(l.ow*2+3) + (w+w_offset);
                IBUF[c][h][w] = ifm[ifm_index].data[c];
            }
        }
    }
}

void Export_CONV(DT32* ofm, DT OBUF[32][43][83], int Hx, int Wx, int Cx, layer l)
{
    int tile = l.iw/80;
    int h_offset, w_offset;
    if(tile)
    {
        h_offset = Hx*40 + Hx/tile;
        w_offset = Wx*80 + Wx/tile;
    }
    else
    {
        h_offset = 0;
        w_offset = 0;
    }
    for (int h=1; h<=40; h++)
    {
        for (int w=1; w<=80; w++)
        {
            for (int c=0; c<32; c++)
            {
                int ofm_index = Cx*(l.oh*2+3)*(l.ow*2+3) + (h+h_offset)*(l.ow*2+3) + (w+w_offset);
                ofm[ofm_index].data[c] = OBUF[c][h][w];
            }
        }
    }
}

void Export_POOL(DT32* ofm, DT OBUF[32][43][83], int Hx, int Wx, int Cx, layer l)
{
    int tile = l.iw/80;
    int h_offset = Hx*20 + Hx/tile;
    int w_offset = Wx*40 + Wx/tile;
    for (int h=1; h<=20; h++)
    {
        for (int w=1; w<=40; w++)
        {
            for (int c=0; c<32; c++)
            {
                int ofm_index = Cx*(l.oh*2+3)*(l.ow*2+3) + (h+h_offset)*(l.ow*2+3) + (w+w_offset);
                ofm[ofm_index].data[c] = OBUF[c][h][w];
            }
        }
    }
}

void Load_FM1(DT32* ifm, DT IBUF[32][43][83], int Cx)
{
    for (int h=0; h<43; h++)
    {
        for (int w=0; w<83; w++)
        {
            int ifm_index = Cx*43*83 + h*83 + w;
            for (int c=0; c<32; c++)
            {
                IBUF[c][h][w] = ifm[ifm_index].data[c];
            }
        }
    }
}

void Export_CONV1(DT32* ofm, DT OBUF[32][43][83], int Cx)
{
    for (int h=0; h<43; h++)
    {
        for (int c=0; c<32; c++)
        {
            OBUF[c][h][41] = 0;
        }
    }
    for (int w=0; w<83; w++)
    {
        for (int c=0; c<32; c++)
        {
            OBUF[c][21][w] = 0;
        }
    }
    for (int h=1; h<42; h++)
    {
        for (int w=1; w<82; w++)
        {
            for (int c=0; c<32; c++)
            {
                int ofm_index = Cx*43*83 + h*83 + w;
                ofm[ofm_index].data[c] = OBUF[c][h][w];
            }
        }
    }
}

void Add_Bias(DT FM[32][43][83], DT BBUF[32], int relu)
{
    for(int h=1; h<=41; h++){
        for(int w=1; w<=81; w++){
            for(int c=0; c<32; c++){
                DT odata = FM[c][h][w];
                odata += BBUF[c];
                if(relu==1)
                {
                    if(odata<0)
                        FM[c][h][w] = 0;
                    else
                        FM[c][h][w] = odata;
                }
            }
        }
    }
}

void Clear_FM(DT FM[32][43][83])
{
    for(int h=0; h<43; h++)
    {
        for(int w=0; w<83; w++)
        {
            for(int c=0; c<32; c++)
            {
                FM[c][h][w] = 0;
            }
        }
    }
}

void Compare(DT FM1[32][43][83], DT FM2[32][43][83])
{
    int error = 0;
    for(int h=1; h<41; h++){
        for(int w=1; w<81; w++){
            for(int c=0; c<32; c++){
                if(abs(FM1[c][h][w]-FM2[c][h][w])>0.001)
                    error++;
                    //printf("FM1[%d][%d][%d]=%f, FM2[%d][%d][%d]=%f\n", c,h,w,FM1[c][h][w],c,h,w,FM2[c][h][w]);
            }
        }
    }
    printf("error count: %d\n", error);
}


int pool1_offset, pool2_offset, pwconv3_offset, pool3_offset, dwconv4_offset, pwconv4_offset, dwconv5_offset, pwconv5_offset, reorg_offset, dwconv6_offset, pwconv6_offset, conv7_offset;
void SkyNet_(DT32* ifm, DT32* ofm, DT32* parameter)
{
    DT FM1[32][43][83]={0};
    DT FM2[32][43][83]={0};
    DT FM3[32][43][83]={0};
    DT FM4[32][43][83]={0};
    DT FM5[32][43][83]={0};

    DT WBUF3x3[4][32][3][3]={0};
    DT WBUF1x1[4][32][32]={0};
    DT BBUF[4][32]={0};

    /*********************************DWCONV1+PWCONV1********************************/
    int weight_offset = 0;
    int bias_offset = weight_offset + config[1].oc*config[1].k*config[1].k/32;
    pool1_offset = 0;
    Load_WBUF3x3(parameter + weight_offset, WBUF3x3[0], 0, config[1]);
    Load_BBUF(parameter + bias_offset, BBUF[0], 0);
    weight_offset = bias_offset + config[1].oc/32;
    bias_offset = weight_offset + config[2].oc*config[2].ic*config[2].k*config[2].k/32;
    Load_WBUF1x1(parameter + weight_offset, WBUF1x1[0], 0, 0, config[2]);
    Load_WBUF1x1(parameter + weight_offset, WBUF1x1[1], 1, 0, config[2]);
    Load_BBUF(parameter + bias_offset, BBUF[1], 0);
    Load_BBUF(parameter + bias_offset, BBUF[2], 1);
    
    for(int Hx=0; Hx<8; Hx++)
    {	
        Load_FM(ifm, FM1, Hx, 0, 0, config[1]);
		for(int Wx=0; Wx<8; Wx++)
        {
            if(Wx%2==0)
            {
                Load_FM(ifm, FM2, Hx, Wx+1, 0, config[1]);
                {
                    DWCONV3X3(FM1, FM3, WBUF3x3[0]);
                    Add_Bias(FM3, BBUF[0], 1);
                }
            }
            else
            {
                Load_FM(ifm, FM1, Hx, Wx+1, 0, config[1]);
                {
                    DWCONV3X3(FM2, FM3, WBUF3x3[0]);
                    Add_Bias(FM3, BBUF[0], 1);
                }
            }
            for(int Cx=0; Cx<2; Cx++)
            {
                PWCONV1X1(FM3, FM4, WBUF1x1[Cx]);
                Add_Bias(FM4, BBUF[Cx+1], 1);
                POOL(FM4, FM5);
                Export_POOL(ofm + pool1_offset, FM5, Hx, Wx, Cx, config[3]);
                Clear_FM(FM4);
            }
            Clear_FM(FM3);
		}
	}

    /*********************************DWCONV2+PWCONV2********************************/
    weight_offset = bias_offset + config[2].oc/32;
    bias_offset = weight_offset + config[4].oc*config[4].k*config[4].k/32;
    pool2_offset = pool1_offset + config[3].oc*(2*config[3].oh+3)*(2*config[3].ow+3)/32;
    Load_WBUF3x3(parameter + weight_offset, WBUF3x3[0], 0, config[4]);
    Load_WBUF3x3(parameter + weight_offset, WBUF3x3[1], 1, config[4]);
    Load_BBUF(parameter + bias_offset, BBUF[0], 0);
    Load_BBUF(parameter + bias_offset, BBUF[1], 1);
    weight_offset = bias_offset + config[4].oc/32;
    bias_offset = weight_offset + config[5].oc*config[5].ic/32;
    Clear_FM(FM2);
    Clear_FM(FM4);
    for(int Hx=0; Hx<4; Hx++)
    {
        for(int Wx=0; Wx<4; Wx++)
        {
            Load_FM(ofm + pool1_offset, FM1, Hx, Wx, 0, config[4]);
            DWCONV3X3(FM1, FM2, WBUF3x3[0]);
            Add_Bias(FM2, BBUF[0], 1);
            //Export_CONV(dwconv2, FM2, Hx, Wx, 0, config[4]);

            Load_FM(ofm + pool1_offset, FM1, Hx, Wx, 1, config[4]);
            DWCONV3X3(FM1, FM3, WBUF3x3[1]);
            Add_Bias(FM3, BBUF[1], 1);
            //Export_CONV(dwconv2, FM3, Hx, Wx, 1, config[4]);
            for(int Mx=0; Mx<3; Mx++)
            {
                Load_WBUF1x1(parameter + weight_offset, WBUF1x1[0], Mx, 0, config[5]);
                PWCONV1X1(FM2, FM4, WBUF1x1[0]);

                Load_WBUF1x1(parameter + weight_offset, WBUF1x1[1], Mx, 1, config[5]);
                PWCONV1X1(FM3, FM4, WBUF1x1[1]);

                Load_BBUF(parameter + bias_offset, BBUF[2], Mx);
                Add_Bias(FM4, BBUF[2], 1);
                
                //Export_CONV(pwconv2, FM4, Hx, Wx, Mx, config[5]);

                POOL(FM4, FM5);
                Export_POOL(ofm + pool2_offset, FM5, Hx, Wx, Mx, config[6]);
                Clear_FM(FM4);
            }
            Clear_FM(FM2);
            Clear_FM(FM3);
        }
    }

    /*********************************DWCONV3+PWCONV3********************************/
    weight_offset = bias_offset + config[6].oc/32;
    bias_offset = weight_offset + config[7].oc*config[7].k*config[7].k/32;
    pwconv3_offset = pool2_offset + config[6].oc*(2*config[6].oh+3)*(2*config[6].ow+3)/32;
    pool3_offset = pwconv3_offset + config[8].oc*(2*config[8].oh+3)*(2*config[8].ow+3)/32;
    Load_WBUF3x3(parameter + weight_offset, WBUF3x3[0], 0, config[7]);
    Load_WBUF3x3(parameter + weight_offset, WBUF3x3[1], 1, config[7]);
    Load_WBUF3x3(parameter + weight_offset, WBUF3x3[2], 2, config[7]);
    Load_BBUF(parameter + bias_offset, BBUF[0], 0);
    Load_BBUF(parameter + bias_offset, BBUF[1], 1);
    Load_BBUF(parameter + bias_offset, BBUF[2], 2);
    weight_offset = bias_offset + config[7].oc/32;
    bias_offset = weight_offset + config[8].oc*config[8].ic/32;
    Clear_FM(FM2);
    Clear_FM(FM3);
    Clear_FM(FM4);

    for(int Hx=0; Hx<2; Hx++)
    {
        for(int Wx=0; Wx<2; Wx++)
        {
            Load_FM(ofm + pool2_offset, FM1, Hx, Wx, 0, config[7]);
            DWCONV3X3(FM1, FM2, WBUF3x3[0]);
            Add_Bias(FM2, BBUF[0], 1);
            //Export_CONV(dwconv, FM2, Hx, Wx, 0, config[7]);
            
            Load_FM(ofm + pool2_offset, FM1, Hx, Wx, 1, config[7]);
            DWCONV3X3(FM1, FM3, WBUF3x3[1]);
            Add_Bias(FM3, BBUF[1], 1);
            //Export_CONV(dwconv, FM3, Hx, Wx, 1, config[7]);
            
            Load_FM(ofm + pool2_offset, FM1, Hx, Wx, 2, config[7]);
            DWCONV3X3(FM1, FM4, WBUF3x3[2]);
            Add_Bias(FM4, BBUF[2], 1);
            //Export_CONV(dwconv, FM4, Hx, Wx, 2, config[7]);

            Clear_FM(FM1);
            for(int Mx=0; Mx<6; Mx++)
            {
                Load_WBUF1x1(parameter + weight_offset, WBUF1x1[0], Mx, 0, config[8]);
                PWCONV1X1(FM2, FM1, WBUF1x1[0]);

                Load_WBUF1x1(parameter + weight_offset, WBUF1x1[1], Mx, 1, config[8]);
                PWCONV1X1(FM3, FM1, WBUF1x1[1]);

                Load_WBUF1x1(parameter + weight_offset, WBUF1x1[2], Mx, 2, config[8]);
                PWCONV1X1(FM4, FM1, WBUF1x1[2]);

                Load_BBUF(parameter + bias_offset, BBUF[3], Mx);
                Add_Bias(FM1, BBUF[3], 1);

                Export_CONV(ofm + pwconv3_offset, FM1, Hx, Wx, Mx, config[8]);
                POOL(FM1, FM5);
                Export_POOL(ofm + pool3_offset, FM5, Hx, Wx, Mx, config[10]);
                Clear_FM(FM1);
            }

            Clear_FM(FM2);
            Clear_FM(FM3);
            Clear_FM(FM4);
        }
    }

    /*********************************DWCONV4+PWCONV4********************************/
    weight_offset = bias_offset + config[8].oc/32;
    bias_offset = weight_offset + config[11].oc*config[11].k*config[11].k/32;
    dwconv4_offset = pool3_offset + config[10].oc*(2*config[10].oh+3)*(2*config[10].ow+3)/32;
    pwconv4_offset = dwconv4_offset + config[11].oc*(2*config[11].oh+3)*(2*config[11].ow+3)/32;
    Load_FM1(ofm + pool3_offset, FM1, 0);
    for(int Nx=0; Nx<6; Nx++)
    {
        Load_WBUF3x3(parameter + weight_offset, WBUF3x3[0], Nx, config[11]);
        Load_BBUF(parameter + bias_offset, BBUF[0], Nx);
        if(Nx%2==0)
        {
            Load_FM1(ofm + pool3_offset, FM3, Nx+1);
            DWCONV3X3(FM1, FM2, WBUF3x3[0]);
        }
        else
        {
            Load_FM1(ofm + pool3_offset, FM1, Nx+1);
            DWCONV3X3(FM3, FM2, WBUF3x3[0]);
        }
        Add_Bias(FM2, BBUF[0], 1);
        Export_CONV1(ofm + dwconv4_offset, FM2, Nx);
        Clear_FM(FM2);
    }

    weight_offset = bias_offset + config[11].oc/32;
    bias_offset = weight_offset + config[12].oc*config[12].ic/32;

    for(int Mx=0; Mx<12; Mx++)
    {
        Load_BBUF(parameter + bias_offset, BBUF[0], Mx);
        Load_FM1(ofm + dwconv4_offset, FM1, 0);
        for(int Nx=0; Nx<6; Nx++)
        {
            Load_WBUF1x1(parameter + weight_offset, WBUF1x1[0], Mx, Nx, config[12]);
            if(Nx%2==0)
            {
                Load_FM1(ofm + dwconv4_offset, FM3, Nx+1);
                PWCONV1X1(FM1, FM2, WBUF1x1[0]);
            }
            else
            {
                Load_FM1(ofm + dwconv4_offset, FM1, Nx+1);
                PWCONV1X1(FM3, FM2, WBUF1x1[0]);
            }
        }
        Add_Bias(FM2, BBUF[0], 1);
        Export_CONV1(ofm + pwconv4_offset, FM2, Mx);
        Clear_FM(FM2);
    }
    /*********************************DWCONV5+PWCONV5********************************/
    weight_offset = bias_offset + config[12].oc/32;
    bias_offset = weight_offset + config[13].oc*config[13].k*config[13].k/32;
    dwconv5_offset = pwconv4_offset + config[12].oc*(2*config[12].oh+3)*(2*config[12].ow+3)/32;
    pwconv5_offset = dwconv5_offset + config[13].oc*(2*config[13].oh+3)*(2*config[13].ow+3)/32;

    Load_FM1(ofm + pwconv4_offset, FM1, 0);
    for(int Nx=0; Nx<12; Nx++)
    {
        Load_WBUF3x3(parameter + weight_offset, WBUF3x3[0], Nx, config[13]);
        Load_BBUF(parameter + bias_offset, BBUF[0], Nx);
        if(Nx%2==0)
        {
            Load_FM1(ofm + pwconv4_offset, FM3, Nx+1);
            DWCONV3X3(FM1, FM2, WBUF3x3[0]);
        }
        else
        {
            Load_FM1(ofm + pwconv4_offset, FM1, Nx+1);
            DWCONV3X3(FM3, FM2, WBUF3x3[0]);
        }
        Add_Bias(FM2, BBUF[0], 1);
        Export_CONV1(ofm + dwconv5_offset, FM2, Nx);
        Clear_FM(FM2);
    }

    weight_offset = bias_offset + config[13].oc/32;
    bias_offset = weight_offset + config[14].oc*config[14].ic/32;

    for(int Mx=0; Mx<16; Mx++)
    {
        Load_FM1(ofm + dwconv5_offset, FM1, 0);
        for(int Nx=0; Nx<12; Nx++)
        {
            Load_WBUF1x1(parameter + weight_offset, WBUF1x1[0], Mx, Nx, config[14]);
            if(Nx%2==0)
            {
                Load_FM1(ofm + dwconv5_offset, FM3, Nx+1);
                PWCONV1X1(FM1, FM2, WBUF1x1[0]);
            }
            else
            {
                Load_FM1(ofm + dwconv5_offset, FM1, Nx+1);
                PWCONV1X1(FM3, FM2, WBUF1x1[0]);
            }
        }
        Load_BBUF(parameter + bias_offset, BBUF[0], Mx);
        Add_Bias(FM2, BBUF[0], 1);
        Export_CONV1(ofm + pwconv5_offset, FM2, Mx);
        Clear_FM(FM2);
    }

    /*********************************REORG+DWCONV6********************************/
    reorg_offset = pwconv5_offset + config[14].oc*(2*config[14].oh+3)*(2*config[14].ow+3)/32;
    weight_offset = bias_offset + config[14].oc/32;
    bias_offset = weight_offset + config[16].oc*config[16].k*config[16].k/32;
    dwconv6_offset = reorg_offset + config[15].oc*(2*config[15].oh+3)*(2*config[15].ow+3)/32;
    for(int Nx=0; Nx<6; Nx++)
    {
        Reorg1(ofm + pwconv3_offset, FM1, Nx);
        //Export_CONV1(ofm + reorg_offset, FM1, Nx);
        Load_WBUF3x3(parameter + weight_offset, WBUF3x3[0], Nx, config[16]);
        Load_BBUF(parameter + bias_offset, BBUF[0], Nx);
        DWCONV3X3(FM1, FM2, WBUF3x3[0]);
        Add_Bias(FM2,BBUF[0],1);
        Export_CONV1(ofm + dwconv6_offset, FM2, Nx);
        Clear_FM(FM2);

        Reorg2(ofm + pwconv3_offset, FM1, Nx);
        //Export_CONV1(ofm + reorg_offset, FM1, Nx+6);
        Load_WBUF3x3(parameter + weight_offset, WBUF3x3[0], Nx+6, config[16]);
        Load_BBUF(parameter + bias_offset, BBUF[0], Nx+6);
        DWCONV3X3(FM1, FM2, WBUF3x3[0]);
        Add_Bias(FM2,BBUF[0],1);
        Export_CONV1(ofm + dwconv6_offset, FM2, Nx+6);
        Clear_FM(FM2);

        Reorg3(ofm + pwconv3_offset, FM1, Nx);
        //Export_CONV1(ofm + reorg_offset, FM1, Nx+12);
        Load_WBUF3x3(parameter + weight_offset, WBUF3x3[0], Nx+12, config[16]);
        Load_BBUF(parameter + bias_offset, BBUF[0], Nx+12);
        DWCONV3X3(FM1, FM2, WBUF3x3[0]);
        Add_Bias(FM2,BBUF[0],1);
        Export_CONV1(ofm + dwconv6_offset, FM2, Nx+12);
        Clear_FM(FM2);

        Reorg4(ofm + pwconv3_offset, FM1, Nx);
        //Export_CONV1(ofm + reorg_offset, FM1, Nx+18);
        Load_WBUF3x3(parameter + weight_offset, WBUF3x3[0], Nx+18, config[16]);
        Load_BBUF(parameter + bias_offset, BBUF[0], Nx+18);
        DWCONV3X3(FM1, FM2, WBUF3x3[0]);
        Add_Bias(FM2,BBUF[0],1);
        Export_CONV1(ofm + dwconv6_offset, FM2, Nx+18);
        Clear_FM(FM2);
    }

    Load_FM1(ofm + pwconv5_offset, FM1, 0);
    for(int Nx=0; Nx<16; Nx++)
    {
        Load_WBUF3x3(parameter + weight_offset, WBUF3x3[0], Nx+24, config[16]);
        Load_BBUF(parameter + bias_offset, BBUF[0], Nx+24);
        if(Nx%2==0)
        {
            Load_FM1(ofm + pwconv5_offset, FM3, Nx+1);
            DWCONV3X3(FM1, FM2, WBUF3x3[0]);
        }
        else
        {
            Load_FM1(ofm + pwconv5_offset, FM1, Nx+1);
            DWCONV3X3(FM3, FM2, WBUF3x3[0]);
        }
        Add_Bias(FM2,BBUF[0],1);
        Export_CONV1(ofm + dwconv6_offset, FM2, Nx+24);
        Clear_FM(FM2);
    }
    /*********************************PWCONV6********************************/
    weight_offset = bias_offset + config[16].oc/32;
    bias_offset = weight_offset + config[17].ic*config[17].oc/32;
    pwconv6_offset = dwconv6_offset + config[16].oc*(2*config[16].oh+3)*(2*config[16].ow+3)/32;
    for(int Mx=0; Mx<3; Mx++)
    {
        Load_FM1(ofm + dwconv6_offset, FM1, 0);
        for(int Nx=0; Nx<40; Nx++)
        {
            Load_WBUF1x1(parameter + weight_offset, WBUF1x1[0], Mx, Nx, config[17]);
            if(Nx%2==0)
            {
                Load_FM1(ofm + dwconv6_offset, FM3, Nx+1);
                PWCONV1X1(FM1, FM2, WBUF1x1[0]);
            }
            else
            {
                Load_FM1(ofm + dwconv6_offset, FM1, Nx+1);
                PWCONV1X1(FM3, FM2, WBUF1x1[0]);
            }
            
        }
        Load_BBUF(parameter + bias_offset, BBUF[0], Mx);
        Add_Bias(FM2, BBUF[0], 1);
        Export_CONV1(ofm + pwconv6_offset, FM2, Mx);
        Clear_FM(FM2);
    }
    /*********************************CONV7********************************/
    weight_offset = bias_offset + config[17].oc/32;
    bias_offset = weight_offset + config[18].ic*config[18].oc/32;
    conv7_offset = pwconv6_offset + config[17].oc*(2*config[17].oh+3)*(2*config[17].ow+3)/32;
    for(int Nx=0; Nx<3; Nx++)
    {
        Load_FM1(ofm + pwconv6_offset, FM1, Nx);
        Load_WBUF1x1(parameter + weight_offset, WBUF1x1[0], 0, Nx, config[18]);
        PWCONV1X1(FM1, FM2, WBUF1x1[0]);
    }
    Export_CONV1(ofm + conv7_offset, FM2, 0);
}

DT32* parameter;
DT* data[4];
DT* data_blob;
DT32* data_blob32;
DT* ofm_blob;
DT32* ofm_blob32;
DT* ofm[4];
void SkyNet_init()
{
    for(int p=0; p<4; p++)
    {
        data[p] = (DT*)sds_alloc(384*160*320*sizeof(DT));
        ofm[p] = (DT*)sds_alloc(64*320*640*sizeof(DT));
    }
    data_blob = (DT*)sds_alloc(384*323*643*sizeof(DT));
    data_blob32 = (DT32*)sds_alloc(384*323*643*sizeof(DT));
    parameter = (DT32*)sds_alloc(444768*sizeof(DT));
    ofm_blob32 = (DT32*)sds_alloc(32*999999*sizeof(DT));
    ofm_blob = (DT*)sds_alloc(64*643*323*sizeof(DT));
}

void SkyNet()
{
    load_weight(parameter, 444768);
    for(int p=0; p<4; p++)
        load_fm(data[p], config[0]);
    stitch(data, data_blob, config[0]);
    fm_DT_2_DT32(data_blob, data_blob32, config[0]);

    SkyNet_(data_blob32, ofm_blob32, parameter);

    fm_DT32_2_DT(&ofm_blob32[pool1_offset], ofm_blob, config[3]);
    distitch(ofm_blob, ofm, config[3]);
    for(int p=0; p<4; p++)
    {
        check_fm(ofm[p], config[3]);
    }

    fm_DT32_2_DT(&ofm_blob32[pool2_offset], ofm_blob, config[6]);
    distitch(ofm_blob, ofm, config[6]);
    for(int p=0; p<4; p++)
    {
        check_fm(ofm[p], config[6]);
    }

    fm_DT32_2_DT(&ofm_blob32[pool3_offset], ofm_blob, config[10]);
    distitch(ofm_blob, ofm, config[10]);
    for(int p=0; p<4; p++)
    {
        check_fm(ofm[p], config[10]);
    }
    
    fm_DT32_2_DT(&ofm_blob32[dwconv4_offset], ofm_blob, config[11]);
    distitch(ofm_blob, ofm, config[11]);
    for(int p=0; p<4; p++)
    {
        check_fm(ofm[p], config[11]);
    }

    fm_DT32_2_DT(&ofm_blob32[pwconv4_offset], ofm_blob, config[12]);
    distitch(ofm_blob, ofm, config[12]);
    for(int p=0; p<4; p++)
    {
        check_fm(ofm[p], config[12]);
    }

    fm_DT32_2_DT(&ofm_blob32[dwconv5_offset], ofm_blob, config[13]);
    distitch(ofm_blob, ofm, config[13]);
    for(int p=0; p<4; p++)
    {
        check_fm(ofm[p], config[13]);
    }

    fm_DT32_2_DT(&ofm_blob32[pwconv5_offset], ofm_blob, config[14]);
    distitch(ofm_blob, ofm, config[14]);
    for(int p=0; p<4; p++)
    {
        check_fm(ofm[p], config[14]);
    }

    fm_DT32_2_DT(&ofm_blob32[reorg_offset], ofm_blob, config[9]);
    distitch(ofm_blob, ofm, config[9]);
    for(int p=0; p<4; p++)
    {
        check_fm(ofm[p], config[9]);
    }

    fm_DT32_2_DT(&ofm_blob32[dwconv6_offset], ofm_blob, config[16]);
    distitch(ofm_blob, ofm, config[16]);
    for(int p=0; p<4; p++)
    {
        check_fm(ofm[p], config[16]);
    }

    fm_DT32_2_DT(&ofm_blob32[pwconv6_offset], ofm_blob, config[17]);
    distitch(ofm_blob, ofm, config[17]);
    for(int p=0; p<4; p++)
    {
        check_fm(ofm[p], config[17]);
    }

    fm_DT32_2_DT(&ofm_blob32[conv7_offset], ofm_blob, config[18]);
    distitch(ofm_blob, ofm, config[18]);
    for(int p=0; p<4; p++)
    {
        check_fm(ofm[p], config[18]);
    }
}