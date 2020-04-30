#include "CNN.h"

void generate_fm(DT* fm, layer l)
{
    for(int c=0; c<l.oc; c++)
    {
        for(int h=0; h<l.oh; h++)
        {
            for(int w=0; w<l.ow; w++)
            {
                int fm_index = c*l.oh*l.ow + h*l.ow + w;
                fm[fm_index] = h + w;
            }
        }
    }
}

void check(DT* result, DT* golden, int len, layer l)
{
    int err = 0;
    for (int j = 0; j < len; j++)
    {
        if (((result[j] - golden[j]) > check_scale) || ((result[j] - golden[j]) < -check_scale))
        {
            err++;
            //printf("[%d] correct=%f,wrong=%f\n", j, tmp[j], fm[j]);
        }
    }

    if (err > 0)
        printf("%s error cnt= %d\n", l.name, err);
    else
        printf("%s correct \n", l.name);
}

void load_fm(DT* fm, layer l)
{
    char nstr[50];

    sprintf(nstr, "../blobs/%s.bb", l.name);
    FILE *fp = fopen(nstr, "rb");
    fread(fm, 1, l.ow*l.oh*l.oc * sizeof(DT), fp);
    fclose(fp);
}

void load_weight(DT32* weight , int length)
{
    char nstr[50];
    sprintf(nstr, "../weights/SkyNetT.wt");
    FILE *fp = fopen(nstr, "rb");
    fread(weight, 1, length*sizeof(DT), fp);
    fclose(fp);
}

void load_weight_dt(DT* weight , int length, layer l)
{
    char nstr[50];
    sprintf(nstr, "../weights/SkyNet.wt");
    //sprintf(nstr, "../weights/%s.wt", l.name);
    FILE *fp = fopen(nstr, "rb");
    fread(weight, 1, length*sizeof(DT), fp);
    fclose(fp);
}

void load_bias(DT* bias , int length, layer l)
{
    char nstr[50];
    sprintf(nstr, "../weights/%s.bs", l.name);
    FILE *fp = fopen(nstr, "rb");
    fread(bias, 1, length*sizeof(DT), fp);
    fclose(fp);
}

void show_fm(DT* fm, layer l)
{
    for (int c=0;c<l.oc;c++)
    {
        for (int h=0;h<l.oh;h++)
        {
            for (int w=0;w<l.ow;w++)
            {
                int i = c*l.oh*l.ow + h*l.ow + w;
                std::cout << fm[i]<<", ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
}

void check_fm(DT* fm, layer l)
{
    int len = l.oc*l.ow*l.oh;
    DT *tmp = (DT *)malloc(sizeof(DT)*len);

    char nstr[50];
    sprintf(nstr, "../blobs/%s.bb", l.name);
    FILE *fp = fopen(nstr, "rb");
    fread(tmp, 1, len*sizeof(DT), fp);
    fclose(fp);

    int err = 0;
    int zero;
    for(int c=0; c<l.oc; c++)
    {
        int channel_error = 0;
        zero = 0;
        for(int h=0; h<l.oh; h++)
        {
            for(int w=0; w<l.ow; w++)
            {
                int index = c*l.oh*l.ow + h*l.ow + w;
                if (((fm[index] - tmp[index]) > check_scale) || ((fm[index] - tmp[index]) < -check_scale))
                {
                    err++;
                    channel_error++;
                    if(fm[index]==0)
                        zero++;
                    //printf("!!![%d][%d][%d] correct=%f, wrong=%f\n", c, h, w, tmp[index], fm[index]);
                }
                else
                {
                    //printf("[%d][%d][%d] correct=%f\n", c, h, w, tmp[index]);
                    if(fm[index]==0)
                        zero++;
                }
                
            }
        }
    }

    if (err > 0)
        printf("%s error cnt= %d\n", l.name, err);
    else
        printf("%s correct \n", l.name);

    free(tmp);
}


void compare_dt32(DT32* data1, DT32* data2, int len)
{
    int err = 0;
    for(int i=0; i<len; i++)
    {
        for(int j=0; j<32; j++)
        {
            if (((data1[i].data[j] - data2[i].data[j]) > check_scale) || ((data1[i].data[j] - data2[i].data[j]) < -check_scale))
                err++;
        }
    }
    printf("error: %d\n", err);
}
