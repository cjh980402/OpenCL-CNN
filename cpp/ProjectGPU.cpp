#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include "MyOpencl.hpp"
#include "ImageProcessing.hpp"

void printMatrix(float *m, int row, int col)
{
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            printf("%4f ", m[i * col + j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[])
{
    FILE *file = NULL;
    const char *weight_files[4] = {"conv1.txt", "conv2.txt", "linear1.txt", "linear2.txt"};
    int weight_sizes[4] = {32 * 1 * 3 * 3, 64 * 32 * 3 * 3, 256 * 3136, 10 * 256};
    float *layers[4];
    for (int i = 0; i < 4; i++)
    {
        layers[i] = new float[weight_sizes[i]];
        file = fopen(weight_files[i], "r");
        if (file == NULL)
        {
            printf("Fail to open file\n");
            _exit(1);
        }
        float weight = 0;
        for (int j = 0; fscanf(file, "%f", &weight) != -1; j++)
        {
            layers[i][j] = weight;
        }
        printf("layer %d\n", i);
        printMatrix(layers[i], 1, 10);
        fclose(file);
    }

    const char *input_image_name = "letter.bmp";
    const char *cl_file_name = "Project.cl";

    OpenclClient client(cl_file_name, 64);

    BMPHEADER bmpHeader;
    unsigned char *image = read_bmp(input_image_name, &bmpHeader);
    float *grayed_img = new float[bmpHeader.biWidth * bmpHeader.biWidth]; // (28 * 28) * 1
    client.launch("kernel_gray_threshold", image, bmpHeader.biWidth, bmpHeader.biWidth, grayed_img);

    float first[32 * 28 * 28];
    client.launch("kernel_convolution", grayed_img, 28, 28, 1, layers[0], 3, 32, first);
    client.launch("kernel_relu", first, 25088, 1);

    float second[32 * 14 * 14];
    client.launch("kernel_avg_pooling", first, 28, 28, 2, 32, second);

    float third[64 * 14 * 14];
    client.launch("kernel_convolution", second, 14, 14, 32, layers[1], 3, 64, third);
    client.launch("kernel_relu", third, 12544, 1);

    float fourth[64 * 7 * 7];
    client.launch("kernel_max_pooling", third, 14, 14, 2, 64, fourth);

    float fifth[256];
    client.launch("kernel_multiply", layers[2], 256, 3136, fourth, 3136, 1, fifth);
    client.launch("kernel_relu", fifth, 256, 1);

    float sixth[10];
    client.launch("kernel_multiply", layers[3], 10, 256, fifth, 256, 1, sixth);

    printf("Result of OCR\n");
    printMatrix(sixth, 1, 10);

    float max = sixth[0];
    int maxIndex = 0;
    for (int i = 1; i < 10; i++)
    {
        if (sixth[i] > max)
        {
            max = sixth[i];
            maxIndex = i;
        }
    }
    printf("Result of prediction\n%d\n", maxIndex);

    for (int i = 0; i < 4; i++)
    {
        delete[] layers[i];
    }
    delete[] image;
    delete[] grayed_img;

    _exit(0);
}
