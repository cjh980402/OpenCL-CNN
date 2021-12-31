#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <byteswap.h>
#include "ImageProcessing.hpp"

unsigned char *read_bmp(const char *filename, BMPHEADER *bmpHeader)
{
    FILE *filePtr;              // our file pointer
    unsigned char *bitmapImage; // store image data
    int imageIdx = 0;           // image index counter
    unsigned char tempRGB;      // our swap variable

    // open filename in read binary mode
    filePtr = fopen(filename, "rb");
    if (filePtr == NULL)
        return NULL;
    // read the bitmap file header
    fread(bmpHeader, sizeof(char), sizeof(BMPHEADER), filePtr);

    // verify that this is a bmp file by check bitmap id
    if (bmpHeader->bfType[0] != 'B' || bmpHeader->bfType[1] != 'M')
    {
        fclose(filePtr);
        return NULL;
    }
    // move file point to the begging of bitmap data
    fseek(filePtr, bmpHeader->bfOffBits, SEEK_SET);

    // allocate enough memory for the bitmap image data
    bitmapImage = new unsigned char[bmpHeader->biSizeImage];
    // read in the bitmap image data
    fread(bitmapImage, bmpHeader->biSizeImage, 1, filePtr);

    // make sure bitmap image data was read
    if (bitmapImage == NULL)
    {
        fclose(filePtr);
        return NULL;
    }

    // swap the r and b values to get RGB (bitmap is BGR)
    for (imageIdx = 0; imageIdx < bmpHeader->biSizeImage; imageIdx += 3) // fixed semicolon
    {
        tempRGB = bitmapImage[imageIdx];
        bitmapImage[imageIdx] = bitmapImage[imageIdx + 2];
        bitmapImage[imageIdx + 2] = tempRGB;
    }

    // close file and return bitmap iamge data
    fclose(filePtr);
    return bitmapImage;
}

int write_bmp(const char *filename, int width, int height, unsigned char *rgb)
{
    int i, j, ipos;
    int bytesPerLine;
    unsigned char *line;

    FILE *file;
    struct BMPHeader bmph;

    /* The length of each line must be a multiple of 4 bytes */

    bytesPerLine = (3 * (width + 1) / 4) * 4;

    strncpy(bmph.bfType, "BM", 2);
    bmph.bfOffBits = 54;
    bmph.bfSize = bmph.bfOffBits + bytesPerLine * height;
    bmph.bfReserved = 0;
    bmph.biSize = 40;
    bmph.biWidth = width;
    bmph.biHeight = height;
    bmph.biPlanes = 1;
    bmph.biBitCount = 24;
    bmph.biCompression = 0;
    bmph.biSizeImage = bytesPerLine * height;
    bmph.biXPelsPerMeter = 0;
    bmph.biYPelsPerMeter = 0;
    bmph.biClrUsed = 0;
    bmph.biClrImportant = 0;

    file = fopen(filename, "wb");
    if (file == NULL)
        return 0;

    fwrite(&bmph.bfType, 2, 1, file);
    fwrite(&bmph.bfSize, 4, 1, file);
    fwrite(&bmph.bfReserved, 4, 1, file);
    fwrite(&bmph.bfOffBits, 4, 1, file);
    fwrite(&bmph.biSize, 4, 1, file);
    fwrite(&bmph.biWidth, 4, 1, file);
    fwrite(&bmph.biHeight, 4, 1, file);
    fwrite(&bmph.biPlanes, 2, 1, file);
    fwrite(&bmph.biBitCount, 2, 1, file);
    fwrite(&bmph.biCompression, 4, 1, file);
    fwrite(&bmph.biSizeImage, 4, 1, file);
    fwrite(&bmph.biXPelsPerMeter, 4, 1, file);
    fwrite(&bmph.biYPelsPerMeter, 4, 1, file);
    fwrite(&bmph.biClrUsed, 4, 1, file);
    fwrite(&bmph.biClrImportant, 4, 1, file);

    line = new unsigned char[bytesPerLine];

    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++)
        {
            ipos = 3 * (width * i + j);
            line[3 * j] = rgb[ipos + 2];
            line[3 * j + 1] = rgb[ipos + 1];
            line[3 * j + 2] = rgb[ipos];
        }
        fwrite(line, bytesPerLine, 1, file);
    }

    delete[] line;
    fclose(file);

    return 1;
}
