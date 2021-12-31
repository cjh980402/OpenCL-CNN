// OpenCL kernel

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void kernel_convolution(__global float *m, int row, int col, int inputChannel,
                                 __global float *filter, int filterSize, int outputChannel,
                                 __global float *result)
{
    int globalId = get_global_id(0);
    if (globalId >= outputChannel * row * col)
        return;

    int nowOutChannel = globalId / (row * col);
    int i = (globalId % (row * col)) / col;
    int j = (globalId % (row * col)) % col;

    int ele = nowOutChannel * row * col + i * col + j;
    result[ele] = 0; // result[nowOutChannel][i][j] = 0
    for (int nowInChannel = 0; nowInChannel < inputChannel; nowInChannel++)
    {
        for (int a = 0; a < filterSize; a++)
        {
            for (int b = 0; b < filterSize; b++)
            {
                int convRow = i + a - filterSize / 2;
                int convCol = j + b - filterSize / 2;
                // zero padding, m[nowInChannel][convRow][convCol]
                float inputValue = convRow < 0 || convRow >= row || convCol < 0 || convCol >= col ? 0 : m[nowInChannel * row * col + convRow * col + convCol];
                // filter[nowOutChannel][nowInChannel][a][b]
                float filterValue = filter[nowOutChannel * inputChannel * filterSize * filterSize
                + nowInChannel * filterSize * filterSize
                + a * filterSize
                + b];
                result[ele] += inputValue * filterValue;
            }
        }
    }
}

__kernel void kernel_multiply(__global float *m1, int row1, int col1,
                              __global float *m2, int row2, int col2,
                              __global float *result)
{
    int globalId = get_global_id(0);
    if (col1 != row2 || globalId >= row1 * col2)
        return;

    int i = globalId / col2;
    int j = globalId % col2;
    
    int ele = i * col2 + j;
    result[ele] = 0; // result[i][j] = 0
    for (int k = 0; k < col1; k++)
        result[ele] += m1[i * col1 + k] * m2[k * col2 + j]; // result[i][j] += m1[i][k] * m2[k][j]
}

__kernel void kernel_add(__global float *m1, int row1, int col1,
                         __global float *m2, int row2, int col2,
                         __global float *result)
{
    int globalId = get_global_id(0);
    if (row1 != row2 || col1 != col2 || globalId >= row1 * col1)
        return;

    int i = globalId / col1;
    int j = globalId % col1;

    int ele = i * col1 + j;
    result[ele] = m1[ele] + m2[ele]; // result[i][j] = m1[i][j] + m2[i][j]
}

__kernel void kernel_avg_pooling(__global float *m, int row, int col, int filterSize, int channel, __global float *result)
{
    int globalId = get_global_id(0);
    if (globalId >= channel * row * col)
        return;

    int nowChannel = globalId / (row * col);
    int i = (globalId % (row * col)) / col;
    int j = (globalId % (row * col)) % col;
    if (i % filterSize != 0 || j % filterSize != 0)
        return;

    int ele = nowChannel * (row / filterSize) * (col / filterSize) + (i / filterSize) * (col / filterSize) + (j / filterSize);
    result[ele] = 0; // result[nowChannel][i/filterSize][j/filterSize] = 0
    for (int a = 0; a < filterSize; a++)
    {
        for (int b = 0; b < filterSize; b++)
        {
            int convRow = i + a;
            int convCol = j + b;
            float inputValue = convRow >= row || convCol >= col ? 0 : m[nowChannel * row * col + convRow * col + convCol];
            result[ele] += inputValue;
        }
    }
    result[ele] /= filterSize * filterSize;
}

__kernel void kernel_max_pooling(__global float *m, int row, int col, int filterSize, int channel, __global float *result)
{
    int globalId = get_global_id(0);
    if (globalId >= channel * row * col)
        return;

    int nowChannel = globalId / (row * col);
    int i = (globalId % (row * col)) / col;
    int j = (globalId % (row * col)) % col;
    if (i % filterSize != 0 || j % filterSize != 0)
        return;

    int ele = nowChannel * (row / filterSize) * (col / filterSize) + (i / filterSize) * (col / filterSize) + (j / filterSize);
    result[ele] = m[nowChannel * row * col + i * col + j]; // result[nowChannel][i/filterSize][j/filterSize] = m[nowChannel][i][j]
    for (int a = 0; a < filterSize; a++)
    {
        for (int b = 0; b < filterSize; b++)
        {
            int convRow = i + a;
            int convCol = j + b;
            float inputValue = convRow >= row || convCol >= col ? 0 : m[nowChannel * row * col + convRow * col + convCol];
            if (inputValue > result[ele])
                result[ele] = inputValue;
        }
    }
}

__kernel void kernel_relu(__global float *m, int row, int col)
{
    int globalId = get_global_id(0);
    if (globalId >= row * col)
        return;

    int i = globalId / col;
    int j = globalId % col;

    int ele = i * col + j;
    if (m[ele] < 0)
        m[ele] = 0; // m[i][j] = 0
}

__kernel void kernel_gray_threshold(__global unsigned char *src, int height, int width, __global float *dst)
{
    int globalId = get_global_id(0);
    if (globalId >= width * height)
        return;

    int row = globalId / width;
    int col = globalId % width;

    int pix = row * width + col;
    float red = src[pix * 3 + 0] * 0.2126;
    float green = src[pix * 3 + 1] * 0.7152;
    float blue = src[pix * 3 + 2] * 0.0722;
    float gray = red + green + blue;

    if (gray < 120)
        dst[pix] = 1 - (gray / 255);
    else
        dst[pix] = 0;
}
