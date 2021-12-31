#ifndef __MY_OPENCL_H__
#define __MY_OPENCL_H__

#include <CL/opencl.h>

class OpenclClient // Wrapper class of OpenCL
{
private:
    cl_int err;                // OpenCL error code
    cl_platform_id cpPlatform; // OpenCL platform
    cl_device_id device_id;    // device ID
    cl_context context;        // context
    cl_command_queue queue;    // command queue
    cl_program program;        // program

    cl_kernel *kernels;        // kernels
    const char **kernel_names; // kernel names
    size_t kernel_count;       // kernel count

    size_t localSize; // OpenCL local size

    cl_kernel getKernel(const char *kernel_name);

public:
    const char *kernel_file_name;

    OpenclClient(const char *file_name, size_t localSize);
    ~OpenclClient();
    void launch(const char *kernel_name, float *m, int row, int col, int inputChannel, float *filter, int filterSize, int outputChannel, float *result);
    void launch(const char *kernel_name, float *m1, int row1, int col1, float *m2, int row2, int col2, float *result);
    void launch(const char *kernel_name, float *m, int row, int col, int filterSize, int channel, float *result);
    void launch(const char *kernel_name, float *m, int row, int col);
    void launch(const char *kernel_name, unsigned char *m, int row, int col, float *result);
};

#endif