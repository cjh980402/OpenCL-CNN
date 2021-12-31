#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "MyOpencl.hpp"

#define checkCL(expression)                                                  \
    {                                                                        \
        cl_int cl_err = (expression);                                        \
        if (cl_err < 0 && cl_err > -64)                                      \
        {                                                                    \
            printf("Error on line %d (error code: %d)\n", __LINE__, cl_err); \
            _exit(0);                                                        \
        }                                                                    \
    }

OpenclClient::OpenclClient(const char *file_name, size_t localSize)
    : kernel_file_name(file_name), localSize(localSize)
{
    FILE *file_handle = fopen(file_name, "r");
    if (file_handle == NULL)
    {
        printf("Couldn't find the file");
        _exit(1);
    }

    // Read kernel file
    fseek(file_handle, 0, SEEK_END);
    size_t kernel_file_size = ftell(file_handle);
    rewind(file_handle);
    char *kernel_file_buffer = new char[kernel_file_size + 1];
    kernel_file_buffer[kernel_file_size] = '\0';
    fread(kernel_file_buffer, sizeof(char), kernel_file_size, file_handle);
    fclose(file_handle);

    // Bind to platform
    checkCL(clGetPlatformIDs(1, &cpPlatform, NULL));

    // Get ID for the device
    checkCL(clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL));

    // Create a context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);

    // Create a command queue
    queue = clCreateCommandQueue(context, device_id, 0, &err);
    checkCL(err);

    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char **)&kernel_file_buffer, &kernel_file_size, &err);
    checkCL(err);
    delete[] kernel_file_buffer;

    // Build the program executable
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *file_log = new char[log_size];
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, file_log, NULL);
        printf("%s\n", file_log);
        delete[] file_log;
        _exit(1);
    }

    kernels = new cl_kernel[10];
    kernel_names = new const char *[10];
    kernel_count = 0;
}

OpenclClient::~OpenclClient()
{
    // Release OpenCL resources
    checkCL(clReleaseContext(context));
    checkCL(clReleaseProgram(program));
    checkCL(clReleaseCommandQueue(queue));
    for (int i = 0; i < kernel_count; i++)
    {
        checkCL(clReleaseKernel(kernels[i]));
    }

    delete[] kernels;
    delete[] kernel_names;
}

cl_kernel OpenclClient::getKernel(const char *kernel_name)
{
    for (int i = 0; i < kernel_count; i++)
    {
        if (strcmp(kernel_name, kernel_names[i]) == 0)
        {
            return kernels[i];
        }
    }

    // Create the compute kernel in the program we wish to run
    cl_kernel kernel = clCreateKernel(program, kernel_name, &err);
    checkCL(err);

    kernels[kernel_count] = kernel;
    kernel_names[kernel_count] = kernel_name;
    kernel_count++;

    return kernel;
}

void OpenclClient::launch(const char *kernel_name, float *m, int row, int col, int inputChannel, float *filter, int filterSize, int outputChannel, float *result)
{
    cl_kernel kernel = getKernel(kernel_name);
    // Number of work items
    size_t n = outputChannel * row * col;

    // Number of total work items - localSize must be devisor
    size_t grid = n / localSize + (n % localSize ? 1 : 0);
    size_t globalSize = grid * localSize;

    // Create the input and output arrays in device memory for our calculation
    cl_mem d_m = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * inputChannel * row * col, NULL, &err);
    checkCL(err);
    cl_mem d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * outputChannel * inputChannel * filterSize * filterSize, NULL, &err);
    checkCL(err);
    cl_mem d_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * outputChannel * row * col, NULL, &err);
    checkCL(err);

    // Write our data set into the input array in device memory
    checkCL(clEnqueueWriteBuffer(queue, d_m, CL_TRUE, 0, sizeof(float) * inputChannel * row * col, m, 0, NULL, NULL));
    checkCL(clEnqueueWriteBuffer(queue, d_filter, CL_TRUE, 0, sizeof(float) * outputChannel * inputChannel * filterSize * filterSize, filter, 0, NULL, NULL));

    // Set the arguments to our compute kernel
    checkCL(clSetKernelArg(kernel, 0, sizeof(d_m), &d_m));
    checkCL(clSetKernelArg(kernel, 1, sizeof(row), &row));
    checkCL(clSetKernelArg(kernel, 2, sizeof(col), &col));
    checkCL(clSetKernelArg(kernel, 3, sizeof(inputChannel), &inputChannel));
    checkCL(clSetKernelArg(kernel, 4, sizeof(d_filter), &d_filter));
    checkCL(clSetKernelArg(kernel, 5, sizeof(filterSize), &filterSize));
    checkCL(clSetKernelArg(kernel, 6, sizeof(outputChannel), &outputChannel));
    checkCL(clSetKernelArg(kernel, 7, sizeof(d_result), &d_result));

    clock_t start = clock();

    // Execute the kernel over the entire range of the data set
    checkCL(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL));
    // Wait for the command queue to get serviced before reading back results
    checkCL(clFinish(queue));

    // Read the results from the device
    checkCL(clEnqueueReadBuffer(queue, d_result, CL_TRUE, 0, sizeof(float) * outputChannel * row * col, result, 0, NULL, NULL));

    clock_t end = clock();
    printf("GPUtime: %lf ms\n", 1000.0 * (end - start) / CLOCKS_PER_SEC);

    // Release OpenCL object
    checkCL(clReleaseMemObject(d_m));
    checkCL(clReleaseMemObject(d_filter));
    checkCL(clReleaseMemObject(d_result));
}

void OpenclClient::launch(const char *kernel_name, float *m1, int row1, int col1, float *m2, int row2, int col2, float *result)
{
    cl_kernel kernel = getKernel(kernel_name);
    // Number of work items
    size_t n = row1 * col2;

    // Number of total work items - localSize must be devisor
    size_t grid = n / localSize + (n % localSize ? 1 : 0);
    size_t globalSize = grid * localSize;

    // Create the input and output arrays in device memory for our calculation
    cl_mem d_m1 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * row1 * col1, NULL, &err);
    checkCL(err);
    cl_mem d_m2 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * row2 * col2, NULL, &err);
    checkCL(err);
    cl_mem d_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * row1 * col2, NULL, &err);
    checkCL(err);

    // Write our data set into the input array in device memory
    checkCL(clEnqueueWriteBuffer(queue, d_m1, CL_TRUE, 0, sizeof(float) * row1 * col1, m1, 0, NULL, NULL));
    checkCL(clEnqueueWriteBuffer(queue, d_m2, CL_TRUE, 0, sizeof(float) * row2 * col2, m2, 0, NULL, NULL));

    // Set the arguments to our compute kernel
    checkCL(clSetKernelArg(kernel, 0, sizeof(d_m1), &d_m1));
    checkCL(clSetKernelArg(kernel, 1, sizeof(row1), &row1));
    checkCL(clSetKernelArg(kernel, 2, sizeof(col1), &col1));
    checkCL(clSetKernelArg(kernel, 3, sizeof(d_m2), &d_m2));
    checkCL(clSetKernelArg(kernel, 4, sizeof(row2), &row2));
    checkCL(clSetKernelArg(kernel, 5, sizeof(col2), &col2));
    checkCL(clSetKernelArg(kernel, 6, sizeof(d_result), &d_result));

    clock_t start = clock();

    // Execute the kernel over the entire range of the data set
    checkCL(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL));
    // Wait for the command queue to get serviced before reading back results
    checkCL(clFinish(queue));

    // Read the results from the device
    checkCL(clEnqueueReadBuffer(queue, d_result, CL_TRUE, 0, sizeof(float) * row1 * col2, result, 0, NULL, NULL));

    clock_t end = clock();
    printf("GPUtime: %lf ms\n", 1000.0 * (end - start) / CLOCKS_PER_SEC);

    // Release OpenCL object
    checkCL(clReleaseMemObject(d_m1));
    checkCL(clReleaseMemObject(d_m2));
    checkCL(clReleaseMemObject(d_result));
}

void OpenclClient::launch(const char *kernel_name, float *m, int row, int col, int filterSize, int channel, float *result)
{
    cl_kernel kernel = getKernel(kernel_name);
    // Number of work items
    size_t n = channel * row * col;

    // Number of total work items - localSize must be devisor
    size_t grid = n / localSize + (n % localSize ? 1 : 0);
    size_t globalSize = grid * localSize;

    // Create the input and output arrays in device memory for our calculation
    cl_mem d_m = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * channel * row * col, NULL, &err);
    checkCL(err);
    cl_mem d_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * channel * (row / 2) * (col / 2), NULL, &err);
    checkCL(err);

    // Write our data set into the input array in device memory
    checkCL(clEnqueueWriteBuffer(queue, d_m, CL_TRUE, 0, sizeof(float) * channel * row * col, m, 0, NULL, NULL));

    // Set the arguments to our compute kernel
    checkCL(clSetKernelArg(kernel, 0, sizeof(d_m), &d_m));
    checkCL(clSetKernelArg(kernel, 1, sizeof(row), &row));
    checkCL(clSetKernelArg(kernel, 2, sizeof(col), &col));
    checkCL(clSetKernelArg(kernel, 3, sizeof(filterSize), &filterSize));
    checkCL(clSetKernelArg(kernel, 4, sizeof(channel), &channel));
    checkCL(clSetKernelArg(kernel, 5, sizeof(d_result), &d_result));

    clock_t start = clock();

    // Execute the kernel over the entire range of the data set
    checkCL(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL));
    // Wait for the command queue to get serviced before reading back results
    checkCL(clFinish(queue));

    // Read the results from the device
    checkCL(clEnqueueReadBuffer(queue, d_result, CL_TRUE, 0, sizeof(float) * channel * (row / 2) * (col / 2), result, 0, NULL, NULL));

    clock_t end = clock();
    printf("GPUtime: %lf ms\n", 1000.0 * (end - start) / CLOCKS_PER_SEC);

    // Release OpenCL object
    checkCL(clReleaseMemObject(d_m));
    checkCL(clReleaseMemObject(d_result));
}

void OpenclClient::launch(const char *kernel_name, float *m, int row, int col)
{
    cl_kernel kernel = getKernel(kernel_name);
    // Number of work items
    size_t n = row * col;

    // Number of total work items - localSize must be devisor
    size_t grid = n / localSize + (n % localSize ? 1 : 0);
    size_t globalSize = grid * localSize;

    // Create the input and output arrays in device memory for our calculation
    cl_mem d_m = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * row * col, NULL, &err);
    checkCL(err);

    // Write our data set into the input array in device memory
    checkCL(clEnqueueWriteBuffer(queue, d_m, CL_TRUE, 0, sizeof(float) * row * col, m, 0, NULL, NULL));

    // Set the arguments to our compute kernel
    checkCL(clSetKernelArg(kernel, 0, sizeof(d_m), &d_m));
    checkCL(clSetKernelArg(kernel, 1, sizeof(row), &row));
    checkCL(clSetKernelArg(kernel, 2, sizeof(col), &col));

    clock_t start = clock();

    // Execute the kernel over the entire range of the data set
    checkCL(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL));
    // Wait for the command queue to get serviced before reading back results
    checkCL(clFinish(queue));

    // Read the results from the device
    checkCL(clEnqueueReadBuffer(queue, d_m, CL_TRUE, 0, sizeof(float) * row * col, m, 0, NULL, NULL));

    clock_t end = clock();
    printf("GPUtime: %lf ms\n", 1000.0 * (end - start) / CLOCKS_PER_SEC);

    // Release OpenCL object
    checkCL(clReleaseMemObject(d_m));
}

void OpenclClient::launch(const char *kernel_name, unsigned char *m, int row, int col, float *result)
{
    cl_kernel kernel = getKernel(kernel_name);
    // Number of work items
    size_t n = row * col;

    // Number of total work items - localSize must be devisor
    size_t grid = n / localSize + (n % localSize ? 1 : 0);
    size_t globalSize = grid * localSize;

    // Create the input and output arrays in device memory for our calculation
    cl_mem d_m = clCreateBuffer(context, CL_MEM_READ_ONLY, 3 * sizeof(unsigned char) * row * col, NULL, &err); // Input image is 3 channel
    checkCL(err);
    cl_mem d_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * row * col, NULL, &err); // Output image is 1 channel
    checkCL(err);

    // Write our data set into the input array in device memory
    checkCL(clEnqueueWriteBuffer(queue, d_m, CL_TRUE, 0, 3 * sizeof(unsigned char) * row * col, m, 0, NULL, NULL));

    // Set the arguments to our compute kernel
    checkCL(clSetKernelArg(kernel, 0, sizeof(d_m), &d_m));
    checkCL(clSetKernelArg(kernel, 1, sizeof(row), &row));
    checkCL(clSetKernelArg(kernel, 2, sizeof(col), &col));
    checkCL(clSetKernelArg(kernel, 3, sizeof(d_result), &d_result));

    clock_t start = clock();

    // Execute the kernel over the entire range of the data set
    checkCL(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL));
    // Wait for the command queue to get serviced before reading back results
    checkCL(clFinish(queue));

    // Read the results from the device
    checkCL(clEnqueueReadBuffer(queue, d_result, CL_TRUE, 0, sizeof(float) * row * col, result, 0, NULL, NULL));

    clock_t end = clock();
    printf("GPUtime: %lf ms\n", 1000.0 * (end - start) / CLOCKS_PER_SEC);

    // Release OpenCL object
    checkCL(clReleaseMemObject(d_m));
    checkCL(clReleaseMemObject(d_result));
}