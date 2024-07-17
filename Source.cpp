#include <iostream>             // Standard input-output stream
#include <fstream>              // File stream
#include <opencv2/opencv.hpp>  // OpenCV library
#include <opencv2/core/ocl.hpp> // OpenCV OpenCL support

#ifdef _WIN32
#include <CL/cl.h>             // OpenCL headers for Windows
#else
#include <OpenCL/cl.h>         // OpenCL headers for other platforms
#endif

// Updated OpenCL kernel source code for grayscale conversion
const char* customKernel =
"__kernel void convertToGrayscale(__global uchar4* inputImage, __global uchar4* outputImage, const int imgWidth, const int imgHeight) {\n"
"    int x = get_global_id(0);\n"
"    int y = get_global_id(1);\n"
"    int index = y * imgWidth + x;\n"
"    uchar4 px = inputImage[index];\n"
"    uchar grayscale = (px.x + px.y + px.z) / 3;\n"
"    outputImage[index] = (uchar4)(grayscale, grayscale, grayscale, px.w);\n"
"}\n";

int main() {

    // Redirect standard output and error streams to /dev/null to suppress console output
    std::ofstream devnull;
    devnull.open("/dev/null");
    std::streambuf* coutbuf = std::cout.rdbuf();
    std::streambuf* cerrbuf = std::cerr.rdbuf();
    std::cout.rdbuf(devnull.rdbuf());
    std::cerr.rdbuf(devnull.rdbuf());

    // Load input image using OpenCV
    cv::Mat inputImg = cv::imread("ISIC_0073502.jpg", cv::IMREAD_COLOR);
    if (inputImg.empty()) {
        std::cerr << "Couldn't find the input image with specified link." << std::endl;
        return 1;
    }

    // Enable OpenCL support in OpenCV
    cv::ocl::setUseOpenCL(true);

    // Convert input image to CV_8UC4 format
    cv::Mat inputImgRGBA;
    cv::cvtColor(inputImg, inputImgRGBA, cv::COLOR_BGR2RGBA);

    // Initialize OpenCL
    cl_int errorVar;
    cl_platform_id platformId;
    errorVar = clGetPlatformIDs(1, &platformId, NULL);
    if (errorVar != CL_SUCCESS) {
        std::cerr << "Failed to get the platform ID." << std::endl;
        return 1;
    }

    cl_device_id devId;
    errorVar = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, 1, &devId, NULL);
    if (errorVar != CL_SUCCESS) {
        std::cerr << "Failed to get the device ID." << std::endl;
        return 1;
    }

    cl_context context = clCreateContext(NULL, 1, &devId, NULL, NULL, &errorVar);
    if (errorVar != CL_SUCCESS) {
        std::cerr << "Failed to create the context." << std::endl;
        return 1;
    }

    cl_command_queue cmdQueue = clCreateCommandQueueWithProperties(context, devId, NULL, &errorVar);
    if (errorVar != CL_SUCCESS) {
        std::cerr << "Failed to create the command queue." << std::endl;
        return 1;
    }

    // Create OpenCL program and kernel
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&customKernel, NULL, &errorVar);
    if (errorVar != CL_SUCCESS) {
        std::cerr << "Failed to create the program." << std::endl;
        return 1;
    }

    errorVar = clBuildProgram(program, 1, &devId, NULL, NULL, NULL);
    if (errorVar != CL_SUCCESS) {
        std::cerr << "Failed to build the program." << std::endl;
        return 1;
    }

    cl_kernel clKernel = clCreateKernel(program, "convertToGrayscale", &errorVar);
    if (errorVar != CL_SUCCESS) {
        std::cerr << "Failed to create the kernel." << std::endl;
        return 1;
    }

    // Set kernel arguments
    int width = inputImgRGBA.cols;
    int height = inputImgRGBA.rows;
    cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uchar) * 4 * width * height, inputImgRGBA.data, &errorVar);
    cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(uchar) * 4 * width * height, NULL, &errorVar);

    errorVar = clSetKernelArg(clKernel, 0, sizeof(cl_mem), &inputBuffer);
    errorVar |= clSetKernelArg(clKernel, 1, sizeof(cl_mem), &outputBuffer);
    errorVar |= clSetKernelArg(clKernel, 2, sizeof(int), &width);
    errorVar |= clSetKernelArg(clKernel, 3, sizeof(int), &height);
    if (errorVar != CL_SUCCESS) {
        std::cerr << "Error: Failed to set kernel arguments." << std::endl;
        return 1;
    }

    // Execute the kernel
    size_t globalWorkSize[2] = { static_cast<size_t>(width), static_cast<size_t>(height) };
    errorVar = clEnqueueNDRangeKernel(cmdQueue, clKernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    if (errorVar != CL_SUCCESS) {
        std::cerr << "Error: Failed to execute kernel." << std::endl;
        return 1;
    }

    // Read back the output image
    cv::Mat outputImgRGBA(height, width, CV_8UC4);
    errorVar = clEnqueueReadBuffer(cmdQueue, outputBuffer, CL_TRUE, 0, sizeof(uchar) * 4 * width * height, outputImgRGBA.data, 0, NULL, NULL);
    if (errorVar != CL_SUCCESS) {
        std::cerr << "Error: Failed to read buffer." << std::endl;
        return 1;
    }

    // Convert output image to BGR format
    cv::Mat outputImg;
    cv::cvtColor(outputImgRGBA, outputImg, cv::COLOR_RGBA2BGR);

    // Save output image
    cv::imwrite("GreyScaledImage.jpg", outputImg);

    // Cleanup
    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);
    clReleaseKernel(clKernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(cmdQueue);
    clReleaseContext(context);

    // Restore standard output and error streams
    std::cout.rdbuf(coutbuf);
    std::cerr.rdbuf(cerrbuf);

    std::cout << "Grayscale conversion has been completed. The Output saved as GreyScaledImage.jpg" << std::endl;

    return 0;
}

