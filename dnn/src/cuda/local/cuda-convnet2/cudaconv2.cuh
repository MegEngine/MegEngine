#ifndef COMMON_CUH
#define COMMON_CUH

#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#include "helper_cuda.h"  // helper functions CUDA error checking and initialization
#include "nvmatrix.cuh"

namespace megdnn {
namespace cuda {

enum FILTER_OUTPUT_ORDER { MODULE_FILTER_IMAGE, FILTER_MODULE_IMAGE };

void convFilterActs(
        cudaStream_t stream, NVMatrix& images, NVMatrix& filters, NVMatrix& targets,
        int imgSizeY, int numModulesY, int numModulesX, int paddingStart,
        int moduleStride, int numImgColors, int numGroups);
void convFilterActs(
        cudaStream_t stream, NVMatrix& images, NVMatrix& filters, NVMatrix& targets,
        int imgSizeY, int numModulesY, int numModulesX, int paddingStart,
        int moduleStride, int numImgColors, int numGroups, float scaleTargets,
        float scaleOutput);

void localFilterActs(
        cudaStream_t stream, NVMatrix& images, NVMatrix& filters, NVMatrix& targets,
        int imgSizeY, int numModulesY, int numModulesX, int paddingStart,
        int moduleStride, int numImgColors, int numGroups);
void localFilterActs(
        cudaStream_t stream, NVMatrix& images, NVMatrix& filters, NVMatrix& targets,
        int imgSizeY, int numModulesY, int numModulesX, int paddingStart,
        int moduleStride, int numImgColors, int numGroups, float scaleTargets,
        float scaleOutput);

void convImgActs(
        cudaStream_t stream, NVMatrix& hidActs, NVMatrix& filters, NVMatrix& targets,
        int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride,
        int numImgColors, int numGroups);
void convImgActs(
        cudaStream_t stream, NVMatrix& hidActs, NVMatrix& filters, NVMatrix& targets,
        int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride,
        int numImgColors, int numGroups, float scaleTargets, float scaleOutput);

void localImgActs(
        cudaStream_t stream, NVMatrix& hidActs, NVMatrix& filters, NVMatrix& targets,
        int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride,
        int numImgColors, int numGroups);
void localImgActs(
        cudaStream_t stream, NVMatrix& hidActs, NVMatrix& filters, NVMatrix& targets,
        int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride,
        int numImgColors, int numGroups, float scaleTargets, float scaleOutput);

void convWeightActs(
        cudaStream_t stream, NVMatrix& images, NVMatrix& hidActs, NVMatrix& targets,
        int imgSizeY, int numModulesY, int numModulesX, int filterSize,
        int paddingStart, int moduleStride, int numImgColors, int numGroups,
        int sumWidth);
void convWeightActs(
        cudaStream_t stream, NVMatrix& images, NVMatrix& hidActs, NVMatrix& targets,
        int imgSizeY, int numModulesY, int numModulesX, int filterSize,
        int paddingStart, int moduleStride, int numImgColors, int numGroups,
        int sumWidth, float scaleTargets, float scaleOutput);

void localWeightActs(
        cudaStream_t stream, NVMatrix& images, NVMatrix& hidActs, NVMatrix& targets,
        int imgSizeY, int numModulesY, int numModulesX, int filterSize,
        int paddingStart, int moduleStride, int numImgColors, int numGroups);

void localWeightActs(
        cudaStream_t stream, NVMatrix& images, NVMatrix& hidActs, NVMatrix& targets,
        int imgSizeY, int numModulesY, int numModulesX, int filterSize,
        int paddingStart, int moduleStride, int numImgColors, int numGroups,
        float scaleTargets, float scaleOutput);
}  // namespace cuda
}  // namespace megdnn

#endif /* COMMON_CUH */
