/*
 * Copyright (C) 2022 Andrew R. Willis
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

// Standard Library includes
#include <iomanip>
#include <sstream>
#include <fstream>

#include <third_party/log.h>
#include <third_party/cxxopts.hpp>

#include <cuGridSearch.cuh>

// this declaration needs to be in any C++ compiled target for CPU
//#define CUDAFUNCTION

#include <charlotte_sar_api.hpp>
#include <uncc_sar_globals.hpp>

#include <uncc_sar_focusing.hpp>
#include <uncc_sar_matio.hpp>

#include "../gpuBackProjection/cuda_sar_focusing/cuda_sar_focusing.hpp"
#include "../cpuBackProjection/cpuBackProjection.hpp"

#include "gridSearchErrorFunctions_dft.cuh"

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#include <Eigen/Dense>

#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

#include <nvVectorNd.h>
#include <chrono>

typedef float NumericType;

#define grid_dimension_quadratic 10    // the dimension of the grid, e.g., 1 => 1D grid, 2 => 2D grid, 3=> 3D grid, etc.
#define grid_dimension_linear 7        // the dimension of the grid, e.g., 1 => 1D grid, 2 => 2D grid, 3=> 3D grid, etc.

typedef float grid_precision;   // the type of values in the grid, e.g., float, double, int, etc.
typedef float func_precision;   // the type of values taken by the error function, e.g., float, double, int, etc.
typedef float pixel_precision; // the type of values in the image, e.g., float, double, int, etc.

// TODO: THIS WILL NEED TO BE CHANGED TO FIT THE ERROR FUNCTION (Look at changes to error function)

typedef func_byvalue_t<func_precision, grid_precision, grid_dimension_linear,
        cufftComplex *,
        int, int,
        NumericType, NumericType,
        NumericType, NumericType,
        NumericType, NumericType,
        NumericType *,
        NumericType *,
        NumericType *,
        NumericType *,
        NumericType *,
        SAR_ImageFormationParameters<NumericType> *,
        NumericType *> image_err_func_byvalue_linear;

typedef func_byvalue_t<func_precision, grid_precision, grid_dimension_quadratic,
        cufftComplex *,
        int, int,
        NumericType, NumericType,
        NumericType, NumericType,
        NumericType, NumericType,
        NumericType *,
        NumericType *,
        NumericType *,
        NumericType *,
        NumericType *,
        SAR_ImageFormationParameters<NumericType> *,
        NumericType *> image_err_func_byvalue_quadratic;

// TODO: THIS WILL ALSO NEED TO BE CHANGED TO FIT THE ERROR FUNCTION
__device__ image_err_func_byvalue_linear dev_func_byvalue_ptr_linear = kernelWrapper<func_precision, grid_precision, grid_dimension_linear, NumericType>;
__device__ image_err_func_byvalue_quadratic dev_func_byvalue_ptr_quadratic = kernelWrapper<func_precision, grid_precision, grid_dimension_quadratic, NumericType>;

// Generic functor
template<typename _Scalar, int NX = Eigen::Dynamic, int NY = Eigen::Dynamic>
struct Functor {
    typedef _Scalar Scalar;
    enum {
        InputsAtCompileTime = NX,
        ValuesAtCompileTime = NY
    };
    typedef Eigen::Matrix<Scalar, InputsAtCompileTime, 1> InputType;
    typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, 1> ValueType;
    typedef Eigen::Matrix <Scalar, ValuesAtCompileTime, InputsAtCompileTime> JacobianType;

    int m_inputs, m_values;

    Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}

    Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}

    int inputs() const { return m_inputs; }

    int values() const { return m_values; }

};

template<typename _Scalar, int NX = Eigen::Dynamic, int NY = Eigen::Dynamic>
struct PGAFunctor {
    typedef _Scalar Scalar;
    enum {
        InputsAtCompileTime = NX,
        ValuesAtCompileTime = NY
    };
    typedef Eigen::Matrix<Scalar, InputsAtCompileTime, 1> InputType;
    typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, 1> ValueType;
    typedef Eigen::Matrix <Scalar, ValuesAtCompileTime, InputsAtCompileTime> JacobianType;

    int m_inputs, m_values, numSamples, numRange;

    PGAFunctor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}

    PGAFunctor(int inputs, int values) : m_inputs(inputs), m_values(values) {}

    int inputs() const { return m_inputs; }

    int values() const { return m_values; }

    int operator()(const Eigen::VectorXcf &x, Eigen::VectorXf &fvec) const {
        // Implement the ()
        // Energy of the phase, all of the phase given the phase shift
        // Python code up to phi (remove linear trend), try it with and without

        cufftComplex *G_dot = new cufftComplex[numSamples * numRange]; // Holds the data difference
        float *phi_dot = new float[numSamples]; // Holds phi_dot and will also hold phi for simplicity// Store values into G_dot

        for (int pulseNum = 0; pulseNum < numSamples; pulseNum++) {
            for (int rangeNum = 0; rangeNum < numRange - 1; rangeNum++) { // Because it's a difference
                G_dot[rangeNum + pulseNum * numRange].x =
                        x(rangeNum + pulseNum * numRange).real() - x(rangeNum + pulseNum * numRange + 1).real();
                G_dot[rangeNum + pulseNum * numRange].y =
                        x(rangeNum + pulseNum * numRange).imag() - x(rangeNum + pulseNum * numRange + 1).imag();
            }
            // To follow the python code where they append the final sample difference to the matrix to make the size the same as the original
            G_dot[numRange - 1 + pulseNum * numRange].x = G_dot[numRange - 2 + pulseNum * numRange].x;
            G_dot[numRange - 1 + pulseNum * numRange].y = G_dot[numRange - 2 + pulseNum * numRange].y;
        }

        for (int pulseNum = 0; pulseNum < numSamples; pulseNum++) {
            float G_norm = 0; // Something to temporarily hold the summed data Norm for that sample
            for (int rangeNum = 0; rangeNum < numRange; rangeNum++) {
                int idx = rangeNum + pulseNum * numRange;
                phi_dot[pulseNum] += (x(idx).real() * G_dot[idx].y) +
                                     (-1 * x(idx).imag() * G_dot[idx].x); // Only the imaginary component is needed
                G_norm += sqrt(x(idx).real() * x(idx).real() + x(idx).imag() * x(idx).imag());
            }
            phi_dot[pulseNum] /= G_norm;
        }

        for (int pulseNum = 1; pulseNum < numSamples; pulseNum++) { // Integrate to get phi
            phi_dot[pulseNum] = phi_dot[pulseNum] + phi_dot[pulseNum - 1];
        }

        delete[] G_dot;
        delete[] phi_dot;
        return 0;
    }

    int df(const Eigen::VectorXcf &x, Eigen::MatrixXf &fjac) const {
        for (int iii = 0; iii < x.size(); iii++) {
            // Still need to figure out how x will look like (Array of complex vectors?)
            float b = x(iii).imag(); // Something like this, .y or .imag()
            fjac(0, iii) = b * b;
        }
        return 0;
    }
};

template<typename func_precision, typename grid_precision, unsigned int D, typename ... Types>
__global__ void evaluationKernel_by_value_stream_optuna(func_precision *result,
                                                 uint32_t STREAM_BLOCK_DIM,
                                                 uint32_t stream_block_idx,
                                                 uint32_t stream_idx,
                                                 uint32_t result_size,
                                                 func_byvalue_t<func_precision, grid_precision, D, Types...> op,
                                                 nv_ext::Vec<grid_precision, D> gridpt,
                                                 Types ... arg_vals) {

    int streamIndex = stream_block_idx * STREAM_BLOCK_DIM + stream_idx;
    if (streamIndex > result_size) {
        return;
    }

    if(threadIdx.x == 0) 
        *(result + streamIndex) = 0;
    __syncthreads();
    *(result + streamIndex) = (*op)(gridpt, arg_vals...);
}

template<typename func_precision, typename grid_precision, unsigned int D, typename ... Types>
void search_by_value_stream_optuna(CudaTensor<func_precision, D> &result, func_byvalue_t<func_precision, grid_precision, D, Types ...> errorFunction,
                            int STREAM_BLOCK_DIM, int numCores, Types &... arg_vals) {
    
    std::ifstream gridPoints_csv("search_grid_points.csv");
    CudaTensor<func_precision, D> *_result;
    _result = &result;
    std::string line;
    std::vector<grid_precision> point(D, 0);
    uint32_t total_samples = result.size();
    std::cout << "Optuna Grid has " << total_samples << " search samples." << std::endl;

    // allocate and initialize an array of stream handles
    // Found out that creating 10000 kernel streams takes up 1 GB of GPU memory
    // Need to use blocks to save on memory
    int numStreamBlocks = 1;
    int numStreams = STREAM_BLOCK_DIM;
    if (total_samples < STREAM_BLOCK_DIM) {
        numStreams = total_samples;
    } else {
        numStreamBlocks = (total_samples + STREAM_BLOCK_DIM) / STREAM_BLOCK_DIM;
    }
    int numBlocks = 1;
    int numThreads = numCores;
    if (numThreads > MAX_THREADS_PER_BLOCK) {
        numThreads = MAX_THREADS_PER_BLOCK;
        numBlocks = (numCores + MAX_THREADS_PER_BLOCK) / MAX_THREADS_PER_BLOCK;
    }
    cudaStream_t *streams = new cudaStream_t[numStreams];

    for (int i = 0; i < numStreams; i++) {
        checkCudaErrors(cudaStreamCreate(&(streams[i])));
    }

    // create CUDA event handles
    cudaEvent_t start_event, stop_event;
    checkCudaErrors(cudaEventCreate(&start_event));
    checkCudaErrors(cudaEventCreate(&stop_event));

    // the events are used for synchronization only and hence do not need to
    // record timings this also makes events not introduce global sync points when
    // recorded which is critical to get overlap

    cudaEvent_t *kernelEvent = new cudaEvent_t[numStreams];


    for (int i = 0; i < numStreams; i++) {
        checkCudaErrors(
                cudaEventCreateWithFlags(&(kernelEvent[i]), cudaEventDisableTiming));
    }

    // create grid point and result image
    // nv_ext::Vec<grid_precision, D> pt((grid_precision) 0.0f);

    cudaEventRecord(start_event, 0);
    std::chrono::time_point<std::chrono::system_clock> start, start2, end, end2;

    start = std::chrono::system_clock::now();
    start2 = std::chrono::system_clock::now();
    // queue nkernels in separate streams and record when they are done
    for (int i = 0; i < numStreamBlocks; ++i) {
        // printf("Beginning Stream block %d\n",i);
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        printf("\rProgress %.2f%% (%d/%d search blocks) (time taken: %f)", (float) (i + 1) * 100 / numStreamBlocks, i + 1,
                numStreamBlocks, elapsed_seconds.count());
        start = std::chrono::system_clock::now();
        for (int j = 0; j < numStreams; ++j) {
            std::getline(gridPoints_csv, line);
            size_t start_pos = 0;
            size_t end_pos = 0;
            int d = 0;
            end_pos = line.find(',');
            while (end_pos < line.length()) {
                point[d] = std::stod(line.substr(start_pos, end_pos - start_pos));
                start_pos = end_pos + 1;
                end_pos = line.find(',', start_pos);
                d = d + 1;
            }
            nv_ext::Vec<grid_precision, D> pt(point.data());
            evaluationKernel_by_value_stream_optuna<func_precision, grid_precision, D> <<< numBlocks, numThreads, 0, streams[j]>>>((*_result).data(),
                                                                                        STREAM_BLOCK_DIM, i, j,
                                                                                        total_samples,
                                                                                        errorFunction,
                                                                                        pt, arg_vals...);
        }
    }
    checkCudaErrors(cudaDeviceSynchronize());
    end2 = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds2 = end2 - start2;
    printf("\nFinal time: %fs\n", elapsed_seconds2.count());
    delete[] streams;
    delete[] kernelEvent;

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    gridPoints_csv.close();
}

template<typename func_precision, typename grid_precision, uint32_t D, typename __Tp>
__global__ void lmKernelWrapper(nv_ext::Vec<grid_precision, D> parameters, cufftComplex *sampleData,
                                int numRangeSamples, int numAzimuthSamples,
                                __Tp delta_x_m_per_pix, __Tp delta_y_m_per_pix,
                                __Tp left, __Tp bottom,
                                __Tp rmin, __Tp rmax,
                                __Tp *Ant_x,
                                __Tp *Ant_y,
                                __Tp *Ant_z,
                                __Tp *slant_range,
                                __Tp *startF,
                                SAR_ImageFormationParameters<__Tp> *sar_image_params,
                                __Tp *range_vec,
                                func_precision *output) {

    *output = kernelWrapper<func_precision, grid_precision, D, NumericType>(parameters, sampleData, numRangeSamples,
                                                                            numAzimuthSamples, delta_x_m_per_pix,
                                                                            delta_y_m_per_pix, left, bottom, rmin, rmax,
                                                                            Ant_x, Ant_y,
                                                                            Ant_z, slant_range, startF,
                                                                            sar_image_params, range_vec);
}

int numRSamples_nl, numASamples_nl;
cufftComplex *data_nl;
NumericType *ax_nl;
NumericType *ay_nl;
NumericType *az_nl;
NumericType *sr_nl;
NumericType *sf_nl;
SAR_ImageFormationParameters<NumericType> *sip_nl;
NumericType *rv_nl;
NumericType delta_x_m_per_pix_nl, delta_y_m_per_pix_nl, left_m_nl, bottom_m_nl, minRange_nl, maxRange_nl;

struct my_functor : Functor<float> {
    my_functor(void) : Functor<float>(grid_dimension_quadratic, grid_dimension_quadratic) {}

    int operator()(const Eigen::VectorXf &x, Eigen::VectorXf &fvec) const {
        // Plan A
        /*
        Make a separate Kernel that does the same logic as the grid search
        Have it use a variable stored on a GPU for the output (kinda like output_image)
        Move the variable back to CPU and assign it to fvec(0)
        
        Complexity: 
        Making a kernalWrapper2 with different inputs
        Assigning everything correctly
        Assuming that this Eigen function runs sequentially
        */

        // Plan B
        /*
        Make a CPU version of the kernelWrapper code

        Complexity:
        Having to un-thread the kernelWrapper code and everything related to it
        */

        float minParams[grid_dimension_quadratic] = {0};
        printf("Performing search on: ");
        for (int i = 0; i < grid_dimension_quadratic; i++) {
            printf("%f ", x(i));
            minParams[i] = x(i);
        }
        printf("\n");
        nv_ext::Vec<float, grid_dimension_quadratic> minParamsVec(minParams);
        fvec(0) = 0;
        func_precision output = 0;
        func_precision *output_d;
        cudaMalloc(&output_d, sizeof(func_precision));
        lmKernelWrapper<func_precision, grid_precision, grid_dimension_quadratic, NumericType><<<1, 451>>>(minParamsVec,
                                                                                                           data_nl,
                                                                                                           numRSamples_nl,
                                                                                                           numASamples_nl,
                                                                                                           delta_x_m_per_pix_nl,
                                                                                                           delta_y_m_per_pix_nl,
                                                                                                           left_m_nl,
                                                                                                           bottom_m_nl,
                                                                                                           minRange_nl,
                                                                                                           maxRange_nl,
                                                                                                           ax_nl, ay_nl,
                                                                                                           az_nl, sr_nl,
                                                                                                           sf_nl,
                                                                                                           sip_nl,
                                                                                                           rv_nl,
                                                                                                           output_d);
        cudaMemcpy(&output, output_d, sizeof(func_precision), cudaMemcpyDeviceToHost);
        cudaFree(output_d);
        // if (output < 33) output = 1e6f;
        fvec(0) = output;
        printf("Output for this search is: %f\n", output);
        for (int i = 1; i < grid_dimension_quadratic; i++)
            fvec(i) = 0;
        return 0;
    }
};



template<typename __nTp>
std::vector<__nTp> vectorDiff(std::vector<__nTp> values) {
    std::vector<__nTp> temp;
    for (int i = 0; i < values.size() - 1; i++)
        temp.push_back(values[i + 1] - values[i]);

    return temp;
}

template<typename __nTp>
std::vector<__nTp> generateDiffEstimate(__nTp slope, __nTp constant, int N) {
    std::vector<__nTp> temp;
    for (int i = 0; i < N; i++)
        temp.push_back(slope * i + constant);
    return temp;
}

template<typename __nTp>
std::vector<__nTp> vectorAppendCumSum(__nTp start, std::vector<__nTp> values) {
    std::vector<__nTp> temp;
    __nTp sum = start;
    temp.push_back(start);
    for (int i = 0; i < values.size(); i++) {
        sum += values[i];
        temp.push_back(sum);
    }

    return temp;
}

template<typename __nTp>
void bestFit(__nTp *coeffs, std::vector<__nTp> values, int nPulse, int skip) {
    // double sumX = 0.0;
    // double sumY = 0.0;
    // double N = values.size();
    // double sumXY = 0.0;
    // double sumXX = 0.0;

    // for (int i =0; i < values.size(); i++) {
    //     sumX += (__nTp)i;
    //     sumY += values[i];
    //     sumXY += ((__nTp)i*values[i]);
    //     sumXX += ((__nTp)i * (__nTp)i);
    // }

    // double numS = N * sumXY - sumX * sumY;
    // double den = N * sumXX - sumX * sumX;

    // double numC = sumY * sumXX - sumX * sumXY;

    // double temp1 = numS/den;
    // double temp2 = numC/den;

    // float tempa = (float)temp1;
    // float tempb = (float)temp2;

    // coeffs[0] = 0;
    // coeffs[1] = tempa;
    // coeffs[2] = tempb;

    int numN = nPulse;
    double N = 0;
    double x1 = 0;
    double x2 = 0;
    double f0 = 0;
    double f1 = 0;

    for (int i = 0; i < numN; i += skip) {
        N += 1;
        x1 += i;
        x2 += i * i;
        f0 += values[i];
        f1 += values[i] * i;
    }

    double D = -1 * x1 * x1 + N * x2;

    double a = N * f1 - f0 * x1;
    double b = f0 * x2 - f1 * x1;
    printf("N = %f\nx1 = %f\nx2 = %f\nf0 = %f\nf1 = %f\n", N, x1, x2, f0, f1);
    printf("a = %f\nb = %f\n D = %f\n", a, b, D);
    a /= D;
    b /= D;
    printf("a1 = %f\nb2 = %f\n", a, b);
    float temp_a = (float) a;
    float temp_b = (float) b;
    printf("temp_a = %f\ntemp_b = %f\n", temp_a, temp_b);
    coeffs[0] = temp_b;
    coeffs[1] = temp_a;
}

template<typename __nTp>
void quadFit(__nTp *coeffs, std::vector<__nTp> values, int nPulse, int skip) {
    int numN = nPulse;
    double N = 0;
    double x1 = 0;
    double x2 = 0;
    double x3 = 0;
    double x4 = 0;
    double f0 = 0;
    double f1 = 0;
    double f2 = 0;

    for (int i = 0; i < numN; i += skip) {
        N += 1;
        x1 += i;
        x2 += i * i;
        x3 += i * i * i;
        x4 += i * i * i * i;
        f0 += values[i];
        f1 += values[i] * i;
        f2 += values[i] * i * i;
    }

    double D = x4 * x1 * x1 - 2 * x1 * x2 * x3 + x2 * x2 * x2 - N * x4 * x2 + N * x3 * x3;

    double a = f2 * x1 * x1 - f1 * x1 * x2 - f0 * x3 * x1 + f0 * x2 * x2 - N * f2 * x2 + N * f1 * x3;
    double b = f1 * x2 * x2 - N * f1 * x4 + N * f2 * x3 + f0 * x1 * x4 - f0 * x2 * x3 - f2 * x1 * x2;
    double c = f2 * x2 * x2 - f1 * x2 * x3 - f0 * x4 * x2 + f0 * x3 * x3 - f2 * x1 * x3 + f1 * x1 * x4;

    a /= D;
    b /= D;
    c /= D;

    float temp_a = (float) a;
    float temp_b = (float) b;
    float temp_c = (float) c;

    coeffs[0] = temp_c;
    coeffs[1] = temp_b;
    coeffs[2] = temp_a;
}

// CPU implementation of autofocus
template<typename __nTp>
void autofocus_cpu(Complex<__nTp> *data, int numSamples, int numRange, int numIterations) {
    // TODO: ADD ifft/shift before G_dot stuff
    // Samples are number of columns, range is the number of rows
    // Data is column-major ordered
    // CArray<__nTp> G_dot(numSamples * numRange);
    Complex<__nTp> *G = new Complex<__nTp>[numSamples * numRange];
    Complex<__nTp> *G_dot = new Complex<__nTp>[numSamples * numRange]; // Holds the data difference
    double *phi_dot = new double[numSamples]; // Holds phi_dot and will also hold phi for simplicity

    int iii = 0;
    for (; iii < numIterations; iii++) {
        for(int j = 0; j < numSamples; j++) {
            phi_dot[j] = 0;
        }
        CArray<__nTp>temp_g(data, numSamples*numRange);

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// Shifting the max values
// For each row
        for (int rangeIndex = 0; rangeIndex < numRange; rangeIndex++) {
            CArray<__nTp> phaseData = temp_g[std::slice(rangeIndex, numSamples, numRange)];
            CArray<__nTp> tempData = temp_g[std::slice(rangeIndex, numSamples, numRange)];

            // Need to shift phaseData around so max is in the beginning
            int maxIdx = 0;
            __nTp maxVal = 0;
            for (int pulseIndex = 0; pulseIndex < numSamples; pulseIndex++) {
                __nTp tempVal = Complex<__nTp>::abs(phaseData[pulseIndex]);
                if(tempVal > maxVal) {
                    maxVal = tempVal;
                    maxIdx = pulseIndex;
                }
            }

            phaseData = phaseData.cshift(phaseData.size() / 2 + maxIdx);

            for(int pulseIndex = 0; pulseIndex < numSamples; pulseIndex++) {
                temp_g[rangeIndex + pulseIndex * numRange] = phaseData[pulseIndex];
            }
        }

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// Windowing
        double avg_window[numSamples] = {0};
        for (int rangeIndex = 0; rangeIndex < numRange; rangeIndex++) {
            for(int pulseIndex = 0; pulseIndex < numSamples; pulseIndex++) {
                int idx = rangeIndex + pulseIndex * numRange;
                Complex<__nTp> tempHolder = Complex<__nTp>::conj(data[idx]) * data[idx];
                avg_window[pulseIndex] += tempHolder.real();
            }
        }

        double window_maxVal = 0;
        for(int pulseIndex = 0; pulseIndex < numSamples; pulseIndex++) {
            if(window_maxVal < avg_window[pulseIndex]) window_maxVal = avg_window[pulseIndex];
        }

        for(int pulseIndex = 0; pulseIndex < numSamples; pulseIndex++) {
            avg_window[pulseIndex] = 10 * log10(avg_window[pulseIndex] / window_maxVal);
        }

        int leftIdx = -1;
        int rightIdx = -1;
        for(int i = 0; i < numSamples / 2; i++) {
            if(avg_window[i] < -30) leftIdx = i;
            if(avg_window[i + numSamples / 2] < -30 && rightIdx == -1) rightIdx = i;
        }

        if (leftIdx == -1) leftIdx = 0;
        if (rightIdx == -1) rightIdx = numSamples;
        
        Complex<__nTp> tempZero(0, 0);
        for (int rangeIndex = 0; rangeIndex < numRange; rangeIndex++) {
            for(int pulseIndex = 0; pulseIndex < leftIdx; pulseIndex++) {
                int idx = rangeIndex + pulseIndex * numRange;
                temp_g[idx] *= tempZero;
            }
            for(int pulseIndex = rightIdx; pulseIndex < numSamples; pulseIndex++) {
                int idx = rangeIndex + pulseIndex * numRange;
                temp_g[idx] *= tempZero;
            }
        }

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// Getting G

        // For each row
        for (int rangeIndex = 0; rangeIndex < numRange; rangeIndex++) {
            CArray<__nTp> phaseData = temp_g[std::slice(rangeIndex, numSamples, numRange)];

            // ifftw(phaseData);
            // CArray<__nTp> compressed_range = fftshift(phaseData);
            CArray<__nTp> compressed_range = fftshift(phaseData);
            fftw(compressed_range);
            // CArray<__nTp> compressed_range = fftshift(phaseData);
            // ifftw(compressed_range);

            for(int pulseIndex = 0; pulseIndex < numSamples; pulseIndex++) {
                G[rangeIndex + pulseIndex * numRange] = compressed_range[pulseIndex];
                // G[rangeIndex + pulseIndex * numRange] = phaseData[pulseIndex];
            }
        }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Finding G_dot
        // For each row (diff by row)
        for (int rangeNum = 0; rangeNum < numRange; rangeNum++) {
            for(int pulseNum = 0; pulseNum < numSamples - 1; pulseNum++) {
                G_dot[rangeNum + pulseNum * numRange] = G[rangeNum + (pulseNum + 1) * numRange] - G[rangeNum + pulseNum * numRange];
            }
            // G_dot[rangeNum + (numSamples - 1) * numRange] = G_dot[rangeNum + (numSamples - 2) * numRange];
        }

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Finding Phi_dot and Phi
        phi_dot[0] = 0;
        for (int pulseNum = 0; pulseNum < numSamples-1; pulseNum++) {
            double G_norm = 0; // Something to temporarily hold the summed data Norm for that sample
            for (int rangeNum = 0; rangeNum < numRange; rangeNum++) {
                int idx = rangeNum + pulseNum * numRange;
                Complex<__nTp> temp = Complex<__nTp>::conj(G[idx]) * G_dot[idx];
                phi_dot[pulseNum + 1] += temp.imag();
                // phi_dot[pulseNum] += (G[idx]._M_real * G_dot[idx]._M_imag) +
                //                      (-1 * G[idx]._M_imag * G_dot[idx]._M_real); // Only the imaginary component is needed
                // G_norm += sqrt(data[idx]._M_real * data[idx]._M_real - data[idx]._M_imag * data[idx]._M_imag);
                G_norm += (Complex<__nTp>::abs(G[idx]) * Complex<__nTp>::abs(G[idx]));
            }
            // printf("idx = %d, val = %e, g_norm = %e\n", pulseNum, phi_dot[pulseNum], G_norm);
            phi_dot[pulseNum + 1] /= G_norm;
            // printf("idx = %d, val = %e\n", pulseNum, G_norm);
        }

        for (int pulseNum = 1; pulseNum < numSamples; pulseNum++) { // Integrate to get phi
            // printf("idx = %d, val = %f\n", pulseNum, phi_dot[pulseNum]);
            phi_dot[pulseNum] += phi_dot[pulseNum - 1];
            // printf("idx = %d, val = %f\n", pulseNum, phi_dot[pulseNum]);
        }

////////////////////////////////////////////////////////////////////////////////////////////////////
// Removing the linear trend
        // Don't know if removing the linar trend is needed, will check after applying the correction
        // TODO: Try removing the linear trend
        //       Figure out what's causing the numerical instability

        double sumX = 0.0;
        double sumY = 0.0;
        double sumXY = 0.0;
        double sumXX = 0.0;

        for (int i = 0; i < numSamples; i++) {
            sumX += i;
            sumY += phi_dot[i];
            sumXY += i*phi_dot[i];
            sumXX += i * i;
        }

        double numS = numSamples * sumXY - sumX * sumY;
        double den = numSamples * sumXX - sumX * sumX;

        double numC = sumY * sumXX - sumX * sumXY;

        double temp1 = numS/den;
        double temp2 = numC/den;
        double tempa =  temp1;
        double tempb =  temp2;
        // printf("tempa = %f\ntempb = %f\na = %f\nb = %f\nD = %f\n", tempa, tempb, a, b, D);
        for(int i = 0; i < numSamples; i++) {
            phi_dot[i] -= (tempa * i + tempb);
            // printf("idx = %d, val = %f\n", i, phi_dot[i]);
        }

////////////////////////////////////////////////////////////////////////////////////////////
// Condition Check

        double rms = 0;
        for(int i = 0; i < numSamples; i++) {
            // printf("iteration = %d, idx = %d, phi_dot = %e\n", iii, i, phi_dot[i]);
            rms += (phi_dot[i] * phi_dot[i]);
        }
        rms /= numSamples;
        rms = sqrt(rms);
        printf("rms = %f\n", rms);
        if(rms < 0.1) break;

/////////////////////////////////////////////////////////////////////////////////////////////
// Applying the correction
        CArray<__nTp>temp_img(data, numSamples*numRange);

        // For each row
        double alpha = 1;
        for (int rangeNum = 0; rangeNum < numRange; rangeNum++) {
            CArray<__nTp> phaseData = temp_img[std::slice(rangeNum, numSamples, numRange)];
            fftw(phaseData);                                                 
            // ifftw(phaseData);
            // CArray<__nTp> compressed_range = fftshift(phaseData);
            for (int pulseNum = 0; pulseNum < numSamples; pulseNum++) {
                Complex<__nTp> tempExp(cos(alpha * phi_dot[pulseNum]),
                                        sin(-1 * alpha * phi_dot[pulseNum])); // Something to represent e^(-j*phi)
                // int idx = rangeNum + pulseNum * numRange;
                // compressed_range[pulseNum] *= tempExp;
                phaseData[pulseNum] *= tempExp;
            }

            // fftw(compressed_range);
            ifftw(phaseData);
            CArray<__nTp> compressed_range = phaseData.cshift((phaseData.size()));
            // ifftw(compressed_range);
            // compressed_range = fftshift(compressed_range);
            // compressed_range = fftshift(compressed_range);
            for (int pulseNum = 0; pulseNum < numSamples; pulseNum++) {
                int idx = rangeNum + pulseNum * numRange;
                data[idx] = compressed_range[pulseNum];
                // data[idx] = phaseData[rangeNum];
            }
        }

//////////////////////////////////////////////////////////////////////////////////////////////////////
// Done
    }

    printf("Stopping iteration: %d\n", iii);
    delete[] G;
    delete[] G_dot;
    delete[] phi_dot;
}

// Alternative to the Levenburg Marquardt Algorithm (taking too long to debug, compile issues)
// Currently going to be single thread for simplicity
// Assumes data is already ift, shifted, and normed
// TODO: Add dummy error to the data to check it
__global__ void autofocus(cufftComplex *data, int numSamples, int numRange, int numIterations) {
    // Samples are number of columns, range is the number of rows
    // Data is column-major ordered

    cufftComplex *G_dot = new cufftComplex[numSamples * numRange]; // Holds the data difference
    float *phi_dot = new float[numSamples]; // Holds phi_dot and will also hold phi for simplicity

    for (int iii = 0; iii < numIterations; iii++) {
        // Store values into G_dot
        for (int pulseNum = 0; pulseNum < numSamples; pulseNum++) {
            for (int rangeNum = 0; rangeNum < numRange - 1; rangeNum++) { // Because it's a difference
                G_dot[rangeNum + pulseNum * numRange].x =
                        data[rangeNum + pulseNum * numRange].x - data[rangeNum + pulseNum * numRange + 1].x;
                G_dot[rangeNum + pulseNum * numRange].y =
                        data[rangeNum + pulseNum * numRange].y - data[rangeNum + pulseNum * numRange + 1].y;
            }
            // To follow the python code where they append the final sample difference to the matrix to make the size the same as the original
            G_dot[numRange - 1 + pulseNum * numRange].x = G_dot[numRange - 2 + pulseNum * numRange].x;
            G_dot[numRange - 1 + pulseNum * numRange].y = G_dot[numRange - 2 + pulseNum * numRange].y;
        }

        for (int pulseNum = 0; pulseNum < numSamples; pulseNum++) {
            float G_norm = 0; // Something to temporarily hold the summed data Norm for that sample
            for (int rangeNum = 0; rangeNum < numRange; rangeNum++) {
                int idx = rangeNum + pulseNum * numRange;
                phi_dot[pulseNum] += (data[idx].x * G_dot[idx].y) +
                                     (-1 * data[idx].y * G_dot[idx].x); // Only the imaginary component is needed
                G_norm += sqrt(data[idx].x * data[idx].x + data[idx].y * data[idx].y);
            }
            phi_dot[pulseNum] /= G_norm;
        }

        for (int pulseNum = 1; pulseNum < numSamples; pulseNum++) { // Integrate to get phi
            phi_dot[pulseNum] = phi_dot[pulseNum] + phi_dot[pulseNum - 1];
        }

        // Don't know if removing the linar trend is needed, will check after applying the correction

        // TODO: Need to figure out what's being done from the np.tile from the python code (mainly a mental visualization issue)
        // Needed for the correction
        for (int pulseNum = 0; pulseNum < numSamples; pulseNum++) {
            Complex<float> tempExp(cos(phi_dot[pulseNum]),
                                   -1 * sin(phi_dot[pulseNum])); // Something to represent e^(-j*phi)
            for (int rangeNum = 0; rangeNum < numRange; rangeNum++) {
                int idx = rangeNum + pulseNum * numRange;
                data[idx].x = data[idx].x * tempExp.real() + data[idx].y * tempExp.imag();
                data[idx].y = data[idx].x * tempExp.imag() + data[idx].y * tempExp.real();
            }
        }
    }

    delete[] G_dot;
    delete[] phi_dot;
}

// TODO: Need to work on setting up the grid search

template<typename __nTp, typename __nTpParams>
void grid_cuda_focus_SAR_image(const SAR_Aperture<__nTp> &sar_data,
                               const SAR_ImageFormationParameters<__nTpParams> &sar_image_params,
                               CArray<__nTp> &output_image, std::ofstream *myfile, int multiRes, int style,
                               int pulseSkip, std::string output_filename) {

    switch (sar_image_params.algorithm) {
        case SAR_ImageFormationParameters<__nTpParams>::ALGORITHM::BACKPROJECTION:
            std::cout << "Selected backprojection algorithm for focusing." << std::endl;
            //run_bp(sar_data, sar_image_params, output_image);
            break;
        case SAR_ImageFormationParameters<__nTpParams>::ALGORITHM::MATCHED_FILTER:
            std::cout << "Selected matched filtering algorithm for focusing." << std::endl;
            //run_mf(SARData, SARImgParams, output_image);
            //break;
        default:
            std::cout << "focus_SAR_image()::Algorithm requested is not recognized or available." << std::endl;
            return;
    }

    // Display maximum scene size and resolution
    std::cout << "Maximum Scene Size:  " << std::fixed << std::setprecision(2) << sar_image_params.max_Wy_m
              << " m range, "
              << sar_image_params.max_Wx_m << " m cross-range" << std::endl;
    std::cout << "Maximum Resolution:  " << std::fixed << std::setprecision(2) << sar_image_params.slant_rangeResolution
              << "m range, "
              << sar_image_params.azimuthResolution << " m cross-range" << std::endl;
    GPUMemoryManager cuda_res;

    if (initialize_GPUMATLAB(cuda_res.deviceId) == EXIT_FAILURE) {
        std::cout << "cuda_focus_SAR_image::Could not initialize the GPU. Exiting..." << std::endl;
        return;
    }

    if (initialize_CUDAResources(sar_data, sar_image_params, cuda_res) == EXIT_FAILURE) {
        std::cout << "cuda_focus_SAR_image::Problem found initializing resources on the GPU. Exiting..." << std::endl;
        return;
    }

    // Calculate range bins for range compression-based algorithms, e.g., backprojection
    RangeBinData<__nTp> range_bin_data;
    range_bin_data.rangeBins.shape.push_back(sar_image_params.N_fft);
    range_bin_data.rangeBins.shape.push_back(1);
    range_bin_data.rangeBins.data.resize(sar_image_params.N_fft);
    __nTp *rangeBins = &range_bin_data.rangeBins.data[0]; //[sar_image_params.N_fft];
    __nTp minRange = range_bin_data.minRange;
    __nTp maxRange = range_bin_data.maxRange;

    minRange = std::numeric_limits<float>::infinity();
    maxRange = -std::numeric_limits<float>::infinity();
    for (int rIdx = 0; rIdx < sar_image_params.N_fft; rIdx++) {
        // -maxWr/2:maxWr/Nfft:maxWr/2
        //float rVal = ((float) rIdx / Nfft - 0.5f) * maxWr;
        __nTp rVal = RANGE_INDEX_TO_RANGE_VALUE(rIdx, sar_image_params.max_Wy_m, sar_image_params.N_fft);
        rangeBins[rIdx] = rVal;
        if (minRange > rangeBins[rIdx]) {
            minRange = rangeBins[rIdx];
        }
        if (maxRange < rangeBins[rIdx]) {
            maxRange = rangeBins[rIdx];
        }
    }

    cuda_res.copyToDevice("range_vec", (void *) &range_bin_data.rangeBins.data[0],
                          range_bin_data.rangeBins.data.size() * sizeof(range_bin_data.rangeBins.data[0]));

    std::cout << cuda_res << std::endl;
    int numSamples = sar_data.sampleData.data.size();
    int newSize = pow(2, ceil(log(sar_data.sampleData.data.size()) / log(2)));

    clock_t c0, c1, c2;

    c0 = clock();
    //std::cout << printf("N_fft: %d, numAzimuthSamples: %d, numSamples: %d\n\n",sar_image_params.N_fft, sar_data.numAzimuthSamples, newSize);
    cuifft(cuda_res.getDeviceMemPointer<cufftComplex>("sampleData"), sar_image_params.N_fft,
           sar_data.numAzimuthSamples);
    cufftNormalize_1DBatch(cuda_res.getDeviceMemPointer<cufftComplex>("sampleData"), sar_image_params.N_fft,
                           sar_data.numAzimuthSamples);
    cufftShift_1DBatch<cufftComplex>(cuda_res.getDeviceMemPointer<cufftComplex>("sampleData"), sar_image_params.N_fft,
                                     sar_data.numAzimuthSamples);
    c1 = clock();
    printf("INFO: CUDA FFT kernels took %f ms.\n", (float) (c1 - c0) * 1000 / CLOCKS_PER_SEC);

    __nTp delta_x_m_per_pix = sar_image_params.Wx_m / (sar_image_params.N_x_pix - 1);
    __nTp delta_y_m_per_pix = sar_image_params.Wy_m / (sar_image_params.N_y_pix - 1);
    __nTp left_m = sar_image_params.x0_m - sar_image_params.Wx_m / 2;
    __nTp bottom_m = sar_image_params.y0_m - sar_image_params.Wy_m / 2;

    // Set up and run the kernel
    dim3 dimBlock(cuda_res.blockwidth, cuda_res.blockheight, 1);
    dim3 dimGrid(std::ceil((float) sar_image_params.N_x_pix / cuda_res.blockwidth),
    std::ceil((float) sar_image_params.N_y_pix / cuda_res.blockheight));
    c0 = clock();

    // LINE FITTING BASED ON PULSE
    float *xCoeffs, *yCoeffs, *zCoeffs;

    // TODO: CHANGE PULSE NUMBERS, 10, 20, 30 for linear

    std::vector<NumericType> xPossDiff = vectorDiff(sar_data.Ant_x.data);
    std::vector<NumericType> yPossDiff = vectorDiff(sar_data.Ant_y.data);
    std::vector<NumericType> zPossDiff = vectorDiff(sar_data.Ant_z.data);

    // Put all of this outside the for loop
    int numRSamples = sar_data.numRangeSamples, numASamples = sar_data.numAzimuthSamples;
    cufftComplex *data_p = cuda_res.getDeviceMemPointer<cufftComplex>("sampleData");
    __nTp *ax_p = cuda_res.getDeviceMemPointer<__nTp>("Ant_x");
    __nTp *ay_p = cuda_res.getDeviceMemPointer<__nTp>("Ant_y");
    __nTp *az_p = cuda_res.getDeviceMemPointer<__nTp>("Ant_z");
    __nTp *sr_p = cuda_res.getDeviceMemPointer<__nTp>("slant_range");
    __nTp *sf_p = cuda_res.getDeviceMemPointer<__nTp>("startF");
    SAR_ImageFormationParameters<__nTpParams> *sip_p = cuda_res.getDeviceMemPointer<SAR_ImageFormationParameters<__nTpParams >>(
            "sar_image_params");
    __nTp *rv_p = cuda_res.getDeviceMemPointer<__nTp>("range_vec");
    cufftComplex *oi_p = cuda_res.getDeviceMemPointer<cufftComplex>("output_image");
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 24 * (1ULL << 30)));

    numRSamples_nl = numRSamples;
    numASamples_nl = numASamples;
    data_nl = data_p;
    ax_nl = ax_p;
    ay_nl = ay_p;
    az_nl = az_p;
    sr_nl = sr_p;
    sf_nl = sf_p;
    sip_nl = sip_p;
    rv_nl = rv_p;
    delta_x_m_per_pix_nl = delta_x_m_per_pix;
    delta_y_m_per_pix_nl = delta_y_m_per_pix;
    left_m_nl = left_m;
    bottom_m_nl = bottom_m;
    minRange_nl = minRange;
    maxRange_nl = maxRange;

    // GET GRID SEARCH RANGE
    // grid_precision gridDiff = 1e-4f;
    grid_precision gridDiff = 1.3f;
    grid_precision gridN = 5;

    float totalTime = 0;

    if (style == 0) {
        gridDiff = 2.6f;
        gridN = 23;
        printf("Using Linear Model\n");
        xCoeffs = new float[2];
        yCoeffs = new float[2];
        zCoeffs = new float[2];
        grid_precision minParams[grid_dimension_linear] = {0};

        bestFit<NumericType>(xCoeffs, sar_data.Ant_x.data, sar_data.numAzimuthSamples, pulseSkip);
        bestFit<NumericType>(yCoeffs, sar_data.Ant_y.data, sar_data.numAzimuthSamples, pulseSkip);
        bestFit<NumericType>(zCoeffs, sar_data.Ant_z.data, sar_data.numAzimuthSamples, pulseSkip);

        printf("X - Slope coeff = %f\n    Const coeff = %f\n", xCoeffs[1], xCoeffs[0]);
        printf("Y - Slope coeff = %f\n    Const coeff = %f\n", yCoeffs[1], yCoeffs[0]);
        printf("Z - Slope coeff = %f\n    Const coeff = %f\n", zCoeffs[1], zCoeffs[0]);

        *myfile << "gt," << xCoeffs[1] << ',' << xCoeffs[0] << ','
                << yCoeffs[1] << ',' << yCoeffs[0] << ','
                << zCoeffs[1] << ',' << zCoeffs[0] << ',';

        grid_precision *covar_matrix = new grid_precision[(grid_dimension_linear)*(grid_dimension_linear)];
        std::vector<grid_precision> start_point = {(grid_precision) xCoeffs[0], (grid_precision) xCoeffs[1] - gridDiff,
                                                   (grid_precision) yCoeffs[0], (grid_precision) yCoeffs[1] - gridDiff,
                                                   (grid_precision) zCoeffs[0], (grid_precision) zCoeffs[1] - gridDiff,
                                                   (grid_precision) 0.9};
        std::vector<grid_precision> end_point = {(grid_precision) xCoeffs[0], (grid_precision) xCoeffs[1] + gridDiff,//(grid_precision) xCoeffs[1] + gridDiff,
                                                 (grid_precision) yCoeffs[0], (grid_precision) yCoeffs[1] + gridDiff,//(grid_precision) yCoeffs[1] + gridDiff,
                                                 (grid_precision) zCoeffs[0], (grid_precision) zCoeffs[1] + gridDiff,//(grid_precision) zCoeffs[1] + gridDiff,
                                                 (grid_precision) 1.1};
        std::vector<grid_precision> grid_numSamples = {(grid_precision) 1, (grid_precision) gridN,
                                                       (grid_precision) 1, (grid_precision) gridN,
                                                       (grid_precision) 1, (grid_precision) gridN,
                                                       (grid_precision) 5};
        image_err_func_byvalue_linear host_func_byval_ptr;
        // Copy device function pointer for the function having by-value parameters to host side
        cudaMemcpyFromSymbol(&host_func_byval_ptr, dev_func_byvalue_ptr_linear,
                             sizeof(dev_func_byvalue_ptr_linear));

        for (int iii = 0; iii < multiRes; iii++) {
            CudaGrid<grid_precision, grid_dimension_linear> grid;
            ck(cudaMalloc(&grid.data(), grid.bytesSize()));

            grid.setStartPoint(start_point);
            grid.setEndPoint(end_point);
            grid.setNumSamples(grid_numSamples);
            grid.display("grid");

            grid_precision axis_sample_counts[grid_dimension_linear];
            grid.getAxisSampleCounts(axis_sample_counts);

            CudaTensor<func_precision, grid_dimension_linear> func_values(axis_sample_counts);
            ck(cudaMalloc(&func_values._data, func_values.bytesSize()));

            // first template argument is the error function return type
            // second template argument is the grid point value type
            CudaGridSearcher<func_precision, grid_precision, grid_dimension_linear> gridsearcher(grid, func_values);

            c1 = clock();
            gridsearcher.search_by_value_stream(host_func_byval_ptr, 1000, sar_image_params.N_x_pix,
                    // gridsearcher.search_by_value(host_func_byval_ptr,
                                                data_p,
                                                numRSamples, numASamples,
                                                delta_x_m_per_pix, delta_y_m_per_pix,
                                                left_m, bottom_m,
                                                minRange, maxRange,
                                                ax_p,
                                                ay_p,
                                                az_p,
                                                sr_p,
                                                sf_p,
                                                sip_p,
                                                rv_p);
            c2 = clock();
            float searchTime = (float) (c2 - c1) * 1000 / CLOCKS_PER_SEC;
            printf("INFO: cuGridSearch took %f ms.\n", searchTime);

            totalTime += searchTime;

            func_precision min_value;
            int32_t min_value_index1d;
            func_values.find_extrema(min_value, min_value_index1d);

            grid_precision min_grid_point[grid_dimension_linear];
            grid.getGridPoint(min_grid_point, min_value_index1d);
            std::cout << "Minimum found at point p = { ";
            for (int d = 0; d < grid_dimension_linear; d++) {
                minParams[d] = min_grid_point[d];
                std::cout << min_grid_point[d] << ((d < grid_dimension_linear - 1) ? ", " : " ");

                start_point[d] = min_grid_point[d] - (end_point[d] - start_point[d]) / 4;
                end_point[d] = min_grid_point[d] + (end_point[d] - start_point[d]) / 4;
                // Update grid here or outside (basically somewhere near here)
            }
            std::cout << "}" << std::endl;

            for (int i = 0; i < (grid_dimension_linear)*(grid_dimension_linear); i++) {
                covar_matrix[i] = 0;
            }
            gridsearcher.covariance_by_value(covar_matrix);

            ck(cudaFree(grid.data()));
            ck(cudaFree(func_values.data()));

        }

        printf("Total time took: %f\n", totalTime);
        *myfile << "time," << totalTime << ',';

        printf("MinParams[");
        for (int d = 0; d < grid_dimension_linear; d++) {
            printf("%e,", minParams[d]);
        }
        printf("]\n");

        printf("Covariance matrix[");
        for (int d = 0; d < (grid_dimension_linear)*(grid_dimension_linear); d++) {
            printf("%e,", covar_matrix[d]);
        }
        printf("]\n");
        delete[] covar_matrix;
        *myfile << "found,";
        for (int i = 0; i < grid_dimension_linear; i++)
            *myfile << minParams[i] << ',';

        nv_ext::Vec<grid_precision, grid_dimension_linear> minParamsVec(minParams);
        computeImageKernel<func_precision, grid_precision, grid_dimension_linear, __nTp><<<1, sar_image_params.N_x_pix>>>(minParamsVec,
                                                                                                     data_p,
                                                                                                     numRSamples,
                                                                                                     numASamples,
                                                                                                     delta_x_m_per_pix,
                                                                                                     delta_y_m_per_pix,
                                                                                                     left_m, bottom_m,
                                                                                                     minRange, maxRange,
                                                                                                     ax_p,
                                                                                                     ay_p,
                                                                                                     az_p,
                                                                                                     sr_p,
                                                                                                     sf_p,
                                                                                                     sip_p,
                                                                                                     rv_p,
                                                                                                     oi_p);

    } else {
        // TODO: Increase the search space
        // Need to figure out where it'll break
        printf("Using Quadratic Model\n");
        xCoeffs = new float[3];
        yCoeffs = new float[3];
        zCoeffs = new float[3];
        grid_precision minParams[grid_dimension_quadratic] = {0};

        quadFit<NumericType>(xCoeffs, sar_data.Ant_x.data, sar_data.numAzimuthSamples, pulseSkip);
        quadFit<NumericType>(yCoeffs, sar_data.Ant_y.data, sar_data.numAzimuthSamples, pulseSkip);
        quadFit<NumericType>(zCoeffs, sar_data.Ant_z.data, sar_data.numAzimuthSamples, pulseSkip);

        printf("X - Quad coeff = %f\n    Slope coeff = %f\n    Const coeff = %f\n", xCoeffs[2], xCoeffs[1], xCoeffs[0]);
        printf("Y - Quad coeff = %f\n    Slope coeff = %f\n    Const coeff = %f\n", yCoeffs[2], yCoeffs[1], yCoeffs[0]);
        printf("Z - Quad coeff = %f\n    Slope coeff = %f\n    Const coeff = %f\n", zCoeffs[2], zCoeffs[1], zCoeffs[0]);

        *myfile << "gt," << xCoeffs[2] << ',' << xCoeffs[1] << ',' << xCoeffs[0] << ','
                << yCoeffs[2] << ',' << yCoeffs[1] << ',' << yCoeffs[0] << ','
                << zCoeffs[2] << ',' << zCoeffs[1] << ',' << zCoeffs[0] << ',';

        grid_precision *covar_matrix = new grid_precision[(grid_dimension_quadratic)*(grid_dimension_quadratic)];
        std::vector<grid_precision> start_point = {(grid_precision) xCoeffs[0], (grid_precision) xCoeffs[1] - gridDiff,
                                                   (grid_precision) xCoeffs[2] - gridDiff/2,
                                                   (grid_precision) yCoeffs[0], (grid_precision) yCoeffs[1] - gridDiff,
                                                   (grid_precision) yCoeffs[2] - gridDiff/2,
                                                   (grid_precision) zCoeffs[0], (grid_precision) zCoeffs[1] - gridDiff,
                                                   (grid_precision) zCoeffs[2] - gridDiff/2,
                                                   (grid_precision) 0.9};
        std::vector<grid_precision> end_point = {(grid_precision) xCoeffs[0], (grid_precision) xCoeffs[1] + gridDiff,
                                                 (grid_precision) xCoeffs[2] + gridDiff/2,
                                                 (grid_precision) yCoeffs[0], (grid_precision) yCoeffs[1] + gridDiff,
                                                 (grid_precision) yCoeffs[2] + gridDiff/2,
                                                 (grid_precision) zCoeffs[0], (grid_precision) zCoeffs[1] + gridDiff,
                                                 (grid_precision) zCoeffs[2] + gridDiff/2,
                                                 (grid_precision) 1.1};
        std::vector<grid_precision> grid_numSamples = {(grid_precision) 1, (grid_precision) gridN, (grid_precision) gridN,
                                                       (grid_precision) 1, (grid_precision) gridN, (grid_precision) gridN,
                                                       (grid_precision) 1, (grid_precision) gridN, (grid_precision) gridN,
                                                       (grid_precision) 5};
        
        std::ofstream bounds_json("search_bounds.json");
        int n_trials = 1000;
        if (bounds_json.is_open()) {
            // Start with trial
            bounds_json << "{\n\t\"ntrials\": " << n_trials << ",\n";
    
            bounds_json << "\t\"xCoeffs0\": {\n\t\t\"low\": " << xCoeffs[0] << ",\n";
            bounds_json << "\t\t\"high\": " << xCoeffs[0] << "\n\t},\n";
    
            bounds_json << "\t\"xCoeffs1\": {\n\t\t\"low\": " << xCoeffs[1] - gridDiff << ",\n";
            bounds_json << "\t\t\"high\": " << xCoeffs[1] + gridDiff << "\n\t},\n";
            
            bounds_json << "\t\"xCoeffs2\": {\n\t\t\"low\": " << xCoeffs[2] - gridDiff/2 << ",\n";
            bounds_json << "\t\t\"high\": " << xCoeffs[2] + gridDiff/2 << "\n\t},\n";

            bounds_json << "\t\"yCoeffs0\": {\n\t\t\"low\": " << yCoeffs[0] << ",\n";
            bounds_json << "\t\t\"high\": " << yCoeffs[0] << "\n\t},\n";
    
            bounds_json << "\t\"yCoeffs1\": {\n\t\t\"low\": " << yCoeffs[1] - gridDiff << ",\n";
            bounds_json << "\t\t\"high\": " << yCoeffs[1] + gridDiff << "\n\t},\n";
            
            bounds_json << "\t\"yCoeffs2\": {\n\t\t\"low\": " << yCoeffs[2] - gridDiff/2 << ",\n";
            bounds_json << "\t\t\"high\": " << yCoeffs[2] + gridDiff/2 << "\n\t},\n";

            bounds_json << "\t\"zCoeffs0\": {\n\t\t\"low\": " << zCoeffs[0] << ",\n";
            bounds_json << "\t\t\"high\": " << zCoeffs[0] << "\n\t},\n";
    
            bounds_json << "\t\"zCoeffs1\": {\n\t\t\"low\": " << zCoeffs[1] - gridDiff << ",\n";
            bounds_json << "\t\t\"high\": " << zCoeffs[1] + gridDiff << "\n\t},\n";
            
            bounds_json << "\t\"zCoeffs2\": {\n\t\t\"low\": " << zCoeffs[2] - gridDiff/2 << ",\n";
            bounds_json << "\t\t\"high\": " << zCoeffs[2] + gridDiff/2 << "\n\t},\n";

            bounds_json << "\t\"vel_percent\": {\n\t\t\"low\": " << 0.9 << ",\n";
            bounds_json << "\t\t\"high\": " << 1.1 << "\n\t}\n";

            // ***********************************************************
            // Verify test

            // bounds_json << "\t\"xCoeffs0\": {\n\t\t\"low\": " << xCoeffs[0] << ",\n";
            // bounds_json << "\t\t\"high\": " << xCoeffs[0] << "\n\t},\n";
    
            // bounds_json << "\t\"xCoeffs1\": {\n\t\t\"low\": " << xCoeffs[1]  << ",\n";
            // bounds_json << "\t\t\"high\": " << xCoeffs[1]  << "\n\t},\n";
            
            // bounds_json << "\t\"xCoeffs2\": {\n\t\t\"low\": " << xCoeffs[2]  << ",\n";
            // bounds_json << "\t\t\"high\": " << xCoeffs[2]  << "\n\t},\n";

            // bounds_json << "\t\"yCoeffs0\": {\n\t\t\"low\": " << yCoeffs[0] << ",\n";
            // bounds_json << "\t\t\"high\": " << yCoeffs[0] << "\n\t},\n";
    
            // bounds_json << "\t\"yCoeffs1\": {\n\t\t\"low\": " << yCoeffs[1]  << ",\n";
            // bounds_json << "\t\t\"high\": " << yCoeffs[1]  << "\n\t},\n";
            
            // bounds_json << "\t\"yCoeffs2\": {\n\t\t\"low\": " << yCoeffs[2]  << ",\n";
            // bounds_json << "\t\t\"high\": " << yCoeffs[2]  << "\n\t},\n";

            // bounds_json << "\t\"zCoeffs0\": {\n\t\t\"low\": " << zCoeffs[0] << ",\n";
            // bounds_json << "\t\t\"high\": " << zCoeffs[0] << "\n\t},\n";
    
            // bounds_json << "\t\"zCoeffs1\": {\n\t\t\"low\": " << zCoeffs[1]  << ",\n";
            // bounds_json << "\t\t\"high\": " << zCoeffs[1]  << "\n\t},\n";
            
            // bounds_json << "\t\"zCoeffs2\": {\n\t\t\"low\": " << zCoeffs[2]  << ",\n";
            // bounds_json << "\t\t\"high\": " << zCoeffs[2]  << "\n\t},\n";

            // bounds_json << "\t\"vel_percent\": {\n\t\t\"low\": " << 1 << ",\n";
            // bounds_json << "\t\t\"high\": " << 1 << "\n\t}\n";
            
            // **************************************************************************
            bounds_json << "}" << std::endl;
            bounds_json.close();
        } else {
            std::cerr << "Error opening search_bounds.json" << std::endl;
        }
        int optuna_results = system("/usr/bin/python3 optunaSearchCaller.py");

        image_err_func_byvalue_quadratic host_func_byval_ptr;
        // Copy device function pointer for the function having by-value parameters to host side
        cudaMemcpyFromSymbol(&host_func_byval_ptr, dev_func_byvalue_ptr_quadratic,
                             sizeof(dev_func_byvalue_ptr_quadratic));

        for (int iii = 0; iii < multiRes; iii++) {
            // CudaGrid<grid_precision, grid_dimension_quadratic> grid;
            // ck(cudaMalloc(&grid.data(), grid.bytesSize()));

            // grid.setStartPoint(start_point);
            // grid.setEndPoint(end_point);
            // grid.setNumSamples(grid_numSamples);
            // grid.display("grid");

            grid_precision axis_sample_counts[grid_dimension_quadratic];
            // grid.getAxisSampleCounts(axis_sample_counts);
            axis_sample_counts[0] = n_trials;

            CudaTensor<func_precision, grid_dimension_quadratic> func_values(axis_sample_counts);
            ck(cudaMalloc(&func_values._data, func_values.bytesSize()));

            // first template argument is the error function return type
            // second template argument is the grid point value type
            // CudaGridSearcher<func_precision, grid_precision, grid_dimension_quadratic> gridsearcher(grid, func_values);

            c1 = clock();
            search_by_value_stream_optuna<func_precision, grid_precision, grid_dimension_quadratic> (func_values, host_func_byval_ptr, 1000, sar_image_params.N_x_pix,
            // gridsearcher.search_by_value_block(host_func_byval_ptr, sar_image_params.N_x_pix,
                    // gridsearcher.search_by_value(host_func_byval_ptr,
                                                data_p,
                                                numRSamples, numASamples,
                                                delta_x_m_per_pix, delta_y_m_per_pix,
                                                left_m, bottom_m,
                                                minRange, maxRange,
                                                ax_p,
                                                ay_p,
                                                az_p,
                                                sr_p,
                                                sf_p,
                                                sip_p,
                                                rv_p);
            c2 = clock();
            float searchTime = (float) (c2 - c1) * 1000 / CLOCKS_PER_SEC;
            printf("INFO: cuGridSearch took %f ms.\n", searchTime);

            totalTime += searchTime;

            func_precision min_value;
            int32_t min_value_index1d;
            func_values.find_extrema(min_value, min_value_index1d);

            grid_precision min_grid_point[grid_dimension_quadratic];

            // float covar_diag[7] = {covar_matrix[11], covar_matrix[22], covar_matrix[44], covar_matrix[55], covar_matrix[77], covar_matrix[88], covar_matrix[99]};
            
            // min_value_index1d = gridsearcher.get_second_min();
            std::cout << "Min idx: " << min_value_index1d << std::endl;
            // grid.getGridPoint(min_grid_point, min_value_index1d);
            int line_idx = 0;
            std::ifstream gridPoints_csv("search_grid_points.csv");
            std::string line;
            do {
                std::getline(gridPoints_csv, line);
            } while (line_idx < min_value_index1d);
            gridPoints_csv.close();

            size_t start_pos = 0;
            size_t end_pos = 0;
            int param_d = 0;
            end_pos = line.find(',');
            while (end_pos < line.length()) {
                min_grid_point[param_d] = std::stod(line.substr(start_pos, end_pos - start_pos));
                start_pos = end_pos + 1;
                end_pos = line.find(',', start_pos);
                param_d = param_d + 1;
            }

            std::cout << "Minimum found at point p = { ";
            for (int d = 0; d < grid_dimension_quadratic; d++) {
                minParams[d] = min_grid_point[d];
                std::cout << min_grid_point[d] << ((d < grid_dimension_quadratic - 1) ? ", " : " ");

                start_point[d] = min_grid_point[d] - (end_point[d] - start_point[d]) / 4;
                end_point[d] = min_grid_point[d] + (end_point[d] - start_point[d]) / 4;
                // Update grid here or outside (basically somewhere near here)
            }
            std::cout << "}" << std::endl;

            for (int i = 0; i < (grid_dimension_quadratic)*(grid_dimension_quadratic); i++) {
                covar_matrix[i] = 0;
            }
            // gridsearcher.covariance_by_value(covar_matrix);
            
            // min_grid_point[0] = xCoeffs[0];
            // min_grid_point[1] = xCoeffs[1];
            // min_grid_point[2] = xCoeffs[2];
            // min_grid_point[3] = yCoeffs[0];
            // min_grid_point[4] = yCoeffs[1];
            // min_grid_point[5] = yCoeffs[2];
            // min_grid_point[6] = zCoeffs[0];
            // min_grid_point[7] = zCoeffs[1];
            // min_grid_point[8] = zCoeffs[2];
            // min_grid_point[9] = 1.0;

            // ck(cudaFree(grid.data()));
            ck(cudaFree(func_values.data()));

        }

        printf("Total time took: %f\n", totalTime);
        *myfile << "time," << totalTime << ',';

        printf("MinParams[");
        for (int d = 0; d < grid_dimension_quadratic; d++) {
            printf("%e,", minParams[d]);
        }
        printf("]\n");

        
        printf("Covariance matrix[");
        for (int d = 0; d < (grid_dimension_quadratic)*(grid_dimension_quadratic); d++) {
            printf("%e,", covar_matrix[d]);
        }
        printf("]\n");
        delete[] covar_matrix;

        *myfile << "found,";
        for (int i = 0; i < grid_dimension_quadratic; i++)
            *myfile << minParams[i] << ',';

        // // Non-linear optimizer
        // // By curve 
        // Eigen::VectorXf x( grid_dimension_quadratic);
        // for(int i = 0; i <  grid_dimension_quadratic; i++)
        //     x(i) = minParams[i];
        // std::cout << "x: " << x << std::endl;

        // my_functor functor;
        // Eigen::NumericalDiff<my_functor> numDiff(functor);
        // Eigen::LevenbergMarquardt<Eigen::NumericalDiff<my_functor>,float> lm(numDiff);

        // lm.parameters.maxfev = 2000;
        // lm.parameters.xtol = 1.0e-10;

        // int ret = lm.minimize(x);
        // std::cout << "Iterations: " << lm.iter << ", Return code: " << ret << std::endl;

        // std::cout << "x that minimizes the function: " << x << std::endl;

        // // Place Found minimums to minParams
        // for(int i = 0; i <  grid_dimension_quadratic; i++)
        //     minParams[i] = x(i);

//         // PGA
//         // LM can't do complex numbers 
//         /*
//         In file included from /usr/include/c++/7/bits/char_traits.h:39:0,
//                  from /usr/include/c++/7/ios:40,
//                  from /usr/include/c++/7/ostream:38,
//                  from /usr/include/c++/7/iostream:39,
//                  from /home/grey/uncc_sar_focusing_gridSearch/cuGridSearch/src/cpu/cpuNonLinearOptimizer.cpp:4:
// /usr/include/c++/7/bits/stl_algobase.h: In instantiation of ‘constexpr const _Tp& std::min(const _Tp&, const _Tp&) [with _Tp = std::complex<double>]’:
// /usr/local/include/eigen3/unsupported/Eigen/src/NonLinearOptimization/LevenbergMarquardt.h:284:31:   required from ‘Eigen::LevenbergMarquardtSpace::Status Eigen::LevenbergMarquardt<FunctorType, Scalar>::minimizeOneStep(Eigen::LevenbergMarquardt<FunctorType, Scalar>::FVectorType&) [with FunctorType = Eigen::NumericalDiff<my_functor>; Scalar = std::complex<double>; Eigen::LevenbergMarquardt<FunctorType, Scalar>::FVectorType = Eigen::Matrix<std::complex<double>, -1, 1>]’
// /usr/local/include/eigen3/unsupported/Eigen/src/NonLinearOptimization/LevenbergMarquardt.h:164:33:   required from ‘Eigen::LevenbergMarquardtSpace::Status Eigen::LevenbergMarquardt<FunctorType, Scalar>::minimize(Eigen::LevenbergMarquardt<FunctorType, Scalar>::FVectorType&) [with FunctorType = Eigen::NumericalDiff<my_functor>; Scalar = std::complex<double>; Eigen::LevenbergMarquardt<FunctorType, Scalar>::FVectorType = Eigen::Matrix<std::complex<double>, -1, 1>]’
// /home/grey/uncc_sar_focusing_gridSearch/cuGridSearch/src/cpu/cpuNonLinearOptimizer.cpp:61:28:   required from here
// /usr/include/c++/7/bits/stl_algobase.h:200:15: error: no match for ‘operator<’ (operand types are ‘const std::complex<double>’ and ‘const std::complex<double>’)
//        if (__b < __a)
//            ~~~~^~~~~
//         */
        // Eigen::VectorXcf xf(numASamples * numRSamples);
        // PGAFunctor<std::complex<float>> pgaFunctor(numASamples*numRSamples, numASamples*numRSamples);
        // cufftComplex* data_cpu_p = cuda_res.getHostMemPointer<cufftComplex>("sampleData");
        // for(int i = 0; i < numASamples * numRSamples; i++) {
        //     xf(i).real(data_cpu_p[i].x);
        //     xf(i).imag(data_cpu_p[i].y);
        // }   
        // Eigen::LevenbergMarquardt<PGAFunctor<std::complex<float>>,float> lm(pgaFunctor);
        // lm.parameters.maxfev = 2000;
        // lm.parameters.xtol = 1.0e-10;

        // int ret = lm.minimize(xf);
        // std::cout << "Iterations: " << lm.iter << ", Return code: " << ret << std::endl;

        nv_ext::Vec<grid_precision, grid_dimension_quadratic> minParamsVec(minParams);
        computeImageKernel<func_precision, grid_precision, grid_dimension_quadratic, __nTp><<<1, sar_image_params.N_x_pix>>>(minParamsVec,
                                                                                                        data_p,
                                                                                                        numRSamples,
                                                                                                        numASamples,
                                                                                                        delta_x_m_per_pix,
                                                                                                        delta_y_m_per_pix,
                                                                                                        left_m,
                                                                                                        bottom_m,
                                                                                                        minRange,
                                                                                                        maxRange,
                                                                                                        ax_p,
                                                                                                        ay_p,
                                                                                                        az_p,
                                                                                                        sr_p,
                                                                                                        sf_p,
                                                                                                        sip_p,
                                                                                                        rv_p,
                                                                                                        oi_p);

    }

//    grid_precision testHolder[] = {xCoeffs[0], xCoeffs[1], yCoeffs[0], yCoeffs[1], zCoeffs[0], zCoeffs[1]};
//    nv_ext::Vec<grid_precision, grid_dimension_quadratic> testHolderVec(testHolder);

    /* NOTE: COMMENT IF GRID ONLY */
    c1 = clock();
    printf("INFO: CUDA Backprojection kernel launch took %f ms.\n", (float) (c1 - c0) * 1000 / CLOCKS_PER_SEC);
    if (cudaDeviceSynchronize() != cudaSuccess)
        printf("\nERROR: threads did NOT synchronize! DO NOT TRUST RESULTS!\n\n");
    c2 = clock();
    printf("INFO: CUDA Backprojection execution took %f ms.\n", (float) (c2 - c1) * 1000 / CLOCKS_PER_SEC);
    printf("INFO: CUDA Backprojection total time took %f ms.\n", (float) (c2 - c0) * 1000 / CLOCKS_PER_SEC);
    /**/

    int num_img_bytes = sizeof(cufftComplex) * sar_image_params.N_x_pix * sar_image_params.N_y_pix;
    std::vector<cufftComplex> image_data(sar_image_params.N_x_pix * sar_image_params.N_y_pix);
    //cuda_res.copyFromDevice("output_image", &output_image[0], num_img_bytes);
    cuda_res.copyFromDevice("output_image", image_data.data(), num_img_bytes);
    for (int idx = 0; idx < sar_image_params.N_x_pix * sar_image_params.N_y_pix; idx++) {
        output_image[idx]._M_real = image_data[idx].x;
        output_image[idx]._M_imag = image_data[idx].y;
    }
    // std::string tempStr("_beforeAF.bmp");
    // std::string beforeAF = output_filename + tempStr;
    // writeBMPFile(sar_image_params, output_image, beforeAF); // NOTE: Debugging only
    // Complex<__nTp> temp_out[sar_image_params.N_x_pix * sar_image_params.N_y_pix];
    // for(int iii = 0; iii < sar_image_params.N_x_pix * sar_image_params.N_y_pix; iii++)
    //     temp_out[iii] = output_image[iii];
    // autofocus_cpu<__nTp>(temp_out, sar_image_params.N_x_pix, sar_image_params.N_y_pix, 30);
    // for(int iii = 0; iii < sar_image_params.N_x_pix * sar_image_params.N_y_pix; iii++)
    //     output_image[iii] = temp_out[iii];
    // std::string afterAF("sar_image_afterAF.bmp");
    // writeBMPFile(sar_image_params, output_image, afterAF); // NOTE: Debugging only
    cuda_res.freeGPUMemory("range_vec");

    delete[] xCoeffs;
    delete[] yCoeffs;
    delete[] zCoeffs;

    if (finalize_CUDAResources(sar_data, sar_image_params, cuda_res) == EXIT_FAILURE) {
        std::cout << "cuda_focus_SAR_image::Problem found de-allocating and free resources on the GPU. Exiting..."
                  << std::endl;
        return;
    }
    std::cout << cuda_res << std::endl;
}

void cxxopts_integration_local(cxxopts::Options &options) {

    options.add_options()
            ("i,input", "Input file", cxxopts::value<std::vector<std::string>>())
            ("k,pulseSkip", "Number of pulses to skip for estimation", cxxopts::value<int>()->default_value("1"))
            ("m,multi", "Multiresolution Value", cxxopts::value<int>()->default_value("1"))
            ("n,numPulse", "Number of pulses to focus", cxxopts::value<int>()->default_value("0"))
            ("s,style", "Linear or Quadratic Calculation", cxxopts::value<int>()->default_value("0"))
            //("f,format", "Data format {GOTCHA, Sandia, <auto>}", cxxopts::value<std::string>()->default_value("auto"))
            ("p,polarity", "Polarity {HH,HV,VH,VV,<any>}", cxxopts::value<std::string>()->default_value("any"))
            ("d,debug", "Enable debugging", cxxopts::value<bool>()->default_value("false"))
            ("v,verbose", "Enable verbose output", cxxopts::value<bool>(verbose))
            ("r,dynrange", "Dynamic Range (dB) <70 dB>", cxxopts::value<float>()->default_value("70"))
            ("o,output", "Output file <sar_image.bmp>", cxxopts::value<std::string>()->default_value("sar_image.bmp"))
            ("h,help", "Print usage");
}

int main(int argc, char **argv) {
    ComplexType test[] = {1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0};
    ComplexType out[8];
    ComplexArrayType data(test, 8);
    std::unordered_map<std::string, matvar_t *> matlab_readvar_map;

    cxxopts::Options options("cpuBackProjection",
                             "UNC Charlotte Machine Vision Lab SAR Back Projection focusing code.");
    cxxopts_integration_local(options);

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }
    bool debug = result["debug"].as<bool>();
    int multiRes = result["multi"].as<int>();
    int style = result["style"].as<int>();
    int nPulse = result["numPulse"].as<int>();
    int pulseSkip = result["pulseSkip"].as<int>();

    initialize_Sandia_SPHRead(matlab_readvar_map);
    initialize_GOTCHA_MATRead(matlab_readvar_map);

    std::string inputfile;
    std::vector<std::string> inputfiles;
    if (result.count("input")) {
        // inputfile = result["input"].as<std::string>();
        auto& ff = result["input"].as<std::vector<std::string>>();
        for (const auto& f : ff) {
            inputfiles.push_back(f);
        }
    inputfile = inputfiles[0];
    } else {
        std::stringstream ss;

        // Sandia SAR DATA FILE LOADING
        int file_idx = 9; // 1-10 for Sandia Rio Grande, 1-9 for Sandia Farms
        std::string fileprefix = Sandia_RioGrande_fileprefix;
        std::string filepostfix = Sandia_RioGrande_filepostfix;
        //        std::string fileprefix = Sandia_Farms_fileprefix;
        //        std::string filepostfix = Sandia_Farms_filepostfix;
        ss << std::setfill('0') << std::setw(2) << file_idx;


        // GOTCHA SAR DATA FILE LOADING
//        int azimuth = 1; // 1-360 for all GOTCHA polarities=(HH,VV,HV,VH) and pass=[pass1,...,pass7]
        //        std::string fileprefix = GOTCHA_fileprefix;
        //        std::string filepostfix = GOTCHA_filepostfix;
        //        ss << std::setfill('0') << std::setw(3) << azimuth;

        inputfile = fileprefix + ss.str() + filepostfix + ".mat";
    }

    std::cout << "Successfully opened MATLAB file " << inputfile << "." << std::endl;

    SAR_Aperture<NumericType> SAR_aperture_data;
    SAR_Aperture<NumericType> SAR_aperture_data_temp;
    SAR_Aperture<NumericType> SAR_aperture_data_temp2;

    if (read_MAT_Variables(inputfile, matlab_readvar_map, SAR_aperture_data) == EXIT_FAILURE) {
        std::cout << "Could not read all desired MATLAB variables from " << inputfile << " exiting." << std::endl;
        return EXIT_FAILURE;
    }

    for (int file_idx = 1; file_idx < inputfiles.size(); file_idx++) {
        if (read_MAT_Variables(inputfiles[file_idx], matlab_readvar_map, SAR_aperture_data_temp) == EXIT_FAILURE) {
            std::cout << "Could not read all desired MATLAB variables from " << inputfile << " exiting." << std::endl;
            return EXIT_FAILURE;
        }
        // Double check uncc_sar_focusing.hpp if things were correctly appended
        SAR_aperture_data.sampleData.data.insert(SAR_aperture_data.sampleData.data.end(), SAR_aperture_data_temp.sampleData.data.begin(), SAR_aperture_data_temp.sampleData.data.end());
        SAR_aperture_data.Ant_x.data.insert(SAR_aperture_data.Ant_x.data.end(), SAR_aperture_data_temp.Ant_x.data.begin(), SAR_aperture_data_temp.Ant_x.data.end());
        SAR_aperture_data.Ant_y.data.insert(SAR_aperture_data.Ant_y.data.end(), SAR_aperture_data_temp.Ant_y.data.begin(), SAR_aperture_data_temp.Ant_y.data.end());
        SAR_aperture_data.Ant_z.data.insert(SAR_aperture_data.Ant_z.data.end(), SAR_aperture_data_temp.Ant_z.data.begin(), SAR_aperture_data_temp.Ant_z.data.end());
        SAR_aperture_data.slant_range.data.insert(SAR_aperture_data.slant_range.data.end(), SAR_aperture_data_temp.slant_range.data.begin(), SAR_aperture_data_temp.slant_range.data.end());
        SAR_aperture_data.theta.data.insert(SAR_aperture_data.theta.data.end(), SAR_aperture_data_temp.theta.data.begin(), SAR_aperture_data_temp.theta.data.end());
        SAR_aperture_data.phi.data.insert(SAR_aperture_data.phi.data.end(), SAR_aperture_data_temp.phi.data.begin(), SAR_aperture_data_temp.phi.data.end());
        SAR_aperture_data.af.r_correct.data.insert(SAR_aperture_data.af.r_correct.data.end(), SAR_aperture_data_temp.af.r_correct.data.begin(), SAR_aperture_data_temp.af.r_correct.data.end());
        SAR_aperture_data.af.ph_correct.data.insert(SAR_aperture_data.af.ph_correct.data.end(), SAR_aperture_data_temp.af.ph_correct.data.begin(), SAR_aperture_data_temp.af.ph_correct.data.end());

        SAR_aperture_data.sampleData.shape[1] += SAR_aperture_data_temp.sampleData.shape[1];
        SAR_aperture_data.Ant_x.shape[1] += SAR_aperture_data_temp.Ant_x.shape[1];
        SAR_aperture_data.Ant_y.shape[1] += SAR_aperture_data_temp.Ant_y.shape[1];
        SAR_aperture_data.Ant_z.shape[1] += SAR_aperture_data_temp.Ant_z.shape[1];
        SAR_aperture_data.slant_range.shape[1] += SAR_aperture_data_temp.slant_range.shape[1];
        SAR_aperture_data.theta.shape[1] += SAR_aperture_data_temp.theta.shape[1];
        SAR_aperture_data.phi.shape[1] += SAR_aperture_data_temp.phi.shape[1];
        SAR_aperture_data.af.r_correct.shape[1] += SAR_aperture_data_temp.af.r_correct.shape[1];
        SAR_aperture_data.af.ph_correct.shape[1] += SAR_aperture_data_temp.af.ph_correct.shape[1];
    }

    int shape_count = 0;

    // Shrink the data to requested values
    for (int idx = 0; idx < nPulse && idx < SAR_aperture_data.Ant_x.shape[1]; idx += pulseSkip) {
        SAR_aperture_data_temp2.Ant_x.data.push_back(SAR_aperture_data.Ant_x.data[idx]);
        SAR_aperture_data_temp2.Ant_y.data.push_back(SAR_aperture_data.Ant_y.data[idx]);
        SAR_aperture_data_temp2.Ant_z.data.push_back(SAR_aperture_data.Ant_z.data[idx]);

        SAR_aperture_data_temp2.slant_range.data.push_back(SAR_aperture_data.slant_range.data[idx]);
        SAR_aperture_data_temp2.theta.data.push_back(SAR_aperture_data.theta.data[idx]);
        SAR_aperture_data_temp2.phi.data.push_back(SAR_aperture_data.phi.data[idx]);
        SAR_aperture_data_temp2.af.r_correct.data.push_back(SAR_aperture_data.af.r_correct.data[idx]);
        SAR_aperture_data_temp2.af.ph_correct.data.push_back(SAR_aperture_data.af.ph_correct.data[idx]);

        SAR_aperture_data_temp2.sampleData.data.insert(
            SAR_aperture_data_temp2.sampleData.data.end(), 
            &SAR_aperture_data.sampleData.data[idx*SAR_aperture_data.sampleData.shape[0]], 
            &SAR_aperture_data.sampleData.data[idx*SAR_aperture_data.sampleData.shape[0] + SAR_aperture_data.sampleData.shape[0]]
        );
        shape_count += 1;
    }

    SAR_aperture_data.sampleData.data = SAR_aperture_data_temp2.sampleData.data;
    SAR_aperture_data.Ant_x.data = SAR_aperture_data_temp2.Ant_x.data;
    SAR_aperture_data.Ant_y.data = SAR_aperture_data_temp2.Ant_y.data;
    SAR_aperture_data.Ant_z.data = SAR_aperture_data_temp2.Ant_z.data;
    SAR_aperture_data.slant_range.data = SAR_aperture_data_temp2.slant_range.data;
    SAR_aperture_data.theta.data = SAR_aperture_data_temp2.theta.data;
    SAR_aperture_data.phi.data = SAR_aperture_data_temp2.phi.data;
    SAR_aperture_data.af.r_correct.data = SAR_aperture_data_temp2.af.r_correct.data;
    SAR_aperture_data.af.ph_correct.data = SAR_aperture_data_temp2.af.ph_correct.data;

    SAR_aperture_data.sampleData.shape[1] = shape_count;
    SAR_aperture_data.Ant_x.shape[1] = shape_count;
    SAR_aperture_data.Ant_y.shape[1] = shape_count;
    SAR_aperture_data.Ant_z.shape[1] = shape_count;
    SAR_aperture_data.slant_range.shape[1] = shape_count;
    SAR_aperture_data.theta.shape[1] = shape_count;
    SAR_aperture_data.phi.shape[1] = shape_count;
    SAR_aperture_data.af.r_correct.shape[1] = shape_count;
    SAR_aperture_data.af.ph_correct.shape[1] = shape_count;

    // Print out raw data imported from file
    std::cout << SAR_aperture_data << std::endl;

    // Sandia SAR data is multi-channel having up to 4 polarities
    // 1 = HH, 2 = HV, 3 = VH, 4 = VVbandwidth = 0:freq_per_sample:(numRangeSamples-1)*freq_per_sample;
    std::string polarity = result["polarity"].as<std::string>();
    if ((polarity == "HH" || polarity == "any") && SAR_aperture_data.sampleData.shape.size() >= 1) {
        SAR_aperture_data.polarity_channel = 0;
    } else if (polarity == "HV" && SAR_aperture_data.sampleData.shape.size() >= 2) {
        SAR_aperture_data.polarity_channel = 1;
    } else if (polarity == "VH" && SAR_aperture_data.sampleData.shape.size() >= 3) {
        SAR_aperture_data.polarity_channel = 2;
    } else if (polarity == "VV" && SAR_aperture_data.sampleData.shape.size() >= 4) {
        SAR_aperture_data.polarity_channel = 3;
    } else {
        std::cout << "Requested polarity channel " << polarity << " is not available." << std::endl;
        return EXIT_FAILURE;
    }
    if (SAR_aperture_data.sampleData.shape.size() > 2) {
        SAR_aperture_data.format_GOTCHA = false;
        // the dimensional index of the polarity index in the 
        // multi-dimensional array (for Sandia SPH SAR data)
        SAR_aperture_data.polarity_dimension = 2;
    }

    initialize_SAR_Aperture_Data(SAR_aperture_data);

    SAR_ImageFormationParameters<NumericType> SAR_image_params =
            SAR_ImageFormationParameters<NumericType>();

    // to increase the frequency samples to a power of 2
    // SAR_image_params.N_fft = (int) 0x01 << (int) (ceil(log2(SAR_aperture_data.numRangeSamples)));
    SAR_image_params.N_fft = (int) SAR_aperture_data.numRangeSamples;
    //SAR_image_params.N_fft = aperture.numRangeSamples;
    SAR_image_params.N_x_pix = (int) SAR_aperture_data.numAzimuthSamples;
    //SAR_image_params.N_y_pix = image_params.N_fft;
    SAR_image_params.N_y_pix = (int) SAR_aperture_data.numRangeSamples;
    // focus image on target phase center
    // Determine the maximum scene size of the image (m)
    // max down-range/fast-time/y-axis extent of image (m)
    SAR_image_params.max_Wy_m = CLIGHT / (2.0 * SAR_aperture_data.mean_deltaF);
    // max cross-range/fast-time/x-axis extent of image (m)
    SAR_image_params.max_Wx_m =
            CLIGHT / (2.0 * std::abs(SAR_aperture_data.mean_Ant_deltaAz) * SAR_aperture_data.mean_startF);

    // default view is 100% of the maximum possible view
    SAR_image_params.Wx_m = 1.00 * SAR_image_params.max_Wx_m;
    SAR_image_params.Wy_m = 1.00 * SAR_image_params.max_Wy_m;
    // make reconstructed image equal size in (x,y) dimensions
    SAR_image_params.N_x_pix = (int) ((float) SAR_image_params.Wx_m * SAR_image_params.N_y_pix) / SAR_image_params.Wy_m;
    // Determine the resolution of the image (m)
    SAR_image_params.slant_rangeResolution = CLIGHT / (2.0 * SAR_aperture_data.mean_bandwidth);
    SAR_image_params.ground_rangeResolution =
            SAR_image_params.slant_rangeResolution / std::sin(SAR_aperture_data.mean_Ant_El);
    SAR_image_params.azimuthResolution = CLIGHT / (2.0 * SAR_aperture_data.Ant_totalAz * SAR_aperture_data.mean_startF);

    // Print out data after critical data fields for SAR focusing have been computed
    std::cout << SAR_aperture_data << std::endl;

    SAR_Aperture<NumericType> SAR_focusing_data;
    if (!SAR_aperture_data.format_GOTCHA) {
        //SAR_aperture_data.exportData(SAR_focusing_data, SAR_aperture_data.polarity_channel);
        SAR_aperture_data.exportData(SAR_focusing_data, 2);
    } else {
        SAR_focusing_data = SAR_aperture_data;
    }

    //    SAR_ImageFormationParameters<NumericType> SAR_image_params =
    //            SAR_ImageFormationParameters<NumericType>::create<NumericType>(SAR_focusing_data);

    if (nPulse > 2 && nPulse < SAR_focusing_data.numAzimuthSamples) {
        SAR_focusing_data.numAzimuthSamples = nPulse;
    }

    std::cout << "Data for focusing" << std::endl;
    std::cout << SAR_focusing_data << std::endl;

    std::ofstream myfile;
    myfile.open("collectedData.txt", std::ios::out | std::ios::app);
    myfile << inputfile.c_str() << ',';

    printf("Main: deltaAz = %f, deltaF = %f, mean_startF = %f\nmaxWx_m = %f, maxWy_m = %f, Wx_m = %f, Wy_m = %f\nX_pix = %d, Y_pix = %d\nNum Az = %d, Num range = %d\n",
           SAR_aperture_data.mean_Ant_deltaAz, SAR_aperture_data.mean_startF, SAR_aperture_data.mean_deltaF,
           SAR_image_params.max_Wx_m, SAR_image_params.max_Wy_m, SAR_image_params.Wx_m, SAR_image_params.Wy_m,
           SAR_image_params.N_x_pix, SAR_image_params.N_y_pix, SAR_aperture_data.numAzimuthSamples,
           SAR_aperture_data.numRangeSamples);
    ComplexArrayType output_image(SAR_image_params.N_y_pix * SAR_image_params.N_x_pix);

    if (multiRes < 1) multiRes = 1;
    std::string output_filename = result["output"].as<std::string>();
    grid_cuda_focus_SAR_image(SAR_focusing_data, SAR_image_params, output_image, &myfile, multiRes, style, 1, output_filename);

    // Required parameters for output generation manually overridden by command line arguments
    // std::string output_filename = result["output"].as<std::string>();
    SAR_image_params.dyn_range_dB = result["dynrange"].as<float>();

    writeBMPFile(SAR_image_params, output_image, output_filename);
    myfile << '\n';
    myfile.close();
    return EXIT_SUCCESS;
}
