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

/* 
 * File:   charlotte_sar_api.hpp
 * Author: local-arwillis
 *
 * Created on April 7, 2022, 10:53 AM
 */

#ifndef EXTERNAL_API_HPP
#define EXTERNAL_API_HPP

#include <vector>
#include <string>
#include <third_party/log.h>

extern std::string outputfile;
extern bool verbose;

template<typename __nTp>
class PhaseHistory {
public:
    int id;
    __nTp sph_MATData_preamble_ADF;
    std::vector<__nTp> sph_MATData_Data_SampleData;
    int numRangeSamples;
    std::vector<__nTp> sph_MATData_Data_StartF;
    std::vector<__nTp> sph_MATData_Data_ChirpRate;
    std::vector<__nTp> sph_MATData_Data_radarCoordinateFrame_x;
    std::vector<__nTp> sph_MATData_Data_radarCoordinateFrame_y;
    std::vector<__nTp> sph_MATData_Data_radarCoordinateFrame_z;
    int numAzimuthSamples;
};

extern int
readData(const std::string &inputfile, const int MAX_PULSES,
         const std::string &polarity, PhaseHistory<float> &ph, bool verbose);

template<typename __nTp>
int sph_sar_data_callback_cpu(
        int id,
        __nTp sph_MATData_preamble_ADF,
        __nTp *sph_MATData_Data_SampleData,
        int numRangeSamples,
        __nTp *sph_MATData_Data_StartF,
        __nTp *sph_MATData_Data_ChirpRate,
        __nTp *sph_MATData_Data_radarCoordinateFrame_x,
        __nTp *sph_MATData_Data_radarCoordinateFrame_y,
        __nTp *sph_MATData_Data_radarCoordinateFrame_z,
        int numAzimuthSamples);

template<typename __nTp>
int sph_sar_data_callback_gpu(
        int id,
        __nTp sph_MATData_preamble_ADF,
        __nTp *sph_MATData_Data_SampleData,
        int numRangeSamples,
        __nTp *sph_MATData_Data_StartF,
        __nTp *sph_MATData_Data_ChirpRate,
        __nTp *sph_MATData_Data_radarCoordinateFrame_x,
        __nTp *sph_MATData_Data_radarCoordinateFrame_y,
        __nTp *sph_MATData_Data_radarCoordinateFrame_z,
        int numAzimuthSamples);

template<typename __nTp>
int sar_data_callback(
        int id,
        __nTp sph_MATData_preamble_ADF,
        __nTp *sph_MATData_Data_SampleData,
        int numRangeSamples,
        __nTp *sph_MATData_Data_StartF,
        __nTp *sph_MATData_Data_ChirpRate,
        __nTp *sph_MATData_Data_radarCoordinateFrame_x,
        __nTp *sph_MATData_Data_radarCoordinateFrame_y,
        __nTp *sph_MATData_Data_radarCoordinateFrame_z,
        int numAzimuthSamples) {

    sph_sar_data_callback_cpu<__nTp>(
            id,
            sph_MATData_preamble_ADF,
            sph_MATData_Data_SampleData,
            numRangeSamples,
            sph_MATData_Data_StartF,
            sph_MATData_Data_ChirpRate,
            sph_MATData_Data_radarCoordinateFrame_x,
            sph_MATData_Data_radarCoordinateFrame_y,
            sph_MATData_Data_radarCoordinateFrame_z,
            numAzimuthSamples);

    sph_sar_data_callback_gpu<__nTp>(
            id,
            sph_MATData_preamble_ADF,
            sph_MATData_Data_SampleData,
            numRangeSamples,
            sph_MATData_Data_StartF,
            sph_MATData_Data_ChirpRate,
            sph_MATData_Data_radarCoordinateFrame_x,
            sph_MATData_Data_radarCoordinateFrame_y,
            sph_MATData_Data_radarCoordinateFrame_z,
            numAzimuthSamples);

    return EXIT_SUCCESS;
}

#include <uncc_sar_focusing.hpp>

using NumericType = float;
using ComplexType = Complex<NumericType>;
using ComplexArrayType = CArray<NumericType>;

template<typename __nTp, typename __pTp>
int writeBMPFile(const SAR_ImageFormationParameters<__pTp> &SARImgParams,
                 const CArray<__nTp> &output_image, const std::string &output_filename);

template<typename __nTp>
int create_SARAperture(__nTp sph_MATData_preamble_ADF,
                       __nTp *sph_MATData_Data_SampleData,
                       int numRangeSamples,
                       __nTp *sph_MATData_Data_StartF,
                       __nTp *sph_MATData_Data_ChirpRate,
                       __nTp *sph_MATData_Data_radarCoordinateFrame_x,
                       __nTp *sph_MATData_Data_radarCoordinateFrame_y,
                       __nTp *sph_MATData_Data_radarCoordinateFrame_z,
                       int numAzimuthSamples,
                       SAR_Aperture<__nTp> &SAR_focusing_data) {

    SAR_focusing_data.ADF.shape.push_back(1);
    SAR_focusing_data.ADF.data.push_back(sph_MATData_preamble_ADF);

    int numPulseVec[1] = {numAzimuthSamples};
    int phaseHistoryDims[2] = {numRangeSamples, numAzimuthSamples};
    import_MatrixComplex<__nTp, Complex<__nTp> >(sph_MATData_Data_SampleData, phaseHistoryDims, 2,
                                                 SAR_focusing_data.sampleData);
    import_Vector<__nTp, __nTp>(sph_MATData_Data_StartF, numPulseVec, 1, SAR_focusing_data.startF);
    import_Vector<__nTp, __nTp>(sph_MATData_Data_ChirpRate, numPulseVec, 1, SAR_focusing_data.chirpRate);
    import_Vector<__nTp, __nTp>(sph_MATData_Data_radarCoordinateFrame_x, numPulseVec, 1, SAR_focusing_data.Ant_x);
    import_Vector<__nTp, __nTp>(sph_MATData_Data_radarCoordinateFrame_y, numPulseVec, 1, SAR_focusing_data.Ant_y);
    import_Vector<__nTp, __nTp>(sph_MATData_Data_radarCoordinateFrame_z, numPulseVec, 1, SAR_focusing_data.Ant_z);

    // convert chirp rate to deltaF
    //    std::vector<__nTp> deltaF(numAzimuthSamples);
    //    for (int i = 0; i < numAzimuthSamples; i++) {
    //        deltaF[i] = sph_MATData_Data_ChirpRate[i] / sph_MATData_preamble_ADF;
    //    }
    //    import_Vector<__nTp, __nTp>(sph_MATData_Data_StartF, numPulseVec, 1, SAR_focusing_data.startF);

    SAR_focusing_data.format_GOTCHA = false;
    initialize_SAR_Aperture_Data(SAR_focusing_data);

    return EXIT_SUCCESS;
}

template<typename __nTp>
int create_SARImageFormationParams(const SAR_Aperture<__nTp> &SAR_focusing_data,
                                   SAR_ImageFormationParameters<__nTp> &SAR_image_params) {
    // to increase the frequency samples to a power of 2
    //SAR_image_params.N_fft = (int) 0x01 << (int) (ceil(log2(SAR_aperture_data.numRangeSamples)));
    SAR_image_params.N_fft = SAR_focusing_data.numRangeSamples;
    SAR_image_params.N_x_pix = SAR_focusing_data.numAzimuthSamples;
    //SAR_image_params.N_y_pix = image_params.N_fft;
    SAR_image_params.N_y_pix = SAR_focusing_data.numRangeSamples;
    // focus image on target phase center
    // Determine the maximum scene size of the image (m)
    // max down-range/fast-time/y-axis extent of image (m)
    SAR_image_params.max_Wy_m = CLIGHT / (2.0 * SAR_focusing_data.mean_deltaF);
    // max cross-range/fast-time/x-axis extent of image (m)
    SAR_image_params.max_Wx_m =
            CLIGHT / (2.0 * std::abs(SAR_focusing_data.mean_Ant_deltaAz) * SAR_focusing_data.mean_startF);

    // default view is 100% of the maximum possible view
    SAR_image_params.Wx_m = 1.00 * SAR_image_params.max_Wx_m;
    SAR_image_params.Wy_m = 1.00 * SAR_image_params.max_Wy_m;
    // make reconstructed image equal size in (x,y) dimensions
    SAR_image_params.N_x_pix = (int) ((float) SAR_image_params.Wx_m * SAR_image_params.N_y_pix) / SAR_image_params.Wy_m;
    // Determine the resolution of the image (m)
    SAR_image_params.slant_rangeResolution = CLIGHT / (2.0 * SAR_focusing_data.mean_bandwidth);
    SAR_image_params.ground_rangeResolution =
            SAR_image_params.slant_rangeResolution / std::sin(SAR_focusing_data.mean_Ant_El);
    SAR_image_params.azimuthResolution = CLIGHT / (2.0 * SAR_focusing_data.Ant_totalAz * SAR_focusing_data.mean_startF);

    return EXIT_SUCCESS;
}

template<typename __nTp>
int sph_sar_data_callback_cpu(
        int id,
        __nTp sph_MATData_preamble_ADF,
        __nTp *sph_MATData_Data_SampleData,
        int numRangeSamples,
        __nTp *sph_MATData_Data_StartF,
        __nTp *sph_MATData_Data_ChirpRate,
        __nTp *sph_MATData_Data_radarCoordinateFrame_x,
        __nTp *sph_MATData_Data_radarCoordinateFrame_y,
        __nTp *sph_MATData_Data_radarCoordinateFrame_z,
        int numAzimuthSamples) {

    L_(linfo) << "Received data with Id = " << id;
    SAR_Aperture<__nTp> SAR_focusing_data;

    create_SARAperture(sph_MATData_preamble_ADF,
                       sph_MATData_Data_SampleData,
                       numRangeSamples,
                       sph_MATData_Data_StartF,
                       sph_MATData_Data_ChirpRate,
                       sph_MATData_Data_radarCoordinateFrame_x,
                       sph_MATData_Data_radarCoordinateFrame_y,
                       sph_MATData_Data_radarCoordinateFrame_z,
                       numAzimuthSamples,
                       SAR_focusing_data);

    initialize_SAR_Aperture_Data(SAR_focusing_data);
    if (verbose) {
        std::cout << "Data for focusing:" << std::endl;
        std::cout << SAR_focusing_data << std::endl;
    }
    L_(linfo) << "Data for focusing:" << SAR_focusing_data;

    SAR_ImageFormationParameters<__nTp> SAR_image_params; // =
    //SAR_ImageFormationParameters<__nTp>::create<__nTp>(SAR_focusing_data);

    create_SARImageFormationParams(SAR_focusing_data, SAR_image_params);

    if (verbose)
        std::cout << SAR_image_params << std::endl;
    L_(linfo) << SAR_image_params;

    ComplexArrayType output_image(SAR_image_params.N_y_pix * SAR_image_params.N_x_pix);

    //cuda_focus_SAR_image(SAR_focusing_data, SAR_image_params, output_image);
    focus_SAR_image(SAR_focusing_data, SAR_image_params, output_image);

    // Required parameters for output generation manually overridden by command line arguments
    //std::string output_filename = "output_cpu.bmp";
    SAR_image_params.dyn_range_dB = 70;

    writeBMPFile(SAR_image_params, output_image, outputfile);

    return EXIT_SUCCESS;
}

template<typename __nTp, typename __nTpParams>
void cuda_focus_SAR_image(const SAR_Aperture<__nTp> &sar_data,
                          const SAR_ImageFormationParameters<__nTpParams> &sar_image_params,
                          CArray<__nTp> &output_image);

template<typename __nTp>
int sph_sar_data_callback_gpu(
        int id,
        __nTp sph_MATData_preamble_ADF,
        __nTp *sph_MATData_Data_SampleData,
        int numRangeSamples,
        __nTp *sph_MATData_Data_StartF,
        __nTp *sph_MATData_Data_ChirpRate,
        __nTp *sph_MATData_Data_radarCoordinateFrame_x,
        __nTp *sph_MATData_Data_radarCoordinateFrame_y,
        __nTp *sph_MATData_Data_radarCoordinateFrame_z,
        int numAzimuthSamples) {

    L_(linfo) << "Received data with Id = " << id;
    SAR_Aperture<__nTp> SAR_focusing_data;

    create_SARAperture(sph_MATData_preamble_ADF,
                       sph_MATData_Data_SampleData,
                       numRangeSamples,
                       sph_MATData_Data_StartF,
                       sph_MATData_Data_ChirpRate,
                       sph_MATData_Data_radarCoordinateFrame_x,
                       sph_MATData_Data_radarCoordinateFrame_y,
                       sph_MATData_Data_radarCoordinateFrame_z,
                       numAzimuthSamples,
                       SAR_focusing_data);

    initialize_SAR_Aperture_Data(SAR_focusing_data);

    if (verbose) {
        std::cout << "Data for focusing:" << std::endl;
        std::cout << SAR_focusing_data << std::endl;
    }
    L_(linfo) << "Data for focusing:" << SAR_focusing_data;

    SAR_ImageFormationParameters<__nTp> SAR_image_params; // =
    //SAR_ImageFormationParameters<__nTp>::create<__nTp>(SAR_focusing_data);

    create_SARImageFormationParams(SAR_focusing_data, SAR_image_params);

    if (verbose)
        std::cout << SAR_image_params << std::endl;
    L_(linfo) << SAR_image_params;

    ComplexArrayType output_image(SAR_image_params.N_y_pix * SAR_image_params.N_x_pix);

    cuda_focus_SAR_image(SAR_focusing_data, SAR_image_params, output_image);

    // Required parameters for output generation manually overridden by command line arguments
    // std::string output_filename = "output_gpu.bmp";
    SAR_image_params.dyn_range_dB = 70;

    writeBMPFile(SAR_image_params, output_image, outputfile);

    return EXIT_SUCCESS;
}

#endif /* EXTERNAL_API_HPP */

