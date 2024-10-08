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
 * File:   charlotte_sar_api.cpp
 * Author: local-arwillis
 *
 * Created on April 23, 2022, 10:16 AM
 */

#define CUDAFUNCTION

#include <charlotte_sar_api.hpp>

#include <uncc_sar_focusing.hpp>
#include <uncc_sar_matio.hpp>

using NumericType = float;
using ComplexType = Complex<NumericType>;
//using ComplexArrayType = CArray<NumericType>;

int readData(const std::string &inputfile, const int MAX_PULSES,
             const std::string &polarity, PhaseHistory<float> &ph, bool verbose) {

    std::unordered_map<std::string, matvar_t *> matlab_readvar_map;

    initialize_Sandia_SPHRead(matlab_readvar_map);
    initialize_GOTCHA_MATRead(matlab_readvar_map);

    SAR_Aperture<NumericType> SAR_aperture_data;
    if (read_MAT_Variables(inputfile, matlab_readvar_map, SAR_aperture_data) == EXIT_FAILURE) {
        std::cout << "Could not read all desired MATLAB variables from " << inputfile << " exiting." << std::endl;
        return EXIT_FAILURE;
    }
    // Print out raw data imported from file
    if (verbose)
        std::cout << SAR_aperture_data << std::endl;

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
    }
    if (!SAR_aperture_data.format_GOTCHA) {
        // the dimensional index of the polarity index in the 
        // multi-dimensional array (for Sandia SPH SAR data)
        SAR_aperture_data.polarity_dimension = 2;
    }
    // Print out data after critical data fields for SAR focusing have been computed
    initialize_SAR_Aperture_Data(SAR_aperture_data);

    //std::cout << "SAR aperture data = " << SAR_aperture_data << std::endl;

    SAR_Aperture<NumericType> SAR_focusing_data;
    if (!SAR_aperture_data.format_GOTCHA) {
        //SAR_aperture_data.exportData(SAR_focusing_data, SAR_aperture_data.polarity_channel);
        SAR_aperture_data.exportData(SAR_focusing_data, 2);
    } else {
        SAR_focusing_data = SAR_aperture_data;
    }

    int numPulses = (MAX_PULSES > SAR_focusing_data.numAzimuthSamples) ? SAR_focusing_data.numAzimuthSamples
                                                                       : MAX_PULSES;
    int firstPulseIndex = (SAR_focusing_data.numAzimuthSamples - numPulses) / 2;
    int lastPulseIndex = (SAR_focusing_data.numAzimuthSamples + numPulses - 1) / 2 +
                         ((SAR_focusing_data.numAzimuthSamples + numPulses) % 2);
    std::vector<int> pulseIndices(numPulses);
    for (std::vector<int>::iterator pIter = pulseIndices.begin(); pIter != pulseIndices.end(); ++pIter) {
        *pIter = firstPulseIndex++;
    }

    if (verbose)
        std::cout << "SAR focusing data = " << SAR_focusing_data << std::endl;
    // Allocate memory for data to pass to library function call
    ph.sph_MATData_Data_SampleData.resize(2 * SAR_focusing_data.numRangeSamples * numPulses);
    ph.sph_MATData_Data_StartF.resize(numPulses);
    ph.sph_MATData_Data_ChirpRate.resize(numPulses);
    ph.sph_MATData_Data_radarCoordinateFrame_x.resize(numPulses);
    ph.sph_MATData_Data_radarCoordinateFrame_y.resize(numPulses);
    ph.sph_MATData_Data_radarCoordinateFrame_z.resize(numPulses);

    //std::vector<std::vector<int>> adf_index {{0}};
    //sph_MATData_preamble_ADF = (NumericType) SAR_aperture_data.ADF.getData(adf_index).at(0);
    std::vector<NumericType> vec_ADF = SAR_aperture_data.ADF.toVector<NumericType>();
    ph.sph_MATData_preamble_ADF = vec_ADF[0];

    std::vector<ComplexType> vec_SampleData = SAR_focusing_data.sampleData.toVector<ComplexType>();

    int pulseTgtOffset = 0;
    for (std::vector<int>::iterator pIter = pulseIndices.begin(); pIter != pulseIndices.end(); ++pIter) {
        //int pulseTgtIndex = 0;
        int pulseSrcIndex = *pIter;
        int pulseSrcOffset = pulseSrcIndex * SAR_focusing_data.numRangeSamples;
        for (int frequencyIndex = 0; frequencyIndex < SAR_focusing_data.numRangeSamples; frequencyIndex++) {
            //std::cout << *cmplx_val << std::endl;
            ph.sph_MATData_Data_SampleData[pulseTgtOffset + frequencyIndex * 2 + 0] = vec_SampleData[pulseSrcOffset +
                                                                                                     frequencyIndex]._M_real;
            ph.sph_MATData_Data_SampleData[pulseTgtOffset + frequencyIndex * 2 + 1] = vec_SampleData[pulseSrcOffset +
                                                                                                     frequencyIndex]._M_imag;
            //pulseTgtIndex++;
        }
        pulseTgtOffset += SAR_focusing_data.numRangeSamples * 2;
    }

    std::vector<NumericType> vec_StartF = SAR_focusing_data.startF.toVector<NumericType>();
    std::vector<NumericType> vec_ChirpRate = SAR_focusing_data.chirpRate.toVector<NumericType>();
    std::vector<NumericType> vec_radarCoordinateFrame_x = SAR_focusing_data.Ant_x.toVector<NumericType>();
    std::vector<NumericType> vec_radarCoordinateFrame_y = SAR_focusing_data.Ant_y.toVector<NumericType>();
    std::vector<NumericType> vec_radarCoordinateFrame_z = SAR_focusing_data.Ant_z.toVector<NumericType>();

    int pulseTgtIndex = 0;
    for (std::vector<int>::iterator pIter = pulseIndices.begin(); pIter != pulseIndices.end(); ++pIter) {
        int pulseSrcIndex = *pIter;
        ph.sph_MATData_Data_StartF[pulseTgtIndex] = vec_StartF[pulseSrcIndex];
        ph.sph_MATData_Data_ChirpRate[pulseTgtIndex] = vec_ChirpRate[pulseSrcIndex];
        ph.sph_MATData_Data_radarCoordinateFrame_x[pulseTgtIndex] = vec_radarCoordinateFrame_x[pulseSrcIndex];
        ph.sph_MATData_Data_radarCoordinateFrame_y[pulseTgtIndex] = vec_radarCoordinateFrame_y[pulseSrcIndex];
        ph.sph_MATData_Data_radarCoordinateFrame_z[pulseTgtIndex] = vec_radarCoordinateFrame_z[pulseSrcIndex];
        pulseTgtIndex++;
    }
    ph.numRangeSamples = SAR_focusing_data.numRangeSamples;
    ph.numAzimuthSamples = numPulses;
    return EXIT_SUCCESS;
}
