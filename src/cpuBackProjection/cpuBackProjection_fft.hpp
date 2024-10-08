/*
 * Copyright (C) 2021 Andrew R. Willis
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

#ifndef CPUBACKPROJECTION_FFT_HPP
#define CPUBACKPROJECTION_FFT_HPP

#include <fftw3.h>

#include "cpuBackProjection.hpp"

// See cpuBackProjection_fft.cpp for specializations based on numeric type
template<typename __nTp>
void fftw_engine(CArray<__nTp>& x, int DIR) {
    std::cout << "A specialization for this type is not available in FFTW!" << std::endl;
}

template<>
inline void fftw_engine(CArray<float>& x, int DIR) {
    // http://www.fftw.org/fftw3_doc/Complex-numbers.html
    // Structure must be only two numbers in the order real, imag
    // to be binary compatible with the C99 complex type
    //
    // TODO: Mangle function invocations and linking based on float/double selection
    // all calls change to fftw_plan for type double complex numbers
    // and the program must then link against fftw3 not fftw3f
    fftwf_plan p = fftwf_plan_dft_1d(x.size(),
            reinterpret_cast<fftwf_complex *> (&x[0]),
            reinterpret_cast<fftwf_complex *> (&x[0]),
            DIR, FFTW_ESTIMATE);
    fftwf_execute(p);
    fftwf_destroy_plan(p);
    if (DIR == FFTW_BACKWARD) {
        x /= x.size();
    }
}

template<>
void fftw_engine(CArray<double>& x, int DIR) {
    // http://www.fftw.org/fftw3_doc/Complex-numbers.html
    // Structure must be only two numbers in the order real, imag
    // to be binary compatible with the C99 complex type
    //
    // TODO: Mangle function invocations and linking based on float/double selection
    // all calls change to fftw_plan for type double complex numbers
    // and the program must then link against fftw3 not fftw3f
    fftw_plan p = fftw_plan_dft_1d(x.size(),
            reinterpret_cast<fftw_complex *> (&x[0]),
            reinterpret_cast<fftw_complex *> (&x[0]),
            DIR, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);
    if (DIR == FFTW_BACKWARD) {
        x /= x.size();
    }
}

template<typename __nTp>
void fftw(CArray<__nTp>& x) {
    fftw_engine(x, FFTW_FORWARD);
}

template<typename __nTp>
void ifftw(CArray<__nTp>& x) {
    fftw_engine(x, FFTW_BACKWARD);
}

template<typename __nTp>
CArray<__nTp> fftshift(CArray<__nTp>& fft) {
    return fft.cshift((fft.size() + 1) / 2);
}

// Cooley-Tukey FFT (in-place, breadth-first, decimation-in-frequency)
// Better optimized but less intuitive
// !!! Warning : in some cases this code make result different from not optimized version above (need to fix bug)
// The bug is now fixed @2017/05/30 

template<typename __nTp>
void fft(CArray<__nTp> &x) {
    // DFT
    unsigned int N = x.size(), k = N, n;
    double thetaT = 3.14159265358979323846264338328L / N;
    Complex<__nTp> phiT = Complex<__nTp>(cos(thetaT), -sin(thetaT)), T;
    while (k > 1) {
        n = k;
        k >>= 1;
        phiT = phiT * phiT;
        T = 1.0L;
        for (unsigned int l = 0; l < k; l++) {
            for (unsigned int a = l; a < N; a += n) {
                unsigned int b = a + k;
                Complex<__nTp> t = x[a] - x[b];
                x[a] += x[b];
                x[b] = t * T;
            }
            T *= phiT;
        }
    }
    // Decimate
    unsigned int m = (unsigned int) log2(N);
    for (unsigned int a = 0; a < N; a++) {
        unsigned int b = a;
        // Reverse bits
        b = (((b & 0xaaaaaaaa) >> 1) | ((b & 0x55555555) << 1));
        b = (((b & 0xcccccccc) >> 2) | ((b & 0x33333333) << 2));
        b = (((b & 0xf0f0f0f0) >> 4) | ((b & 0x0f0f0f0f) << 4));
        b = (((b & 0xff00ff00) >> 8) | ((b & 0x00ff00ff) << 8));
        b = ((b >> 16) | (b << 16)) >> (32 - m);
        if (b > a) {
            Complex<__nTp> t = x[a];
            x[a] = x[b];
            x[b] = t;
        }
    }
    //// Normalize (This section make it not working correctly)
    //Complex f = 1.0 / sqrt(N);
    //for (unsigned int i = 0; i < N; i++)
    //	x[i] *= f;
}

template<typename __nTp>
void ifft(CArray<__nTp>& x) {
    // conjugate the complex numbers
    x = x.apply(Complex<__nTp>::conj);

    // forward fft
    fft(x);

    // conjugate the complex numbers again
    x = x.apply(Complex<__nTp>::conj);

    // scale the numbers
    x /= x.size();
}

// Cooley–Tukey FFT (in-place, divide-and-conquer)
// Higher memory requirements and redundancy although more intuitive

template<typename __nTp>
void fft_alt(CArray<__nTp>& x) {
    const size_t N = x.size();
    if (N <= 1) return;

    // divide
    CArray<__nTp> even = x[std::slice(0, N / 2, 2)];
    CArray<__nTp> odd = x[std::slice(1, N / 2, 2)];

    // conquer
    fft_alt(even);
    fft_alt(odd);

    // combine
    for (size_t k = 0; k < N / 2; ++k) {
        //Complex t = Complex::polar(1.0f, -2.0f * PI * k / N) * odd[k];
        Complex<__nTp> t = Complex<__nTp>::polar(1.0f, -2.0f * PI * k / N) * odd[k];
        x[k ] = even[k] + t;
        x[k + N / 2] = even[k] - t;
    }
}

#endif /* CPUBACKPROJECTION_FFT_HPP */

