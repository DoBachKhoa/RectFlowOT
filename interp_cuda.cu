/*********************************************
 * interp_cuda.cu
 *
 * A CUDA translation of the CPU code that:
 *   - Computes summed-area tables (SATs) for an image,
 *   - Defines device-side velocity and RK4 routines,
 *   - Launches a kernel that advects many particles in parallel.
 *
 * To compile:
 *   nvcc -O3 -o interp_cuda interp_cuda.cu
 *
 * Requirements: 
 *   - A CUDA-capable GPU.
 *   - The CUDA toolkit (nvcc).
 *   - The image file "lionOrigami.bmp" and particle file "BNOT_uniformpts8k.txt"
 *     in the same folder.
 *   - stb_image.h and flow_cuda.h must be available in the same folder.
 **********************************************/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "flow_cuda.h"
#include <chrono>
#include <vector>
#include <iostream>

// STB Image is used to load the image on the host.
// Download stb_image.h from https://github.com/nothings/stb (or include it in your project).
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h" 


void test_interp2D(const char* filename_image1, const char* filename_image2, const int N_interp, const char* result_filename) {
     // --- Load the images ---
    int w, h, n;
    unsigned char* data1 = stbi_load(filename_image1, &w, &h, &n, 1);
    unsigned char* data2 = stbi_load(filename_image2, &w, &h, &n, 1);
    if (!data1 || !data2) {
        fprintf(stderr, "Error loading one of the images!\n");
        return;
    }
    int num_pixels = w * h;
    std::vector<double> host_input1(num_pixels), host_input2(num_pixels);
    for (int i = 0; i < num_pixels; i++) {
        host_input1[i] = 255.0 - (double)data1[i];
        host_input2[i] = 255.0 - (double)data2[i];
    }
    stbi_image_free(data1);
    stbi_image_free(data2);

    // --- Load the particle positions ---
    int N = 65536;  // 65536 particles
    std::vector<double> pts(N * 2);
    FILE* fpts = fopen("PeriodicBNOT_uniformpts64k.txt", "r");
    if (!fpts) {
        fprintf(stderr, "Error opening particle file!\n");
        return;
    }
    for (int i = 0; i < N; i++) {
        fscanf(fpts, "%lf %lf\n", &pts[2 * i], &pts[2 * i + 1]);
        pts[2 * i]     *= w;
        pts[2 * i + 1] *= h;
    }
    fclose(fpts);
    // Duplicate particle arrays (one for each flow)
    std::vector<double> pts1 = pts;
    std::vector<double> pts2 = pts;

    // --- Set up device parameters ---
    int padded_width  = w + 1;
    int padded_height = h + 1;
    int padded_size   = padded_width * padded_height;

    // --- Create two CUDA streams for concurrent flow processing ---
    cudaStream_t stream1, stream2;
    cudaCheckError(cudaStreamCreate(&stream1));
    cudaCheckError(cudaStreamCreate(&stream2));

    // --- Allocate separate device buffers for each flow ---
    // Flow 1 buffers:
    double* d_input1;
    cudaCheckError(cudaMalloc((void**)&d_input1, sizeof(double) * num_pixels));
    double *d_denom1, *d_nom_x1, *d_nom_y1;
    cudaCheckError(cudaMalloc((void**)&d_denom1, sizeof(double) * padded_size));
    cudaCheckError(cudaMalloc((void**)&d_nom_x1, sizeof(double) * padded_size));
    cudaCheckError(cudaMalloc((void**)&d_nom_y1, sizeof(double) * padded_size));
    double* d_particles1;
    cudaCheckError(cudaMalloc((void**)&d_particles1, sizeof(double) * N * 2));
    FlowCuda* d_flow1;
    cudaCheckError(cudaMalloc((void**)&d_flow1, sizeof(FlowCuda)));

    // Flow 2 buffers:
    double* d_input2;
    cudaCheckError(cudaMalloc((void**)&d_input2, sizeof(double) * num_pixels));
    double *d_denom2, *d_nom_x2, *d_nom_y2;
    cudaCheckError(cudaMalloc((void**)&d_denom2, sizeof(double) * padded_size));
    cudaCheckError(cudaMalloc((void**)&d_nom_x2, sizeof(double) * padded_size));
    cudaCheckError(cudaMalloc((void**)&d_nom_y2, sizeof(double) * padded_size));
    double* d_particles2;
    cudaCheckError(cudaMalloc((void**)&d_particles2, sizeof(double) * N * 2));
    FlowCuda* d_flow2;
    cudaCheckError(cudaMalloc((void**)&d_flow2, sizeof(FlowCuda)));

    // --- Define grid/block configuration ---
    dim3 blockSize(16, 16);
    dim3 gridSize((w + blockSize.x - 1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y);
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    int num_steps = 50;

    // --- Begin timer ---
    auto start_time = std::chrono::steady_clock::now();

    // --- Process Flow 1 in stream1 ---
    cudaCheckError(cudaMemcpyAsync(d_input1, host_input1.data(), sizeof(double)*num_pixels, cudaMemcpyHostToDevice, stream1));
    cudaCheckError(cudaMemsetAsync(d_denom1, 0, sizeof(double)*padded_size, stream1));
    cudaCheckError(cudaMemsetAsync(d_nom_x1, 0, sizeof(double)*padded_size, stream1));
    cudaCheckError(cudaMemsetAsync(d_nom_y1, 0, sizeof(double)*padded_size, stream1));
    fillSAT<<<gridSize, blockSize, 0, stream1>>>(d_input1, d_denom1, d_nom_x1, d_nom_y1, w, h, padded_width);
    horizontal_scan<<<padded_height, 1, 0, stream1>>>(d_denom1, padded_width, padded_height);
    horizontal_scan<<<padded_height, 1, 0, stream1>>>(d_nom_x1, padded_width, padded_height);
    horizontal_scan<<<padded_height, 1, 0, stream1>>>(d_nom_y1, padded_width, padded_height);
    vertical_scan<<<padded_width, 1, 0, stream1>>>(d_denom1, padded_width, padded_height);
    vertical_scan<<<padded_width, 1, 0, stream1>>>(d_nom_x1, padded_width, padded_height);
    vertical_scan<<<padded_width, 1, 0, stream1>>>(d_nom_y1, padded_width, padded_height);

    double h_sum_denom1, h_sum_nom_x1, h_sum_nom_y1;
    cudaCheckError(cudaMemcpyAsync(&h_sum_denom1, d_denom1 + padded_size - 1, sizeof(double), cudaMemcpyDeviceToHost, stream1));
    cudaCheckError(cudaMemcpyAsync(&h_sum_nom_x1, d_nom_x1 + padded_size - 1, sizeof(double), cudaMemcpyDeviceToHost, stream1));
    cudaCheckError(cudaMemcpyAsync(&h_sum_nom_y1, d_nom_y1 + padded_size - 1, sizeof(double), cudaMemcpyDeviceToHost, stream1));
    cudaCheckError(cudaMemcpyAsync(d_particles1, pts1.data(), sizeof(double)*N*2, cudaMemcpyHostToDevice, stream1));
    FlowCuda h_flow1;
    h_flow1.width = w; h_flow1.height = h;
    h_flow1.padded_width = padded_width; h_flow1.padded_height = padded_height;
    h_flow1.input = d_input1; h_flow1.denom = d_denom1; h_flow1.nom_x = d_nom_x1; h_flow1.nom_y = d_nom_y1;
    h_flow1.sum_denom = h_sum_denom1; h_flow1.sum_nom_x = h_sum_nom_x1; h_flow1.sum_nom_y = h_sum_nom_y1;
    cudaCheckError(cudaMemcpyAsync(d_flow1, &h_flow1, sizeof(FlowCuda), cudaMemcpyHostToDevice, stream1));
    advect_particles_kernel<<<blocks, threadsPerBlock, 0, stream1>>>(d_flow1, d_particles1, N, num_steps);

    // --- Process Flow 2 in stream2 ---
    cudaCheckError(cudaMemcpyAsync(d_input2, host_input2.data(), sizeof(double)*num_pixels, cudaMemcpyHostToDevice, stream2));
    cudaCheckError(cudaMemsetAsync(d_denom2, 0, sizeof(double)*padded_size, stream2));
    cudaCheckError(cudaMemsetAsync(d_nom_x2, 0, sizeof(double)*padded_size, stream2));
    cudaCheckError(cudaMemsetAsync(d_nom_y2, 0, sizeof(double)*padded_size, stream2));
    fillSAT<<<gridSize, blockSize, 0, stream2>>>(d_input2, d_denom2, d_nom_x2, d_nom_y2, w, h, padded_width);
    horizontal_scan<<<padded_height, 1, 0, stream2>>>(d_denom2, padded_width, padded_height);
    horizontal_scan<<<padded_height, 1, 0, stream2>>>(d_nom_x2, padded_width, padded_height);
    horizontal_scan<<<padded_height, 1, 0, stream2>>>(d_nom_y2, padded_width, padded_height);
    vertical_scan<<<padded_width, 1, 0, stream2>>>(d_denom2, padded_width, padded_height);
    vertical_scan<<<padded_width, 1, 0, stream2>>>(d_nom_x2, padded_width, padded_height);
    vertical_scan<<<padded_width, 1, 0, stream2>>>(d_nom_y2, padded_width, padded_height);
    
    double h_sum_denom2, h_sum_nom_x2, h_sum_nom_y2;
    cudaCheckError(cudaMemcpyAsync(&h_sum_denom2, d_denom2 + padded_size - 1, sizeof(double), cudaMemcpyDeviceToHost, stream2));
    cudaCheckError(cudaMemcpyAsync(&h_sum_nom_x2, d_nom_x2 + padded_size - 1, sizeof(double), cudaMemcpyDeviceToHost, stream2));
    cudaCheckError(cudaMemcpyAsync(&h_sum_nom_y2, d_nom_y2 + padded_size - 1, sizeof(double), cudaMemcpyDeviceToHost, stream2));
    cudaCheckError(cudaMemcpyAsync(d_particles2, pts2.data(), sizeof(double)*N*2, cudaMemcpyHostToDevice, stream2));
    FlowCuda h_flow2;
    h_flow2.width = w; h_flow2.height = h;
    h_flow2.padded_width = padded_width; h_flow2.padded_height = padded_height;
    h_flow2.input = d_input2; h_flow2.denom = d_denom2; h_flow2.nom_x = d_nom_x2; h_flow2.nom_y = d_nom_y2;
    h_flow2.sum_denom = h_sum_denom2; h_flow2.sum_nom_x = h_sum_nom_x2; h_flow2.sum_nom_y = h_sum_nom_y2;
    cudaCheckError(cudaMemcpyAsync(d_flow2, &h_flow2, sizeof(FlowCuda), cudaMemcpyHostToDevice, stream2));
    advect_particles_kernel<<<blocks, threadsPerBlock, 0, stream2>>>(d_flow2, d_particles2, N, num_steps);

    // --- Wait for both streams to complete ---
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // --- Copy particle results back to host ---
    cudaCheckError(cudaMemcpy(pts1.data(), d_particles1, sizeof(double)*N*2, cudaMemcpyDeviceToHost));
    cudaCheckError(cudaMemcpy(pts2.data(), d_particles2, sizeof(double)*N*2, cudaMemcpyDeviceToHost));


    // --- Interpolation and image accumulation (as before) ---
    int W = w * N_interp, H = h;
    std::vector<unsigned char> result(W * H, 0);
	
    for (int v = 0; v < N_interp; v++) {
        double alpha = v / (N_interp-1.0);
        for (int i = 0; i < N; i++) {
            double x = pts1[2*i]     * (1.0 - alpha) + pts2[2*i]     * alpha;
            double y = pts1[2*i + 1] * (1.0 - alpha) + pts2[2*i + 1] * alpha;
            int ix = std::min(W - 1, std::max(0, v*w + (int)(x + 0.5)));
            int iy = std::min(H - 1, std::max(0, (int)(y + 0.5)));
            result[iy*W + ix]++;
        }
        unsigned char vmax = 0;
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                vmax = std::max(vmax, result[i*W + v*w + j]);
            }
        }
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                double val = result[i*W + v*w + j];
                result[i*W + v*w + j] = (unsigned char)(255 * (1.0 - val/(double)vmax));
            }
        }
    }
	
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;
    std::cout << "GPU integration elapsed time: " << elapsed_seconds.count() << " s\n";

	
    stbi_write_png(result_filename, W, H, 1, result.data(), W);
    std::cout << "Output image saved as "<<result_filename<<"\n";

    // --- Clean up ---
    cudaFree(d_input1); cudaFree(d_denom1); cudaFree(d_nom_x1); cudaFree(d_nom_y1);
    cudaFree(d_particles1); cudaFree(d_flow1);
    cudaFree(d_input2); cudaFree(d_denom2); cudaFree(d_nom_x2); cudaFree(d_nom_y2);
    cudaFree(d_particles2); cudaFree(d_flow2);
    cudaStreamDestroy(stream1); cudaStreamDestroy(stream2);
}



// ---------------------------------------------------------------------
// Main (host) function
int main()
{
	// both images should have the same size: 256x256 
	// for other resolutions, use other input point sets (not uniformPeriodicBNOT_64k), e.g., a regular grid
	const char* img1 = "caterpillar.png";
	const char* img2 = "butterfly_simple.png";
	const char* result = "out_interp.png";
	const int N_interp = 5;
    
	// produces 5 interpolations steps and save in a single image
	test_interp2D(img1, img2, N_interp, result);

    return 0;
}
