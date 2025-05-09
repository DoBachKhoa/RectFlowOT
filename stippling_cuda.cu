/*********************************************
 * flow_cuda.cu
 *
 * A CUDA translation of the CPU code that:
 *   - Computes summed-area tables (SATs) for an image,
 *   - Defines device-side velocity and RK4 routines,
 *   - Launches a kernel that advects many particles in parallel.
 *
 * To compile:
 *   nvcc -O3 -o stippling_cuda stippling_cuda.cu
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


// ---------------------------------------------------------------------
//  - Loads the image and particle data.
//  - Allocates device memory and computes the SAT.
//  - Launches the particle advection kernel.
//  - Writes the output positions to an SVG file.

void test_stippling(const char* image_filename, int N, const char* uniform_stipples_filename, const char* result_filename) {
   // Load image using stb_image (grayscale, one channel).
    int w, h, n;
    unsigned char* data = stbi_load(image_filename, &w, &h, &n, 1);
    if (!data) {
        fprintf(stderr, "Error loading image!\n");
        return;
    }
    int num_pixels = w * h;
    std::vector<double> host_input(num_pixels);
    for (int i = 0; i < num_pixels; i++) {
        host_input[i] = 255.0 - (double)data[i];
    }
    stbi_image_free(data);

    // Load particle data from file ("BNOT_uniformpts40k.txt").    
    std::vector<double> host_particles(N * 2);
    FILE* fpts = fopen(uniform_stipples_filename, "r");	
    if (!fpts) {
        fprintf(stderr, "Error opening particle file!\n");
        return;
    }
    for (int i = 0; i < N; i++) {
        fscanf(fpts, "%lf %lf\n", &host_particles[2 * i], &host_particles[2 * i + 1]);
        // Scale particle coordinates by image dimensions.
        host_particles[2 * i]     *= w;
        host_particles[2 * i + 1] *= h;
    }
    fclose(fpts);

    // Allocate device memory for the image input.
    double* d_input;
    cudaCheckError(cudaMalloc((void**)&d_input, sizeof(double) * num_pixels));
    cudaCheckError(cudaMemcpy(d_input, host_input.data(), sizeof(double) * num_pixels, cudaMemcpyHostToDevice));

    // Padded dimensions for SAT arrays:
    int padded_width  = w + 1;
    int padded_height = h + 1;
    int padded_size   = padded_width * padded_height;

    double *d_denom, *d_nom_x, *d_nom_y;
    cudaCheckError(cudaMalloc((void**)&d_denom, sizeof(double) * padded_size));
    cudaCheckError(cudaMalloc((void**)&d_nom_x, sizeof(double) * padded_size));
    cudaCheckError(cudaMalloc((void**)&d_nom_y, sizeof(double) * padded_size));

    // Initialize the SAT arrays to zero.
    cudaCheckError(cudaMemset(d_denom, 0, sizeof(double) * padded_size));
    cudaCheckError(cudaMemset(d_nom_x, 0, sizeof(double) * padded_size));
    cudaCheckError(cudaMemset(d_nom_y, 0, sizeof(double) * padded_size));

    auto start = std::chrono::steady_clock::now();

    // Launch the kernel to fill the inner (offset-by-one) region of SAT arrays.
    dim3 blockSize(16, 16);
    dim3 gridSize((w + blockSize.x - 1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y);
    fillSAT<<<gridSize, blockSize>>>(d_input, d_denom, d_nom_x, d_nom_y, w, h, padded_width);
    cudaCheckError(cudaDeviceSynchronize());

    // Perform horizontal scans (one thread per row).
    horizontal_scan<<<padded_height, 1>>>(d_denom, padded_width, padded_height);
    horizontal_scan<<<padded_height, 1>>>(d_nom_x, padded_width, padded_height);
    horizontal_scan<<<padded_height, 1>>>(d_nom_y, padded_width, padded_height);
    cudaCheckError(cudaDeviceSynchronize());

    // Perform vertical scans (one thread per column).
    vertical_scan<<<padded_width, 1>>>(d_denom, padded_width, padded_height);
    vertical_scan<<<padded_width, 1>>>(d_nom_x, padded_width, padded_height);
    vertical_scan<<<padded_width, 1>>>(d_nom_y, padded_width, padded_height);
    cudaCheckError(cudaDeviceSynchronize());

    // Retrieve the full sums from the SAT arrays (last element).
    double host_sum_denom, host_sum_nom_x, host_sum_nom_y;
    cudaCheckError(cudaMemcpy(&host_sum_denom, d_denom + (padded_size - 1), sizeof(double), cudaMemcpyDeviceToHost));
    cudaCheckError(cudaMemcpy(&host_sum_nom_x, d_nom_x + (padded_size - 1), sizeof(double), cudaMemcpyDeviceToHost));
    cudaCheckError(cudaMemcpy(&host_sum_nom_y, d_nom_y + (padded_size - 1), sizeof(double), cudaMemcpyDeviceToHost));

    // Allocate device memory for particles and copy data.
    double* d_particles;
    cudaCheckError(cudaMalloc((void**)&d_particles, sizeof(double) * N * 2));
    cudaCheckError(cudaMemcpy(d_particles, host_particles.data(), sizeof(double) * N * 2, cudaMemcpyHostToDevice));

    // Setup our FlowCuda structure and copy it to device memory.
    FlowCuda h_flow;
    h_flow.width = w;
    h_flow.height = h;
    h_flow.padded_width = padded_width;
    h_flow.padded_height = padded_height;
    h_flow.input = d_input;
    h_flow.denom = d_denom;
    h_flow.nom_x = d_nom_x;
    h_flow.nom_y = d_nom_y;
    h_flow.sum_denom = host_sum_denom;
    h_flow.sum_nom_x = host_sum_nom_x;
    h_flow.sum_nom_y = host_sum_nom_y;

    FlowCuda* d_flow;
    cudaCheckError(cudaMalloc((void**)&d_flow, sizeof(FlowCuda)));
    cudaCheckError(cudaMemcpy(d_flow, &h_flow, sizeof(FlowCuda), cudaMemcpyHostToDevice));

    // Launch kernel to advect particles using RK4 integration.
    int num_steps = 50;
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    
    advect_particles_kernel<<<blocks, threadsPerBlock>>>(d_flow, d_particles, N, num_steps);
    cudaCheckError(cudaDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "GPU elapsed time: " << elapsed_seconds.count() << " s\n";

    // Copy updated particle positions back to host.
    cudaCheckError(cudaMemcpy(host_particles.data(), d_particles, sizeof(double) * N * 2, cudaMemcpyDeviceToHost));

    // Write output positions to an SVG file.
    FILE* fout = fopen(result_filename, "w");
    if (fout) {
        fprintf(fout, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"%d\" height=\"%d\">\n", w, h);
        fprintf(fout, "<g>\n");
        for (int i = 0; i < N; i++) {
            fprintf(fout, "<circle cx=\"%3.3f\" cy=\"%3.3f\" r=\"1\" />\n", host_particles[2 * i], host_particles[2 * i + 1]);
        }
        fprintf(fout, "</g>\n</svg>\n");
        fclose(fout);
        std::cout << "SVG saved to outAdvectedBNOT.svg\n";
    }

    // Free device memory.
    cudaFree(d_input);
    cudaFree(d_denom);
    cudaFree(d_nom_x);
    cudaFree(d_nom_y);
    cudaFree(d_particles);
    cudaFree(d_flow);
}



// ---------------------------------------------------------------------
// Main (host) function
int main()
{
	const char* image = "lionOrigami.bmp";
	int N_stipples = 16384;
	const char* uniform_stipples = "BNOT_uniformpts16k.txt";
	const char* result = "output.svg";
    test_stippling(image, N_stipples, uniform_stipples, result);
    return 0;
}
