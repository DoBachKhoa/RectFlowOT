// Constants (as defined in your original code)
#define t_thres 1e-12
#define f_thres 1e-12

// Structure to hold flow information on the device
struct FlowCuda {
    int width, height;
    int padded_width, padded_height;
    double *input;  // input image (on device)
    double *denom;  // SAT for pixel values
    double *nom_x;  // SAT for pixel * x
    double *nom_y;  // SAT for pixel * y
    double sum_denom, sum_nom_x, sum_nom_y; // full-sum values (from last SAT element)
};

// ---------------------------------------------------------------------
// GPU Kernel: fillSAT
// Writes input values into the SAT arrays with an offset of one in each dimension.
// Each thread fills one pixel’s value into the padded SAT.
__global__ void fillSAT(const double* input, double* denom, double* nom_x, double* nom_y,
                          int width, int height, int padded_width)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y; // row index (0 <= i < height)
    int j = blockIdx.x * blockDim.x + threadIdx.x; // column index (0 <= j < width)

    if (i < height && j < width) {
        int padded_index = (i + 1) * padded_width + (j + 1);
        int input_index = i * width + j;
        double val = input[input_index];
        denom[padded_index] = val;
        nom_x[padded_index] = val * (j + 0.5);
        nom_y[padded_index] = val * (i + 0.5);
    }
}

// ---------------------------------------------------------------------
// GPU Kernel: horizontal_scan
// For each row, perform an in-place cumulative sum.
__global__ void horizontal_scan(double* data, int padded_width, int padded_height)
{
    int row = blockIdx.x;  // one block per row
    if (row < padded_height) {
        int row_start = row * padded_width;
        for (int j = 1; j < padded_width; j++) {
            data[row_start + j] += data[row_start + j - 1];
        }
    }
}

// ---------------------------------------------------------------------
// GPU Kernel: vertical_scan
// For each column, perform an in-place cumulative sum.
__global__ void vertical_scan(double* data, int padded_width, int padded_height)
{
    int col = blockIdx.x;  // one block per column
    if (col < padded_width) {
        for (int i = 1; i < padded_height; i++) {
            data[i * padded_width + col] += data[(i - 1) * padded_width + col];
        }
    }
}

// ---------------------------------------------------------------------
// Device inline functions (lerp and integrate) mirroring your CPU-side versions.

// lerp_denom: interpolates in the summed-area table "denom"
__device__ inline double lerp_denom(const FlowCuda* f, int pixel_index, int pixel_index_padded, double rx, double ry)
{
    return f->denom[pixel_index_padded] +
           (f->denom[pixel_index_padded + 1] - f->denom[pixel_index_padded]) * rx +
           (f->denom[pixel_index_padded + f->padded_width] - f->denom[pixel_index_padded]) * ry +
           (f->input[pixel_index] * rx * ry);
}

// lerp_nom_x: interpolates in the summed-area table "nom_x"
__device__ inline double lerp_nom_x(const FlowCuda* f, double avg_x_ix, int pixel_index, int pixel_index_padded, double rx, double ry)
{
    return f->nom_x[pixel_index_padded] +
           (f->nom_x[pixel_index_padded + f->padded_width] - f->nom_x[pixel_index_padded]) * ry +
           (f->denom[pixel_index_padded + 1] - f->denom[pixel_index_padded]) * rx * avg_x_ix +
           (f->input[pixel_index] * rx * avg_x_ix * ry);
}

// lerp_nom_y: interpolates in the summed-area table "nom_y"
__device__ inline double lerp_nom_y(const FlowCuda* f, double avg_y_iy, int pixel_index, int pixel_index_padded, double rx, double ry)
{
    return f->nom_y[pixel_index_padded] +
           (f->denom[pixel_index_padded + f->padded_width] - f->denom[pixel_index_padded]) * ry * avg_y_iy +
           (f->nom_y[pixel_index_padded + 1] - f->nom_y[pixel_index_padded]) * rx +
           (f->input[pixel_index] * ry * rx * avg_y_iy);
}

// integrate: computes integrated values over a rectangular region using interpolation.
__device__ void integrate(const FlowCuda* f, double y_top, double x_left, double y_bottom, double x_right,
                           double* out_d, double* out_x, double* out_y)
{
    int ix_left = (int)floor(x_left);
    int iy_top  = (int)floor(y_top);
    double rx_left = x_left - ix_left;
    double ry_top  = y_top - iy_top;

    int ix_right = (int)floor(x_right);
    int iy_bottom = (int)floor(y_bottom);
    double rx_right = x_right - ix_right;
    double ry_bottom = y_bottom - iy_bottom;

    // Indices into the input array and the padded SAT arrays.
    int index_top_left = iy_top * f->width + ix_left;
    int index_top_left_padded = index_top_left + iy_top;

    int index_top_right = index_top_left - ix_left + ix_right;
    int index_top_right_padded = index_top_left_padded - ix_left + ix_right;

    int index_bottom_left = iy_bottom * f->width + ix_left;
    int index_bottom_left_padded = index_bottom_left + iy_bottom;

    int index_bottom_right = index_bottom_left - ix_left + ix_right;
    int index_bottom_right_padded = index_bottom_left_padded - ix_left + ix_right;

    *out_d = lerp_denom(f, index_bottom_right, index_bottom_right_padded, rx_right, ry_bottom)
           + lerp_denom(f, index_top_left, index_top_left_padded, rx_left, ry_top)
           - lerp_denom(f, index_top_right, index_top_right_padded, rx_right, ry_top)
           - lerp_denom(f, index_bottom_left, index_bottom_left_padded, rx_left, ry_bottom);

    *out_x = lerp_nom_x(f, (x_right + ix_right) * 0.5, index_bottom_right, index_bottom_right_padded, rx_right, ry_bottom)
           + lerp_nom_x(f, (x_left + ix_left) * 0.5, index_top_left, index_top_left_padded, rx_left, ry_top)
           - lerp_nom_x(f, (x_right + ix_right) * 0.5, index_top_right, index_top_right_padded, rx_right, ry_top)
           - lerp_nom_x(f, (x_left + ix_left) * 0.5, index_bottom_left, index_bottom_left_padded, rx_left, ry_bottom);

    *out_y = lerp_nom_y(f, (y_bottom + iy_bottom) * 0.5, index_bottom_right, index_bottom_right_padded, rx_right, ry_bottom)
           + lerp_nom_y(f, (y_top + iy_top) * 0.5, index_top_left, index_top_left_padded, rx_left, ry_top)
           - lerp_nom_y(f, (y_top + iy_top) * 0.5, index_top_right, index_top_right_padded, rx_right, ry_top)
           - lerp_nom_y(f, (y_bottom + iy_bottom) * 0.5, index_bottom_left, index_bottom_left_padded, rx_left, ry_bottom);
}

// ---------------------------------------------------------------------
// Device function: velocity
// Computes the velocity at the point (x, y) for a given time t.
__device__ void velocity(const FlowCuda* f, double x, double y, double t, double* vx, double* vy)
{
    // Clamp the coordinates
    x = fmin(fmax(f_thres, x), f->width - f_thres);
    y = fmin(fmax(f_thres, y), f->height - f_thres);

    double integral_d, integral_x, integral_y;
    if (t >= 1.0 - t_thres) {
        *vx = 0.0;
        *vy = 0.0;
        return;
    }
    if (t < t_thres) {
        integral_d = f->sum_denom;
        integral_x = f->sum_nom_x;
        integral_y = f->sum_nom_y;
    }
    else {
        double r_top    = f->height - (f->height - y) / t;
        double r_bottom = y / t;
        double r_left   = f->width - (f->width - x) / t;
        double r_right  = x / t;

        if (r_top <= f_thres && r_bottom >= f->height - f_thres &&
            r_left <= f_thres && r_right >= f->width - f_thres)
        {
            integral_d = f->sum_denom;
            integral_x = f->sum_nom_x;
            integral_y = f->sum_nom_y;
        }
        else {
            r_top    = fmax(f_thres, r_top);
            r_bottom = fmin((double)f->height - f_thres, r_bottom);
            r_left   = fmax(f_thres, r_left);
            r_right  = fmin((double)f->width - f_thres, r_right);
            integrate(f, r_top, r_left, r_bottom, r_right, &integral_d, &integral_x, &integral_y);
        }
    }
    double x1 = integral_x / integral_d;
    double y1 = integral_y / integral_d;
    *vx = (x1 - x) / (1.0 - t);
    *vy = (y1 - y) / (1.0 - t);

    if (isinf(*vx)) *vx = 0.0;
    if (isinf(*vy)) *vy = 0.0;
}

// ---------------------------------------------------------------------
// Device function: integrate_rk4_device
// Uses RK4 integration to update the particle position.
__device__ void integrate_rk4_device(const FlowCuda* f, double* x, double* y, int num_step)
{
    double t, vx1, vy1, vx2, vy2, vx3, vy3, vx4, vy4;
    double d = 1.0 / (double)num_step;
    for (int i = 0; i < num_step; i++) {
        t = i * d;
        velocity(f, *x, *y, t, &vx1, &vy1);
        velocity(f, *x + vx1 * (d * 0.5), *y + vy1 * (d * 0.5), t + d * 0.5, &vx2, &vy2);
        velocity(f, *x + vx2 * (d * 0.5), *y + vy2 * (d * 0.5), t + d * 0.5, &vx3, &vy3);
        velocity(f, *x + vx3 * d, *y + vy3 * d, t + d, &vx4, &vy4);
        *x += (vx1 + 2.0*(vx2 + vx3) + vx4) * d / 6.0;
        *y += (vy1 + 2.0*(vy2 + vy3) + vy4) * d / 6.0;
    }
}

// ---------------------------------------------------------------------
// Kernel: advect_particles_kernel
// Each thread reads one particle's (x, y), applies RK4 integration,
// and writes back the updated position.
__global__ void advect_particles_kernel(const FlowCuda* f, double* particles, int num_particles, int num_steps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_particles) {
        double x = particles[2 * idx];
        double y = particles[2 * idx + 1];
        integrate_rk4_device(f, &x, &y, num_steps);
        particles[2 * idx]     = x;
        particles[2 * idx + 1] = y;
    }
}

// ---------------------------------------------------------------------
// Helper macro for CUDA error checking.
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

