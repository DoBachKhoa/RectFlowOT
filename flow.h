#include <stdio.h>
#include <stdlib.h>
#include <iostream> 
#include <cmath>

#define t_thres 1e-12
#define f_thres 1e-12

class Flow {
public:
    size_t width, padded_width;
    size_t height, padded_height;
    double* nom_x; // Summed-area table for pixel times x-coord
    double* nom_y; // Summed-area table for pixel times y-coord
    double* denom; // Summed-area table for pixel
    double* input; // Actual input
    double sum_denom, sum_nom_x, sum_nom_y; // entire sums


    // Initializers
    Flow();
    ~Flow() { delete[] denom; delete[] nom_x; delete[] nom_y; };
    Flow(size_t width, size_t height, double* input = 0);
    void initialize(double* input);

    // Lerp operator function
    double lerp_denom(size_t pixel_index, size_t pixel_index_padded, double rx, double ry);
    double lerp_nom_x(double avg_x_ix, size_t pixel_index, size_t pixel_index_padded, double rx, double ry);
    double lerp_nom_y(double avg_y_iy, size_t pixel_index, size_t pixel_index_padded, double rx, double ry);

    // Main calculate functions
    void integrate(double r_up, double r_down, double r_left, double r_right, double& out_d, double& out_x, double& out_y);

    // Velocity functions
    void velocity(double x, double y, double t, double& vx, double& vy);

    // RK4 functions
    void integrate_rk4(double& x, double& y, int num_step);
    void integrate_rk4_backwards(double& x, double& y, int num_step);
};
