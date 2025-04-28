#include <stdio.h>
#include <stdlib.h>
#include <iostream> 
#include <cmath>

#define t_thres 1e-12
#define f_thres 1e-12

class Flow_3d {
public:
    int width, padded_width;
    int height, padded_height;
    int depth, padded_depth;
    int inc_x, inc_y, inc_z;
    double* nom_x; // Summed-area table for pixel times x-coord
    double* nom_y; // Summed-area table for pixel times y-coord
    double* nom_z; // Summed-area table for pixel times z-coord
    double* denom; // Summed-area table for pixel
    double* input; // Actual input
    double sum_denom, sum_nom_x, sum_nom_y, sum_nom_z; // entire sums


    // Initializers
    Flow_3d();
    Flow_3d(int width, int height, int depth, double* input = 0);
    void initialize(double* input);

    // Lerp operator function
    double lerp_denom(int pixel_index, int pixel_index_padded, double rx, double ry, double rz);
    double lerp_nom_x(double avg_x_ix, int pixel_index, int pixel_index_padded, double rx, double ry, double rz);
    double lerp_nom_y(double avg_y_iy, int pixel_index, int pixel_index_padded, double rx, double ry, double rz);
    double lerp_nom_z(double avg_z_iz, int pixel_index, int pixel_index_padded, double rx, double ry, double rz);

    // Main calculate functions
    void integrate(double y1, double x1, double z1, double y2, double x2, double z2, double& out_d, double& out_x, double& out_y, double& out_z);

    // Velocity functions
    void velocity(double x, double y, double z, double t, double& vx, double& vy, double& vz);

    // Interpulate functions
    void integrate_rk4(double& x, double& y, double& z, int num_step);
};
