#include "flow.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>

Flow::Flow() {}

Flow::Flow(size_t width, size_t height, double* input) {
    this->width = width;
    this->height = height;
    padded_width = width + 1;
    padded_height = height + 1;
    nom_x = new double[padded_width* padded_height];
    nom_y = new double[padded_width * padded_height];
    denom = new double[padded_width * padded_height];
	if (input)
		initialize(input);
}

void Flow::initialize(double* input) {
    /*
    Initializes the summed-area table for the input matrix
    Variable 'nom_x' is the summed-area table for the pixel*x values (second/column index)
    Variable 'nom_y' is the summed-area table for the pixel*y values (first/row index)
    Variable 'denom' is the summed-area table for the pixels of the input matrix
        `nom_x', `nom_y` and `denom` have dimension (width+1)*(height+1)
        with the first rows and columns of them being zeros, for convenience
    */
    
    // Assign input
    this->input = input;
	memset(denom, 0, padded_width * padded_height *sizeof(denom[0]));
	memset(nom_x, 0, padded_width * padded_height *sizeof(nom_x[0]));
	memset(nom_y, 0, padded_width * padded_height *sizeof(nom_y[0]));

    // Compute summed area tables for the image, x*image and y*image.
    
    for (size_t i=0; i<height; i ++) {
        for (size_t j=0; j<width; j ++) {
            denom[(i+1)*padded_width + (j+1)] = denom[(i+1) * padded_width + j] + denom[i * padded_width + (j+1)] - denom[i * padded_width +j] + input[i*width + j];
            nom_x[(i+1)*padded_width + (j+1)] = nom_x[(i+1) * padded_width + j] + nom_x[i * padded_width + (j+1)] - nom_x[i * padded_width +j] + input[i*width + j]*(j+0.5);
            nom_y[(i+1)*padded_width + (j+1)] = nom_y[(i+1) * padded_width + j] + nom_y[i * padded_width + (j+1)] - nom_y[i * padded_width +j] + input[i*width + j]*(i+0.5);
        }
    }
    

    sum_denom = denom[height * padded_width + width];
    sum_nom_x = nom_x[height * padded_width + width];
    sum_nom_y = nom_y[height * padded_width + width];
}

double Flow::lerp_denom(size_t pixel_index, size_t pixel_index_padded, double rx, double ry) {
    /*
        "Lerp" operator to access the summed-area table `denom` with float index
    */
    return  denom[pixel_index_padded] +
           (denom[pixel_index_padded + 1]            - denom[pixel_index_padded])*rx +
           (denom[pixel_index_padded + padded_width] - denom[pixel_index_padded])*ry +
           (input[pixel_index]*rx*ry);
}

double Flow::lerp_nom_x(double avg_x_ix, size_t pixel_index, size_t pixel_index_padded, double rx, double ry) {
    /*
        "Lerp" operator to access the summed-area table `nom_x` with float index
    */
    return nom_x[pixel_index_padded] +
           (nom_x[pixel_index_padded + padded_width] - nom_x[pixel_index_padded])*ry +
           (denom[pixel_index_padded + 1] - denom[pixel_index_padded])*rx* avg_x_ix +
           (input[pixel_index]*rx* avg_x_ix*ry);
}

double Flow::lerp_nom_y(double avg_y_iy, size_t pixel_index, size_t pixel_index_padded, double rx, double ry) {
    /*
        "Lerp" operator to access the summed-area table `nom_y` with float index
    */
    return nom_y[pixel_index_padded] +
           (denom[pixel_index_padded + padded_width] - denom[pixel_index_padded])*ry*avg_y_iy +
           (nom_y[pixel_index_padded +1] - nom_y[pixel_index_padded])*rx +
           (input[pixel_index]*ry*rx* avg_y_iy);
}

void Flow::integrate(double y_top, double x_left, double y_bottom, double x_right, double& out_d, double& out_x, double& out_y) {
    /*
        Mean function that updates the variables `out_d`, `out_x` and `out_y` with integrals
        calculated using lerp functions
    */

    // top left corner, integer and fractional parts
    size_t ix_left = floor(x_left), iy_top = floor(y_top);
    double rx_left = x_left - ix_left, ry_top = y_top - iy_top;

    // bottom right corner, integer and fractional parts
    size_t ix_right = floor(x_right), iy_bottom = floor(y_bottom);
    double rx_right = x_right - ix_right, ry_bottom = y_bottom - iy_bottom;

    // factorizing costs of accessing table[i][j] as table[i*width+j]
    size_t index_top_left = iy_top * width + ix_left,                   index_top_left_padded = index_top_left + iy_top;
    size_t index_top_right = index_top_left - ix_left + ix_right,       index_top_right_padded = index_top_left_padded - ix_left + ix_right;
    size_t index_bottom_left = iy_bottom * width + ix_left,             index_bottom_left_padded = index_bottom_left + iy_bottom;
    size_t index_bottom_right = index_bottom_left - ix_left + ix_right, index_bottom_right_padded = index_bottom_left_padded - ix_left + ix_right;

    // standard integration using summed area tables, but with interpolation
    out_d = lerp_denom(index_bottom_right, index_bottom_right_padded, rx_right, ry_bottom)           + lerp_denom(index_top_left, index_top_left_padded, rx_left, ry_top)         - lerp_denom(index_top_right, index_top_right_padded, rx_right, ry_top)         - lerp_denom(index_bottom_left, index_bottom_left_padded, rx_left, ry_bottom);
    out_x = lerp_nom_x((x_right+ix_right)*0.5, index_bottom_right, index_bottom_right_padded, rx_right, ry_bottom)  + lerp_nom_x((x_left+ix_left)*0.5, index_top_left, index_top_left_padded, rx_left, ry_top) - lerp_nom_x((x_right+ix_right)*0.5, index_top_right, index_top_right_padded, rx_right, ry_top) - lerp_nom_x((x_left+ix_left)*0.5, index_bottom_left, index_bottom_left_padded, rx_left, ry_bottom);
    out_y = lerp_nom_y((y_bottom+iy_bottom)*0.5, index_bottom_right, index_bottom_right_padded, rx_right, ry_bottom) + lerp_nom_y((y_top+iy_top)*0.5, index_top_left, index_top_left_padded, rx_left, ry_top)  - lerp_nom_y((y_top+iy_top)*0.5, index_top_right, index_top_right_padded, rx_right, ry_top)   - lerp_nom_y((y_bottom+iy_bottom)*0.5, index_bottom_left, index_bottom_left_padded, rx_left, ry_bottom);

	// naive integral
    //out_d = denom[index_bottom_right_padded] + denom[index_top_left_padded]  - denom[index_top_right_padded] - denom[index_bottom_left_padded];
    //out_x = nom_x[index_bottom_right_padded] + nom_x[index_top_left_padded] - nom_x[index_top_right_padded] - nom_x[index_bottom_left_padded];
   // out_y = nom_y[index_bottom_right_padded] + nom_y[index_top_left_padded] - nom_y[index_top_right_padded] - nom_y[index_bottom_left_padded];



}


void Flow::velocity(double x, double y, double t, double& vx, double& vy) {
    double integral_d, integral_x, integral_y;
    x = std::min(std::max(f_thres, x), width  - f_thres);
    y = std::min(std::max(f_thres, y), height - f_thres);
    if (t >= 1. - t_thres) {vx=0.; vy=0; return;}
    if (t < t_thres) {
        integral_d = sum_denom;
        integral_x = sum_nom_x;
        integral_y = sum_nom_y;
    } else {
        double r_top = height - (height-y)/t,
               r_bottom = y/t,
               r_left = width - (width-x)/t,
               r_right = x/t;
        if (r_top <= f_thres && r_bottom >= height - f_thres && r_left <= f_thres && r_right >= width - f_thres) {
            integral_d = sum_denom;
            integral_x = sum_nom_x;
            integral_y = sum_nom_y;
        } else {
            r_top = std::max(f_thres, r_top);
            r_bottom = std::min(height - f_thres, r_bottom);
            r_left = std::max(f_thres, r_left);
            r_right = std::min(width - f_thres, r_right);
            integrate(r_top, r_left, r_bottom, r_right, integral_d, integral_x, integral_y);
        }
    }
    double x1 = integral_x / integral_d;
    double y1 = integral_y / integral_d;
    vx = (x1-x)/(1.-t);
    vy = (y1-y)/(1.-t);
    if (isinf(vx)) vx = 0;
    if (isinf(vy)) vy = 0;
}


void Flow::integrate_rk4(double& x, double& y, int num_step) {
    /*
    Solver that advects the point (x, y) with a Runge Kutta of order 4 scheme.
        Derivative information given by the velocity function
    The number of velocity function calls is 4x as much as the 1st-order case.
*/
    double t, vx1, vy1, vx2, vy2, vx3, vy3, vx4, vy4;
    double d = 1. / double(num_step);
    for (int i = 0.; i < num_step; i++) {
        t = i*d;
        velocity(x,                   y,                   t,           vx1, vy1);
        velocity(x + vx1 * (d * 0.5), y + vy1 * (d * 0.5), t + d * 0.5, vx2, vy2);
        velocity(x + vx2 * (d * 0.5), y + vy2 * (d * 0.5), t + d * 0.5, vx3, vy3);
        velocity(x + vx3 * d,         y + vy3 * d,         t + d,       vx4, vy4);
        x += (vx1 + (vx2 + vx3) * 2. + vx4) * d / 6.;
        y += (vy1 + (vy2 + vy3) * 2. + vy4) * d / 6.;
    }
}

void Flow::integrate_rk4_backwards(double& x, double& y, int num_step) {
    /*
    Solver that advects the point (x, y) with a Runge Kutta of order 4 scheme.
        Derivative information given by the velocity function
    The number of velocity function calls is 4x as much as the 1st-order case.
*/
    double t, vx1, vy1, vx2, vy2, vx3, vy3, vx4, vy4;
    double d = 1. / double(num_step);
    for (int i = 0.; i < num_step; i++) {
        t = 1-i * d;
        velocity(x, y, t, vx1, vy1);
        velocity(x - vx1 * (d * 0.5), y - vy1 * (d * 0.5), t - d * 0.5, vx2, vy2);
        velocity(x - vx2 * (d * 0.5), y - vy2 * (d * 0.5), t - d * 0.5, vx3, vy3);
        velocity(x - vx3 * d, y - vy3 * d, t - d, vx4, vy4);
        x -= (vx1 + (vx2 + vx3) * 2. + vx4) * d / 6.;
        y -= (vy1 + (vy2 + vy3) * 2. + vy4) * d / 6.;
    }
}