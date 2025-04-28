#include "flow_3d.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>

Flow_3d::Flow_3d() {}

Flow_3d::Flow_3d(int width, int height, int depth, double* input) {
    this->width = width;
    this->height = height;
    this->depth = depth;
    padded_width = width + 1;
    padded_height = height + 1;
    padded_depth = depth + 1;
    nom_x = new double[padded_width * padded_height * padded_depth];
    nom_y = new double[padded_width * padded_height * padded_depth];
    nom_z = new double[padded_width * padded_height * padded_depth];
    denom = new double[padded_width * padded_height * padded_depth];

    // Increment values for the x, y, z coords for denom, nom_x, nom_y, nom_z matrix
    inc_x = padded_depth;
    inc_y = padded_depth * padded_width;
    inc_z = 1;

	if (input)
		initialize(input);


}

void Flow_3d::initialize(double* input) {
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
	memset(denom, 0, padded_width * padded_height * padded_width * sizeof(denom[0]));
	memset(nom_x, 0, padded_width * padded_height * padded_width * sizeof(nom_x[0]));
	memset(nom_y, 0, padded_width * padded_height * padded_width * sizeof(nom_y[0]));
    memset(nom_z, 0, padded_width * padded_height * padded_width * sizeof(nom_z[0]));


    // Compute summed area tables for the image, x*image and y*image.
    for (int i=0; i<height; i ++) {
        for (int j=0; j<width; j ++) {
            for (int k=0; k<depth; k ++) {
                denom[(i+1)*padded_depth*padded_width + (j+1)*padded_depth + (k+1)] = denom[(i)*padded_depth*padded_width + (j+1)*padded_depth + (k+1)] 
                                                                                    + denom[(i+1)*padded_depth*padded_width + (j)*padded_depth + (k+1)] 
                                                                                    + denom[(i+1)*padded_depth*padded_width + (j+1)*padded_depth + (k)] 
                                                                                    - denom[(i)*padded_depth*padded_width + (j)*padded_depth + (k+1)] 
                                                                                    - denom[(i)*padded_depth*padded_width + (j+1)*padded_depth + (k)] 
                                                                                    - denom[(i+1)*padded_depth*padded_width + (j)*padded_depth + (k)] 
                                                                                    + denom[(i)*padded_depth*padded_width + (j)*padded_depth + (k)] 
                                                                                    + input[(i)*depth*width + (j)*depth + (k)];
                nom_x[(i+1)*padded_depth*padded_width + (j+1)*padded_depth + (k+1)] = nom_x[(i)*padded_depth*padded_width + (j+1)*padded_depth + (k+1)] 
                                                                                    + nom_x[(i+1)*padded_depth*padded_width + (j)*padded_depth + (k+1)] 
                                                                                    + nom_x[(i+1)*padded_depth*padded_width + (j+1)*padded_depth + (k)] 
                                                                                    - nom_x[(i)*padded_depth*padded_width + (j)*padded_depth + (k+1)] 
                                                                                    - nom_x[(i)*padded_depth*padded_width + (j+1)*padded_depth + (k)] 
                                                                                    - nom_x[(i+1)*padded_depth*padded_width + (j)*padded_depth + (k)] 
                                                                                    + nom_x[(i)*padded_depth*padded_width + (j)*padded_depth + (k)] 
                                                                                    + input[(i)*depth*width + (j)*depth + (k)]*(j+0.5);
                nom_y[(i+1)*padded_depth*padded_width + (j+1)*padded_depth + (k+1)] = nom_y[(i)*padded_depth*padded_width + (j+1)*padded_depth + (k+1)] 
                                                                                    + nom_y[(i+1)*padded_depth*padded_width + (j)*padded_depth + (k+1)] 
                                                                                    + nom_y[(i+1)*padded_depth*padded_width + (j+1)*padded_depth + (k)] 
                                                                                    - nom_y[(i)*padded_depth*padded_width + (j)*padded_depth + (k+1)] 
                                                                                    - nom_y[(i)*padded_depth*padded_width + (j+1)*padded_depth + (k)] 
                                                                                    - nom_y[(i+1)*padded_depth*padded_width + (j)*padded_depth + (k)] 
                                                                                    + nom_y[(i)*padded_depth*padded_width + (j)*padded_depth + (k)] 
                                                                                    + input[(i)*depth*width + (j)*depth + (k)]*(i+0.5);
                nom_z[(i+1)*padded_depth*padded_width + (j+1)*padded_depth + (k+1)] = nom_z[(i)*padded_depth*padded_width + (j+1)*padded_depth + (k+1)] 
                                                                                    + nom_z[(i+1)*padded_depth*padded_width + (j)*padded_depth + (k+1)] 
                                                                                    + nom_z[(i+1)*padded_depth*padded_width + (j+1)*padded_depth + (k)] 
                                                                                    - nom_z[(i)*padded_depth*padded_width + (j)*padded_depth + (k+1)] 
                                                                                    - nom_z[(i)*padded_depth*padded_width + (j+1)*padded_depth + (k)] 
                                                                                    - nom_z[(i+1)*padded_depth*padded_width + (j)*padded_depth + (k)] 
                                                                                    + nom_z[(i)*padded_depth*padded_width + (j)*padded_depth + (k)] 
                                                                                    + input[(i)*depth*width + (j)*depth + (k)]*(k+0.5);         
                      
            }
        }
    }
    sum_denom = denom[padded_height * padded_width * padded_depth - 1];
    sum_nom_x = nom_x[padded_height * padded_width * padded_depth - 1];
    sum_nom_y = nom_y[padded_height * padded_width * padded_depth - 1];
    sum_nom_z = nom_z[padded_height * padded_width * padded_depth - 1];

}

double Flow_3d::lerp_denom(int pixel_index, int pixel_index_padded, double rx, double ry, double rz) {
    /*
        "Lerp" operator to access the summed-area table `denom` with float index
    */
    double tmp =  denom[pixel_index_padded] +
           (denom[pixel_index_padded + inc_x + inc_y] + denom[pixel_index_padded] - denom[pixel_index_padded + inc_x] - denom[pixel_index_padded + inc_y])*rx*ry +
           (denom[pixel_index_padded + inc_x + inc_z] + denom[pixel_index_padded] - denom[pixel_index_padded + inc_x] - denom[pixel_index_padded + inc_z])*rx*rz +
           (denom[pixel_index_padded + inc_y + inc_z] + denom[pixel_index_padded] - denom[pixel_index_padded + inc_y] - denom[pixel_index_padded + inc_z])*ry*rz +
           (denom[pixel_index_padded + inc_x] - denom[pixel_index_padded])*rx +
           (denom[pixel_index_padded + inc_y] - denom[pixel_index_padded])*ry +
           (denom[pixel_index_padded + inc_z] - denom[pixel_index_padded])*rz +
           (input[pixel_index]*rx*ry*rz);

    return tmp;
}

double Flow_3d::lerp_nom_x(double avg_x_ix, int pixel_index, int pixel_index_padded, double rx, double ry, double rz) {
    /*
        "Lerp" operator to access the summed-area table `nom_x` with float index
    */
    return nom_x[pixel_index_padded] +
           (denom[pixel_index_padded + inc_x + inc_y] + denom[pixel_index_padded] - denom[pixel_index_padded + inc_x] - denom[pixel_index_padded + inc_y])*rx*ry*avg_x_ix +
           (denom[pixel_index_padded + inc_x + inc_z] + denom[pixel_index_padded] - denom[pixel_index_padded + inc_x] - denom[pixel_index_padded + inc_z])*rx*rz*avg_x_ix +
           (nom_x[pixel_index_padded + inc_y + inc_z] + nom_x[pixel_index_padded] - nom_x[pixel_index_padded + inc_y] - nom_x[pixel_index_padded + inc_z])*ry*rz +
           (denom[pixel_index_padded + inc_x] - denom[pixel_index_padded])*rx*avg_x_ix +
           (nom_x[pixel_index_padded + inc_y] - nom_x[pixel_index_padded])*ry +
           (nom_x[pixel_index_padded + inc_z] - nom_x[pixel_index_padded])*rz +
           (input[pixel_index]*rx*ry*rz*avg_x_ix);
}

double Flow_3d::lerp_nom_y(double avg_y_iy, int pixel_index, int pixel_index_padded, double rx, double ry, double rz) {
    /*
        "Lerp" operator to access the summed-area table `nom_y` with float index
    */
    return  nom_y[pixel_index_padded] +
           (denom[pixel_index_padded + inc_x + inc_y] + denom[pixel_index_padded] - denom[pixel_index_padded + inc_x] - denom[pixel_index_padded + inc_y])*rx*ry*avg_y_iy +
           (nom_y[pixel_index_padded + inc_x + inc_z] + nom_y[pixel_index_padded] - nom_y[pixel_index_padded + inc_x] - nom_y[pixel_index_padded + inc_z])*rx*rz +
           (denom[pixel_index_padded + inc_y + inc_z] + denom[pixel_index_padded] - denom[pixel_index_padded + inc_y] - denom[pixel_index_padded + inc_z])*ry*rz*avg_y_iy +
           (nom_y[pixel_index_padded + inc_x] - nom_y[pixel_index_padded])*rx +
           (denom[pixel_index_padded + inc_y] - denom[pixel_index_padded])*ry*avg_y_iy +
           (nom_y[pixel_index_padded + inc_z] - nom_y[pixel_index_padded])*rz +
           (input[pixel_index]*rx*ry*rz*avg_y_iy);
}

double Flow_3d::lerp_nom_z(double avg_z_iz, int pixel_index, int pixel_index_padded, double rx, double ry, double rz) {
    /*
        "Lerp" operator to access the summed-area table `nom_y` with float index
    */
    return  nom_z[pixel_index_padded] +
           (nom_z[pixel_index_padded + inc_x + inc_y] + nom_z[pixel_index_padded] - nom_z[pixel_index_padded + inc_x] - nom_z[pixel_index_padded + inc_y])*rx*ry +
           (denom[pixel_index_padded + inc_x + inc_z] + denom[pixel_index_padded] - denom[pixel_index_padded + inc_x] - denom[pixel_index_padded + inc_z])*rx*rz*avg_z_iz +
           (denom[pixel_index_padded + inc_y + inc_z] + denom[pixel_index_padded] - denom[pixel_index_padded + inc_y] - denom[pixel_index_padded + inc_z])*ry*rz*avg_z_iz +
           (nom_z[pixel_index_padded + inc_x] - nom_z[pixel_index_padded])*rx +
           (nom_z[pixel_index_padded + inc_y] - nom_z[pixel_index_padded])*ry +
           (denom[pixel_index_padded + inc_z] - denom[pixel_index_padded])*rz*avg_z_iz +
           (input[pixel_index]*rx*ry*rz*avg_z_iz);
}

void Flow_3d::integrate(double y_top, double x_left, double z_back, double y_bottom, double x_right, double z_front, double& out_d, double& out_x, double& out_y, double& out_z) {
    /*
        Mean function that updates the variables `out_d`, `out_x` and `out_y` with integrals
        calculated using lerp functions
    */

    // top left corner, integer and fractional parts
    int ix_left = floor(x_left), iy_top = floor(y_top), iz_back = floor(z_back);
    double rx_left = x_left - ix_left, ry_top = y_top - iy_top, rz_back=z_back-iz_back;

    // bottom right corner, integer and fractional parts
    int ix_right = floor(x_right), iy_bottom = floor(y_bottom), iz_front = floor(z_front);
    double rx_right = x_right - ix_right, ry_bottom = y_bottom - iy_bottom, rz_front = z_front-iz_front;

    // factorizing costs of accessing table[i][j] as table[i*width+j]
    int index_top_left_back = iy_top*depth*width + ix_left*depth + iz_back,           index_top_left_back_padded = iy_top*inc_y + ix_left*inc_x + iz_back*inc_z;
    int index_top_left_front = iy_top*depth*width + ix_left*depth + iz_front,         index_top_left_front_padded = iy_top*inc_y + ix_left*inc_x + iz_front*inc_z;
    int index_top_right_back = iy_top*depth*width + ix_right*depth + iz_back,         index_top_right_back_padded = iy_top*inc_y + ix_right*inc_x + iz_back*inc_z;
    int index_top_right_front = iy_top*depth*width + ix_right*depth + iz_front,       index_top_right_front_padded = iy_top*inc_y + ix_right*inc_x + iz_front*inc_z;
    int index_bottom_left_back = iy_bottom*depth*width + ix_left*depth + iz_back,     index_bottom_left_back_padded = iy_bottom*inc_y + ix_left*inc_x + iz_back*inc_z;
    int index_bottom_left_front = iy_bottom*depth*width + ix_left*depth + iz_front,   index_bottom_left_front_padded = iy_bottom*inc_y + ix_left*inc_x + iz_front*inc_z;
    int index_bottom_right_back = iy_bottom*depth*width + ix_right*depth + iz_back,   index_bottom_right_back_padded = iy_bottom*inc_y + ix_right*inc_x + iz_back*inc_z;
    int index_bottom_right_front = iy_bottom*depth*width + ix_right*depth + iz_front, index_bottom_right_front_padded = iy_bottom*inc_y + ix_right*inc_x + iz_front*inc_z;

    // standard integration using summed area tables, but with interpolation
    out_d = lerp_denom(index_bottom_right_front, index_bottom_right_front_padded, rx_right, ry_bottom, rz_front) - lerp_denom(index_bottom_right_back, index_bottom_right_back_padded, rx_right, ry_bottom, rz_back)
          + lerp_denom(index_bottom_left_back, index_bottom_left_back_padded, rx_left, ry_bottom, rz_back) - lerp_denom(index_bottom_left_front, index_bottom_left_front_padded, rx_left, ry_bottom, rz_front)
          + lerp_denom(index_top_right_back, index_top_right_back_padded, rx_right, ry_top, rz_back) - lerp_denom(index_top_right_front, index_top_right_front_padded, rx_right, ry_top, rz_front)
          + lerp_denom(index_top_left_front, index_top_left_front_padded, rx_left, ry_top, rz_front) - lerp_denom(index_top_left_back, index_top_left_back_padded, rx_left, ry_top, rz_back);

    out_x = lerp_nom_x((x_right+ix_right)*0.5, index_bottom_right_front, index_bottom_right_front_padded, rx_right, ry_bottom, rz_front) - lerp_nom_x((x_right+ix_right)*0.5, index_bottom_right_back, index_bottom_right_back_padded, rx_right, ry_bottom, rz_back)
          + lerp_nom_x((x_left+ix_left)*0.5, index_bottom_left_back, index_bottom_left_back_padded, rx_left, ry_bottom, rz_back) - lerp_nom_x((x_left+ix_left)*0.5, index_bottom_left_front, index_bottom_left_front_padded, rx_left, ry_bottom, rz_front)
          + lerp_nom_x((x_right+ix_right)*0.5, index_top_right_back, index_top_right_back_padded, rx_right, ry_top, rz_back) - lerp_nom_x((x_right+ix_right)*0.5, index_top_right_front, index_top_right_front_padded, rx_right, ry_top, rz_front)
          + lerp_nom_x((x_left+ix_left)*0.5, index_top_left_front, index_top_left_front_padded, rx_left, ry_top, rz_front) - lerp_nom_x((x_left+ix_left)*0.5, index_top_left_back, index_top_left_back_padded, rx_left, ry_top, rz_back);

    out_y = lerp_nom_y((y_bottom+iy_bottom)*0.5, index_bottom_right_front, index_bottom_right_front_padded, rx_right, ry_bottom, rz_front) - lerp_nom_y((y_bottom+iy_bottom)*0.5, index_bottom_right_back, index_bottom_right_back_padded, rx_right, ry_bottom, rz_back)
          + lerp_nom_y((y_bottom+iy_bottom)*0.5, index_bottom_left_back, index_bottom_left_back_padded, rx_left, ry_bottom, rz_back) - lerp_nom_y((y_bottom+iy_bottom)*0.5, index_bottom_left_front, index_bottom_left_front_padded, rx_left, ry_bottom, rz_front)
          + lerp_nom_y((y_top+iy_top)*0.5, index_top_right_back, index_top_right_back_padded, rx_right, ry_top, rz_back) - lerp_nom_y((y_top+iy_top)*0.5, index_top_right_front, index_top_right_front_padded, rx_right, ry_top, rz_front)
          + lerp_nom_y((y_top+iy_top)*0.5, index_top_left_front, index_top_left_front_padded, rx_left, ry_top, rz_front) - lerp_nom_y((y_top+iy_top)*0.5, index_top_left_back, index_top_left_back_padded, rx_left, ry_top, rz_back);

    out_z = lerp_nom_z((z_front+iz_front)*0.5, index_bottom_right_front, index_bottom_right_front_padded, rx_right, ry_bottom, rz_front) - lerp_nom_z((z_back+iz_back)*0.5, index_bottom_right_back, index_bottom_right_back_padded, rx_right, ry_bottom, rz_back)
          + lerp_nom_z((z_back+iz_back)*0.5, index_bottom_left_back, index_bottom_left_back_padded, rx_left, ry_bottom, rz_back) - lerp_nom_z((z_front+iz_front)*0.5, index_bottom_left_front, index_bottom_left_front_padded, rx_left, ry_bottom, rz_front)
          + lerp_nom_z((z_back+iz_back)*0.5, index_top_right_back, index_top_right_back_padded, rx_right, ry_top, rz_back) - lerp_nom_z((z_front+iz_front)*0.5, index_top_right_front, index_top_right_front_padded, rx_right, ry_top, rz_front)
          + lerp_nom_z((z_front+iz_front)*0.5, index_top_left_front, index_top_left_front_padded, rx_left, ry_top, rz_front) - lerp_nom_z((z_back+iz_back)*0.5, index_top_left_back, index_top_left_back_padded, rx_left, ry_top, rz_back);
}


void Flow_3d::velocity(double x, double y, double z, double t, double& vx, double& vy, double& vz) {
    double integral_d, integral_x, integral_y, integral_z;
    x = std::min(std::max(f_thres, x), width  - f_thres);
    y = std::min(std::max(f_thres, y), height - f_thres);
    z = std::min(std::max(f_thres, z), depth  - f_thres);
    if (t >= 1.) { vx = 0.; vy = 0; vz = 0;  return; }
    if (t < t_thres) {
        integral_d = sum_denom;
        integral_x = sum_nom_x;
        integral_y = sum_nom_y;
        integral_z = sum_nom_z;
    } else {
        double r_top = height - (height-y)/t,
               r_bottom = y/t,
               r_left = width - (width-x)/t,
               r_right = x/t,
               r_back = depth - (depth-z)/t,
               r_front = z/t;
        if (r_top <= f_thres && r_bottom >= height - f_thres && r_left <= f_thres && r_right >= width - f_thres && r_back <= f_thres && r_front > depth-f_thres) {
            integral_d = sum_denom;
            integral_x = sum_nom_x;
            integral_y = sum_nom_y;
            integral_z = sum_nom_z;
        } else {
            r_top = std::max(f_thres, r_top);
            r_bottom = std::min(height - f_thres, r_bottom);
            r_left = std::max(f_thres, r_left);
            r_right = std::min(width - f_thres, r_right);
            r_back = std::max(f_thres, r_back);
            r_front = std::min(depth-f_thres, r_front);
            integrate(r_top, r_left, r_back, r_bottom, r_right, r_front, integral_d, integral_x, integral_y, integral_z);
        }
    }
    double x1 = integral_x / integral_d;
    double y1 = integral_y / integral_d;
    double z1 = integral_z / integral_d;
    vx = (x1-x)/(1.-t);
    vy = (y1-y)/(1.-t);
    vz = (z1-z)/(1.-t);

}


void Flow_3d::integrate_rk4(double& x, double& y, double& z, int num_step) {
    /*
    Solver that advects the point (x, y) with a Runge Kutta of order 4 scheme.
        Derivative information given by the velocity function
    The number of velocity function calls is 4x as much as the 1st-order case.
*/
    double t, vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3, vx4, vy4, vz4;
    double d = 1. / double(num_step);
    for (int i = 0.; i < num_step; i++) {
        t = i*d;
        velocity(x,                   y,                   z,                   t,           vx1, vy1, vz1);
        velocity(x + vx1 * (d * 0.5), y + vy1 * (d * 0.5), z + vz1 * (d * 0.5), t + d * 0.5, vx2, vy2, vz2);
        velocity(x + vx2 * (d * 0.5), y + vy2 * (d * 0.5), z + vz2 * (d * 0.5), t + d * 0.5, vx3, vy3, vz3);
        velocity(x + vx3 * d,         y + vy3 * d,         z + vz3 * d,         t + d,       vx4, vy4, vz4);
        x += (vx1 + (vx2 + vx3) * 2. + vx4) * d / 6.;
        y += (vy1 + (vy2 + vy3) * 2. + vy4) * d / 6.;
        z += (vz1 + (vz2 + vz3) * 2. + vz4) * d / 6.;
    }
}
