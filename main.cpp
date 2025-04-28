
#include <iostream>
#include "flow.h"


#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"



void save_svg(const std::vector<double>& pts, std::string filename, int W, int H) {
    FILE* f = fopen(filename.c_str(), "w+");
    fprintf(f, "<svg xmlns = \"http://www.w3.org/2000/svg\" width = \"%u\" height = \"%u\">\n", W, H);
    fprintf(f, "<g>\n");
    for (int j = 0; j < pts.size()/2; j++) {
        fprintf(f, "<circle cx=\" %3.3f\" cy=\"%3.3f\" r=\"1\" /> \nn", pts[j*2+0], pts[j*2+1]);
    }
    fprintf(f, "</g>\n");
    fprintf(f, "</svg>\n");
    fclose(f);
}

void load_pts(std::vector<double>& pts, int N, std::string filename) {
    pts.resize(N*2);
    FILE* f = fopen(filename.c_str(), "r+");
    for (int j = 0; j < N; j++) {
        fscanf(f, "%lf %lf\n", &pts[j*2+0], &pts[j*2+1]);
    }
    fclose(f);
}

int main()
{
    int w,h,n;
    int N = 40000;
    int numsteps = 50;
    unsigned char *data = stbi_load("lionOrigami.bmp", &w, &h, &n, 1);
    std::vector<double> datad(w * h);
    double s = 0;
    for (int i = 0; i < w * h; i++) {
        datad[i] = 255-data[i];
    }

    std::vector<double> pts(N);
    load_pts(pts, N, "BNOT_uniformpts40k.txt");
    //save_svg(pts, "uniform.svg");
    for (int i = 0; i < N; i++) {
        pts[i*2+0] *= w;
        pts[i*2+1] *= h;
    }

    const auto start = std::chrono::steady_clock::now();
    Flow f(w, h, &datad[0]);
#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        f.integrate_rk4(pts[i*2], pts[i*2+1], numsteps);
    }
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed_seconds = end - start;

    std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
    save_svg(pts, "outAdvectedBNOT.svg", w, h);
 
    
}

