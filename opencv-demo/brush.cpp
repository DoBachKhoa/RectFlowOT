#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/ml/ml.hpp>
#include<iostream>

#include "../flow.h"
#include "lutLDBN.h"


using namespace std;
int radius =40;
int radius_erase =50;

cv::Mat img;
int ix = 1;
int iy = 1;

cv::Mat stippling;

bool hasChanged = false;

bool draw=false;

bool rightclick = false;


double bump(int i, int j ,int x ,int y, int radius)
{
  if ((i-x)*(i-x) + (j-y)*(j-y) >= radius*radius) return 0.0;
  double t= 1.0*((i-x)*(i-x) + (j-y)*(j-y)) / ((double)radius*radius);
  return std::pow(1-t,6)*(4*t+1);
}
void splat( int x, int y, int radius)
{
  int di =std::max(0,x-radius);
  int ddi =std::min(x+radius, img.cols);
  int dj =  std::max(0,y-radius);
  int ddj = std::min(y+radius, img.rows);
  
  for (auto i =  di; i <ddi ; ++i)
  {
    for (auto j = dj; j <ddj; ++j)
    {
      img.at<uchar>(j, i) = cv::saturate_cast<uchar>(img.at<uchar>(j, i)   - 64*bump(i,j,x,y,radius));
    }
  }
}

//mouse callback function
void drawCircle(int event, int x, int y, int, void* param)
{
  if ((event == cv::EVENT_LBUTTONDOWN) || (event == cv::EVENT_RBUTTONDOWN))
  {
    draw = true;
    if (event == cv::EVENT_RBUTTONDOWN)
      rightclick = true;
  }
  else
    if (event == cv::EVENT_MOUSEMOVE)
    {
      if (draw)
      {
        if (rightclick)
            cv::circle(img, cv::Point(x, y), radius_erase, cv::Scalar(255,255,255),cv::FILLED);
          else
            splat(x,y,radius);
        //cv::circle(img, cv::Point(x, y), radius, cv::Scalar(0,0,0),cv::FILLED);
        
        hasChanged = true;
      }
    }
  if ((event == cv::EVENT_LBUTTONUP) || (event == cv::EVENT_RBUTTONUP))
  {
    draw = false;
    hasChanged = true;
    rightclick = false;
  }
  
}


void drawPts(std::vector<Point> &samples)
{
  for(auto i=0; i < samples.size();++i)
    cv::circle(stippling, cv::Point(samples[i][0], samples[i][1]), 2, cv::Scalar(0,0,0),cv::FILLED);
  
  cv::putText(stippling, //target image
              std::string("N = ")+std::to_string(samples.size()), //text
              cv::Point(10, 40), //top-left position
              cv::FONT_HERSHEY_DUPLEX,
              1.0,
              CV_RGB(118, 185, 0), //font color
              2);
}

void copyToDouble(cv::Mat &img_float, double *dest)
{
  // Convert each element from float to double and copy it to the double array
  for (int i = 0; i < img_float.rows; i++) {
    for (int j = 0; j < img_float.cols; j++) {
      // Access the float value and convert it to double
      dest[i * img_float.cols + j] =255- static_cast<double>(img_float.at<uchar>(i, j));
    }
  }
}



int main() {
  
  img = cv::imread("../sig2025.png",cv::IMREAD_GRAYSCALE);
  
  if (img.empty()) {
    cout << "\nerror reading image" << endl;
    return -1;
  }
  
  cv::namedWindow("stippling",1);
  stippling = cv::Mat(img.rows, img.cols, CV_8UC3);
  stippling = cv::Scalar(255,255,255);

  //img = cv::Scalar(0);
  cv::namedWindow("Image",1);
  cv::imshow("Image", img);
  
  cv::setMouseCallback("Image", drawCircle);
  
  // Get the type of the cv::Mat
  int type = img.type();
  
  // Decode the type
  int depth = CV_MAT_DEPTH(type);
  int channels = CV_MAT_CN(type);
  
  std::cout << "Type: " << type << std::endl;
  std::cout << "Depth: " << depth << std::endl;
  std::cout << "Channels: " << channels << std::endl;
      
  //Sampling
  auto NBPTS = 1000;
  std::vector<Point> samples;
  initSamplers();
  
  ldbnBNOT(NBPTS, samples);
  for(auto &s : samples)
  {
    s[0] *= img.cols;
    s[1] *= img.rows;
  }
  
  drawPts(samples);
  cv::imshow("stippling", stippling);
  
  //Flow
  Flow flow(img.cols, img.rows);
  double* density = new double[img.cols*img.rows];
  copyToDouble(img, density);
  flow.initialize(density);
  
  int key;
  while ((key = cv::waitKey(20)) != 27)  // wait until ESC is pressed
  {
    
    cv::imshow("Image", img);
    if (key == 32)
      hasChanged = true;
    
    if (key == 'u')
    {
      NBPTS +=100;
      std::cout<< NBPTS << std::endl;
      hasChanged = true;
    }
    
    
    if (hasChanged)
    {
      ldbnBNOT(NBPTS, samples);
#pragma omp for schedule(static)
      for(auto &s : samples)
      {
        s[0] *= img.cols;
        s[1] *= img.rows;
      }      
      copyToDouble(img, density);
      flow.initialize(density);
      
#pragma omp for schedule(static)
      for(auto i = 0 ; i < samples.size(); ++i)
      {
        flow.integrate_rk4(samples[i][0], samples[i][1], 50);
      }
      
      stippling = cv::Scalar(255,255,255);
      drawPts(samples);
      cv::imshow("stippling", stippling);
      hasChanged = false;
    }
  }
  
  delete[] density;
  return 0;
}
