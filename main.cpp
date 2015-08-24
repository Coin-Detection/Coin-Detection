#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace  cv;

static const char* keys =
{
    "{ |  image   |         |  Source image }"
    "{ |  method  |  Hough  |  A method to work with }"
};

/** @function main */
int main(int argc, char** argv)
{
    cv::CommandLineParser parser( argc, argv, keys );
    // Parse and validate input parameters
    std::string image_file = parser.get<std::string>( "image" );
    std::string method = parser.get<std::string>( "method" );

    Mat src, src_gray;
    
    /// Read the image
    src = imread( image_file );

  if( !src.data )
    { return -1; }

  /// Convert it to gray
  cvtColor( src, src_gray, CV_BGR2GRAY );

  /// Reduce the noise so we avoid false circle detection
  GaussianBlur( src_gray, src_gray, Size(9, 9), 2, 2);

  vector<Vec3f> circles;

  /// Apply the Hough Transform to find the circles
  HoughCircles( 
      src_gray, // image in gray
      circles, // storage
      CV_HOUGH_GRADIENT, // method - ?
      1, // depths
      src_gray.rows/8 // minimal distance between objects
      );

  /// Draw the circles detected
  for( size_t i = 0; i < circles.size(); i++ )
  {
      Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
      int radius = cvRound(circles[i][2]);
      // circle center
      circle( src, center, 3, Scalar(0,255,0), -1, 8, 0 );
      // circle outline
      circle( src, center, radius, Scalar(0,0,255), 3, 8, 0 );
   }
    
    std::cout<<"number of coins:\t"<<circles.size()<<std::endl;
    Canny( src_gray, src_gray, 0, 50 );
  /// Show your results
  imshow( "Hough Circle Transform Demo", src );
  imshow( "Gray scale", src_gray ); 

  waitKey(0);
  return 0;
}
