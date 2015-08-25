#include "detector.hpp"

using namespace cv;

class Hough : public Detector {
public:
    Hough() {};
    
    void init( Mat &src )
    {
        image = src;
    }

    int count()
    {
        Mat gray_image;
        cvtColor( image, gray_image, CV_BGR2GRAY );
        /// Reduce the noise so we avoid false circle detection
        GaussianBlur( gray_image, gray_image, Size(9, 9), 2, 2 );
        /// Apply the Hough Transform to find the circles
        HoughCircles( 
            gray_image, // image in gray
            circles, // storage
            CV_HOUGH_GRADIENT, // method - ?
            1, // depths
            gray_image.rows/8 // minimal distance between objects
            ); 
        return circles.size();
    }

    void draw()
    {
        /// Draw the circles detected
        for( size_t i = 0; i < circles.size(); i++ )
        {
            Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
            int radius = cvRound(circles[i][2]);
            // circle center
            circle( image, center, 3, Scalar(0,255,0), -1, 8, 0 );
            // circle outline
            circle( image, center, radius, Scalar(0,0,255), 3, 8, 0 );
        }
        std::cout<<"number of coins:\t"<<circles.size()<<std::endl;
        /// Show your results
        imshow( "Hough Circle Transform", image );
        waitKey(0);
    }

private:
    vector<Vec3f> circles;
    Mat image;
};

cv::Ptr<Detector> createDetectorHough() 
{
    return cv::Ptr<Detector>(new Hough());
}