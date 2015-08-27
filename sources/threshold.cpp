#include "detector.hpp"

using namespace cv;

class Threshold : public Detector {
public:
    Threshold() {};
    
    void init( Mat &src )
    {
        image = src;
    }

    int count()
    {
        int thresh = 100;
        int max_thresh = 255;
        RNG rng(12345);
        int counter = 0;

        Mat gray_image;
        imshow( " ", image );
        int size_height = image.size().height;
        int size_width = image.size().width;
        if (size_height > size_width )
        {
            cv::transpose(image, image);
            cv::flip(image, image, 1);
            size_height = image.size().height;
            size_width = image.size().width;
        }

        if (size_width > 640 )
        {
            Mat scaled_image;
            resize( image, scaled_image, Size(640,480) );
            cvtColor( scaled_image, gray_image, CV_BGR2GRAY );
        }
        else
            cvtColor( image, gray_image, CV_BGR2GRAY );
        /// Reduce the noise so we avoid false circle detection
        GaussianBlur( gray_image, gray_image, Size(9, 9), 2, 2 );
        Mat kernel;
        kernel = Mat::ones( 6, 6, CV_8UC1);
        kernel = kernel * 255;
        morphologyEx(gray_image, gray_image, MORPH_OPEN, kernel);
        Mat threshold_output;
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;

        /// Detect edges using Threshold
        threshold( gray_image, threshold_output, thresh, 255, THRESH_OTSU);

        /// Filling empty spots with Morph 

        imshow("tre", threshold_output);

        /// Find contours
        findContours( threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

        /// Find the rotated rectangles and ellipses for each contour
        vector<RotatedRect> minRect( contours.size() );
        vector<RotatedRect> minEllipse( contours.size() );
        for( int i = 0; i < contours.size(); i++ )
        {     
            minRect[i] = minAreaRect( Mat(contours[i]));
            if ( contours[i].size() > 5 )
            { 
                double area = cv::contourArea( contours[i]);
                if ( ( area < 57600 ) && ( area > 300 ) )
                {
                      minEllipse[i] = fitEllipse( Mat(contours[i]) );
                      counter++ ;
                }
            }
        }

        /// Draw contours + rotated rects + ellipses
        Mat drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );
        for( int i = 0; i< contours.size(); i++ )
        {
             Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
             // contour
             drawContours( drawing, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
              // ellipse
            ellipse( drawing, minEllipse[i], color, 2, 8 );
        }
        imshow("Threshold", drawing  );
        std::cout<< "number of coins:\t" << counter << std::endl;
        waitKey(0);
        return counter; 
    }

    void draw()
    {}

private:
    vector<Vec3f> circles;
    Mat image;
};

cv::Ptr<Detector> createDetectorThreshold() 
{
    return cv::Ptr<Detector>(new Threshold());
}