#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <stdio.h>
#include "detector.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2\opencv.hpp"

using namespace cv;

static const char* keys =
{
    "{ |  image   |         |  Source image }"
    "{ |  method  |  Hough  |  A method to work with }"
    "{ |  camera  |  false  |  Option for camera}"
    "{ |  strike  |  false  |  Option for image but not camera}"
};

int main(int argc, char** argv)
{
    cv::CommandLineParser parser( argc, argv, keys );
    // Parse and validate input parameters
    std::string image_file = parser.get<std::string>( "image" );
    std::string method_name = parser.get<std::string>( "method" );
    bool camera = parser.get<bool>( "camera" );
    bool strike = parser.get<bool>( "strike" );
    Mat src, dst;

    /// Read the image
    if (!image_file.empty()) {
        src = imread( image_file );
        CV_Assert( !src.empty() );
    }
    else if (camera) {
        VideoCapture cap(0);
        Mat image_obj;
        std::string image_obj_path;
        std::cout<<"Enter path for object to find"<<std::endl;
        std::cin>>image_obj_path;
        image_obj=imread(image_obj_path);
        cvtColor(image_obj, image_obj, CV_BGR2GRAY);
        GaussianBlur(image_obj, image_obj, Size(5, 5), 2, 2);
        Mat frame;
        vector<Mat> corners;
        for(;;)
        {
            cap >> frame; // Получить очередной фрейм из камеры
            Mat img_scene = frame; 
            if( !image_obj.data || !img_scene.data ) // Проверка наличия информации в матрице изображения
                std::cout<<"Reading mistake, image_obj or image_scene, No Data"<<std::endl;

            resize(img_scene, img_scene, img_scene.size(), 0.5, 0.5);
            //-- Этап 1. Нахождение ключевых точек.
            int minHessian = 400;

            SURF detector( minHessian );

            std::vector<KeyPoint> keypoints_object, keypoints_scene;

            detector.detect( image_obj, keypoints_object );
            detector.detect( img_scene, keypoints_scene );

            CV_Assert(!keypoints_object.empty());
            CV_Assert(!keypoints_scene.empty());


            //-- Этап 2. Вычисление дескрипторов.
            SURF extractor;

            Mat descriptors_object, descriptors_scene;

            extractor.compute( image_obj, keypoints_object, descriptors_object );
            extractor.compute( img_scene, keypoints_scene, descriptors_scene );

            CV_Assert(!descriptors_object.empty());
            CV_Assert(!descriptors_scene.empty());


            //-- Этап 3: Необходимо сматчить вектора дескрипторов.
            FlannBasedMatcher matcher;
            vector< DMatch > matches;
            matcher.match( descriptors_object, descriptors_scene, matches );

            //-- Вычисление максимального и минимального расстояния среди всех дескрипторов
                        // в пространстве признаков
            double max_dist = 0; double min_dist = 100;

            for( int i = 0; i < descriptors_object.rows; i++ )
            { 
                double dist = matches[i].distance;
                if( dist < min_dist ) min_dist = dist;
                if( dist > max_dist ) max_dist = dist;
            }

            printf("-- Max dist : %f \n", max_dist );
            printf("-- Min dist : %f \n", min_dist );


            //-- Отобрать только хорошие матчи, расстояние меньше чем 3 * min_dist
            vector< DMatch > good_matches;

            for( int i = 0; i < descriptors_object.rows; i++ )
            { 
                if( matches[i].distance < 3 * min_dist )
                { 
                    good_matches.push_back( matches[i]); 
                }
            }  

            Mat img_matches;
    

            //-- Нарисовать хорошие матчи
            drawMatches( image_obj, keypoints_object, img_scene, keypoints_scene, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS ); 
   
            //-- Локализация объектов
            vector<Point2f> obj;
            vector<Point2f> scene;

            for( int i = 0; i < good_matches.size(); i++ )
            {
                obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
                scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt ); 
            }

            Mat H = findHomography( obj, scene, CV_RANSAC );

            
            //-- Получить "углы" изображения с целевым объектом
            vector<Point2f> obj_corners(4);
            obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( image_obj.cols, 0 );
            obj_corners[2] = cvPoint( image_obj.cols, image_obj.rows ); obj_corners[3] = cvPoint( 0, image_obj.rows );

            //-- Отобразить углы целевого объекта, используя найденное преобразование, на сцену
            vector<Point2f> scene_corners(4);
            perspectiveTransform( obj_corners, scene_corners, H);


            //-- Соеденить отображенные углы
            line( img_matches, scene_corners[0] + Point2f( image_obj.cols, 0), scene_corners[1] + Point2f( image_obj.cols, 0), Scalar(0, 255, 0), 4 );
            line( img_matches, scene_corners[1] + Point2f( image_obj.cols, 0), scene_corners[2] + Point2f( image_obj.cols, 0), Scalar( 0, 255, 0), 4 );
            line( img_matches, scene_corners[2] + Point2f( image_obj.cols, 0), scene_corners[3] + Point2f( image_obj.cols, 0), Scalar( 0, 255, 0), 4 );
            line( img_matches, scene_corners[3] + Point2f( image_obj.cols, 0), scene_corners[0] + Point2f( image_obj.cols, 0), Scalar( 0, 255, 0), 4 );


            //-- Show detected matches
            imshow( "Good Matches & Object detection", img_matches );

            if(waitKey(30) >= 0) break;
        }
    }
    else if (strike) {
        Mat image_obj;
        std::string image_obj_path;
        std::cout<<"Enter path for object to find"<<std::endl;
        std::cin>>image_obj_path;
        image_obj=imread(image_obj_path);
        cvtColor(image_obj, image_obj, CV_BGR2GRAY);
        GaussianBlur(image_obj, image_obj, Size(5, 5), 2, 2);
        Mat frame;
        frame=imread("rubl.jpg");
        Mat img_scene = frame; 
        if( !image_obj.data || !img_scene.data ) // Проверка наличия информации в матрице изображения
        std::cout<<"Reading mistake, image_obj or image_scene Data"<<std::endl;

        cvtColor(img_scene, img_scene, CV_BGR2GRAY);
        GaussianBlur(img_scene, img_scene, Size(5, 5), 2, 2);

        resize(image_obj, image_obj, image_obj.size(), 0.15, 0.15);
        resize(img_scene, img_scene, img_scene.size(), 0.15, 0.15);


        //-- Этап 1. Нахождение ключевых точек.
        int minHessian = 400;

        SURF detector( minHessian );

        std::vector<KeyPoint> keypoints_object, keypoints_scene;

        detector.detect( image_obj, keypoints_object );
        detector.detect( img_scene, keypoints_scene );

        CV_Assert(!keypoints_object.empty());
        CV_Assert(!keypoints_scene.empty());


        //-- Этап 2. Вычисление дескрипторов.
        SURF extractor;

        Mat descriptors_object, descriptors_scene;

        extractor.compute( image_obj, keypoints_object, descriptors_object );
        extractor.compute( img_scene, keypoints_scene, descriptors_scene );

        CV_Assert(!descriptors_object.empty());
        CV_Assert(!descriptors_scene.empty());


        //-- Этап 3: Необходимо сматчить вектора дескрипторов.
        FlannBasedMatcher matcher;
        vector< DMatch > matches;
        matcher.match( descriptors_object, descriptors_scene, matches );

        double max_dist = 0; double min_dist = 100;


        //-- Вычисление максимального и минимального расстояния среди всех дескрипторов
        // в пространстве признаков
        for( int i = 0; i < descriptors_object.rows; i++ )
        { 
            double dist = matches[i].distance;
            if( dist < min_dist ) min_dist = dist;
            if( dist > max_dist ) max_dist = dist;
        }

        printf("-- Max dist : %f \n", max_dist );
        printf("-- Min dist : %f \n", min_dist );


        //-- Отобрать только хорошие матчи, расстояние меньше чем 3 * min_dist
        vector< DMatch > good_matches;

        for( int i = 0; i < descriptors_object.rows; i++ )
        { 
            if( matches[i].distance < 3 * min_dist )
            { 
                good_matches.push_back( matches[i]); 
            }
        }      

        //-- Нарисовать хорошие матчи
        Mat img_matches;
        drawMatches( image_obj, keypoints_object, img_scene, keypoints_scene, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS); 


        //-- Локализация объектов
        vector<Point2f> obj;
        vector<Point2f> scene;

        for( int i = 0; i < good_matches.size(); i++ )
        {
            obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
            scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt ); 
        }

        Mat H = findHomography( obj, scene, CV_RANSAC );

            
        //-- Получить "углы" изображения с целевым объектом
        std::vector<Point2f> obj_corners(4);
        obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( image_obj.cols, 0 );
        obj_corners[2] = cvPoint( image_obj.cols, image_obj.rows ); obj_corners[3] = cvPoint( 0, image_obj.rows );


        //-- Отобразить углы целевого объекта, используя найденное преобразование, на сцену
        std::vector<Point2f> scene_corners(4);
        perspectiveTransform( obj_corners, scene_corners, H);


        //-- Соеденить отображенные углы
        line( img_matches, scene_corners[0] + Point2f( image_obj.cols, 0), scene_corners[1] + Point2f( image_obj.cols, 0), Scalar(0, 255, 0), 4 );
        line( img_matches, scene_corners[1] + Point2f( image_obj.cols, 0), scene_corners[2] + Point2f( image_obj.cols, 0), Scalar( 0, 255, 0), 4 );
        line( img_matches, scene_corners[2] + Point2f( image_obj.cols, 0), scene_corners[3] + Point2f( image_obj.cols, 0), Scalar( 0, 255, 0), 4 );
        line( img_matches, scene_corners[3] + Point2f( image_obj.cols, 0), scene_corners[0] + Point2f( image_obj.cols, 0), Scalar( 0, 255, 0), 4 );


        //-- Show detected matches
        imshow( "Good Matches & Object detection", img_matches );
        waitKey();
    }

    /*
    Ptr<Detector> detector(detectorCreation(method_name));
    detector->init(src);
    detector->count();
    detector->draw();
    */
    return 0;
}