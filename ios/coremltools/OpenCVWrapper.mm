//
//  OpenCVWrapper.m
//  coremltools
//
//  Created by Shabao Ma on 2019/02/12.
//  Copyright © 2019年 Shabao Ma. All rights reserved.
//

#import <opencv2/opencv.hpp>
#import "OpenCVWrapper.h"
#import <CoreML/CoreML.h>
#include "opencv2/imgcodecs/ios.h"

@implementation OpenCVWrapper

- (MLMultiArray *)preprocessInput:(UIImage *)inputImage {
    cv::Mat imageArray;
    UIImageToMat(inputImage, imageArray);
    cv::cvtColor(imageArray, imageArray, cv::COLOR_RGBA2RGB);
    // NSLog(@"%d", imageArray.channels());
    if (imageArray.channels() != 3) {
        // std::cout << "imageArray =" << imageArray.rowRange(0, 1) << std::endl << std::endl;
        return nil;
    }
    
    cv::resize(imageArray, imageArray, cv::Size(256, 256), 0, 0, cv::INTER_NEAREST);
    cv::cvtColor(imageArray, imageArray, cv::COLOR_RGB2BGR);
    imageArray.convertTo(imageArray, CV_64F);
    imageArray = imageArray - cv::Scalar(103.939, 116.779, 123.68); // important
    
    NSArray *shape = @[@3, @256, @256];
    MLMultiArrayDataType dataType = MLMultiArrayDataTypeDouble;
    NSError *error = nil;
    MLMultiArray *outputArray =  [[MLMultiArray alloc] initWithShape:(NSArray*)shape
                                                            dataType:(MLMultiArrayDataType)dataType
                                                               error:&error];
    
    for (int z=0; z<3; z++) {
        for (int i=0; i<256; i++) {
            for (int j=0; j<256; j++) {
                double value = imageArray.at<double>(i,3*j+z);
                NSArray *index = [NSArray arrayWithObjects:
                                  [NSNumber numberWithInteger: z],
                                  [NSNumber numberWithInteger: i],
                                  [NSNumber numberWithInteger: j], nil];
                [outputArray setObject:[NSNumber numberWithDouble:value] forKeyedSubscript: index];
            }
        }
    }
    
    return outputArray;
}

- (UIImage *)findScreenConer:(UIImage *)imageX imageY:(UIImage *)imageY {
    cv::Mat imageArray;
    cv::Mat binaryArray;
    UIImageToMat(imageX, imageArray); // 3024 x 4032
    UIImageToMat(imageY, binaryArray); // 256 x 256
    // NSLog(@"%d", imageArray.channels());
    // NSLog(@"%d", binaryArray.channels());
    
    if (binaryArray.channels() != 1) {
        // std::cout << "binaryArray=" << binaryArray << std::endl << std::endl;
        return nil;
    }
    
    // value <= 242.25 -> 0, value > 242.25 -> 255
    cv::threshold(binaryArray, binaryArray, 242.25, 255, cv::THRESH_BINARY);
    
    // contours
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binaryArray, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    
    int maxIndex = 0;
    double maxArea = 0;
    for (int i=0; i<contours.size(); i++) {
        std::vector<cv::Point> closeContour = contours[i];
        double area = cv::contourArea(closeContour);
        if (area > maxArea) {
            maxIndex = i;
            maxArea = area;
        }
    }
    
    int imageSize = 256;
    std::vector<cv::Point> maxAreaContours = contours[maxIndex];
    cv::Point ul = [self findNearestPoint:maxAreaContours cornerPoint:cv::Point {0, 0}];
    cv::Point ur = [self findNearestPoint:maxAreaContours cornerPoint:cv::Point {imageSize, 0}];
    cv::Point lr = [self findNearestPoint:maxAreaContours cornerPoint:cv::Point {imageSize, imageSize}];
    cv::Point ll = [self findNearestPoint:maxAreaContours cornerPoint:cv::Point {0, imageSize}];
    std::vector<cv::Point> cornerPoints = {ul, ur, lr, ll};
    cv::polylines(imageArray, cornerPoints, true, cv::Scalar(255), 15);
    
    UIImage* outputImage = MatToUIImage(imageArray);
    return outputImage;
}

- (cv::Point)findNearestPoint:(std::vector<cv::Point>)maxAreaContours cornerPoint:(cv::Point)cornerPoint {
    
    int minIndex = 0;
    double minDistance = INFINITY;
    for (int i=0; i<maxAreaContours.size(); i++) {
        cv::Point p1 = maxAreaContours[i];
        cv::Point p2 = cornerPoint;
        double distance = [self calcDistance:p1 p2:p2];
        if (distance < minDistance) {
            minIndex = i;
            minDistance = distance;
        }
    }
    
    cv::Point zoomedPoint = maxAreaContours[minIndex];
    zoomedPoint.x = zoomedPoint.x * (4032.0 / 256.0);
    zoomedPoint.y = zoomedPoint.y * (3024.0 / 256.0);
    return zoomedPoint;
}

- (double)calcDistance:(cv::Point)p1 p2:(cv::Point)p2 {
    return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y);
}

@end
