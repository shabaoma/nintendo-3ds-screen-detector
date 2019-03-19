//
//  OpenCVWrapper.h
//  coremltools
//
//  Created by 馬 沙暴 on 2019/02/12.
//  Copyright © 2019年 Shabao Ma. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#import <CoreML/CoreML.h>

NS_ASSUME_NONNULL_BEGIN

@interface OpenCVWrapper : NSObject
- (MLMultiArray *)preprocessInput:(UIImage *)inputImage;
- (UIImage *)findScreenConer:(UIImage *)imageX imageY:(UIImage *)imageY;
@end

NS_ASSUME_NONNULL_END
