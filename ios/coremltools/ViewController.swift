//
//  ViewController.swift
//  coremltools
//
//  Created by Shabao Ma on 2019/02/01.
//  Copyright © 2019年 Shabao Ma. All rights reserved.
//

import UIKit
import CoreML

class ViewController: UIViewController {

    @IBOutlet var imageView: UIImageView!
    let model = core_model()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        
        let openCVWrapper = OpenCVWrapper()
        let inputImage = UIImage(named: "2019-02-05 22.57.08.jpg")
        let outputArray = openCVWrapper.preprocessInput(inputImage!)
        let prediction = try? model.prediction(photo: outputArray)
        for i in 0..<prediction!.screen.count {
            let value = prediction?.screen[i]
            if value!.doubleValue > 0.95 {
                prediction!.screen[i] = 255.0
            } else {
                prediction!.screen[i] = 0.0
            }
        }
        let outputImage = prediction?.screen.image()
        let predGameScreen = openCVWrapper.findScreenConer(inputImage!, imageY: outputImage!)
        imageView.image = predGameScreen
    }
}
