/// Copyright (c) 2019 Razeware LLC
///
/// Permission is hereby granted, free of charge, to any person obtaining a copy
/// of this software and associated documentation files (the "Software"), to deal
/// in the Software without restriction, including without limitation the rights
/// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
/// copies of the Software, and to permit persons to whom the Software is
/// furnished to do so, subject to the following conditions:
///
/// The above copyright notice and this permission notice shall be included in
/// all copies or substantial portions of the Software.
///
/// Notwithstanding the foregoing, you may not use, copy, modify, merge, publish,
/// distribute, sublicense, create a derivative work, and/or sell copies of the
/// Software in any work that is designed, intended, or marketed for pedagogical or
/// instructional purposes related to programming, coding, application development,
/// or information technology.  Permission for such use, copying, modification,
/// merger, publication, distribution, sublicensing, creation of derivative works,
/// or sale is expressly withheld.
///
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
/// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
/// THE SOFTWARE.

import AVFoundation
import UIKit
import Vision

class FaceDetectionViewController: UIViewController {
  var sequenceHandler = VNSequenceRequestHandler()

  @IBOutlet var faceView: FaceView!
  @IBOutlet var faceLaserLabel: UILabel!
  @IBOutlet weak var blinkCounterLabel: UILabel!
  @IBOutlet weak var detectionCounterLabel: UILabel!
  @IBOutlet weak var statusLabel: UILabel!
  
  var blinkCounter: Int = 0
  
  // MARK: - Variables for Detection Frame Counter
  var detectionCounterPerSecond: Int = 0
  var detectionCounterMax: Int = 0
  var detectionBaseTime: Double = 0
  var detectionLastTime: Double = 0
  var detectionCurrentTime: Double = 0
  var detectionOnGoing: Bool = false
  
  var timeBeforeRequest: Double = 0
  
  var blinkTimePrevious: Double = -1
  var blinkTimeCurrent: Double = 0
  
  var idle: Double = 0
  let idle_upperbound: Double = 1.0
  let idle_lowerbound: Double = 0.3
  
  var x_est: Double = 4.0
  var x_sam: Double = 0
  var x_var: Double = 0
  let alpha: Double = 1.0 / 4
  let beta: Double = 0.25
  let K: Double = 0.25
  var alertBaseTime: Double = 0
  
  let session = AVCaptureSession()
  var previewLayer: AVCaptureVideoPreviewLayer!
  
  let dataOutputQueue = DispatchQueue(
    label: "video data queue",
    qos: .userInitiated,
    attributes: [],
    autoreleaseFrequency: .workItem)

  var faceViewHidden = false
  
  var maxX: CGFloat = 0.0
  var midY: CGFloat = 0.0
  var maxY: CGFloat = 0.0

  override func viewDidLoad() {
    super.viewDidLoad()
    configureCaptureSession()
    
    maxX = view.bounds.maxX
    midY = view.bounds.midY
    maxY = view.bounds.maxY
    
    session.startRunning()
  }
}

extension FaceDetectionViewController {
  @IBAction func handleTap(_ sender: UITapGestureRecognizer) {
    faceView.isHidden.toggle()
    faceViewHidden = faceView.isHidden
    
    if faceViewHidden {
      faceLaserLabel.text = "Show"
    } else {
      faceLaserLabel.text = "Hide"
    }
  }
}

// MARK: - Video Processing methods

extension FaceDetectionViewController {
  // MARK: - EAR Methods
  
  func configureCaptureSession() {
    // Define the capture device we want to use
    guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera,
                                               for: .video,
                                               position: .front) else {
      fatalError("No front video camera available")
    }
    
    // Connect the camera to the capture session input
    do {
      let cameraInput = try AVCaptureDeviceInput(device: camera)
      session.addInput(cameraInput)
    } catch {
      fatalError(error.localizedDescription)
    }
    
    // Create the video data output
    let videoOutput = AVCaptureVideoDataOutput()
    videoOutput.setSampleBufferDelegate(self, queue: dataOutputQueue)
    videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
    
    // Add the video output to the capture session
    session.addOutput(videoOutput)
    
    let videoConnection = videoOutput.connection(with: .video)
    videoConnection?.videoOrientation = .portrait
    
    // Configure the preview layer
    previewLayer = AVCaptureVideoPreviewLayer(session: session)
    previewLayer.videoGravity = .resizeAspectFill
    previewLayer.frame = view.bounds
    view.layer.insertSublayer(previewLayer, at: 0)
  }
}



// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate methods
extension FaceDetectionViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
  func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
    
    let date = Date()
    
    timeBeforeRequest = date.timeIntervalSince1970
    
    if blinkTimePrevious != -1 && alertBaseTime + 10 < timeBeforeRequest {
      var msg = ""
      if x_est > 4.5 {
        msg = "Blink!"
      }
      else {
        msg = "Good"
      }
      DispatchQueue.global(qos: .background).async {
        DispatchQueue.main.async {
          self.statusLabel.text = msg
        }
      }
      alertBaseTime = timeBeforeRequest
    }
    
    // optimization
    if blinkTimeCurrent + idle > timeBeforeRequest {
      return // skip the request -> energy saving
    }
    
    
    
    // 1
    guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
      return
    }

    // 2
    let detectFaceRequest = VNDetectFaceLandmarksRequest(completionHandler: detectedFace)

    // 3
    do {
      try sequenceHandler.perform(
        [detectFaceRequest],
        on: imageBuffer,
        orientation: .leftMirrored)
    } catch {
      print(error.localizedDescription)
    }
  }
}

extension FaceDetectionViewController {
  func convert(rect: CGRect) -> CGRect {
    // 1
    let origin = previewLayer.layerPointConverted(fromCaptureDevicePoint: rect.origin)

    // 2
    let size = previewLayer.layerPointConverted(fromCaptureDevicePoint: rect.size.cgPoint)
    
    // 3
    return CGRect(origin: origin, size: size.cgSize)
  }

  // 1
  func landmark(point: CGPoint, to rect: CGRect) -> CGPoint {
    // 2
    let absolute = point.absolutePoint(in: rect)

    // 3
    let converted = previewLayer.layerPointConverted(fromCaptureDevicePoint: absolute)

    // 4
    return converted
  }

  func landmark(points: [CGPoint]?, to rect: CGRect) -> [CGPoint]? {
    guard let points = points else {
      return nil
    }

    return points.compactMap { landmark(point: $0, to: rect) }
  }
  
  // EyeClosed
  static var isEyeClosed: Bool = false
  
  func updateFaceView(for result: VNFaceObservation) {
    
    defer {
      DispatchQueue.main.async {
        self.faceView.setNeedsDisplay()
      }
    }

    let box = result.boundingBox
    faceView.boundingBox = convert(rect: box)

    guard let landmarks = result.landmarks else {
      return
    }
    
    // MARK: - EAR methods
    func CGPointDistanceSquared(from: CGPoint, to: CGPoint) -> CGFloat {
        return (from.x - to.x) * (from.x - to.x) + (from.y - to.y) * (from.y - to.y)
    }

    func CGPointDistance(from: CGPoint, to: CGPoint) -> CGFloat {
        return sqrt(CGPointDistanceSquared(from: from, to: to))
    }
    
    // MARK: - Detection Per Second
    
    // MARK: - EAR calculation
    var leftEyeNormalizedPoints : [CGPoint]?
    var rightEyeNormalizedPoints : [CGPoint]?
    var leftFlag = false
    var rightFlag = false
    
    func getEARLeft(eyePoints: [CGPoint]) -> CGFloat {
      let p1 = eyePoints[1]
      let p2 = eyePoints[5]
      let p3 = eyePoints[4]
      let p4 = eyePoints[0]
      let p5 = eyePoints[2]
      let p6 = eyePoints[3]
      return
        (CGPointDistance(from: p2, to: p6) + CGPointDistance(from: p3, to: p5)) /
        (CGPointDistance(from: p1, to: p4))
    }
    
    func getEARRight(eyePoints: [CGPoint]) -> CGFloat {
      let p1 = eyePoints[1]
      let p2 = eyePoints[5]
      let p3 = eyePoints[4]
      let p4 = eyePoints[0]
      let p5 = eyePoints[2]
      let p6 = eyePoints[3]
      return
        (CGPointDistance(from: p2, to: p6) + CGPointDistance(from: p3, to: p5)) /
        (CGPointDistance(from: p1, to: p4))
    }
   
    if let leftEye = landmark(
      points: landmarks.leftEye?.normalizedPoints,
      to: result.boundingBox) {
      leftEyeNormalizedPoints = landmarks.leftEye?.normalizedPoints
      leftFlag = true
      faceView.leftEye = leftEye
    }


    if let rightEye = landmark(
      points: landmarks.rightEye?.normalizedPoints,
      to: result.boundingBox) {
      rightEyeNormalizedPoints = landmarks.rightEye?.normalizedPoints
      rightFlag = true
      faceView.rightEye = rightEye
    }
    

    if (leftFlag && rightFlag) {
      
      
      // update detection per second
      
      if detectionOnGoing == false {
        detectionOnGoing = true
        detectionBaseTime = detectionCurrentTime
        detectionCounterPerSecond = 1
      }
      else {
        
        detectionCounterPerSecond += 1
        if detectionBaseTime + 1 < detectionCurrentTime {
          detectionBaseTime = detectionCurrentTime
          detectionCounterMax = max(detectionCounterPerSecond, detectionCounterMax)
          detectionCounterPerSecond = 1
        }
      }
      
      DispatchQueue.global(qos: .background).async {
        DispatchQueue.main.async {
          self.detectionCounterLabel.text = "\(self.detectionCounterMax)"
        }
      }

      detectionLastTime = detectionCurrentTime
      
      // 1초가 지났으면, 카운터 초기화 후 라벨 업데이트
      
      // if not, increment coutner
      
      let EARLeft = getEARLeft(eyePoints: leftEyeNormalizedPoints!)
      let EARRight = getEARRight(eyePoints: rightEyeNormalizedPoints!)
      let EAR = (EARLeft + EARRight) / 2
      if (EAR < 2.75) {
        if (FaceDetectionViewController.isEyeClosed == false) {
          FaceDetectionViewController.isEyeClosed = true
//        print(EAR)
          DispatchQueue.global(qos: .background).async {
              DispatchQueue.main.async {
                self.blinkCounter += 1
                self.blinkCounterLabel.text = "\(self.blinkCounter)"
              }
          }
        }
        
        // optimization
        blinkTimeCurrent = detectionCurrentTime
        if blinkTimePrevious == -1 {
          blinkTimePrevious = blinkTimeCurrent - 4
        }
        x_sam = blinkTimeCurrent - blinkTimePrevious
        blinkTimePrevious = blinkTimeCurrent
        print("samping: \(x_sam)")
        x_est = alpha * x_sam + (1 - alpha) * x_est
        x_var = beta * abs(x_sam - x_est) + (1 - beta) * x_var
        print("estimation: \(x_est)")
        print("variance: \(x_var)")
        let blinkPerSecond = 1 / x_est
        idle = max(min(K / (blinkPerSecond + 0.02 * x_var), idle_upperbound), idle_lowerbound)
        print("idle: \(idle)")
        
      }
      else {
        if (FaceDetectionViewController.isEyeClosed == true) {
          FaceDetectionViewController.isEyeClosed = false
          
          
        }
      }
    }
    
  }
  
  func detectedFace(request: VNRequest, error: Error?) {
    detectionCurrentTime = Date().timeIntervalSince1970
    guard
      let results = request.results as? [VNFaceObservation],
      let result = results.first
      else {
      
        // detection time out
        if detectionOnGoing == false || detectionLastTime + 0.4 < detectionCurrentTime {
          
          // optimization
          idle = 0 // keep trying to detect face.
          blinkTimePrevious = -1
          blinkTimeCurrent = 0
          x_est = 4.0
          x_var = 0.0
          
          
          detectionCounterPerSecond = 0
          detectionOnGoing = false
          DispatchQueue.global(qos: .background).async {
              DispatchQueue.main.async {
                self.detectionCounterLabel.text = "-"
              }
          }
        }
          faceView.clear()
          return
      }
    // detected -> updaetFaceView
    updateFaceView(for: result)
  }
}
