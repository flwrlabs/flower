/*
  Copyright (c) 2019 M.I. Hollemans

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to
  deal in the Software without restriction, including without limitation the
  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
  sell copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
  IN THE SOFTWARE.
*/

import CoreML

extension MLModel {
  /**
    Returns the MLImageConstraint for the given model input, or nil if that
    input doesn't exist or is not an image.
   */
  public func imageConstraint(forInput inputName: String) -> MLImageConstraint? {
    modelDescription.inputDescriptionsByName[inputName]?.imageConstraint
  }
}

#if canImport(UIKit)
import UIKit

@available(iOS 13.0, tvOS 13.0, *)
extension MLModel {
  /**
    Converts a UIImage into an MLFeatureValue, using the image constraint of
    the specified model input.
   */
  public func featureValue(fromUIImage image: UIImage,
                           forInput inputName: String,
                           orientation: CGImagePropertyOrientation = .up,
                           options: [MLFeatureValue.ImageOption: Any]? = nil)
                           -> MLFeatureValue? {

    guard let cgImage = image.cgImage else {
      print("Error: could not convert UIImage to CGImage")
      return nil
    }

    return featureValue(fromCGImage: cgImage, forInput: inputName,
                        orientation: orientation, options: options)
  }
}

#endif

@available(iOS 13.0, tvOS 13.0, OSX 10.15, *)
extension MLModel {
  /**
    Converts a CGImage into an MLFeatureValue, using the image constraint of
    the specified model input.
   */
  public func featureValue(fromCGImage image: CGImage,
                           forInput inputName: String,
                           orientation: CGImagePropertyOrientation = .up,
                           options: [MLFeatureValue.ImageOption: Any]? = nil)
                           -> MLFeatureValue? {

    guard let constraint = imageConstraint(forInput: inputName) else {
      print("Error: could not get image constraint for input named '\(inputName)'")
      return nil
    }

    guard let featureValue = try? MLFeatureValue(cgImage: image,
                                                 orientation: orientation,
                                                 constraint: constraint,
                                                 options: options) else {
      print("Error: could not get feature value for image \(image)")
      return nil
    }

    return featureValue
  }

  /**
    Converts an image file from a URL into an MLFeatureValue, using the image
    constraint of the specified model input.
   */
  public func featureValue(fromImageAt url: URL,
                           forInput inputName: String,
                           orientation: CGImagePropertyOrientation = .up,
                           options: [MLFeatureValue.ImageOption: Any]? = nil) -> MLFeatureValue? {

    guard let constraint = imageConstraint(forInput: inputName) else {
      print("Error: could not get image constraint for input named '\(inputName)'")
      return nil
    }

    guard let featureValue = try? MLFeatureValue(imageAt: url,
                                                 orientation: orientation,
                                                 constraint: constraint,
                                                 options: options) else {
      print("Error: could not get feature value for image at '\(url)'")
      return nil
    }

    return featureValue
  }
}
