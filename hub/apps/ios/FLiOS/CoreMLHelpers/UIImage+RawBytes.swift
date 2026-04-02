/*
  Copyright (c) 2017-2019 M.I. Hollemans

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

#if canImport(UIKit)

import UIKit

extension UIImage {
  /**
    Converts the image into an array of RGBA bytes.
  */
  @nonobjc public func toByteArrayRGBA() -> [UInt8]? {
    return cgImage?.toByteArrayRGBA()
  }

  /**
    Creates a new UIImage from an array of RGBA bytes.
  */
  @nonobjc public class func fromByteArrayRGBA(_ bytes: [UInt8],
                                               width: Int,
                                               height: Int,
                                               scale: CGFloat = 0,
                                               orientation: UIImage.Orientation = .up) -> UIImage? {
    if let cgImage = CGImage.fromByteArrayRGBA(bytes, width: width, height: height) {
      return UIImage(cgImage: cgImage, scale: scale, orientation: orientation)
    } else {
      return nil
    }
  }

  /**
    Creates a new UIImage from an array of grayscale bytes.
  */
  @nonobjc public class func fromByteArrayGray(_ bytes: [UInt8],
                                               width: Int,
                                               height: Int,
                                               scale: CGFloat = 0,
                                               orientation: UIImage.Orientation = .up) -> UIImage? {
    if let cgImage = CGImage.fromByteArrayGray(bytes, width: width, height: height) {
      return UIImage(cgImage: cgImage, scale: scale, orientation: orientation)
    } else {
      return nil
    }
  }
}

#endif
