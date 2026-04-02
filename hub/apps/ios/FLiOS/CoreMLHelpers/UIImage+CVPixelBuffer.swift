/*
  Copyright (c) 2017-2021 M.I. Hollemans

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
import VideoToolbox

extension UIImage {
  /**
    Converts the image to an ARGB `CVPixelBuffer`.
  */
  public func pixelBuffer() -> CVPixelBuffer? {
    return pixelBuffer(width: Int(size.width), height: Int(size.height))
  }

  /**
    Resizes the image to `width` x `height` and converts it to an ARGB
    `CVPixelBuffer`.
  */
  public func pixelBuffer(width: Int, height: Int) -> CVPixelBuffer? {
    return pixelBuffer(width: width, height: height,
                       pixelFormatType: kCVPixelFormatType_32ARGB,
                       colorSpace: CGColorSpaceCreateDeviceRGB(),
                       alphaInfo: .noneSkipFirst)
  }

  /**
    Converts the image to a grayscale `CVPixelBuffer`.
  */
  public func pixelBufferGray() -> CVPixelBuffer? {
    return pixelBufferGray(width: Int(size.width), height: Int(size.height))
  }

  /**
    Resizes the image to `width` x `height` and converts it to a grayscale
    `CVPixelBuffer`.
  */
  public func pixelBufferGray(width: Int, height: Int) -> CVPixelBuffer? {
    return pixelBuffer(width: width, height: height,
                       pixelFormatType: kCVPixelFormatType_OneComponent8,
                       colorSpace: CGColorSpaceCreateDeviceGray(),
                       alphaInfo: .none)
  }

  /**
    Resizes the image to `width` x `height` and converts it to a `CVPixelBuffer`
    with the specified pixel format, color space, and alpha channel.
  */
  public func pixelBuffer(width: Int, height: Int,
                          pixelFormatType: OSType,
                          colorSpace: CGColorSpace,
                          alphaInfo: CGImageAlphaInfo) -> CVPixelBuffer? {
    var maybePixelBuffer: CVPixelBuffer?
    let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
                 kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue]
    let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                     width,
                                     height,
                                     pixelFormatType,
                                     attrs as CFDictionary,
                                     &maybePixelBuffer)

    guard status == kCVReturnSuccess, let pixelBuffer = maybePixelBuffer else {
      return nil
    }

    let flags = CVPixelBufferLockFlags(rawValue: 0)
    guard kCVReturnSuccess == CVPixelBufferLockBaseAddress(pixelBuffer, flags) else {
      return nil
    }
    defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, flags) }

    guard let context = CGContext(data: CVPixelBufferGetBaseAddress(pixelBuffer),
                                  width: width,
                                  height: height,
                                  bitsPerComponent: 8,
                                  bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer),
                                  space: colorSpace,
                                  bitmapInfo: alphaInfo.rawValue)
    else {
      return nil
    }

    UIGraphicsPushContext(context)
    context.translateBy(x: 0, y: CGFloat(height))
    context.scaleBy(x: 1, y: -1)
    self.draw(in: CGRect(x: 0, y: 0, width: width, height: height))
    UIGraphicsPopContext()

    return pixelBuffer
  }
}

extension UIImage {
  /**
    Creates a new UIImage from a CVPixelBuffer.

    - Note: Not all CVPixelBuffer pixel formats support conversion into a
            CGImage-compatible pixel format.
  */
  public convenience init?(pixelBuffer: CVPixelBuffer) {
    if let cgImage = CGImage.create(pixelBuffer: pixelBuffer) {
      self.init(cgImage: cgImage)
    } else {
      return nil
    }
  }

  /*
  // Alternative implementation:
  public convenience init?(pixelBuffer: CVPixelBuffer) {
    // This converts the image to a CIImage first and then to a UIImage.
    // Does not appear to work on the simulator but is OK on the device.
    self.init(ciImage: CIImage(cvPixelBuffer: pixelBuffer))
  }
  */

  /**
    Creates a new UIImage from a CVPixelBuffer, using a Core Image context.
  */
  public convenience init?(pixelBuffer: CVPixelBuffer, context: CIContext) {
    if let cgImage = CGImage.create(pixelBuffer: pixelBuffer, context: context) {
      self.init(cgImage: cgImage)
    } else {
      return nil
    }
  }
}

#endif
