/*
  Copyright (c) 2017-2020 M.I. Hollemans

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

import Accelerate
import CoreML

public protocol MultiArrayType: Comparable {
  static var multiArrayDataType: MLMultiArrayDataType { get }
  static func +(lhs: Self, rhs: Self) -> Self
  static func -(lhs: Self, rhs: Self) -> Self
  static func *(lhs: Self, rhs: Self) -> Self
  static func /(lhs: Self, rhs: Self) -> Self
  init(_: Int)
  var toUInt8: UInt8 { get }
}

extension Double: MultiArrayType {
  public static var multiArrayDataType: MLMultiArrayDataType { return .double }
  public var toUInt8: UInt8 { return UInt8(self) }
}

extension Float: MultiArrayType {
  public static var multiArrayDataType: MLMultiArrayDataType { return .float32 }
  public var toUInt8: UInt8 { return UInt8(self) }
}

extension Int32: MultiArrayType {
  public static var multiArrayDataType: MLMultiArrayDataType { return .int32 }
  public var toUInt8: UInt8 { return UInt8(self) }
}

extension MLMultiArray {
  /**
    Converts the multi-array to a CGImage.

    The multi-array must have at least 2 dimensions for a grayscale image, or
    at least 3 dimensions for a color image.

    The default expected shape is (height, width) or (channels, height, width).
    However, you can change this using the `axes` parameter. For example, if
    the array shape is (1, height, width, channels), use `axes: (3, 1, 2)`.

    If `channel` is not nil, only converts that channel to a grayscale image.
    This lets you visualize individual channels from a multi-array with more
    than 4 channels.

    Otherwise, converts all channels. In this case, the number of channels in
    the multi-array must be 1 for grayscale, 3 for RGB, or 4 for RGBA.

    Use the `min` and `max` parameters to put the values from the array into
    the range [0, 255], if not already:

    - `min`: should be the smallest value in the data; this will be mapped to 0.
    - `max`: should be the largest value in the data; will be mapped to 255.

    For example, if the range of the data in the multi-array is [-1, 1], use
    `min: -1, max: 1`. If the range is already [0, 255], then use the defaults.
  */
  public func cgImage(min: Double = 0,
                      max: Double = 255,
                      channel: Int? = nil,
                      axes: (Int, Int, Int)? = nil) -> CGImage? {
    switch self.dataType {
    case .double:
      return _image(min: min, max: max, channel: channel, axes: axes)
    case .float32:
      return _image(min: Float(min), max: Float(max), channel: channel, axes: axes)
    case .int32:
      return _image(min: Int32(min), max: Int32(max), channel: channel, axes: axes)
    @unknown default:
      fatalError("Unsupported data type \(dataType.rawValue)")
    }
  }

  /**
    Helper function that allows us to use generics. The type of `min` and `max`
    is also the dataType of the MLMultiArray.
  */
  private func _image<T: MultiArrayType>(min: T,
                                         max: T,
                                         channel: Int?,
                                         axes: (Int, Int, Int)?) -> CGImage? {
    if let (b, w, h, c) = toRawBytes(min: min, max: max, channel: channel, axes: axes) {
      if c == 1 {
        return CGImage.fromByteArrayGray(b, width: w, height: h)
      } else {
        return CGImage.fromByteArrayRGBA(b, width: w, height: h)
      }
    }
    return nil
  }

  /**
    Converts the multi-array into an array of RGBA or grayscale pixels.

    - Note: This is not particularly fast, but it is flexible. You can change
            the loops to convert the multi-array whichever way you please.

    - Note: The type of `min` and `max` must match the dataType of the
            MLMultiArray object.

    - Returns: tuple containing the RGBA bytes, the dimensions of the image,
               and the number of channels in the image (1, 3, or 4).
  */
  public func toRawBytes<T: MultiArrayType>(min: T,
                                            max: T,
                                            channel: Int? = nil,
                                            axes: (Int, Int, Int)? = nil)
                  -> (bytes: [UInt8], width: Int, height: Int, channels: Int)? {
    // MLMultiArray with unsupported shape?
    if shape.count < 2 {
      print("Cannot convert MLMultiArray of shape \(shape) to image")
      return nil
    }

    // Figure out which dimensions to use for the channels, height, and width.
    let channelAxis: Int
    let heightAxis: Int
    let widthAxis: Int
    if let axes = axes {
      channelAxis = axes.0
      heightAxis = axes.1
      widthAxis = axes.2
      guard channelAxis >= 0 && channelAxis < shape.count &&
            heightAxis >= 0 && heightAxis < shape.count &&
            widthAxis >= 0 && widthAxis < shape.count else {
        print("Invalid axes \(axes) for shape \(shape)")
        return nil
      }
    } else if shape.count == 2 {
      // Expected shape for grayscale is (height, width)
      heightAxis = 0
      widthAxis = 1
      channelAxis = -1 // Never be used
    } else {
      // Expected shape for color is (channels, height, width)
      channelAxis = 0
      heightAxis = 1
      widthAxis = 2
    }

    let height = self.shape[heightAxis].intValue
    let width = self.shape[widthAxis].intValue
    let yStride = self.strides[heightAxis].intValue
    let xStride = self.strides[widthAxis].intValue

    let channels: Int
    let cStride: Int
    let bytesPerPixel: Int
    let channelOffset: Int

    // MLMultiArray with just two dimensions is always grayscale. (We ignore
    // the value of channelAxis here.)
    if shape.count == 2 {
      channels = 1
      cStride = 0
      bytesPerPixel = 1
      channelOffset = 0

    // MLMultiArray with more than two dimensions can be color or grayscale.
    } else {
      let channelDim = self.shape[channelAxis].intValue
      if let channel = channel {
        if channel < 0 || channel >= channelDim {
          print("Channel must be -1, or between 0 and \(channelDim - 1)")
          return nil
        }
        channels = 1
        bytesPerPixel = 1
        channelOffset = channel
      } else if channelDim == 1 {
        channels = 1
        bytesPerPixel = 1
        channelOffset = 0
      } else {
        if channelDim != 3 && channelDim != 4 {
          print("Expected channel dimension to have 1, 3, or 4 channels, got \(channelDim)")
          return nil
        }
        channels = channelDim
        bytesPerPixel = 4
        channelOffset = 0
      }
      cStride = self.strides[channelAxis].intValue
    }

    // Allocate storage for the RGBA or grayscale pixels. Set everything to
    // 255 so that alpha channel is filled in if only 3 channels.
    let count = height * width * bytesPerPixel
    var pixels = [UInt8](repeating: 255, count: count)

    // Grab the pointer to MLMultiArray's memory.
    var ptr = UnsafeMutablePointer<T>(OpaquePointer(self.dataPointer))
    ptr = ptr.advanced(by: channelOffset * cStride)

    // Loop through all the pixels and all the channels and copy them over.
    for c in 0..<channels {
      for y in 0..<height {
        for x in 0..<width {
          let value = ptr[c*cStride + y*yStride + x*xStride]
          let scaled = (value - min) * T(255) / (max - min)
          let pixel = clamp(scaled, min: T(0), max: T(255)).toUInt8
          pixels[(y*width + x)*bytesPerPixel + c] = pixel
        }
      }
    }
    return (pixels, width, height, channels)
  }
}

/**
  Fast conversion from MLMultiArray to CGImage using the vImage framework.

  - Parameters:
    - features: A multi-array with data type FLOAT32 and three dimensions
                (3, height, width).
    - min: The smallest value in the multi-array. This value, as well as any
           smaller values, will be mapped to 0 in the output image.
    - max: The largest value in the multi-array. This and any larger values
           will be will be mapped to 255 in the output image.

  - Returns: a new CGImage or nil if the conversion fails
*/
public func createCGImage(fromFloatArray features: MLMultiArray,
                          min: Float = 0,
                          max: Float = 255) -> CGImage? {
  assert(features.dataType == .float32)
  assert(features.shape.count == 3)

  let ptr = UnsafeMutablePointer<Float>(OpaquePointer(features.dataPointer))

  let height = features.shape[1].intValue
  let width = features.shape[2].intValue
  let channelStride = features.strides[0].intValue
  let rowStride = features.strides[1].intValue
  let srcRowBytes = rowStride * MemoryLayout<Float>.stride

  var blueBuffer = vImage_Buffer(data: ptr,
                                 height: vImagePixelCount(height),
                                 width: vImagePixelCount(width),
                                 rowBytes: srcRowBytes)
  var greenBuffer = vImage_Buffer(data: ptr.advanced(by: channelStride),
                                  height: vImagePixelCount(height),
                                  width: vImagePixelCount(width),
                                  rowBytes: srcRowBytes)
  var redBuffer = vImage_Buffer(data: ptr.advanced(by: channelStride * 2),
                                height: vImagePixelCount(height),
                                width: vImagePixelCount(width),
                                rowBytes: srcRowBytes)

  let destRowBytes = width * 4

  var error: vImage_Error = 0
  var pixels = [UInt8](repeating: 0, count: height * destRowBytes)

  pixels.withUnsafeMutableBufferPointer { ptr in
    var destBuffer = vImage_Buffer(data: ptr.baseAddress!,
                                   height: vImagePixelCount(height),
                                   width: vImagePixelCount(width),
                                   rowBytes: destRowBytes)

    error = vImageConvert_PlanarFToBGRX8888(&blueBuffer,
                                            &greenBuffer,
                                            &redBuffer,
                                            Pixel_8(255),
                                            &destBuffer,
                                            [max, max, max],
                                            [min, min, min],
                                            vImage_Flags(0))
  }

  if error == kvImageNoError {
    return CGImage.fromByteArrayRGBA(pixels, width: width, height: height)
  } else {
    return nil
  }
}

#if canImport(UIKit)

import UIKit

extension MLMultiArray {
  public func image(min: Double = 0,
                    max: Double = 255,
                    channel: Int? = nil,
                    axes: (Int, Int, Int)? = nil) -> UIImage? {
    let cgImg = cgImage(min: min, max: max, channel: channel, axes: axes)
    return cgImg.map { UIImage(cgImage: $0) }
  }
}

public func createUIImage(fromFloatArray features: MLMultiArray,
                          min: Float = 0,
                          max: Float = 255) -> UIImage? {
  let cgImg = createCGImage(fromFloatArray: features, min: min, max: max)
  return cgImg.map { UIImage(cgImage: $0) }
}

#endif
