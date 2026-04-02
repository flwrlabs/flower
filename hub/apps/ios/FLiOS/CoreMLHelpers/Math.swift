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

import Foundation
import Accelerate

/**
  Returns the index and value of the largest element in the array.

  - Parameters:
    - count: If provided, only look at the first `count` elements of the array,
             otherwise look at the entire array.
*/
public func argmax(_ array: [Float], count: Int? = nil) -> (Int, Float) {
  var maxValue: Float = 0
  var maxIndex: vDSP_Length = 0
  vDSP_maxvi(array, 1, &maxValue, &maxIndex, vDSP_Length(count ?? array.count))
  return (Int(maxIndex), maxValue)
}

/**
  Returns the index and value of the largest element in the array.

  - Parameters:
    - ptr: Pointer to the first element in memory.
    - count: How many elements to look at.
    - stride: The distance between two elements in memory.
*/
public func argmax(_ ptr: UnsafePointer<Float>, count: Int, stride: Int = 1) -> (Int, Float) {
  var maxValue: Float = 0
  var maxIndex: vDSP_Length = 0
  vDSP_maxvi(ptr, vDSP_Stride(stride), &maxValue, &maxIndex, vDSP_Length(count))
  return (Int(maxIndex), maxValue)
}

/**
  Returns the index and value of the largest element in the array.

  - Parameters:
    - count: If provided, only look at the first `count` elements of the array,
             otherwise look at the entire array.
*/
public func argmax(_ array: [Double], count: Int? = nil) -> (Int, Double) {
  var maxValue: Double = 0
  var maxIndex: vDSP_Length = 0
  vDSP_maxviD(array, 1, &maxValue, &maxIndex, vDSP_Length(count ?? array.count))
  return (Int(maxIndex), maxValue)
}

/**
  Returns the index and value of the largest element in the array.

  - Parameters:
    - ptr: Pointer to the first element in memory.
    - count: How many elements to look at.
    - stride: The distance between two elements in memory.
*/
public func argmax(_ ptr: UnsafePointer<Double>, count: Int, stride: Int = 1) -> (Int, Double) {
  var maxValue: Double = 0
  var maxIndex: vDSP_Length = 0
  vDSP_maxviD(ptr, vDSP_Stride(stride), &maxValue, &maxIndex, vDSP_Length(count))
  return (Int(maxIndex), maxValue)
}

/** Ensures that `x` is in the range `[min, max]`. */
public func clamp<T: Comparable>(_ x: T, min: T, max: T) -> T {
  if x < min { return min }
  if x > max { return max }
  return x
}

/** Logistic sigmoid. */
public func sigmoid(_ x: Float) -> Float {
  return 1 / (1 + exp(-x))
}

/** Logistic sigmoid. */
public func sigmoid(_ x: Double) -> Double {
  return 1 / (1 + exp(-x))
}

/* In-place logistic sigmoid: x = 1 / (1 + exp(-x)) */
public func sigmoid(_ x: UnsafeMutablePointer<Float>, count: Int) {
  vDSP_vneg(x, 1, x, 1, vDSP_Length(count))
  var cnt = Int32(count)
  vvexpf(x, x, &cnt)
  var y: Float = 1
  vDSP_vsadd(x, 1, &y, x, 1, vDSP_Length(count))
  vvrecf(x, x, &cnt)
}

/* In-place logistic sigmoid: x = 1 / (1 + exp(-x)) */
public func sigmoid(_ x: UnsafeMutablePointer<Double>, count: Int) {
  vDSP_vnegD(x, 1, x, 1, vDSP_Length(count))
  var cnt = Int32(count)
  vvexp(x, x, &cnt)
  var y: Double = 1
  vDSP_vsaddD(x, 1, &y, x, 1, vDSP_Length(count))
  vvrec(x, x, &cnt)
}

/**
  Computes the "softmax" function over an array.

  Based on code from https://github.com/nikolaypavlov/MLPNeuralNet/

  This is what softmax looks like in "pseudocode" (actually using Python
  and numpy):

      x -= np.max(x)
      exp_scores = np.exp(x)
      softmax = exp_scores / np.sum(exp_scores)

  First we shift the values of x so that the highest value in the array is 0.
  This ensures numerical stability with the exponents, so they don't blow up.
*/
public func softmax(_ x: [Float]) -> [Float] {
  var x = x
  let len = vDSP_Length(x.count)

  // Find the maximum value in the input array.
  var max: Float = 0
  vDSP_maxv(x, 1, &max, len)

  // Subtract the maximum from all the elements in the array.
  // Now the highest value in the array is 0.
  max = -max
  vDSP_vsadd(x, 1, &max, &x, 1, len)

  // Exponentiate all the elements in the array.
  var count = Int32(x.count)
  vvexpf(&x, x, &count)

  // Compute the sum of all exponentiated values.
  var sum: Float = 0
  vDSP_sve(x, 1, &sum, len)

  // Divide each element by the sum. This normalizes the array contents
  // so that they all add up to 1.
  vDSP_vsdiv(x, 1, &sum, &x, 1, len)

  return x
}
