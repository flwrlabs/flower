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

public struct BoundingBox {
  /** Index of the predicted class. */
  public let classIndex: Int

  /** Confidence score. */
  public let score: Float

  /** Normalized coordinates between 0 and 1. */
  public let rect: CGRect

  public init(classIndex: Int, score: Float, rect: CGRect) {
    self.classIndex = classIndex
    self.score = score
    self.rect = rect
  }
}

/**
  Computes intersection-over-union overlap between two bounding boxes.
*/
public func IOU(_ a: CGRect, _ b: CGRect) -> Float {
  let areaA = a.width * a.height
  if areaA <= 0 { return 0 }

  let areaB = b.width * b.height
  if areaB <= 0 { return 0 }

  let intersectionMinX = max(a.minX, b.minX)
  let intersectionMinY = max(a.minY, b.minY)
  let intersectionMaxX = min(a.maxX, b.maxX)
  let intersectionMaxY = min(a.maxY, b.maxY)
  let intersectionArea = max(intersectionMaxY - intersectionMinY, 0) *
                         max(intersectionMaxX - intersectionMinX, 0)
  return Float(intersectionArea / (areaA + areaB - intersectionArea))
}

/**
  Removes bounding boxes that overlap too much with other boxes that have
  a higher score.
*/
public func nonMaxSuppression(boundingBoxes: [BoundingBox],
                              iouThreshold: Float,
                              maxBoxes: Int) -> [Int] {
  return nonMaxSuppression(boundingBoxes: boundingBoxes,
                           indices: Array(boundingBoxes.indices),
                           iouThreshold: iouThreshold,
                           maxBoxes: maxBoxes)
}

/**
  Removes bounding boxes that overlap too much with other boxes that have
  a higher score.

  Based on code from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/non_max_suppression_op.cc

  - Note: This version of NMS ignores the class of the bounding boxes. Since it
    selects the bounding boxes in a greedy fashion, if a certain class has many
    boxes that are selected, then it is possible none of the boxes of the other
    classes get selected.

  - Parameters:
    - boundingBoxes: an array of bounding boxes and their scores
    - indices: which predictions to look at
    - iouThreshold: used to decide whether boxes overlap too much
    - maxBoxes: the maximum number of boxes that will be selected

  - Returns: the array indices of the selected bounding boxes
*/
public func nonMaxSuppression(boundingBoxes: [BoundingBox],
                              indices: [Int],
                              iouThreshold: Float,
                              maxBoxes: Int) -> [Int] {

  // Sort the boxes based on their confidence scores, from high to low.
  let sortedIndices = indices.sorted { boundingBoxes[$0].score > boundingBoxes[$1].score }

  var selected: [Int] = []

  // Loop through the bounding boxes, from highest score to lowest score,
  // and determine whether or not to keep each box.
  for i in 0..<sortedIndices.count {
    if selected.count >= maxBoxes { break }

    var shouldSelect = true
    let boxA = boundingBoxes[sortedIndices[i]]

    // Does the current box overlap one of the selected boxes more than the
    // given threshold amount? Then it's too similar, so don't keep it.
    for j in 0..<selected.count {
      let boxB = boundingBoxes[selected[j]]
      if IOU(boxA.rect, boxB.rect) > iouThreshold {
        shouldSelect = false
        break
      }
    }

    // This bounding box did not overlap too much with any previously selected
    // bounding box, so we'll keep it.
    if shouldSelect {
      selected.append(sortedIndices[i])
    }
  }

  return selected
}

/**
  Multi-class version of non maximum suppression.

  Where `nonMaxSuppression()` does not look at the class of the predictions at
  all, the multi-class version first selects the best bounding boxes for each
  class, and then keeps the best ones of those.

  With this method you can usually expect to see at least one bounding box for
  each class (unless all the scores for a given class are really low).

  Based on code from: https://github.com/tensorflow/models/blob/master/object_detection/core/post_processing.py

  - Parameters:
    - numClasses: the number of classes
    - boundingBoxes: an array of bounding boxes and their scores
    - scoreThreshold: used to only keep bounding boxes with a high enough score
    - iouThreshold: used to decide whether boxes overlap too much
    - maxPerClass: the maximum number of boxes that will be selected per class
    - maxTotal: maximum number of boxes that will be selected over all classes

  - Returns: the array indices of the selected bounding boxes
*/
public func nonMaxSuppressionMultiClass(numClasses: Int,
                                        boundingBoxes: [BoundingBox],
                                        scoreThreshold: Float,
                                        iouThreshold: Float,
                                        maxPerClass: Int,
                                        maxTotal: Int) -> [Int] {
  var selectedBoxes: [Int] = []

  // Look at all the classes one-by-one.
  for c in 0..<numClasses {
    var filteredBoxes = [Int]()

    // Look at every bounding box for this class.
    for p in 0..<boundingBoxes.count {
      let prediction = boundingBoxes[p]
      if prediction.classIndex == c {

        // Only keep the box if its score is over the threshold.
        if prediction.score > scoreThreshold {
          filteredBoxes.append(p)
        }
      }
    }

    // Only keep the best bounding boxes for this class.
    let nmsBoxes = nonMaxSuppression(boundingBoxes: boundingBoxes,
                                     indices: filteredBoxes,
                                     iouThreshold: iouThreshold,
                                     maxBoxes: maxPerClass)

    // Add the indices of the surviving boxes to the big list.
    selectedBoxes.append(contentsOf: nmsBoxes)
  }

  // Sort all the surviving boxes by score and only keep the best ones.
  let sortedBoxes = selectedBoxes.sorted { boundingBoxes[$0].score > boundingBoxes[$1].score }
  return Array(sortedBoxes.prefix(maxTotal))
}
