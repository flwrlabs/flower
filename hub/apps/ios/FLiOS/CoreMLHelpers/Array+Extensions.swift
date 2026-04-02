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

import Swift

extension Array where Element: Comparable {
  /**
    Returns the index and value of the largest element in the array.

    - Note: This method is slow. For faster results, use the standalone
            version of argmax() instead.
  */
  public func argmax() -> (Int, Element) {
    precondition(self.count > 0)
    var maxIndex = 0
    var maxValue = self[0]
    for i in 1..<self.count where self[i] > maxValue {
      maxValue = self[i]
      maxIndex = i
    }
    return (maxIndex, maxValue)
  }

  /**
    Returns the indices of the array's elements in sorted order.
  */
  public func argsort(by areInIncreasingOrder: (Element, Element) -> Bool) -> [Array.Index] {
    return indices.sorted { areInIncreasingOrder(self[$0], self[$1]) }
  }

  /**
    Returns a new array containing the elements at the specified indices.
  */
  public func gather(indices: [Array.Index]) -> [Element] {
    return indices.map { self[$0] }
  }
}
