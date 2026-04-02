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
import Combine

@available(iOS 13.0, tvOS 13.0, OSX 10.15, *)
extension Publisher where Self.Output: MLFeatureProvider {
  /**
   Operator that lets you run a Core ML model as part of a Combine chain.

   It accepts an MLFeatureProvider object as input, and, if all goes well,
   returns another MLFeatureProvider with the model outputs.

   Since Core ML can give errors, we put everything in a Result object.

   Use the `compactMap` version to always ignore errors, or `tryMap` to
   complete the subscription upon the first error.

   To perform the Core ML request on a background thread, it's probably a good
   idea to write a custom Publisher class, but for simple use cases `map` works
   well enough.
  */
  public func prediction(model: MLModel) -> Publishers.Map<Self, Result<MLFeatureProvider, Error>> {
    map { input in
      do {
        return .success(try model.prediction(from: input))
      } catch {
        return .failure(error)
      }
    }
  }

  public func prediction(model: MLModel) -> Publishers.CompactMap<Self, MLFeatureProvider> {
    compactMap { input in try? model.prediction(from: input) }
  }

  public func prediction(model: MLModel) -> Publishers.TryMap<Self, MLFeatureProvider?> {
    tryMap { input in try model.prediction(from: input) }
  }
}
