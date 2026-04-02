//
//  MLModelInspect.swift
//  FlowerSDK
//
//  Created by Daniel Nugraha on 16.03.22.
//

import Foundation
import flwr

struct MLModelInspect {
    
    private let mlModel: CoreML_Specification_Model?
    
    init(serializedData data: Data) {
        mlModel = try? CoreML_Specification_Model(serializedData: data)
    }
    
    func getLayerWrappers() -> [MLLayerWrapper] {
        return (mlModel?.neuralNetwork.layers.compactMap { getLayerWrapper(layer: $0) })!
    }
    
    private func getLayerWrapper(layer: CoreML_Specification_NeuralNetworkLayer) -> MLLayerWrapper? {
        if let layerType = layer.layer {
            switch layerType {
            case .convolution:
                let convolution = layer.convolution
                //shape definition = [outputChannels, kernelChannels, kernelHeight, kernelWidth]
                let shape = [Int16(convolution.outputChannels), Int16(convolution.kernelChannels), Int16(convolution.kernelSize[0]), Int16(convolution.kernelSize[1])]
                return MLLayerWrapper(shape: shape,
                                      name: layer.name,
                                      weights: convolution.weights.floatValue,
                                      isUpdatable: layer.isUpdatable)
            case .innerProduct:
                let innerProduct = layer.innerProduct
                //shape definition = [C_out, C_in].
                let shape = [Int16(innerProduct.outputChannels), Int16(innerProduct.inputChannels)]
                return MLLayerWrapper(shape: shape,
                                      name: layer.name,
                                      weights: innerProduct.weights.floatValue,
                                      isUpdatable: layer.isUpdatable)
            default:
                return nil
            }
        }
        return nil
    }
}
