//
//  File.swift
//  
//
//  Created by Daniel Nugraha on 20.01.23.
//

import Foundation
import CoreML
import os
import flwr

typealias Model = CoreML_Specification_Model
typealias NeuralNetwork = CoreML_Specification_NeuralNetwork
typealias NeuralNetworkLayer = CoreML_Specification_NeuralNetworkLayer

/// Container for train and test dataset.
public struct MLDataLoader {
    public let trainBatchProvider: MLBatchProvider
    public let testBatchProvider: MLBatchProvider
    
    public init(trainBatchProvider: MLBatchProvider, testBatchProvider: MLBatchProvider) {
        self.trainBatchProvider = trainBatchProvider
        self.testBatchProvider = testBatchProvider
    }
}

/// Container for neural network layer information.
public struct MLLayerWrapper {
    let shape: [Int16]
    let name: String
    var weights: [Float]
    let isUpdatable: Bool
    
    public init(shape: [Int16], name: String, weights: [Float], isUpdatable: Bool) {
        self.shape = shape
        self.name = name
        self.weights = weights
        self.isUpdatable = isUpdatable
    }
}

struct MLResult {
    let loss: Double
    let numSamples: Int
    let accuracy: Double
}

/// A class responsible for loading and retrieving model parameters to and from the CoreML model.
@available(iOS 14.0, *)
public class MLParameter {
    private var parameterConverter = ParameterConverter.shared
    
    var layerWrappers: [MLLayerWrapper]
    var model: Model?
    let modelUrl: URL
    let compiledModelUrl: URL
    
    private let log = Logger(subsystem: Bundle.main.bundleIdentifier ?? "flwr.Flower",
                                    category: String(describing: MLParameter.self))
    
    /// Inits MLParameter class that contains information about the model parameters and implements routines for their update and transformation.
    ///
    /// - Parameters:
    ///   - layerWrappers: Information about the layer provided with primitive data types.
    public init(layerWrappers: [MLLayerWrapper], modelUrl: URL, compiledModelUrl: URL) {
        self.layerWrappers = layerWrappers
        self.modelUrl = modelUrl
        self.compiledModelUrl = compiledModelUrl
        self.model = try? Model(serializedData: try Data(contentsOf: modelUrl))
    }
    
    /// Converts the Parameters structure to MLModelConfiguration to interface with CoreML.
    ///
    /// - Parameters:
    ///   - parameters: The parameters of the model passed as Parameters struct.
    /// - Returns: Specification of the machine learning model configuration in the CoreML structure.
    public func parametersToWeights(parameters: Parameters) -> MLModelConfiguration {
        let config = MLModelConfiguration()
        
        guard parameters.tensors.count == self.layerWrappers.count else {
            log.info("parameters received is not valid")
            return config
        }
        
        for (index, data) in parameters.tensors.enumerated() {
            let expectedNumberOfElements = layerWrappers[index].shape.map({Int($0)}).reduce(1, *)
            if let weightsArray = parameterConverter.dataToArray(data: data) {
                guard weightsArray.count == expectedNumberOfElements else {
                    log.info("array received has wrong number of elements")
                    continue
                }
                
                layerWrappers[index].weights = weightsArray
                if layerWrappers[index].isUpdatable {
                    guard model?.neuralNetwork.layers != nil else {
                        continue
                    }
                    for (indexB, neuralNetworkLayer) in model!.neuralNetwork.layers.enumerated() {
                        guard layerWrappers[index].name == neuralNetworkLayer.name else {
                            continue
                        }
                        switch neuralNetworkLayer.layer! {
                        case .convolution:
                            model!.neuralNetwork.layers[indexB].convolution.weights.floatValue = layerWrappers[index].weights
                        case .innerProduct:
                            model!.neuralNetwork.layers[indexB].innerProduct.weights.floatValue = layerWrappers[index].weights
                        default:
                            log.info("unexpected layer \(neuralNetworkLayer.name)")
                            continue
                        }
                    }
                }
            }
        }
        
        exportModel()
        return config
    }
    
    /// Returns the weights of the current layer wrapper in parameter format
    ///
    /// - Returns: The weights of the current layer wrapper in parameter format
    public func weightsToParameters() -> Parameters {
        let dataArray = layerWrappers.compactMap { parameterConverter.arrayToData(array: $0.weights, shape: $0.shape) }
        if dataArray.count != layerWrappers.count {
            log.info("dataArray size != layerWrappers size")
        }
        return Parameters(tensors: dataArray, tensorType: "ndarray")
    }
    
    private func exportModel()  {
        let modelFileName = modelUrl.deletingPathExtension().lastPathComponent
        let fileManager = FileManager.default
        let tempModelUrl = appDirectory.appendingPathComponent("temp\(modelFileName).mlmodel")
        try? model?.serializedData().write(to: tempModelUrl)
        if let compiledTempModelUrl = try? MLModel.compileModel(at: tempModelUrl) {
            _ = try? fileManager.replaceItemAt(compiledModelUrl, withItemAt: compiledTempModelUrl)
        }
    }
    
    func initializeParameters() {
        guard ((model?.neuralNetwork.layers) != nil) else {
            return
        }
        for (indexA, neuralNetworkLayer) in model!.neuralNetwork.layers.enumerated() {
            for (indexB, layer) in layerWrappers.enumerated() {
                if layer.name != neuralNetworkLayer.name { continue }
                switch neuralNetworkLayer.layer! {
                case .convolution:
                    let convolution = neuralNetworkLayer.convolution
                    //shape definition = [outputChannels, kernelChannels, kernelHeight, kernelWidth]
                    let upperLower = Float(6.0 / Float(Int16(convolution.outputChannels) + Int16(convolution.kernelChannels) + Int16(convolution.kernelSize[0]) + Int16(convolution.kernelSize[1]))).squareRoot()
                    let initialise = (0..<(neuralNetworkLayer.convolution.weights.floatValue.count)).map { _ in Float.random(in: -upperLower...upperLower) }
                    model?.neuralNetwork.layers[indexA].convolution.weights.floatValue = initialise
                    layerWrappers[indexB].weights = initialise
                case .innerProduct:
                    let innerProduct = neuralNetworkLayer.innerProduct
                    //shape definition = [C_out, C_in].
                    let upperLower = Float(6.0 / Float(Int16(innerProduct.outputChannels) + Int16(innerProduct.inputChannels))).squareRoot()
                    let initialise = (0..<(neuralNetworkLayer.innerProduct.weights.floatValue.count)).map { _ in Float.random(in: -upperLower...upperLower) }
                    model?.neuralNetwork.layers[indexA].innerProduct.weights.floatValue = initialise
                    layerWrappers[indexB].weights = initialise
                default:
                    log.info("unexpected layer \(neuralNetworkLayer.name)")
                    continue
                }
            }
        }
        exportModel()
    }
    
    /// Updates the layers given the CoreML update context
    ///
    /// - Parameters:
    ///   - context: The context of the update procedure of the CoreML model.
    public func updateLayerWrappers(context: MLUpdateContext) {
        for (index, layer) in self.layerWrappers.enumerated() {
            if layer.isUpdatable {
                let paramKey = MLParameterKey.weights.scoped(to: layer.name)
                if let weightsMultiArray = try? context.model.parameterValue(for: paramKey) as? MLMultiArray {
                    let weightsShape = Array(weightsMultiArray.shape.map({ Int16(truncating: $0) }).drop(while: { $0 < 2 }))
                    guard weightsShape == layer.shape else {
                        log.info("shape \(weightsShape) is not the same as \(layer.shape)")
                        continue
                    }
                    
                    if let pointer = try? UnsafeBufferPointer<Float>(weightsMultiArray) {
                        let array = pointer.compactMap{$0}
                        self.layerWrappers[index].weights = array
                    }
                }
            }
        }
    }
}
