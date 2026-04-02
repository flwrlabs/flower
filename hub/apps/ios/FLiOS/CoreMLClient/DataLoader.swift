//
//  DataLoader.swift
//  FlowerSDK
//
//  Created by Daniel Nugraha on 25.05.22.
//  Adapted from https://github.com/JacopoMangiavacchi/MNIST-CoreML-Training

import Foundation
import CoreML
import Compression
import UIKit
import os

let appDirectory = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!

class DataLoader {
    
    private static let log = Logger(subsystem: Bundle.main.bundleIdentifier ?? "flwr.Flower",
                             category: String(describing: DataLoader.self))
    
    static func trainBatchProvider(scenario: Constants.ScenarioTypes, progressHandler: @escaping (Int) -> Void) -> MLBatchProvider {
        return prepareMLBatchProvider(filePath: extractTrainFile(scenario: scenario), scenario: scenario, progressHandler: progressHandler)
    }
    
    static func testBatchProvider(scenario: Constants.ScenarioTypes, progressHandler: @escaping (Int) -> Void) -> MLBatchProvider {
        return prepareMLBatchProvider(filePath: extractTestFile(scenario: scenario), scenario: scenario, progressHandler: progressHandler)
    }

    /// Extract file
    ///
    /// - parameter sourceURL:URL of source file
    /// - parameter destinationFilename: Choosen destination filename
    ///
    /// - returns: Temporary path of extracted file
    fileprivate static func extractFile(from sourceURL: URL, to destinationURL: URL, scenario: Constants.ScenarioTypes) -> String {
        let sourceFileHandle = try! FileHandle(forReadingFrom: sourceURL)
        var isDir:ObjCBool = true
        let fileManager = FileManager.default
        if !fileManager.fileExists(atPath: appDirectory.path, isDirectory: &isDir) {
            try! fileManager.createDirectory(at: appDirectory, withIntermediateDirectories: true)
        }
        if fileManager.fileExists(atPath: destinationURL.path) {
            return destinationURL.path
        }
        FileManager.default.createFile(atPath: destinationURL.path,
                                       contents: nil,
                                       attributes: nil)
        
        let destinationFileHandle = try! FileHandle(forWritingTo: destinationURL)
        let bufferSize = 65536
        
        let filter = try! OutputFilter(.decompress, using: .lzfse, bufferCapacity: 655360) { data in
            if let data = data {
                destinationFileHandle.write(data)
            }
        }
        
        while true {
            let data = sourceFileHandle.readData(ofLength: bufferSize)
            
            try! filter.write(data)
            if data.count < bufferSize {
                break
            }
        }
        
        sourceFileHandle.closeFile()
        destinationFileHandle.closeFile()

        return destinationURL.path
    }

    /// Extract train file
    ///
    /// - returns: Temporary path of extracted file
    private static func extractTrainFile(scenario: Constants.ScenarioTypes) -> String {
        let dataset = scenario.description
        let sourceURL = Bundle.main.url(forResource: dataset + "_train", withExtension: "csv.lzfse")!
        let destinationURL = appDirectory.appendingPathComponent(dataset + "_train.csv")
        return extractFile(from: sourceURL, to: destinationURL, scenario: scenario)
    }

    /// Extract test file
    ///
    /// - returns: Temporary path of extracted file
    private static func extractTestFile(scenario: Constants.ScenarioTypes) -> String {
        let dataset = scenario.description
        let sourceURL = Bundle.main.url(forResource: dataset + "_test", withExtension: "csv.lzfse")!
        let destinationURL = appDirectory.appendingPathComponent(dataset + "_test.csv")
        return extractFile(from: sourceURL, to: destinationURL, scenario: scenario)
    }
    
    private static func prepareMLBatchProvider(filePath: String, scenario: Constants.ScenarioTypes, progressHandler: @escaping (Int) -> Void) -> MLBatchProvider {
        var featureProviders = [MLFeatureProvider]()
        
        var count = 0
        errno = 0
        
        if freopen(filePath, "r", stdin) == nil {
            log.error("error opening file")
        }
        var lengthEntry = 1
        scenario.shapeData.enumerated().forEach { index, value in
            lengthEntry = Int(truncating: value) * lengthEntry
        }

        // MARK: Fails if commas occur in the values of csv
        while let line = readLine()?.split(separator: ",") {
            count += 1
            progressHandler(count)
            let imageMultiArr = try! MLMultiArray(shape: scenario.shapeData, dataType: .float32)
            let outputMultiArr = try! MLMultiArray(shape: scenario.shapeTarget, dataType: .int32)
            for i in 0..<lengthEntry {
                imageMultiArr[i] = NSNumber(value: Float(String(line[i]))! / scenario.normalization)
            }
            outputMultiArr[0] = NSNumber(value: Float(String(line.last!))!)
            let imageValue = MLFeatureValue(multiArray: imageMultiArr)
            let outputValue = MLFeatureValue(multiArray: outputMultiArr)
            let dataPointFeatures: [String: MLFeatureValue] = ["image": imageValue,
                                                               "output_true": outputValue]
            if let provider = try? MLDictionaryFeatureProvider(dictionary: dataPointFeatures) {
                featureProviders.append(provider)
            }
        }

        return MLArrayBatchProvider(array: featureProviders)
    }
    
    static func predictionBatchProvider(scenario: Constants.ScenarioTypes) -> MLBatchProvider {
        var featureProviders = [MLFeatureProvider]()
        
        var count = 0
        errno = 0
        
        let testFilePath = extractTestFile(scenario: scenario)
        if freopen(testFilePath, "r", stdin) == nil {
            log.error("error opening file")
        }
        var lengthEntry = 1
        scenario.shapeData.enumerated().forEach { index, value in
            lengthEntry = Int(truncating: value) * lengthEntry
        }
        while let line = readLine()?.split(separator: ",") {
            count += 1
            let imageMultiArr = try! MLMultiArray(shape: scenario.shapeData, dataType: .float32)
            for i in 0..<lengthEntry {
                imageMultiArr[i] = NSNumber(value: Float(String(line[i]))! / scenario.normalization)
            }
            let imageValue = MLFeatureValue(multiArray: imageMultiArr)
            let dataPointFeatures: [String: MLFeatureValue] = ["image": imageValue]
            if let provider = try? MLDictionaryFeatureProvider(dictionary: dataPointFeatures) {
                featureProviders.append(provider)
            }
        }
        return MLArrayBatchProvider(array: featureProviders)
    }
}
