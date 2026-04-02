//
//  FLiOSModel.swift
//  FLiOS
//
//  Created by Daniel Nugraha on 18.01.23.
//

import Foundation
import CoreML
import flwr
import os

public class FLiOSModel: ObservableObject {
    @Published public var scenarioSelection = Constants.ScenarioTypes.MNIST {
        didSet {
            self.resetPreperation()
        }
    }
    private var trainingBatchProvider: MLBatchProvider?
    @Published public var trainingBatchStatus = Constants.PreparationStatus.notPrepared
    let scenarios = Constants.ScenarioTypes.allCases
    
    private var testBatchProvider: MLBatchProvider?
    @Published public var testBatchStatus = Constants.PreparationStatus.notPrepared
    
    private var localClient: LocalClient?
    @Published public var localClientStatus = Constants.PreparationStatus.notPrepared
    @Published public var epoch: Int = 5
    @Published public var localTrainingStatus = Constants.TaskStatus.idle
    @Published public var localTestStatus = Constants.TaskStatus.idle
    
    private var mlFlwrClient: MLFlwrClient?
    @Published public var mlFlwrClientStatus = Constants.PreparationStatus.notPrepared
    
    private var flwrGRPC: FlwrGRPC?
    @Published public var hostname: String = "localhost"
    @Published public var port: Int = 8080
    @Published public var federatedServerStatus = Constants.TaskStatus.idle
    
    public var benchmarkSuite = BenchmarkSuite.shared
    private let log = Logger(subsystem: Bundle.main.bundleIdentifier ?? "flwr.Flower",
                             category: String(describing: DataLoader.self))
    
    public func resetPreperation() {
        self.trainingBatchStatus = .notPrepared
        self.testBatchStatus = .notPrepared
        self.localClientStatus = .notPrepared
        self.mlFlwrClientStatus = .notPrepared
    }
    
    public func prepareTrainDataset() {
        trainingBatchStatus = .preparing(count: 0)
        self.benchmarkSuite.takeActionSnapshot(snapshot: ActionSnaptshot(action: "preparing train dataset " + scenarioSelection.description))
        DispatchQueue.global(qos: .userInitiated).async {
            let batchProvider = DataLoader.trainBatchProvider(scenario: self.scenarioSelection) { count in
                DispatchQueue.main.async {
                    self.trainingBatchStatus = .preparing(count: count)
                }
            }
            DispatchQueue.main.async {
                self.trainingBatchProvider = batchProvider
                self.trainingBatchStatus = .ready
                self.benchmarkSuite.takeActionSnapshot(snapshot: ActionSnaptshot(action: "finished preparing train dataset"))
            }
        }
    }
    
    public func prepareTestDataset() {
        testBatchStatus = .preparing(count: 0)
        self.benchmarkSuite.takeActionSnapshot(snapshot: ActionSnaptshot(action: "preparing test dataset " + scenarioSelection.description))
        DispatchQueue.global(qos: .userInitiated).async {
            let batchProvider = DataLoader.testBatchProvider(scenario: self.scenarioSelection) { count in
                DispatchQueue.main.async {
                    self.testBatchStatus = .preparing(count: count)
                }
            }
            DispatchQueue.main.async {
                self.testBatchProvider = batchProvider
                self.testBatchStatus = .ready
                self.benchmarkSuite.takeActionSnapshot(snapshot: ActionSnaptshot(action: "finished test dataset"))
            }
        }
    }
    
    public func initLocalClient() {
        self.benchmarkSuite.takeActionSnapshot(snapshot: ActionSnaptshot(action: "init local client with " + scenarioSelection.modelName))
        self.localClientStatus = .preparing(count: 0)
        if self.localClient == nil {
            DispatchQueue.global(qos: .userInitiated).async {
                let dataLoader = MLDataLoader(trainBatchProvider: self.trainingBatchProvider!, testBatchProvider: self.testBatchProvider!)
                if let modelUrl = Bundle.main.url(forResource: self.scenarioSelection.modelName, withExtension: "mlmodel") {
                    self.initClient(modelUrl: modelUrl, dataLoader: dataLoader, clientType: .local)
                    DispatchQueue.main.async {
                        self.localClientStatus = .ready
                        self.benchmarkSuite.takeActionSnapshot(snapshot: ActionSnaptshot(action: "local client ready"))
                    }
                }
            }
        } else {
            self.localClientStatus = .ready
            self.benchmarkSuite.takeActionSnapshot(snapshot: ActionSnaptshot(action: "local client ready"))
        }
    }
    
    public func initMLFlwrClient() {
        self.mlFlwrClientStatus = .preparing(count: 0)
        self.benchmarkSuite.takeActionSnapshot(snapshot: ActionSnaptshot(action: "init ML Flwr Client with " + scenarioSelection.modelName))
        if self.mlFlwrClient == nil {
            DispatchQueue.global(qos: .userInitiated).async {
                let dataLoader = MLDataLoader(trainBatchProvider: self.trainingBatchProvider!, testBatchProvider: self.testBatchProvider!)
                if let modelUrl = Bundle.main.url(forResource: self.scenarioSelection.modelName, withExtension: "mlmodel") {
                    self.initClient(modelUrl: modelUrl, dataLoader: dataLoader, clientType: .federated)
                    DispatchQueue.main.async {
                        self.mlFlwrClientStatus = .ready
                        self.benchmarkSuite.takeActionSnapshot(snapshot: ActionSnaptshot(action: "ML Flwr Client ready"))
                    }
                }
            }
        } else {
            self.mlFlwrClientStatus = .ready
            self.benchmarkSuite.takeActionSnapshot(snapshot: ActionSnaptshot(action: "ML Flwr Client ready"))
        }
    }
    
    private func initClient(modelUrl url: URL, dataLoader: MLDataLoader, clientType: Constants.ClientType) {
        do {
            let compiledModelUrl = try MLModel.compileModel(at: url)
            switch clientType {
            case .federated:
                let modelInspect = try MLModelInspect(serializedData: Data(contentsOf: url))
                let layerWrappers = modelInspect.getLayerWrappers()
                self.mlFlwrClient = MLFlwrClient(layerWrappers: layerWrappers,
                                                 dataLoader: dataLoader,
                                                 compiledModelUrl: compiledModelUrl,
                                                 modelUrl: url)
            case .local:
                self.localClient = LocalClient(dataLoader: dataLoader, compiledModelUrl: compiledModelUrl)
            }
            
        } catch {
            log.error("\(error)")
        }
    }
    
    public func startLocalTrain() {
        let statusHandler: (Constants.TaskStatus) -> Void = { status in
            DispatchQueue.main.async {
                self.localTrainingStatus = status
                self.benchmarkSuite.takeActionSnapshot(snapshot: ActionSnaptshot(action: "local train status: \(status)"))
            }
        }
        localClient!.runMLTask(statusHandler: statusHandler, numEpochs: self.epoch, task: .train)
    }
    
    public func startLocalTest() {
        let statusHandler: (Constants.TaskStatus) -> Void = { status in
            DispatchQueue.main.async {
                self.localTestStatus = status
                self.benchmarkSuite.takeActionSnapshot(snapshot: ActionSnaptshot(action: "local test status: \(status)"))
            }
        }
        localClient!.runMLTask(statusHandler: statusHandler, numEpochs: 1, task: .test)
    }
    
    public func startFederatedLearning() {
        log.info("starting federated learning")
        self.federatedServerStatus = .ongoing(info: "Starting federated learning")
        self.benchmarkSuite.takeActionSnapshot(snapshot: ActionSnaptshot(action: "starting federated learning"))
        if self.flwrGRPC == nil {
            self.flwrGRPC = FlwrGRPC(serverHost: hostname, serverPort: port, extendedInterceptor: BenchmarkInterceptor())
        }
        
        self.flwrGRPC?.startFlwrGRPC(client: self.mlFlwrClient!) {
            DispatchQueue.main.async {
                self.federatedServerStatus = .completed(info: "Federated learning completed")
                self.flwrGRPC = nil
                self.benchmarkSuite.takeActionSnapshot(snapshot: ActionSnaptshot(action: "Federated learning completed"))
            }
        }
    }
    
    public func abortFederatedLearning() {
        log.info("aborting federated learning")
        self.flwrGRPC?.abortGRPCConnection(reasonDisconnect: .powerDisconnected) {
            DispatchQueue.main.async {
                self.federatedServerStatus = .completed(info: "Federated learning aborted")
                self.flwrGRPC = nil
                self.benchmarkSuite.takeActionSnapshot(snapshot: ActionSnaptshot(action: "Federated learning aborted"))
            }
        }
    }
}

class LocalClient {
    private var dataLoader: MLDataLoader
    private var compiledModelUrl: URL
    private let log = Logger(subsystem: Bundle.main.bundleIdentifier ?? "flwr.Flower",
                                    category: String(describing: LocalClient.self))
    
    init(dataLoader: MLDataLoader, compiledModelUrl: URL) {
        self.dataLoader = dataLoader
        self.compiledModelUrl = compiledModelUrl
    }
    
    func runMLTask(statusHandler: @escaping (Constants.TaskStatus) -> Void,
                   numEpochs: Int,
                   task: MLTask
    ) {
        let dataset: MLBatchProvider
        let configuration = MLModelConfiguration()
        let epochs = MLParameterKey.epochs
        configuration.parameters = [epochs:numEpochs]
        
        switch task {
        case .train:
            dataset = self.dataLoader.trainBatchProvider
        case .test:
            dataset = self.dataLoader.testBatchProvider
        }
        
        var startTime = Date()
        let progressHandler = { (contextProgress: MLUpdateContext) in
            switch contextProgress.event {
            case .trainingBegin:
                let taskStatus: Constants.TaskStatus = .ongoing(info: "Started to \(task) locally")
                statusHandler(taskStatus)
            case .epochEnd:
                let taskStatus: Constants.TaskStatus
                let loss = String(format: "%.4f", contextProgress.metrics[.lossValue] as! Double)
                switch task {
                case .train:
                    let epochIndex = contextProgress.metrics[.epochIndex] as! Int
                    taskStatus = .ongoing(info: "Epoch \(epochIndex + 1) end with loss \(loss)")
                case .test:
                    taskStatus = .ongoing(info: "Local test end with loss \(loss)")
                }
                statusHandler(taskStatus)
            default:
                self.log.info("Unknown event")
            }
        }
        
        let completionHandler = { (finalContext: MLUpdateContext) in
            let loss = String(format: "%.4f", finalContext.metrics[.lossValue] as! Double)
            let taskStatus: Constants.TaskStatus = .completed(info: "Local \(task) completed with loss: \(loss) in \(Int(Date().timeIntervalSince(startTime))) secs")
            statusHandler(taskStatus)
        }
        
        
        let progressHandlers = MLUpdateProgressHandlers(
            forEvents: [.trainingBegin, .epochEnd],
            progressHandler: progressHandler,
            completionHandler: completionHandler
        )
        
        startTime = Date()
        do {
            let updateTask = try MLUpdateTask(forModelAt: compiledModelUrl,
                                              trainingData: dataset,
                                              configuration: configuration,
                                              progressHandlers: progressHandlers)
            updateTask.resume()
            
        } catch let error {
            log.error("\(error)")
        }
    }
}
