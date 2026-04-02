//
//  ContentView.swift
//  FlowerCoreML
//
//  Created by Daniel Nugraha on 24.06.22.
//

import SwiftUI

struct ContentView: View {
    @ObservedObject var model = FLiOSModel()
    @State var preparedExport = false
    
    var numberFormatter: NumberFormatter = {
        var nf = NumberFormatter()
        nf.usesGroupingSeparator = false
        nf.numberStyle = .none
        return nf
    }()
    
    var body: some View {
        GeometryReader { geometry in
            VStack(spacing: 0) {
                Spacer().frame(height: 50)
                Text("Flower iOS Prototype")
                    .font(.largeTitle)
                Spacer()
                Form {
                    Section(header: Text("Scenario")) {
                        HStack{
                            Picker("Select a Scenario", selection: $model.scenarioSelection) {
                                ForEach(model.scenarios, id: \.self) {
                                    Text($0.description)
                                }
                            }
                        }
                    }
                    Section(header: Text("Prepare Dataset")) {
                        HStack {
                            Text("Training Dataset: \(self.model.trainingBatchStatus.description)")
                            Spacer()
                            Button(action: {
                                model.prepareTrainDataset()
                            }) {
                                switch self.model.trainingBatchStatus {
                                case .notPrepared:
                                    Text("Start")
                                case .ready:
                                    Image(systemName: "checkmark")
                                default:
                                    ProgressView()
                                }
                            }
                        }
                        HStack {
                            Text("Test Dataset: \(self.model.testBatchStatus.description)")
                            Spacer()
                            Button(action: {
                                model.prepareTestDataset()
                            }) {
                                switch self.model.testBatchStatus {
                                case .notPrepared:
                                    Text("Start")
                                case .ready:
                                    Image(systemName: "checkmark")
                                default:
                                    ProgressView()
                                }
                            }
                            .disabled(model.trainingBatchStatus != Constants.PreparationStatus.ready)
                        }
                    }
                    Section(header: Text("Local Training")) {
                        HStack {
                            Text("Prepare Local Client")
                            Spacer()
                            Button(action: {
                                model.initLocalClient()
                            }) {
                                switch model.localClientStatus {
                                case .notPrepared:
                                    Text("Start")
                                case .ready:
                                    Image(systemName: "checkmark")
                                default:
                                    ProgressView()
                                }
                            }
                            .disabled(model.testBatchStatus != .ready || model.trainingBatchStatus != .ready)
                        }
                        Stepper(value: $model.epoch, in: 1...10, label: { Text("Epoch: \(model.epoch)")})
                            .disabled(model.localClientStatus != .ready)
                        HStack {
                            switch model.localTrainingStatus {
                            case .idle:
                                Text("Local Train")
                            default:
                                Text(model.localTrainingStatus.description)
                            }
                            Spacer()
                            Button(action: {
                                model.startLocalTrain()
                            }) {
                                switch model.localTrainingStatus {
                                case .idle:
                                    Text("Start")
                                case .completed:
                                    Image(systemName: "checkmark")
                                default:
                                    ProgressView()
                                }
                            }
                            .disabled(model.localClientStatus != .ready)
                        }
                        HStack {
                            switch model.localTestStatus {
                            case .idle:
                                Text("Local Test")
                            default:
                                Text(model.localTestStatus.description)
                            }
                            Spacer()
                            Button(action: {
                                model.startLocalTest()
                            }) {
                                switch model.localTestStatus {
                                case .idle:
                                    Text("Start")
                                case .completed:
                                    Image(systemName: "checkmark")
                                default:
                                    ProgressView()
                                }
                            }
                            .disabled(model.localTrainingStatus != .completed(info: ""))
                        }
                    }
                    Section(header: Text("Federated Learning")) {
                        HStack {
                            Text("Prepare Federated Client")
                            Spacer()
                            Button(action: {
                                model.initMLFlwrClient()
                            }) {
                                switch model.mlFlwrClientStatus {
                                case .notPrepared:
                                    Text("Start")
                                case .ready:
                                    Image(systemName: "checkmark")
                                default:
                                    ProgressView()
                                }
                            }
                            .disabled(model.testBatchStatus != .ready || model.trainingBatchStatus != .ready)
                        }
                        HStack {
                            Text("Server Hostname: ")
                            TextField("Server Hostname", text: $model.hostname)
                                .multilineTextAlignment(.trailing)
                        }
                        HStack {
                            Text("Server Port: ")
                            TextField( "Server Port", value: $model.port, formatter: numberFormatter)
                                .multilineTextAlignment(.trailing)
                        }
                        HStack {
                            if model.federatedServerStatus == .ongoing(info: "") {
                                Button(action: {
                                    model.abortFederatedLearning()
                                }) {
                                    Text("Stop").foregroundColor(.red)
                                }
                            }
                            Spacer()
                            Button(action: {
                                model.startFederatedLearning()
                            }) {
                                switch model.federatedServerStatus {
                                case .idle:
                                    Text("Start")
                                case .completed:
                                    Text("Rerun FL")
                                default:
                                    ProgressView()
                                }
                                
                            }
                            .disabled(model.mlFlwrClientStatus != .ready)
                        }
                    }
                    Section(header: Text("Benchmark")) {
                        HStack{
                            Text("Prepare Benchmark Export")
                            Spacer()
                            Button(action: {
                                model.benchmarkSuite.exportBenchmark()
                                preparedExport = true
                            }) {
                                Text("Start").disabled(preparedExport)
                            }
                        }
                        
                        if model.benchmarkSuite.benchmarkExists() || preparedExport {
                            ShareLink(item:model.benchmarkSuite.getBenchmarkFileUrl())
                        }
                    }
                }
            }
            .background(Color(UIColor.systemGray6))
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
