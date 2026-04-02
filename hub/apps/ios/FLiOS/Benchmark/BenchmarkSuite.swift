//
//  BenchmarkSuite.swift
//
//
//  Created by Christoph Weinhuber on 10.01.23.
//

import Foundation
import SwiftUI
import os

public class BenchmarkSuite: ObservableObject {
    
    public static let shared = BenchmarkSuite(interval: 60.0)

    private var batteryHistory : [BatterySnapshot] = []
    
    // stores every data ever sent
    private var networkHistory : [NetworkSnapshot] = []
    
    // stores every action which happened
    private var actionHistory : [ActionSnaptshot] = []
    
    // interval in seconds
    private var interval: Double
    
    private var deviceID: String
    
    private let log = Logger(subsystem: Bundle.main.bundleIdentifier ?? "flwr.Flower",
                                    category: String(describing: BenchmarkSuite.self))
    
    private init(interval: Double = 60.0) {
        
        self.interval = interval
        self.deviceID = UIDevice.current.identifierForVendor!.uuidString
        
        UIDevice.current.isBatteryMonitoringEnabled = true
        
        self.startMonitoring()
    }
    
    
    public func takeBatterySnapshot() {
        let battery = Int(UIDevice.current.batteryLevel * 100)
        self.batteryHistory.append(BatterySnapshot(batteryLevel: battery))

    }
    
    public func takeNetworkSnaptshot(snapshot: NetworkSnapshot) {
        self.networkHistory.append(snapshot)
        self.takeBatterySnapshot()
        
    }
    
    public func takeActionSnapshot(snapshot: ActionSnaptshot) {
        self.actionHistory.append(snapshot)
        self.takeBatterySnapshot()
    }
    
    private func startMonitoring() {
        self.takeBatterySnapshot()
        
        Timer.scheduledTimer(withTimeInterval: self.interval, repeats: true) { _ in
            self.takeBatterySnapshot()
            }
        }
    
    private func getDocumentsDirectory() -> URL {
        // find all possible documents directories for this user
        let paths = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)

        // just send back the first one, which ought to be the only one
        return paths[0]
    }
    
    public func getBenchmarkFileUrl() -> URL {
        return getDocumentsDirectory().appendingPathComponent("benchmark\(self.deviceID).json")
    }
    
    public func benchmarkExists() -> Bool {
        return FileManager.default.fileExists(atPath: self.getBenchmarkFileUrl().path)
    }
    
    public func exportBenchmark() {
        let url = self.getBenchmarkFileUrl()
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        
        do {
            let completeBenchmark = try encoder.encode(BenchmarkSnaptshot(batteryHistory: self.batteryHistory, networkHistory: self.networkHistory, actionHistory: self.actionHistory, deviceID: self.deviceID))
            let benchmarkString = String(data: completeBenchmark, encoding: .utf8)!
            try benchmarkString.write(to: url, atomically: true, encoding: .utf8)
        } catch {
            log.error("Error while encoding complete Benchmark history.")
        }
    }
}


