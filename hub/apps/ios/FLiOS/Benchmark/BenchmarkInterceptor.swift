//
//  BenchmarkInterceptor.swift
//  FLiOS
//
//  Created by Christoph Weinhuber on 22.01.23.
//

import Foundation
import flwr

class BenchmarkInterceptor: InterceptorExtension {
    let benchmarkSuite = BenchmarkSuite.shared
    
    func receive(part: GRPCPartWrapper) {
        switch part {
        
        case .metadata(_):
            benchmarkSuite.takeNetworkSnaptshot(snapshot: NetworkSnapshot(type: NetworkAction.received(size: "0 Bytes"), description: "Received headers"))
            
        case let .message(content):
            let size = NetworkSnapshot.calcSize(message: content)
            benchmarkSuite.takeNetworkSnaptshot(snapshot: NetworkSnapshot(type: NetworkAction.received(size: size), description: "Received response"))
            
        case .end(_, _):
            benchmarkSuite.takeActionSnapshot(snapshot: ActionSnaptshot(action: "Response stream closed"))
            benchmarkSuite.takeNetworkSnaptshot(snapshot: NetworkSnapshot(type: NetworkAction.received(size: "0 Bytes"), description: "Response stream closed"))
        }
        
    }
    
    
    func send(part: GRPCPartWrapper) {
        switch part {
        
        case .metadata(_):
            benchmarkSuite.takeNetworkSnaptshot(snapshot: NetworkSnapshot(type: NetworkAction.sent(size: "0 Bytes"), description: "Sent headers"))
            
        case let .message(content):
            let size = NetworkSnapshot.calcSize(message: content)
            benchmarkSuite.takeNetworkSnaptshot(snapshot: NetworkSnapshot(type: NetworkAction.sent(size: size), description: "Sent request"))
            
        case .end(_, _):
            benchmarkSuite.takeNetworkSnaptshot(snapshot: NetworkSnapshot(type: NetworkAction.sent(size: "0 Bytes"), description: "Closing request stream"))
            
        }
    }
}
