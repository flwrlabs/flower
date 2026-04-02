//
//  Constants.swift
//  FLiOS
//
//  Created by Maximilian Kapsecker on 14.02.23.
//

import Foundation

public enum Constants: Equatable {
    public enum ScenarioTypes: CustomStringConvertible, CaseIterable {
        // Add here further scenarios
        case MNIST
        /*case CIFAR
        case BOSTON
         */
        
        public var description: String {
            switch self {
            case .MNIST:
                return "MNIST"
            /*case .CIFAR:
                 return "CIFAR"
            case .BOSTON:
                 return "BOSTON"
             */
            }
        }
        public var modelName: String {
            switch self {
            case .MNIST:
                return "MNIST_Model"
                /*case .CIFAR:
                 return "CIFAR_Model"
            case .BOSTON:
                 return "BOSTON_Model"
                */
            }
        }
        public var shapeData: [NSNumber] {
          switch self {
          case .MNIST:
              return [1, 28, 28]
              /*case .CIFAR:
              return [1, 32, 32, 3]
          case .BOSTON:
              return [1, 13]*/
          }
        }
        public var shapeTarget: [NSNumber] {
          switch self {
          case .MNIST:
              return [1]
              /*case .CIFAR:
              return [1]
          case .BOSTON:
              return [1]
              */
          }
        }
        
        public var normalization: Float {
          switch self {
          case .MNIST:
              return 255.0
              /*case .CIFAR:
              return 255.0
          case .BOSTON:
              return 1.0
              */
          }
        }
    }
    
    public enum PreparationStatus: Comparable {
        case notPrepared
        case preparing(count: Int)
        case ready
        
        var description: String {
            switch self {
            case .notPrepared:
                return "Not Prepared"
            case .preparing(let count):
                return "Preparing \(count)"
            case .ready:
                return "Ready"
            }
        }
    }

    public enum TaskStatus: Equatable {
        case idle
        case ongoing(info: String)
        case completed(info: String)
        
        var description: String {
            switch self {
            case .idle:
                return "Not Yet Started"
            case .ongoing(let info):
                return info
            case .completed(let info):
                return info
            }
        }
        
        public static func ==(lhs: TaskStatus, rhs: TaskStatus) -> Bool {
            switch(lhs, rhs) {
            case (.idle, .idle):
                return true
            case (.ongoing(_), .ongoing(_)):
                return true
            case (.completed(_), .completed(_)):
                return true
            default:
                return false
            }
        }
    }

    public enum ClientType {
        case federated
        case local
    }
}
