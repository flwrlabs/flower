cmake_minimum_required(VERSION 3.16)
project(SimpleCppFlowerClient
  VERSION 1.0
  DESCRIPTION "Creates a simple C++ Flower client that trains a linear model on synthetic data."
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(FLWR_SOURCE_ROOT "" CACHE PATH "Path to the Flower repository root")
set(GRPC_CPP_PLUGIN_EXECUTABLE "" CACHE FILEPATH "Path to grpc_cpp_plugin")

find_package(Threads REQUIRED)
find_package(Protobuf REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(gRPC CONFIG QUIET)
find_package(PkgConfig QUIET)

if(TARGET protobuf::libprotobuf)
  set(QUICKSTART_PROTOBUF_LIB protobuf::libprotobuf)
else()
  set(QUICKSTART_PROTOBUF_LIB ${Protobuf_LIBRARIES})
endif()

if(gRPC_FOUND)
  set(QUICKSTART_GRPC_LIB gRPC::grpc++)
elseif(PkgConfig_FOUND)
  pkg_check_modules(GRPCPP REQUIRED IMPORTED_TARGET grpc++)
  set(QUICKSTART_GRPC_LIB PkgConfig::GRPCPP)
else()
  message(FATAL_ERROR "Could not locate gRPC. Install gRPC C++ or provide a CMake package.")
endif()

if(NOT GRPC_CPP_PLUGIN_EXECUTABLE)
  find_program(GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)
endif()
if(NOT GRPC_CPP_PLUGIN_EXECUTABLE)
  message(FATAL_ERROR "Could not find grpc_cpp_plugin")
endif()

if(FLWR_SOURCE_ROOT)
  get_filename_component(FLWR_SOURCE_ROOT "${FLWR_SOURCE_ROOT}" ABSOLUTE)
else()
  get_filename_component(FLWR_SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
endif()

set(FLWR_SDK_PATH "${FLWR_SOURCE_ROOT}/framework/cc/flwr")
set(FLWR_PROTO_ROOT "${FLWR_SOURCE_ROOT}/framework/proto")
if(NOT EXISTS "${FLWR_SDK_PATH}/include/client.h")
  message(FATAL_ERROR "Could not find Flower C++ SDK headers at ${FLWR_SDK_PATH}")
endif()
if(NOT EXISTS "${FLWR_PROTO_ROOT}/flwr/proto/fleet.proto")
  message(FATAL_ERROR "Could not find Flower proto sources at ${FLWR_PROTO_ROOT}")
endif()

set(GENERATED_PROTO_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(GLOB FLWR_PROTO_FILES CONFIGURE_DEPENDS "${FLWR_PROTO_ROOT}/flwr/proto/*.proto")
set(GENERATED_PROTO_SRCS)
set(GENERATED_PROTO_HDRS)

foreach(PROTO_FILE ${FLWR_PROTO_FILES})
  get_filename_component(PROTO_NAME "${PROTO_FILE}" NAME_WE)
  set(PB_CC "${GENERATED_PROTO_DIR}/flwr/proto/${PROTO_NAME}.pb.cc")
  set(PB_H "${GENERATED_PROTO_DIR}/flwr/proto/${PROTO_NAME}.pb.h")
  set(GRPC_CC "${GENERATED_PROTO_DIR}/flwr/proto/${PROTO_NAME}.grpc.pb.cc")
  set(GRPC_H "${GENERATED_PROTO_DIR}/flwr/proto/${PROTO_NAME}.grpc.pb.h")

  add_custom_command(
    OUTPUT "${PB_CC}" "${PB_H}" "${GRPC_CC}" "${GRPC_H}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${GENERATED_PROTO_DIR}"
    COMMAND ${Protobuf_PROTOC_EXECUTABLE}
      --proto_path=${FLWR_PROTO_ROOT}
      --cpp_out=${GENERATED_PROTO_DIR}
      --grpc_out=${GENERATED_PROTO_DIR}
      --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN_EXECUTABLE}
      "${PROTO_FILE}"
    DEPENDS "${PROTO_FILE}"
    VERBATIM)

  list(APPEND GENERATED_PROTO_SRCS "${PB_CC}" "${GRPC_CC}")
  list(APPEND GENERATED_PROTO_HDRS "${PB_H}" "${GRPC_H}")
endforeach()

add_library(flwr_proto STATIC ${GENERATED_PROTO_SRCS} ${GENERATED_PROTO_HDRS})
target_include_directories(flwr_proto PUBLIC
  "${GENERATED_PROTO_DIR}"
  ${Protobuf_INCLUDE_DIRS}
)
target_link_libraries(flwr_proto PUBLIC
  ${QUICKSTART_GRPC_LIB}
  ${QUICKSTART_PROTOBUF_LIB}
  Threads::Threads
)

file(GLOB FLWR_CLIENT_SRCS CONFIGURE_DEPENDS src/*.cc)
add_executable(flwr_client ${FLWR_CLIENT_SRCS})
target_include_directories(flwr_client PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${GENERATED_PROTO_DIR}"
  "${FLWR_SDK_PATH}/include"
)
target_link_libraries(flwr_client PRIVATE flwr_proto ${QUICKSTART_GRPC_LIB})
target_link_libraries(flwr_client PRIVATE OpenSSL::Crypto)
