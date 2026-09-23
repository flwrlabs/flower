cmake_minimum_required(VERSION 3.16)
project(SimpleCppFlowerClient
  VERSION 1.0
  DESCRIPTION "Creates a simple C++ Flower client that trains a linear model on synthetic data."
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(FLWR_SOURCE_ROOT "" CACHE PATH "Path to the Flower repository root")

find_package(Threads REQUIRED)

# Prefer protobuf's own CMake package (CONFIG mode). CMake's built-in
# FindProtobuf module and protobuf's own package both define `protobuf::*`
# targets, and gRPC's package pulls in the latter. Mixing the two trips the
# export-set check in protobuf-targets.cmake with "Some (but not all) targets in
# this export set were already defined". MODULE_COMPATIBLE keeps the
# FindProtobuf-style `Protobuf_*` variables this file relies on available.
set(protobuf_MODULE_COMPATIBLE ON CACHE BOOL
  "Expose FindProtobuf-compatible Protobuf_* variables")
find_package(Protobuf CONFIG QUIET)
if(NOT Protobuf_FOUND)
  # No protobuf CMake package: fall back to CMake's FindProtobuf module.
  find_package(Protobuf REQUIRED)
endif()

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

# `find_program` populates the cache entry itself. Declaring
# `GRPC_CPP_PLUGIN_EXECUTABLE` as an empty CACHE variable first would make the
# search a no-op, so the automatic lookup below would silently fail and the
# build would depend on passing -DGRPC_CPP_PLUGIN_EXECUTABLE explicitly.
find_program(GRPC_CPP_PLUGIN_EXECUTABLE
  NAMES grpc_cpp_plugin
  DOC "Path to grpc_cpp_plugin")
if(NOT GRPC_CPP_PLUGIN_EXECUTABLE)
  message(FATAL_ERROR "Could not find grpc_cpp_plugin. Install the gRPC C++ "
    "tools or pass -DGRPC_CPP_PLUGIN_EXECUTABLE=/path/to/grpc_cpp_plugin")
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

  # The Flower protos import each other (e.g. message.proto imports
  # recorddict.proto), so every generated source depends on the whole proto set.
  # Depending only on ${PROTO_FILE} would leave stale *.pb.cc files behind when
  # an imported proto changes.
  add_custom_command(
    OUTPUT "${PB_CC}" "${PB_H}" "${GRPC_CC}" "${GRPC_H}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${GENERATED_PROTO_DIR}"
    COMMAND ${Protobuf_PROTOC_EXECUTABLE}
      --proto_path=${FLWR_PROTO_ROOT}
      --cpp_out=${GENERATED_PROTO_DIR}
      --grpc_out=${GENERATED_PROTO_DIR}
      --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN_EXECUTABLE}
      "${PROTO_FILE}"
    DEPENDS ${FLWR_PROTO_FILES}
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

# Keep the native and cross-language regressions in the standard CTest suite.
include(CTest)
if(BUILD_TESTING)
  find_package(Python3 3.11 REQUIRED COMPONENTS Interpreter)
  add_executable(test_tensor_order
    test_tensor_order.cc
    src/recorddict_serde.cc
  )
  target_include_directories(test_tensor_order PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${GENERATED_PROTO_DIR}"
    "${FLWR_SDK_PATH}/include"
  )
  target_link_libraries(test_tensor_order PRIVATE flwr_proto ${QUICKSTART_GRPC_LIB})
  add_test(NAME tensor_order COMMAND test_tensor_order)

  add_executable(test_transport_client
    test_transport_client.cc
    src/recorddict_serde.cc
    src/message_handler.cc
    src/supernode_client.cc
  )
  target_include_directories(test_transport_client PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${GENERATED_PROTO_DIR}"
    "${FLWR_SDK_PATH}/include"
  )
  target_link_libraries(test_transport_client PRIVATE
    flwr_proto ${QUICKSTART_GRPC_LIB} OpenSSL::Crypto)

  add_test(NAME python_serialization
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_fedavg_cpp.py" -v)
  add_test(NAME transport_python
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_transport.py"
      --transport-client $<TARGET_FILE:test_transport_client>
      --client $<TARGET_FILE:flwr_client> -v)
  set_tests_properties(python_serialization transport_python PROPERTIES TIMEOUT 60)
endif()
