cmake_minimum_required(VERSION 3.16)
project(flwr VERSION 1.0
  DESCRIPTION "Flower Library that packages gRPC and other dependencies"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Assume gRPC and other dependencies are necessary
include(FetchContent)
FetchContent_Declare(
  gRPC
  GIT_REPOSITORY https://github.com/grpc/grpc
  GIT_TAG        v1.43.2
)
FetchContent_MakeAvailable(gRPC)

# Variables for gRPC and Protocol Buffers
set(_PROTOBUF_LIBPROTOBUF libprotobuf)
set(_REFLECTION grpc++_reflection)
set(_PROTOBUF_PROTOC $<TARGET_FILE:protoc>)
set(_GRPC_GRPCPP grpc++)
if(CMAKE_CROSSCOMPILING)
    find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)
else()
    set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:grpc_cpp_plugin>)
endif()

# Paths and output directories
get_filename_component(FLWR_PROTO_BASE_PATH "../../proto/" ABSOLUTE)
set(INCLUDE_FLWR_PROTO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/flwr/proto")

# Generate source files and copy them
macro(GENERATE_AND_COPY PROTO_NAME)
    set(OUT_PROTO_SRCS "${CMAKE_CURRENT_BINARY_DIR}/flwr/proto/${PROTO_NAME}.pb.cc")
    set(OUT_PROTO_HDRS "${CMAKE_CURRENT_BINARY_DIR}/flwr/proto/${PROTO_NAME}.pb.h")
    set(OUT_GRPC_SRCS "${CMAKE_CURRENT_BINARY_DIR}/flwr/proto/${PROTO_NAME}.grpc.pb.cc")
    set(OUT_GRPC_HDRS "${CMAKE_CURRENT_BINARY_DIR}/flwr/proto/${PROTO_NAME}.grpc.pb.h")
    set(SOURCE_PROTO "${FLWR_PROTO_BASE_PATH}/flwr/proto/${PROTO_NAME}.proto")

    add_custom_command(
        OUTPUT "${OUT_PROTO_SRCS}" "${OUT_PROTO_HDRS}" "${OUT_GRPC_SRCS}" "${OUT_GRPC_HDRS}"
        COMMAND ${_PROTOBUF_PROTOC}
        ARGS  --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
              --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
              -I "${FLWR_PROTO_BASE_PATH}"
              --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
              "${SOURCE_PROTO}"
    )

    add_custom_command(
        OUTPUT "${INCLUDE_FLWR_PROTO_DIR}/${PROTO_NAME}.pb.cc"
               "${INCLUDE_FLWR_PROTO_DIR}/${PROTO_NAME}.pb.h"
               "${INCLUDE_FLWR_PROTO_DIR}/${PROTO_NAME}.grpc.pb.cc"
               "${INCLUDE_FLWR_PROTO_DIR}/${PROTO_NAME}.grpc.pb.h"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${OUT_PROTO_SRCS}" "${OUT_PROTO_HDRS}" "${OUT_GRPC_SRCS}" "${OUT_GRPC_HDRS}"
                "${INCLUDE_FLWR_PROTO_DIR}"
        DEPENDS "${OUT_PROTO_SRCS}" "${OUT_PROTO_HDRS}" "${OUT_GRPC_SRCS}" "${OUT_GRPC_HDRS}"
    )

    set(ALL_PROTO_FILES
        ${ALL_PROTO_FILES}
        "${INCLUDE_FLWR_PROTO_DIR}/${PROTO_NAME}.pb.cc"
        "${INCLUDE_FLWR_PROTO_DIR}/${PROTO_NAME}.pb.h"
        "${INCLUDE_FLWR_PROTO_DIR}/${PROTO_NAME}.grpc.pb.cc"
        "${INCLUDE_FLWR_PROTO_DIR}/${PROTO_NAME}.grpc.pb.h"
        CACHE INTERNAL "All generated proto files"
    )
endmacro()

# Using the above macro for all proto files
GENERATE_AND_COPY(transport)
GENERATE_AND_COPY(node)
GENERATE_AND_COPY(fab)
GENERATE_AND_COPY(run)
GENERATE_AND_COPY(message)
GENERATE_AND_COPY(fleet)
GENERATE_AND_COPY(error)
GENERATE_AND_COPY(recorddict)

add_library(flwr_grpc_proto STATIC ${ALL_PROTO_FILES})

target_include_directories(flwr_grpc_proto 
   PUBLIC 
       $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
   PRIVATE
       ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(flwr_grpc_proto
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF}
)

# For the internal use of flwr
file(GLOB FLWR_SRCS "src/*.cc")
add_library(flwr ${FLWR_SRCS})

target_include_directories(flwr PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# Link gRPC and other dependencies
target_link_libraries(flwr PRIVATE flwr_grpc_proto)
