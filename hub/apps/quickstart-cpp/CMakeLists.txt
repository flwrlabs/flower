cmake_minimum_required(VERSION 3.16)
project(SimpleCppFlowerClient VERSION 0.10
  DESCRIPTION "Creates a Simple C++ Flower client that trains a linear model on synthetic data."
  LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
option(USE_LOCAL_FLWR "Use local Flower directory instead of fetching from GitHub" OFF)

######################
### Download gRPC 

include(FetchContent)
FetchContent_Declare(
  gRPC
  GIT_REPOSITORY https://github.com/grpc/grpc
  GIT_TAG        v1.43.2
)
set(FETCHCONTENT_QUIET OFF)
FetchContent_MakeAvailable(gRPC)

set(_PROTOBUF_LIBPROTOBUF libprotobuf)
set(_REFLECTION grpc++_reflection)
set(_PROTOBUF_PROTOC $<TARGET_FILE:protoc>)
set(_GRPC_GRPCPP grpc++)
if(CMAKE_CROSSCOMPILING)
    find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)
else()
    set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:grpc_cpp_plugin>)
endif()

######################
### FLWR_LIB
if(USE_LOCAL_FLWR)
  set(FLWR_SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
else()
  FetchContent_Declare(
    flwr_repo
    GIT_REPOSITORY https://github.com/flwrlabs/flower.git
    GIT_TAG        main
  )

  FetchContent_GetProperties(flwr_repo)
  if(NOT flwr_repo_POPULATED)
    FetchContent_Populate(flwr_repo)
  endif()

  set(FLWR_SOURCE_ROOT "${flwr_repo_SOURCE_DIR}")
endif()

set(FLWR_SDK_PATH "${FLWR_SOURCE_ROOT}/framework/cc/flwr")

file(GLOB FLWR_SRCS "${FLWR_SDK_PATH}/src/*.cc")
file(GLOB FLWR_PROTO_SRCS "${FLWR_SDK_PATH}/include/flwr/proto/*.cc")
set(FLWR_INCLUDE_DIR "${FLWR_SDK_PATH}/include")

add_library(flwr ${FLWR_SRCS} ${FLWR_PROTO_SRCS})

target_link_libraries(flwr
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF}
)

target_include_directories(flwr PUBLIC
  ${FLWR_INCLUDE_DIR}
)

######################
### FLWR_CLIENT
file(GLOB FLWR_CLIENT_SRCS src/*.cc)
set(EXECUTABLE_NAME flwr_client)
add_executable(${EXECUTABLE_NAME} ${FLWR_CLIENT_SRCS})

target_include_directories(${EXECUTABLE_NAME} PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  ${FLWR_INCLUDE_DIR}
)

target_link_libraries(${EXECUTABLE_NAME}
  flwr
)
