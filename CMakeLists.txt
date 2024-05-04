cmake_minimum_required(VERSION 3.16)

project(surakarta-alphazero)
enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# add_compile_options(-fsanitize=address)
# add_compile_options(-g)
# add_link_options(-fsanitize=address)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Do not enable -Wall -Wextra due to the third-party code
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # Do not enable -Wall -Wextra due to the third-party code
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /w14640")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2")
endif()

add_subdirectory(third-party/surakarta-core)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third-party/surakarta-core/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third-party/tiny-dnn)

SET(SURAKARTA_ALPHAZERO_SOURCE
    src/surakarta_agent_alphazero.cpp
    src/surakarta_alphazero_mcts.cpp
)
add_library(surakarta-alphazero STATIC ${SURAKARTA_ALPHAZERO_SOURCE})
target_link_libraries(surakarta-alphazero surakarta)

SET(SURAKARTA_ALPHAZERO_TRAIN_SOURCE
    src/train.cpp
)
add_executable(surakarta-alphazero-train ${SURAKARTA_ALPHAZERO_TRAIN_SOURCE})
target_link_libraries(surakarta-alphazero-train surakarta-alphazero)

SET(SURAKARTA_ALPHAZERO_BENCHMARK_SOURCE
    src/benchmark.cpp
)
add_executable(surakarta-alphazero-benchmark ${SURAKARTA_ALPHAZERO_BENCHMARK_SOURCE})
target_link_libraries(surakarta-alphazero-benchmark surakarta-alphazero)

install(TARGETS surakarta-alphazero-train surakarta-alphazero-benchmark)
