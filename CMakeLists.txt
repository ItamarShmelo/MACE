cmake_minimum_required(VERSION 3.22)
project(compton_cross_section LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)
find_package(Boost REQUIRED)

include_directories(src)
include_directories(external/CMMC/src)
include_directories(${Boost_INCLUDE_DIRS})

pybind11_add_module(_compton_kernel_quadrature
    src/compton_kernel_quadrature/compton_kernel_quadrature.cpp
    src/compton_kernel_quadrature/bind/_compton_kernel_quadrature.cpp
)
target_compile_options(_compton_kernel_quadrature PRIVATE
    -O3 -Wall -Wextra -Wpedantic
)
set_target_properties(_compton_kernel_quadrature PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/cpp_modules"
)

pybind11_add_module(_units
    src/units/bind_units.cpp
)
target_compile_options(_units PRIVATE
    -O3 -Wall -Wextra -Wpedantic
)
set_target_properties(_units PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/cpp_modules"
)
