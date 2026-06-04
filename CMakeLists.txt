cmake_minimum_required(VERSION 3.22)
project(compton_cross_section LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)
find_package(Boost REQUIRED)

include(FetchContent)
FetchContent_Declare(
  doubledouble
  GIT_REPOSITORY https://github.com/WarrenWeckesser/doubledouble.git
  GIT_TAG 3db2c175e9e1adf2b0b324a9f5514836c6c6bc21
)
FetchContent_Populate(doubledouble)

FetchContent_Declare(
  planck_integral
  GIT_REPOSITORY git@github.com:menahemkrief/planck_integral.git
  GIT_TAG main
)
FetchContent_Populate(planck_integral)

# planck_integral.hpp includes "../units/units.hpp" via a relative path.
# Create a symlink so that path resolves to our src/units/.
file(CREATE_LINK
  "${PROJECT_SOURCE_DIR}/src/units"
  "${planck_integral_SOURCE_DIR}/../units"
  SYMBOLIC
)

include_directories(src)
include_directories(external/CMMC/src)
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${doubledouble_SOURCE_DIR}/include)
include_directories(${planck_integral_SOURCE_DIR})

pybind11_add_module(_compton_common
    src/compton_common/bind/_compton_common.cpp
)
target_compile_options(_compton_common PRIVATE
    -O3 -Wall -Wextra -Wpedantic
)
set_target_properties(_compton_common PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/cpp_modules"
)

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

pybind11_add_module(_compton_kernel_series
    src/compton_kernel_series/compton_kernel_series.cpp
    src/compton_kernel_series/bind/_compton_kernel_series.cpp
)
target_compile_options(_compton_kernel_series PRIVATE
    -O3 -Wall -Wextra -Wpedantic
)
set_target_properties(_compton_kernel_series PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/cpp_modules"
)


pybind11_add_module(_units
    src/units/bind/_units.cpp
)
target_compile_options(_units PRIVATE
    -O3 -Wall -Wextra -Wpedantic
)
set_target_properties(_units PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/cpp_modules"
)

pybind11_add_module(_compton_kernel_solver
    src/compton_kernel_solver/compton_kernel_solver.cpp
    src/compton_kernel_solver/bind/_compton_kernel_solver.cpp
    src/compton_kernel_quadrature/compton_kernel_quadrature.cpp
    src/compton_kernel_series/compton_kernel_series.cpp
)
target_compile_options(_compton_kernel_solver PRIVATE
    -O3 -Wall -Wextra -Wpedantic
)
set_target_properties(_compton_kernel_solver PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/cpp_modules"
)

pybind11_add_module(_compton_multigroup
    src/compton_multigroup/compton_multigroup.cpp
    src/compton_multigroup/weight_function.cpp
    src/compton_multigroup/bind/_compton_multigroup.cpp
    src/compton_kernel_solver/compton_kernel_solver.cpp
    src/compton_kernel_quadrature/compton_kernel_quadrature.cpp
    src/compton_kernel_series/compton_kernel_series.cpp
)
target_compile_options(_compton_multigroup PRIVATE
    -O3 -Wall -Wextra -Wpedantic
)
set_target_properties(_compton_multigroup PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/cpp_modules"
)

pybind11_add_module(_compton_multigroup_misc
    src/compton_multigroup/bind/_compton_multigroup_misc.cpp
)
target_compile_options(_compton_multigroup_misc PRIVATE
    -O3 -Wall -Wextra -Wpedantic
)
set_target_properties(_compton_multigroup_misc PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/cpp_modules"
)
