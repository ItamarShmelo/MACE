cmake_minimum_required(VERSION 3.25)
project(compton_matrix LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)
find_package(Boost REQUIRED)

option(COMPTON_ENABLE_OMP "Enable OpenMP parallelization" ON)
if(COMPTON_ENABLE_OMP)
    find_package(OpenMP REQUIRED)
endif()

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
  "${PROJECT_SOURCE_DIR}/src/utilities"
  "${planck_integral_SOURCE_DIR}/../units"
  SYMBOLIC
)

include_directories(src)
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${doubledouble_SOURCE_DIR}/include)
include_directories(${planck_integral_SOURCE_DIR})

pybind11_add_module(_compton_common
    src/compton_common/bind/_compton_common.cpp
)
target_compile_options(_compton_common PRIVATE
    -O3 -Wall -Wextra -Wpedantic
)

pybind11_add_module(_compton_differential_cross_section
    src/compton_differential_cross_section/bind/_compton_differential_cross_section.cpp
    src/compton_differential_cross_section/compton_kernel_power_series/compton_kernel_power_series.cpp
    src/compton_differential_cross_section/compton_kernel_asymptotic_series/compton_kernel_asymptotic_series.cpp
    src/compton_differential_cross_section/compton_kernel_quadrature/compton_kernel_quadrature.cpp
    src/compton_differential_cross_section/compton_kernel_solver/compton_kernel_solver.cpp
    src/compton_differential_cross_section/compton_kernel_approximate/compton_kernel_approximate.cpp
    src/compton_differential_cross_section/compton_kernel_approximate_solver/compton_kernel_approximate_solver.cpp
)
target_compile_options(_compton_differential_cross_section PRIVATE
    -O3 -Wall -Wextra -Wpedantic
)

pybind11_add_module(_units
    src/utilities/bind/_units.cpp
)
target_compile_options(_units PRIVATE
    -O3 -Wall -Wextra -Wpedantic
)

pybind11_add_module(_compton_multigroup
    src/compton_multigroup/compton_multigroup_deterministic/compton_multigroup_deterministic.cpp
    src/compton_multigroup/compton_multigroup_monte_carlo/compton_multigroup_monte_carlo.cpp
    src/compton_multigroup/weight_function.cpp
    src/compton_multigroup/bind/_compton_multigroup.cpp
    src/compton_differential_cross_section/compton_kernel_solver/compton_kernel_solver.cpp
    src/compton_differential_cross_section/compton_kernel_quadrature/compton_kernel_quadrature.cpp
    src/compton_differential_cross_section/compton_kernel_power_series/compton_kernel_power_series.cpp
    src/compton_differential_cross_section/compton_kernel_asymptotic_series/compton_kernel_asymptotic_series.cpp
)
target_compile_options(_compton_multigroup PRIVATE
    -O3 -Wall -Wextra -Wpedantic
)
if(COMPTON_ENABLE_OMP)
    target_link_libraries(_compton_multigroup PRIVATE OpenMP::OpenMP_CXX)
endif()

pybind11_add_module(_utilities
    src/utilities/bind/_utilities.cpp
)
target_compile_options(_utilities PRIVATE
    -O3 -Wall -Wextra -Wpedantic
)

pybind11_add_module(_compton_kernel_multipliers
    src/kernel_multipliers/bind/_compton_kernel_multipliers.cpp
)
target_compile_options(_compton_kernel_multipliers PRIVATE
    -O3 -Wall -Wextra -Wpedantic
)

install(TARGETS
    _compton_common
    _compton_differential_cross_section
    _compton_multigroup
    _compton_kernel_multipliers
    _units
    _utilities
    LIBRARY DESTINATION compton_matrix
)
