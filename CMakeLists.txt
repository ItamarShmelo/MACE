cmake_minimum_required(VERSION 3.25)
project(compton_matrix LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(MACE_BUILD_PYTHON "Build the pybind11 extension modules" ON)
option(MACE_BUILD_TESTS "Build the C++ library tests" OFF)
option(MACE_ENABLE_MPI "Build the MPI table builder and the build_compton_table tool" ON)
set(MACE_PLANCK_INTEGRAL_HEADER "" CACHE FILEPATH
    "Use this planck_integral.hpp instead of fetching planck_integral over SSH")

if(MACE_BUILD_PYTHON)
    find_package(pybind11 REQUIRED)
endif()
find_package(Boost REQUIRED)

option(COMPTON_ENABLE_OMP "Enable OpenMP parallelization" ON)
option(COMPTON_BUILD_BENCHMARKS "Build approximate-kernel benchmarks" OFF)
if(COMPTON_ENABLE_OMP)
    find_package(OpenMP REQUIRED)
endif()

include(FetchContent)
# doubledouble: the pinned submodule deps/doubledouble when checked out,
# otherwise fetched at the same commit.
if(EXISTS "${PROJECT_SOURCE_DIR}/deps/doubledouble/include/doubledouble.h")
    set(doubledouble_SOURCE_DIR "${PROJECT_SOURCE_DIR}/deps/doubledouble")
else()
    FetchContent_Declare(
      doubledouble
      GIT_REPOSITORY https://github.com/WarrenWeckesser/doubledouble.git
      GIT_TAG 3db2c175e9e1adf2b0b324a9f5514836c6c6bc21
    )
    FetchContent_Populate(doubledouble)
endif()

# planck_integral.hpp includes "../units/units.hpp" via a relative path, so it
# needs our src/utilities next to it as "units": a given header is copied into
# the build tree next to such a link; otherwise the repository is fetched and
# the link is created next to its checkout.
if(MACE_PLANCK_INTEGRAL_HEADER)
    set(planck_integral_SOURCE_DIR "${PROJECT_BINARY_DIR}/mace_deps/planck_integral")
    configure_file("${MACE_PLANCK_INTEGRAL_HEADER}"
        "${planck_integral_SOURCE_DIR}/planck_integral.hpp" COPYONLY)
else()
    FetchContent_Declare(
      planck_integral
      GIT_REPOSITORY git@github.com:menahemkrief/planck_integral.git
      GIT_TAG main
    )
    FetchContent_Populate(planck_integral)
endif()
file(CREATE_LINK
  "${PROJECT_SOURCE_DIR}/src/utilities"
  "${planck_integral_SOURCE_DIR}/../units"
  SYMBOLIC
)

include_directories(src)
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${doubledouble_SOURCE_DIR}/include)
include_directories(${planck_integral_SOURCE_DIR})

# C++ library: the kernel solvers and the deterministic multigroup integrator.
add_library(mace_core STATIC
    src/compton_differential_cross_section/compton_kernel_approximate/compton_kernel_approximate.cpp
    src/compton_differential_cross_section/compton_kernel_approximate_solver/compton_kernel_approximate_solver.cpp
    src/compton_differential_cross_section/compton_kernel_power_series/compton_kernel_power_series.cpp
    src/compton_differential_cross_section/compton_kernel_asymptotic_series/compton_kernel_asymptotic_series.cpp
    src/compton_differential_cross_section/compton_kernel_quadrature/compton_kernel_quadrature.cpp
    src/compton_differential_cross_section/compton_kernel_solver/compton_kernel_solver.cpp
    src/compton_multigroup/compton_multigroup_deterministic/compton_multigroup_deterministic.cpp
    src/compton_multigroup/weight_function.cpp
)
target_include_directories(mace_core PUBLIC
    "${PROJECT_SOURCE_DIR}/src"
    "${doubledouble_SOURCE_DIR}/include"
    "${planck_integral_SOURCE_DIR}"
    ${Boost_INCLUDE_DIRS}
)
target_compile_features(mace_core PUBLIC cxx_std_23)
set_target_properties(mace_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(mace_core PRIVATE -O3 -Wall -Wextra -Wpedantic)
if(COMPTON_ENABLE_OMP)
    target_link_libraries(mace_core PUBLIC OpenMP::OpenMP_CXX)
endif()

if(MACE_BUILD_TESTS)
    enable_testing()
    add_executable(test_mace_core tests/cpp/test_mace_core.cpp)
    target_link_libraries(test_mace_core PRIVATE mace_core)
    add_test(NAME mace_core COMMAND test_mace_core)
endif()

if(MACE_BUILD_PYTHON)

pybind11_add_module(_compton_common
    src/compton_common/bind/_compton_common.cpp
)
target_compile_options(_compton_common PRIVATE
    -O3 -Wall -Wextra -Wpedantic
)

pybind11_add_module(_compton_differential_cross_section
    src/compton_differential_cross_section/bind/_compton_differential_cross_section.cpp
    src/compton_differential_cross_section/compton_kernel_approximate/compton_kernel_approximate.cpp
    src/compton_differential_cross_section/compton_kernel_approximate_solver/compton_kernel_approximate_solver.cpp
    src/compton_differential_cross_section/compton_kernel_power_series/compton_kernel_power_series.cpp
    src/compton_differential_cross_section/compton_kernel_asymptotic_series/compton_kernel_asymptotic_series.cpp
    src/compton_differential_cross_section/compton_kernel_quadrature/compton_kernel_quadrature.cpp
    src/compton_differential_cross_section/compton_kernel_solver/compton_kernel_solver.cpp
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

endif() # MACE_BUILD_PYTHON

if(COMPTON_BUILD_BENCHMARKS)
    add_executable(compton_kernel_approximate_benchmark
        benchmarks/benchmark_approximate_solver.cpp
        src/compton_differential_cross_section/compton_kernel_approximate/compton_kernel_approximate.cpp
        src/compton_differential_cross_section/compton_kernel_approximate_solver/compton_kernel_approximate_solver.cpp
        src/compton_differential_cross_section/compton_kernel_solver/compton_kernel_solver.cpp
        src/compton_differential_cross_section/compton_kernel_power_series/compton_kernel_power_series.cpp
        src/compton_differential_cross_section/compton_kernel_asymptotic_series/compton_kernel_asymptotic_series.cpp
    )
    target_compile_options(compton_kernel_approximate_benchmark PRIVATE
        -O3 -Wall -Wextra -Wpedantic
    )

    add_executable(compton_multigroup_approximate_benchmark
        benchmarks/benchmark_multigroup.cpp
        src/compton_multigroup/compton_multigroup_deterministic/compton_multigroup_deterministic.cpp
        src/compton_multigroup/weight_function.cpp
        src/compton_differential_cross_section/compton_kernel_approximate/compton_kernel_approximate.cpp
        src/compton_differential_cross_section/compton_kernel_approximate_solver/compton_kernel_approximate_solver.cpp
        src/compton_differential_cross_section/compton_kernel_solver/compton_kernel_solver.cpp
        src/compton_differential_cross_section/compton_kernel_power_series/compton_kernel_power_series.cpp
        src/compton_differential_cross_section/compton_kernel_asymptotic_series/compton_kernel_asymptotic_series.cpp
    )
    target_compile_options(compton_multigroup_approximate_benchmark PRIVATE
        -O3 -Wall -Wextra -Wpedantic
    )
    if(COMPTON_ENABLE_OMP)
        target_link_libraries(
            compton_multigroup_approximate_benchmark
            PRIVATE OpenMP::OpenMP_CXX
        )
    endif()

    add_executable(compton_multigroup_approximate_stress_benchmark
        benchmarks/benchmark_multigroup_stress.cpp
        src/compton_multigroup/compton_multigroup_deterministic/compton_multigroup_deterministic.cpp
        src/compton_multigroup/weight_function.cpp
        src/compton_differential_cross_section/compton_kernel_approximate/compton_kernel_approximate.cpp
        src/compton_differential_cross_section/compton_kernel_approximate_solver/compton_kernel_approximate_solver.cpp
        src/compton_differential_cross_section/compton_kernel_solver/compton_kernel_solver.cpp
        src/compton_differential_cross_section/compton_kernel_power_series/compton_kernel_power_series.cpp
        src/compton_differential_cross_section/compton_kernel_asymptotic_series/compton_kernel_asymptotic_series.cpp
    )
    target_compile_options(
        compton_multigroup_approximate_stress_benchmark
        PRIVATE -O3 -Wall -Wextra -Wpedantic
    )
    if(COMPTON_ENABLE_OMP)
        target_link_libraries(
            compton_multigroup_approximate_stress_benchmark
            PRIVATE OpenMP::OpenMP_CXX
        )
    endif()
endif()

if(MACE_BUILD_PYTHON)
install(TARGETS
    _compton_common
    _compton_differential_cross_section
    _compton_multigroup
    _compton_kernel_multipliers
    _units
    _utilities
    LIBRARY DESTINATION compton_matrix
)
endif()
