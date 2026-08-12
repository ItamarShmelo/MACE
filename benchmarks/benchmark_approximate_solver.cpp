#include "compton_differential_cross_section/compton_kernel_approximate/compton_kernel_approximate.hpp"
#include "compton_differential_cross_section/compton_kernel_approximate_solver/compton_kernel_approximate_solver.hpp"
#include "compton_differential_cross_section/compton_kernel_solver/compton_kernel_solver.hpp"
#include "utilities/units.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

struct Point {
    double E;
    double E_prime;
    double xi;
    double T;
};

struct Timing {
    double minimum;
    double median;
    double maximum;
    double checksum;
};

std::vector<Point> make_grid()
{
    std::array<double, 9> const temperatures = {
        0.01,
        0.1,
        1.0,
        5.0,
        10.0,
        25.0,
        50.0,
        70.0,
        100.0};
    std::array<double, 6> const ratios = {1e-3, 1e-2, 0.1, 1.0, 3.0, 10.0};
    std::array<double, 8> const angles = {
        -0.999,
        -0.9,
        -0.5,
        0.0,
        0.5,
        0.9,
        0.99,
        0.999};
    std::array<double, 7> const offsets = {-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0};

    std::vector<Point> points;
    points.reserve(
        temperatures.size() * ratios.size() * angles.size() * offsets.size());
    for (double const kT_kev : temperatures) {
        double const tau = kT_kev * units::kev / units::me_c2;
        for (double const ratio : ratios) {
            double const E = ratio * kT_kev * units::kev;
            double const gamma = E / units::me_c2;
            for (double const xi : angles) {
                double const cold_line =
                    gamma / (1.0 + gamma * (1.0 - xi));
                for (double const offset : offsets) {
                    double const gamma_prime = cold_line * std::exp(
                        offset * std::sqrt(2.0 * tau * (1.0 - xi)));
                    points.push_back(
                        {E,
                         gamma_prime * units::me_c2,
                         xi,
                         kT_kev * units::kev_kelvin});
                }
            }
        }
    }
    return points;
}

template <typename Function>
Timing time_function(Function&& function, std::size_t const evaluations)
{
    std::vector<double> samples;
    double checksum = 0.0;
    for (int trial = 0; trial < 9; ++trial) {
        auto const start = std::chrono::steady_clock::now();
        checksum += function();
        auto const stop = std::chrono::steady_clock::now();
        samples.push_back(
            std::chrono::duration<double, std::nano>(stop - start).count() /
            static_cast<double>(evaluations));
    }
    std::ranges::sort(samples);
    return {samples.front(), samples[4], samples.back(), checksum};
}

double percentile(std::vector<double> values, double const fraction)
{
    std::ranges::sort(values);
    return values[static_cast<std::size_t>(
        fraction * static_cast<double>(values.size() - 1))];
}

} // namespace

int main()
{
    auto const points = make_grid();
    compton::ComptonKernelApproximate approximate;
    compton::ComptonKernelApproximateSolver new_solver;
    compton::ComptonKernelSolver original_solver;

    std::vector<double> relative_errors;
    std::vector<double> approximate_relative_errors;
    std::vector<Point> accepted_approximate_points;
    std::size_t failures = 0;
    std::size_t approximate_failures = 0;
    std::size_t approximate_rejections = 0;
    for (Point const& point : points) {
        auto const reference = original_solver.sigma_E(
            point.E,
            point.E_prime,
            point.xi,
            point.T);
        auto const candidate = new_solver.sigma_E(
            point.E,
            point.E_prime,
            point.xi,
            point.T);
        if (candidate.estimated_abs_error == 1.0 && candidate.value == 0.0) {
            ++failures;
        } else if (reference.value != 0.0) {
            relative_errors.push_back(
                std::abs(candidate.value - reference.value) /
                std::abs(reference.value));
        }

        auto const approximation = approximate.sigma_E(
            point.E,
            point.E_prime,
            point.xi,
            point.T);
        if (approximation.estimated_abs_error == 1.0) {
            ++approximate_failures;
        } else if (approximation.estimated_rel_error >= 3e-4) {
            ++approximate_rejections;
        } else {
            accepted_approximate_points.push_back(point);
            if (reference.value != 0.0) {
                approximate_relative_errors.push_back(
                    std::abs(approximation.value - reference.value) /
                    std::abs(reference.value));
            }
        }
    }

    constexpr int REPEATS = 120;
    std::size_t const evaluations = points.size() * REPEATS;
    auto const approximate_timing = time_function(
        [&] {
            double sum = 0.0;
            for (int repeat = 0; repeat < REPEATS; ++repeat) {
                for (Point const& point : accepted_approximate_points) {
                    sum += approximate
                               .sigma_E(
                                   point.E,
                                   point.E_prime,
                                   point.xi,
                                   point.T)
                               .value;
                }
            }
            return sum;
        },
        accepted_approximate_points.size() * REPEATS);
    auto const new_timing = time_function(
        [&] {
            double sum = 0.0;
            for (int repeat = 0; repeat < REPEATS; ++repeat) {
                for (Point const& point : points) {
                    sum += new_solver
                               .sigma_E(
                                   point.E,
                                   point.E_prime,
                                   point.xi,
                                   point.T)
                               .value;
                }
            }
            return sum;
        },
        evaluations);
    auto const original_timing = time_function(
        [&] {
            double sum = 0.0;
            for (int repeat = 0; repeat < REPEATS; ++repeat) {
                for (Point const& point : points) {
                    sum += original_solver
                               .sigma_E(
                                   point.E,
                                   point.E_prime,
                                   point.xi,
                                   point.T)
                               .value;
                }
            }
            return sum;
        },
        evaluations);

    std::cout << std::setprecision(12)
              << "points " << points.size() << '\n'
              << "failures " << failures << '\n'
              << "approximate_accepted "
              << accepted_approximate_points.size() << '\n'
              << "approximate_failures " << approximate_failures << '\n'
              << "approximate_rejections " << approximate_rejections << '\n'
              << "approximate_relative_max "
              << *std::ranges::max_element(approximate_relative_errors) << '\n'
              << "relative_median " << percentile(relative_errors, 0.5) << '\n'
              << "relative_p95 " << percentile(relative_errors, 0.95) << '\n'
              << "relative_p99 " << percentile(relative_errors, 0.99) << '\n'
              << "relative_max "
              << *std::ranges::max_element(relative_errors) << '\n'
              << "approximate_ns " << approximate_timing.minimum << ' '
              << approximate_timing.median << ' ' << approximate_timing.maximum
              << '\n'
              << "new_solver_ns " << new_timing.minimum << ' '
              << new_timing.median << ' ' << new_timing.maximum << '\n'
              << "original_solver_ns " << original_timing.minimum << ' '
              << original_timing.median << ' ' << original_timing.maximum << '\n'
              << "solver_speedup "
              << original_timing.median / new_timing.median << '\n'
              << "checksums " << approximate_timing.checksum << ' '
              << new_timing.checksum << ' ' << original_timing.checksum << '\n';
}
