#include "compton_differential_cross_section/compton_kernel_approximate_solver/compton_kernel_approximate_solver.hpp"
#include "compton_differential_cross_section/compton_kernel_solver/compton_kernel_solver.hpp"
#include "compton_multigroup/compton_multigroup_deterministic/compton_multigroup_deterministic.hpp"
#include "compton_multigroup/weight_function.hpp"
#include "utilities/units.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <vector>

namespace {

struct Measurement {
    double milliseconds;
    std::vector<double> matrix;
};

template <typename Kernel>
Measurement measure(
    compton::ComptonMultigroupKernel const& multigroup,
    Kernel const& kernel,
    int const angle_bins,
    double const T)
{
    double best = 1e300;
    std::vector<double> best_matrix;
    for (int trial = 0; trial < 3; ++trial) {
        auto const start = std::chrono::steady_clock::now();
        auto matrix = multigroup.compute_sigma_matrix(
            kernel,
            angle_bins,
            T,
            compton::ConstantMultiplier{});
        auto const stop = std::chrono::steady_clock::now();
        double const milliseconds =
            std::chrono::duration<double, std::milli>(stop - start).count();
        if (milliseconds < best) {
            best = milliseconds;
            best_matrix = std::move(matrix);
        }
    }
    return {best, std::move(best_matrix)};
}

void report_accuracy(
    std::vector<double> const& candidate,
    std::vector<double> const& reference,
    int const groups,
    int const angle_bins)
{
    double const reference_max =
        *std::ranges::max_element(reference, {}, [](double const value) {
            return std::abs(value);
        });
    double l1_difference = 0.0;
    double l1_reference = 0.0;
    double max_significant_relative = 0.0;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        double const difference = std::abs(candidate[i] - reference[i]);
        l1_difference += difference;
        l1_reference += std::abs(reference[i]);
        if (std::abs(reference[i]) > 1e-8 * reference_max) {
            max_significant_relative = std::max(
                max_significant_relative,
                difference / std::abs(reference[i]));
        }
    }

    double max_row_sum_relative = 0.0;
    for (int g = 0; g < groups; ++g) {
        double candidate_sum = 0.0;
        double reference_sum = 0.0;
        for (int gp = 0; gp < groups; ++gp) {
            for (int angle = 0; angle < angle_bins; ++angle) {
                auto const index = static_cast<std::size_t>(
                    g * groups * angle_bins + gp * angle_bins + angle);
                candidate_sum += candidate[index];
                reference_sum += reference[index];
            }
        }
        max_row_sum_relative = std::max(
            max_row_sum_relative,
            std::abs(candidate_sum - reference_sum) /
                (std::abs(reference_sum) + 1e-300));
    }

    std::cout << " matrix_l1_relative "
              << l1_difference / (l1_reference + 1e-300)
              << " matrix_max_significant_relative "
              << max_significant_relative
              << " matrix_max_row_sum_relative " << max_row_sum_relative;
}

} // namespace

int main()
{
    std::vector<double> boundaries_kev = {0.1, 1.0, 10.0, 100.0, 1000.0};
    std::vector<double> boundaries;
    boundaries.reserve(boundaries_kev.size());
    for (double const value : boundaries_kev) {
        boundaries.push_back(value * units::kev);
    }

    auto weight = std::make_shared<compton::UniformWeightFunction>();
    compton::MGIntegrationConfig const config(
        std::nullopt,
        24,
        5.0,
        16,
        5.0,
        2.0,
        24,
        24,
        8,
        2.0,
        5.0);
    compton::ComptonMultigroupKernel const multigroup(
        boundaries,
        weight,
        config);
    compton::ComptonKernelSolver const reference_solver;
    compton::ComptonKernelApproximateSolver const approximate_solver;
    constexpr int ANGLE_BINS = 4;
    int const groups = static_cast<int>(boundaries.size()) - 1;

    std::cout << std::setprecision(12);
    for (double const T_kev : {1.0, 10.0, 100.0}) {
        double const T = T_kev * units::kev_kelvin;
        auto const reference =
            measure(multigroup, reference_solver, ANGLE_BINS, T);
        auto const candidate =
            measure(multigroup, approximate_solver, ANGLE_BINS, T);
        std::cout << "T_kev " << T_kev << " original_ms "
                  << reference.milliseconds << " approximate_solver_ms "
                  << candidate.milliseconds << " speedup "
                  << reference.milliseconds / candidate.milliseconds;
        report_accuracy(
            candidate.matrix,
            reference.matrix,
            groups,
            ANGLE_BINS);
        std::cout << '\n';
    }
}
