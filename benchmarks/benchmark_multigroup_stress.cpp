#include "compton_differential_cross_section/compton_kernel_approximate_solver/compton_kernel_approximate_solver.hpp"
#include "compton_differential_cross_section/compton_kernel_solver/compton_kernel_solver.hpp"
#include "compton_multigroup/compton_multigroup_deterministic/compton_multigroup_deterministic.hpp"
#include "compton_multigroup/weight_function.hpp"
#include "utilities/units.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

namespace {

class WienBenchmarkWeight final : public compton::WeightFunction {
  public:
    double weight(double const E, double const T) const override
    {
        double const x = E / (units::k_boltz * T);
        return x * x * x * std::exp(-x);
    }

    double compute_denominator(
        double const E_left,
        double const E_right,
        double const T) const override
    {
        double const kT = units::k_boltz * T;
        double const x_left = E_left / kT;
        double const x_right = E_right / kT;
        return kT * (tail_integral(x_left) - tail_integral(x_right));
    }

    double d_weight_dT(double const E, double const T) const override
    {
        double const x = E / (units::k_boltz * T);
        return weight(E, T) * (x - 3.0) / T;
    }

    double d_log_weight_dT(double const E, double const T) const override
    {
        double const x = E / (units::k_boltz * T);
        return (x - 3.0) / T;
    }

    double d_denominator_dT(
        double const E_left,
        double const E_right,
        double const T) const override
    {
        double const kT = units::k_boltz * T;
        double const x_left = E_left / kT;
        double const x_right = E_right / kT;
        double const integral =
            tail_integral(x_left) - tail_integral(x_right);
        return units::k_boltz *
               (integral + x_left * shape(x_left) -
                x_right * shape(x_right));
    }

    std::optional<double> peak_energy(double const T) const override
    {
        return 3.0 * units::k_boltz * T;
    }

  private:
    static double shape(double const x)
    {
        return x * x * x * std::exp(-x);
    }

    static double tail_integral(double const x)
    {
        return std::exp(-x) *
               (x * x * x + 3.0 * x * x + 6.0 * x + 6.0);
    }
};

struct Measurement {
    double milliseconds;
    std::vector<double> matrix;
};

struct Accuracy {
    double l1_relative;
    double max_significant_relative;
    double max_row_sum_relative;
};

template <typename Kernel>
Measurement measure(
    compton::ComptonMultigroupKernel const& multigroup,
    Kernel const& kernel,
    int const angle_bins,
    double const T)
{
    auto const start = std::chrono::steady_clock::now();
    auto matrix = multigroup.compute_sigma_matrix(
        kernel,
        angle_bins,
        T,
        compton::ConstantMultiplier{});
    auto const stop = std::chrono::steady_clock::now();
    return {
        std::chrono::duration<double, std::milli>(stop - start).count(),
        std::move(matrix)};
}

Accuracy accuracy(
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

    return {
        l1_difference / (l1_reference + 1e-300),
        max_significant_relative,
        max_row_sum_relative};
}

struct Scenario {
    std::string_view name;
    std::vector<double> boundaries_kev;
    int angle_bins;
    bool wien_weight;
};

} // namespace

int main()
{
    std::array const scenarios = {
        Scenario{"standard_uniform", {0.1, 1.0, 10.0, 100.0, 1000.0}, 4, false},
        Scenario{
            "fine_uniform",
            {0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0},
            8,
            false},
        Scenario{
            "broad_uniform",
            {0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0},
            8,
            false},
        Scenario{
            "fine_wien",
            {0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0},
            8,
            true},
    };

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
    compton::ComptonKernelSolver const reference_solver;
    compton::ComptonKernelApproximateSolver const candidate_solver;
    bool passed = true;

    std::cout << std::setprecision(12);
    for (Scenario const& scenario : scenarios) {
        std::vector<double> boundaries;
        boundaries.reserve(scenario.boundaries_kev.size());
        for (double const value : scenario.boundaries_kev) {
            boundaries.push_back(value * units::kev);
        }
        std::shared_ptr<compton::WeightFunction const> weight =
            scenario.wien_weight
                ? std::static_pointer_cast<compton::WeightFunction const>(
                      std::make_shared<WienBenchmarkWeight>())
                : std::static_pointer_cast<compton::WeightFunction const>(
                      std::make_shared<compton::UniformWeightFunction>());
        compton::ComptonMultigroupKernel const multigroup(
            boundaries,
            std::move(weight),
            config);
        int const groups = static_cast<int>(boundaries.size()) - 1;

        constexpr double TEMPERATURES_KEV[] = {
            1.0,
            10.0,
            50.0,
            100.0,
            150.0,
            200.0,
            220.0,
            224.0,
            225.0,
            228.0,
            229.0,
            229.9,
            230.0,
            250.0,
        };
        for (double const T_kev : TEMPERATURES_KEV) {
            double const T = T_kev * units::kev_kelvin;
            auto const reference = measure(
                multigroup,
                reference_solver,
                scenario.angle_bins,
                T);
            auto const candidate = measure(
                multigroup,
                candidate_solver,
                scenario.angle_bins,
                T);
            Accuracy const error = accuracy(
                candidate.matrix,
                reference.matrix,
                groups,
                scenario.angle_bins);
            passed = passed &&
                     std::isfinite(error.max_significant_relative) &&
                     error.max_significant_relative < 0.05;
            std::cout << "scenario " << scenario.name << " T_kev " << T_kev
                      << " original_ms " << reference.milliseconds
                      << " approximate_solver_ms " << candidate.milliseconds
                      << " speedup "
                      << reference.milliseconds / candidate.milliseconds
                      << " matrix_l1_relative " << error.l1_relative
                      << " matrix_max_significant_relative "
                      << error.max_significant_relative
                      << " matrix_max_row_sum_relative "
                      << error.max_row_sum_relative << '\n';
        }
    }

    return passed ? 0 : 1;
}
