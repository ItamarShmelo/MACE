// C++ library checks for mace_core:
//   * compute_sigma_row / compute_dsigma_dT_row reproduce the rows of
//     compute_sigma_matrix / compute_dsigma_dT_matrix bit for bit;
//   * group_index clamps energies on or past the outer grid edges;
//   * compute_logging_enabled(false) suppresses the per-call log file.
// Exit status 0 when every check passes.

#include "compton_differential_cross_section/compton_kernel_solver/compton_kernel_solver.hpp"
#include "compton_multigroup/compton_multigroup_deterministic/compton_multigroup_deterministic.hpp"
#include "compton_multigroup/weight_function.hpp"
#include "kernel_multipliers/kernel_multipliers.hpp"
#include "utilities/compute_logger.hpp"
#include "utilities/units.hpp"

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <format>
#include <memory>
#include <print>
#include <vector>

#include <unistd.h>

namespace {

int failures = 0;

void check(bool const ok, std::string const& what)
{
    std::println("{} {}", ok ? "PASS" : "FAIL", what);
    if (!ok) {
        ++failures;
    }
}

bool rows_match(
    std::vector<double> const& full,
    std::vector<double> const& row,
    int const g,
    std::size_t const row_size)
{
    if (row.size() != row_size) {
        return false;
    }
    for (std::size_t i = 0; i < row_size; ++i) {
        // Bit-for-bit: the row path runs the same arithmetic.
        if (full[static_cast<std::size_t>(g) * row_size + i] != row[i]) {
            return false;
        }
    }
    return true;
}

} // namespace

int main()
{
    using namespace compton;

    int const G = 6;
    int const num_angle_bins = 4;
    std::vector<double> boundaries(G + 1);
    std::vector<double> centers(G);
    for (int i = 0; i <= G; ++i) {
        boundaries[i] = units::kev * std::pow(10.0, -2.0 + 4.0 * i / G);
    }
    for (int i = 0; i < G; ++i) {
        centers[i] = 0.5 * (boundaries[i] + boundaries[i + 1]);
    }

    compute_logging_enabled() = false;
    auto const log_path = std::filesystem::path(
        std::format("compton_multigroup_{}.log", ::getpid()));
    std::filesystem::remove(log_path);

    auto const weight =
        std::make_shared<PlanckWeightFunction>(25.0, boundaries);
    ComptonMultigroupKernel const kernel_mg(
        boundaries,
        weight,
        MGIntegrationConfig());
    ComptonKernelSolver const solver;
    NumberWeightedEnergyTransferMultiplier const multiplier(
        boundaries,
        centers);
    double const T = 5.0 * units::kev_kelvin;

    auto const sigma =
        kernel_mg.compute_sigma_matrix(solver, num_angle_bins, T, multiplier);
    auto const dsigma = kernel_mg.compute_dsigma_dT_matrix(
        solver,
        num_angle_bins,
        T,
        multiplier);
    std::size_t const row_size = static_cast<std::size_t>(G) * num_angle_bins;
    bool sigma_ok = true;
    bool dsigma_ok = true;
    for (int g = 0; g < G; ++g) {
        sigma_ok = sigma_ok &&
                   rows_match(
                       sigma,
                       kernel_mg.compute_sigma_row(
                           solver,
                           num_angle_bins,
                           T,
                           g,
                           multiplier),
                       g,
                       row_size);
        dsigma_ok = dsigma_ok &&
                    rows_match(
                        dsigma,
                        kernel_mg.compute_dsigma_dT_row(
                            solver,
                            num_angle_bins,
                            T,
                            g,
                            multiplier),
                        g,
                        row_size);
    }
    check(sigma_ok, "compute_sigma_row equals the rows of compute_sigma_matrix");
    check(
        dsigma_ok,
        "compute_dsigma_dT_row equals the rows of compute_dsigma_dT_matrix");

    double row_max = 0.0;
    for (double const value : sigma) {
        row_max = std::max(row_max, std::abs(value));
    }
    check(row_max > 0.0 && std::isfinite(row_max), "sigma is finite and nonzero");

    check(group_index(boundaries, 0.5 * boundaries.front()) == 0,
          "group_index clamps below the grid");
    check(group_index(boundaries, boundaries.back()) == G - 1,
          "group_index clamps the top edge");
    check(group_index(boundaries, 2.0 * boundaries.back()) == G - 1,
          "group_index clamps above the grid");
    check(group_index(boundaries, centers[2]) == 2,
          "group_index inside the grid");

    check(!std::filesystem::exists(log_path),
          "no compton_multigroup log with logging off");

    std::println("{} failure(s)", failures);
    return failures == 0 ? 0 : 1;
}
