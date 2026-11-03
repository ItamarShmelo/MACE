#include "compton_table/compton_table.hpp"

#include "compton_differential_cross_section/compton_kernel_approximate_solver/compton_kernel_approximate_solver.hpp"
#include "compton_differential_cross_section/compton_kernel_solver/compton_kernel_solver.hpp"
#include "compton_multigroup/compton_multigroup_deterministic/compton_multigroup_deterministic.hpp"
#include "compton_multigroup/weight_function.hpp"
#include "kernel_multipliers/kernel_multipliers.hpp"
#include "planck_integral.hpp"
#include "utilities/compute_logger.hpp"
#include "utilities/units.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <memory>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cerrno>
#include <cstring>
#include <random>
#include <thread>

#include <fcntl.h>
#include <unistd.h>

#include "mace_version.hpp" // generated: MACE_GIT_COMMIT

namespace compton::table {

namespace {

constexpr int format_version = 2;
constexpr char const* format_magic = "MACE_COMPTON_TABLE";

char const* solver_name(SolverKind const solver)
{
    return solver == SolverKind::exact ? "exact" : "approximate";
}

SolverKind solver_from_name(std::string const& name)
{
    if (name == "exact") {
        return SolverKind::exact;
    }
    if (name == "approximate") {
        return SolverKind::approximate;
    }
    throw std::runtime_error("unknown solver in table: " + name);
}

/// The specification as text, one "name value" per line, doubles with
/// `digits` significant digits (17: exact, for the file header; 12: for the
/// cache key, so grids computed with last-bit differences share a table).
std::string spec_text(TableSpec const& spec, int const digits = 17)
{
    std::string text;
    auto number = [digits](double const value) {
        return std::format("{:.{}g}", value, digits);
    };
    auto add_doubles = [&](char const* name, std::vector<double> const& values) {
        text += name;
        for (double const value : values) {
            text += ' ' + number(value);
        }
        text += '\n';
    };
    add_doubles("boundaries", spec.boundaries);
    add_doubles("centers", spec.centers);
    add_doubles("temperatures", spec.temperatures);
    text += std::format("angle_bins {}\n", spec.num_angle_bins);
    text += std::format("solver {}\n", solver_name(spec.solver));
    text += "weight planck cap_x " + number(spec.planck_cap_x) + '\n';
    text += "multiplier number_weighted_energy_transfer\n";
    text += "total energy_weighted\n";
    text += "cutoff_ratio " + number(spec.cutoff_ratio) + '\n';
    text += "quadrature cold_temperature " + number(spec.cold_temperature) +
            std::format(" e_panel_order_cold {} e_panel_order_hot {} cold_ep_k_cut 10 cold_ep_orders 48\n",
                        spec.e_panel_order_cold,
                        spec.e_panel_order_hot);
    text += "taper " + number(spec.taper_start) + ' ' + number(spec.taper_end) + '\n';
    return text;
}

double parse_double(std::string const& token)
{
    // The header writes doubles with 17 significant digits, which round-trip.
    char* end = nullptr;
    double const value = std::strtod(token.c_str(), &end);
    if (end == token.c_str()) {
        throw std::runtime_error("bad number in Compton table header: " + token);
    }
    return value;
}

std::uint64_t fnv1a(std::string const& text)
{
    std::uint64_t hash = 1469598103934665603ULL;
    for (unsigned char const c : text) {
        hash ^= c;
        hash *= 1099511628211ULL;
    }
    return hash;
}

MGIntegrationConfig quadrature(TableSpec const& spec, bool const cold)
{
    if (cold) {
        return MGIntegrationConfig(
            spec.cutoff_ratio,
            std::nullopt,
            5.0,
            std::nullopt,
            10.0,
            2.0,
            48,
            48,
            spec.e_panel_order_cold);
    }
    return MGIntegrationConfig(
        spec.cutoff_ratio,
        std::nullopt,
        5.0,
        std::nullopt,
        5.0,
        2.0,
        std::nullopt,
        std::nullopt,
        spec.e_panel_order_hot);
}

/// Lower estimate of s = K_min / kT over the box E in g, E' in g', xi in bin a.
/// K vanishes on the cold-recoil surface E' = E / (1 + e (1 - xi)), which
/// increases with E and xi: if the box's recoil range meets g', s = 0.
/// Otherwise K grows with the distance of E' from the surface, so the minimum
/// lies on the face of g' nearest to it; that face is searched on a grid and
/// refined, and the result is lowered by a 5% margin (the taper must not
/// remove allowed transitions).
double box_exponent(
    TableSpec const& spec,
    int const g,
    int const gp,
    double const xi_lo,
    double const xi_hi,
    double const T)
{
    double const mc2 = units::me_c2;
    double const kT = units::k_boltz * T;
    double const a = spec.boundaries[static_cast<std::size_t>(g)] / mc2;
    double const b = spec.boundaries[static_cast<std::size_t>(g) + 1] / mc2;
    double const c = spec.boundaries[static_cast<std::size_t>(gp)] / mc2;
    double const d = spec.boundaries[static_cast<std::size_t>(gp) + 1] / mc2;
    auto recoil = [](double const e, double const xi) { return e / (1.0 + e * (1.0 - xi)); };
    double const recoil_low = recoil(a, xi_lo);
    double const recoil_high = recoil(b, xi_hi);
    if (recoil_high >= c && recoil_low <= d) {
        return 0.0;
    }
    double const ep = recoil_high < c ? c : d;
    // Grid on the face (log e, xi), then coordinate golden-section refinement.
    constexpr int e_points = 33;
    constexpr int xi_points = 17;
    double const log_a = std::log(a);
    double const log_b = std::log(b);
    auto kinetic = [&](double const log_e, double const xi) {
        return minimum_kinetic_energy(std::exp(log_e), ep, xi);
    };
    double best = std::numeric_limits<double>::infinity();
    double best_log_e = log_a;
    double best_xi = xi_lo;
    for (int i = 0; i < e_points; ++i) {
        double const log_e = log_a + (log_b - log_a) * i / (e_points - 1);
        for (int k = 0; k < xi_points; ++k) {
            double const xi = xi_lo + (xi_hi - xi_lo) * k / (xi_points - 1);
            double const value = kinetic(log_e, xi);
            if (value < best) {
                best = value;
                best_log_e = log_e;
                best_xi = xi;
            }
        }
    }
    auto golden = [](auto const& f, double lo, double hi) {
        double const ratio = 0.5 * (std::sqrt(5.0) - 1.0);
        double x1 = hi - ratio * (hi - lo);
        double x2 = lo + ratio * (hi - lo);
        double f1 = f(x1);
        double f2 = f(x2);
        for (int iteration = 0; iteration < 40; ++iteration) {
            if (f1 < f2) {
                hi = x2;
                x2 = x1;
                f2 = f1;
                x1 = hi - ratio * (hi - lo);
                f1 = f(x1);
            } else {
                lo = x1;
                x1 = x2;
                f1 = f2;
                x2 = lo + ratio * (hi - lo);
                f2 = f(x2);
            }
        }
        return f1 < f2 ? x1 : x2;
    };
    double const e_step = (log_b - log_a) / (e_points - 1);
    double const xi_step = (xi_hi - xi_lo) / (xi_points - 1);
    for (int round = 0; round < 3; ++round) {
        best_log_e = golden([&](double const x) { return kinetic(x, best_xi); },
                            std::max(log_a, best_log_e - e_step), std::min(log_b, best_log_e + e_step));
        best_xi = golden([&](double const x) { return kinetic(best_log_e, x); },
                         std::max(xi_lo, best_xi - xi_step), std::min(xi_hi, best_xi + xi_step));
        best = std::min(best, kinetic(best_log_e, best_xi));
    }
    return 0.95 * best * mc2 / kT;
}

/// Centre-form equilibrium group energies B_g and occupations n_g at T, as
/// in the run-time detailed-balance enforcement.
void equilibrium(
    TableSpec const& spec,
    double const T,
    std::vector<double>& B,
    std::vector<double>& n)
{
    std::size_t const G = spec.centers.size();
    B.assign(G, 0.0);
    n.assign(G, 0.0);
    double const kT = units::k_boltz * T;
    double const factor = std::pow(units::clight, 3) /
                          (8.0 * std::numbers::pi * units::planck_constant);
    double const radiation = units::arad * std::pow(T, 4);
    for (std::size_t g = 0; g < G; ++g) {
        double const Bg =
            radiation * planck_integral::planck_integral(
                            spec.boundaries[g] / kT,
                            spec.boundaries[g + 1] / kT);
        double const nu = spec.centers[g] / units::planck_constant;
        double const dnu = (spec.boundaries[g + 1] - spec.boundaries[g]) /
                           units::planck_constant;
        B[g] = Bg;
        n[g] = factor * Bg / (nu * nu * nu * dnu);
    }
}

void write_doubles(std::ofstream& file, std::vector<double> const& values)
{
    static_assert(std::endian::native == std::endian::little,
                  "the table format is little-endian");
    file.write(
        reinterpret_cast<char const*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(double)));
}

void read_doubles(std::ifstream& file, std::vector<double>& values, std::size_t const count)
{
    values.resize(count);
    file.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(count * sizeof(double)));
    if (!file) {
        throw std::runtime_error("Compton table file is truncated");
    }
}

} // namespace

std::size_t Table::index(
    std::size_t const i,
    std::size_t const g,
    std::size_t const gp,
    std::size_t const a) const
{
    std::size_t const G = spec.centers.size();
    std::size_t const A = static_cast<std::size_t>(spec.num_angle_bins);
    return ((i * G + g) * G + gp) * A + a;
}

std::size_t Table::total_index(
    std::size_t const i,
    std::size_t const g,
    std::size_t const a) const
{
    std::size_t const G = spec.centers.size();
    std::size_t const A = static_cast<std::size_t>(spec.num_angle_bins);
    return (i * G + g) * A + a;
}

std::vector<double> default_temperatures()
{
    std::vector<double> temperatures(32);
    for (std::size_t i = 0; i < temperatures.size(); ++i) {
        temperatures[i] =
            1e3 * std::pow(1.2e7, static_cast<double>(i) / (temperatures.size() - 1));
    }
    return temperatures;
}

TableSpec log_grid_spec(double const E_lo, double const E_hi, int const num_groups)
{
    TableSpec spec;
    spec.boundaries.resize(static_cast<std::size_t>(num_groups) + 1);
    spec.centers.resize(static_cast<std::size_t>(num_groups));
    double const ratio = std::pow(E_hi / E_lo, 1.0 / num_groups);
    spec.boundaries[0] = E_lo;
    for (int g = 0; g < num_groups; ++g) {
        spec.boundaries[static_cast<std::size_t>(g) + 1] =
            spec.boundaries[static_cast<std::size_t>(g)] * ratio;
    }
    for (int g = 0; g < num_groups; ++g) {
        spec.centers[static_cast<std::size_t>(g)] =
            0.5 * (spec.boundaries[static_cast<std::size_t>(g)] +
                   spec.boundaries[static_cast<std::size_t>(g) + 1]);
    }
    spec.temperatures = default_temperatures();
    return spec;
}

void validate(TableSpec const& spec)
{
    if (spec.centers.empty() || spec.boundaries.size() != spec.centers.size() + 1) {
        throw std::invalid_argument("a table needs G >= 1 centres and G + 1 boundaries");
    }
    for (std::size_t g = 0; g < spec.centers.size(); ++g) {
        if (!(spec.boundaries[g] > 0.0) || !(spec.boundaries[g] < spec.boundaries[g + 1]) ||
            !(spec.centers[g] > spec.boundaries[g]) || !(spec.centers[g] < spec.boundaries[g + 1])) {
            throw std::invalid_argument("boundaries must increase and contain their centres");
        }
    }
    if (spec.temperatures.empty() || !(spec.temperatures.front() > 0.0) ||
        !std::is_sorted(spec.temperatures.begin(), spec.temperatures.end()) ||
        std::adjacent_find(spec.temperatures.begin(), spec.temperatures.end()) !=
            spec.temperatures.end()) {
        throw std::invalid_argument("temperatures must be positive and strictly increasing");
    }
    if (spec.num_angle_bins < 1) {
        throw std::invalid_argument("num_angle_bins must be >= 1");
    }
    if (!(spec.taper_start > 0.0) || !(spec.taper_end > spec.taper_start)) {
        throw std::invalid_argument("the taper needs 0 < taper_start < taper_end");
    }
}

std::string mace_commit()
{
    return MACE_GIT_COMMIT;
}

namespace {

std::string key_for(TableSpec const& spec, std::string const& commit)
{
    std::string const text = std::format(
        "{} {}\n{}mace_commit {}\n",
        format_magic,
        format_version,
        spec_text(spec, 12),
        commit);
    return std::format("{:016x}", fnv1a(text));
}

} // namespace

std::string table_key(TableSpec const& spec)
{
    return key_for(spec, mace_commit());
}

std::string cache_file_name(TableSpec const& spec)
{
    return std::format("mace_compton_{}.tbl", table_key(spec));
}

double minimum_kinetic_energy(double const e, double const ep, double const xi)
{
    // Energy-momentum conservation for the most favourable electron direction:
    // gamma A - sqrt(gamma^2 - 1) q = C with A = e - e', C = e e' (1 - xi) and
    // q^2 = A^2 + 2C, solvable for 2C gamma^2 + 2AC gamma - (C^2 + q^2) >= 0.
    // gamma - 1 = (-A + Y / (sqrt(4 + Y) + 2)) / 2, Y = 2C + A^2 + 2A^2/C,
    // which avoids the cancellation in q^2 near forward scattering.
    double const A = e - ep;
    double const C = e * ep * (1.0 - xi);
    if (!(C > 0.0)) {
        return A == 0.0 ? 0.0 : std::numeric_limits<double>::infinity();
    }
    double const Y = 2.0 * C + A * A + 2.0 * A * A / C;
    return std::max(0.0, 0.5 * (-A + Y / (std::sqrt(4.0 + Y) + 2.0)));
}

double minimum_lorentz_factor(double const e, double const ep, double const xi)
{
    return 1.0 + minimum_kinetic_energy(e, ep, xi);
}

double suppression_exponent(
    TableSpec const& spec,
    int const g,
    int const gp,
    int const a,
    double const T)
{
    double const dxi = 2.0 / spec.num_angle_bins;
    double const xi_lo = -1.0 + a * dxi;
    double const xi_hi = std::min(-1.0 + (a + 1) * dxi, 1.0 - 1e-12);
    return std::min(
        box_exponent(spec, g, gp, xi_lo, xi_hi, T),
        box_exponent(spec, gp, g, xi_lo, xi_hi, T));
}

double taper_weight(TableSpec const& spec, double const s, double& dw_ds)
{
    dw_ds = 0.0;
    if (s <= spec.taper_start) {
        return 1.0;
    }
    if (s >= spec.taper_end) {
        return 0.0;
    }
    double const width = spec.taper_end - spec.taper_start;
    double const x = (s - spec.taper_start) / width;
    dw_ds = -6.0 * x * (1.0 - x) / width;
    return 1.0 - x * x * (3.0 - 2.0 * x);
}

Table compute_partial(
    TableSpec const& spec,
    int const rank,
    int const size,
    ProgressCallback const& progress)
{
    validate(spec);
    if (size < 1 || rank < 0 || rank >= size) {
        throw std::invalid_argument("invalid rank or size");
    }
    std::size_t const G = spec.centers.size();
    std::size_t const A = static_cast<std::size_t>(spec.num_angle_bins);
    std::size_t const NT = spec.temperatures.size();

    Table table;
    table.spec = spec;
    table.key = table_key(spec);
    table.mace_commit = mace_commit();
    table.sigma.assign(NT * G * G * A, 0.0);
    table.dsigma_dT.assign(NT * G * G * A, 0.0);
    table.total.assign(NT * G * A, 0.0);
    table.dtotal_dT.assign(NT * G * A, 0.0);

    // Per-call log files off while building; restored on every exit.
    struct LoggingGuard {
        bool previous = compute_logging_enabled().exchange(false);
        ~LoggingGuard() { compute_logging_enabled() = previous; }
    } const logging_guard;
    auto const start = std::chrono::steady_clock::now();

    auto const weight =
        std::make_shared<PlanckWeightFunction>(spec.planck_cap_x, spec.boundaries);
    ComptonMultigroupKernel const cold_kernel(spec.boundaries, weight, quadrature(spec, true));
    ComptonMultigroupKernel const hot_kernel(spec.boundaries, weight, quadrature(spec, false));
    NumberWeightedEnergyTransferMultiplier const multiplier(spec.boundaries, spec.centers);
    ConstantMultiplier const unit;
    std::unique_ptr<ComptonKernelSolver> const solver =
        spec.solver == SolverKind::exact
            ? std::make_unique<ComptonKernelSolver>()
            : std::make_unique<ComptonKernelApproximateSolver>();

    std::size_t const items = NT * G;
    std::size_t const mine = items / static_cast<std::size_t>(size) +
                             (static_cast<std::size_t>(rank) < items % static_cast<std::size_t>(size) ? 1 : 0);
    std::size_t done = 0;
    for (std::size_t item = static_cast<std::size_t>(rank); item < items;
         item += static_cast<std::size_t>(size)) {
        std::size_t const i = item / G;
        int const g = static_cast<int>(item % G);
        double const T = spec.temperatures[i];
        ComptonMultigroupKernel const& kernel =
            T < spec.cold_temperature ? cold_kernel : hot_kernel;
        // The effective rates and the energy-weighted rates into each target (multiplier 1, summed into the
        // total below), with their derivatives, from one sweep of each kernel (as compute_sigma_row and
        // compute_dsigma_dT_row, bit for bit).
        std::vector<SigmaRowWithDerivative> const rows =
            kernel.compute_rows_with_derivatives(*solver, spec.num_angle_bins, T, g, {&multiplier, &unit});
        std::vector<double> const& sigma = rows[0].sigma;
        std::vector<double> const& dsigma = rows[0].dsigma_dT;
        std::vector<double> const& plain = rows[1].sigma;
        std::vector<double> const& dplain = rows[1].dsigma_dT;
        for (std::size_t gp = 0; gp < G; ++gp) {
            for (std::size_t a = 0; a < A; ++a) {
                std::size_t const local = gp * A + a;
                double dw_ds = 0.0;
                double const s = suppression_exponent(
                    spec, g, static_cast<int>(gp), static_cast<int>(a), T);
                double const w = taper_weight(spec, s, dw_ds);
                if (w > 0.0 && w < 1.0) {
                    ++table.diagnostics.tapered_entries;
                }
                std::size_t const index = table.index(i, static_cast<std::size_t>(g), gp, a);
                table.sigma[index] = w * sigma[local];
                // ds/dT = -s/T at fixed energies.
                table.dsigma_dT[index] =
                    w * dsigma[local] + sigma[local] * dw_ds * (-s / T);
                std::size_t const total = table.total_index(i, static_cast<std::size_t>(g), a);
                table.total[total] += w * plain[local];
                table.dtotal_dT[total] += w * dplain[local] + plain[local] * dw_ds * (-s / T);
            }
        }
        ++done;
        if (progress) {
            double const seconds = std::chrono::duration<double>(
                                       std::chrono::steady_clock::now() - start)
                                       .count();
            progress(done, mine, seconds);
        }
    }
    table.diagnostics.build_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return table;
}

void finalize(Table& table)
{
    TableSpec const& spec = table.spec;
    std::size_t const G = spec.centers.size();
    std::size_t const A = static_cast<std::size_t>(spec.num_angle_bins);
    TableDiagnostics& diagnostics = table.diagnostics;
    diagnostics.max_detailed_balance_mismatch = 0.0;
    diagnostics.min_remainder_fraction = 1.0;
    diagnostics.negative_remainder_rows = 0;
    diagnostics.max_negative_angular_fraction = 0.0;
    std::vector<double> B;
    std::vector<double> n;
    for (std::size_t i = 0; i < spec.temperatures.size(); ++i) {
        double const T = spec.temperatures[i];
        equilibrium(spec, T, B, n);
        std::vector<double> summed(G * G, 0.0);
        std::vector<double> row_max(G, 0.0);
        for (std::size_t g = 0; g < G; ++g) {
            for (std::size_t gp = 0; gp < G; ++gp) {
                for (std::size_t a = 0; a < A; ++a) {
                    summed[g * G + gp] += table.sigma[table.index(i, g, gp, a)];
                }
                row_max[g] = std::max(row_max[g], std::abs(summed[g * G + gp]));
            }
        }
        for (std::size_t g = 0; g < G; ++g) {
            for (std::size_t gp = g + 1; gp < G; ++gp) {
                double const forward = summed[g * G + gp];
                double const backward = summed[gp * G + g];
                double const floor = 1e-14 * std::max(row_max[g], row_max[gp]);
                if (!(forward > floor) || !(backward > floor) ||
                    B[gp] * spec.centers[g] < std::numeric_limits<double>::min() * 1e40) {
                    continue;
                }
                double const factor = (1.0 + n[gp]) * B[g] * spec.centers[gp] /
                                      ((1.0 + n[g]) * B[gp] * spec.centers[g]);
                if (!std::isfinite(factor) || !(factor > 0.0)) {
                    continue;
                }
                double const mismatch = std::abs(backward / (forward * factor) - 1.0);
                if (mismatch > diagnostics.max_detailed_balance_mismatch) {
                    diagnostics.max_detailed_balance_mismatch = mismatch;
                    diagnostics.mismatch_temperature = T;
                    diagnostics.mismatch_g = static_cast<int>(g);
                    diagnostics.mismatch_gp = static_cast<int>(gp);
                }
            }
        }
        // Direction-only remainder total - sum_{g' != g} sigma, per angle bin and summed.
        for (std::size_t g = 0; g < G; ++g) {
            double total = 0.0;
            double remainder = 0.0;
            double negative = 0.0;
            for (std::size_t a = 0; a < A; ++a) {
                double bin = table.total[table.total_index(i, g, a)];
                total += bin;
                for (std::size_t gp = 0; gp < G; ++gp) {
                    if (gp != g) {
                        bin -= table.sigma[table.index(i, g, gp, a)];
                    }
                }
                remainder += bin;
                negative += std::max(0.0, -bin);
            }
            if (!(total > 0.0)) {
                continue;
            }
            if (remainder < 0.0) {
                ++diagnostics.negative_remainder_rows;
            }
            if (remainder / total < diagnostics.min_remainder_fraction) {
                diagnostics.min_remainder_fraction = remainder / total;
                diagnostics.remainder_temperature = T;
                diagnostics.remainder_g = static_cast<int>(g);
            }
            if (negative / total > diagnostics.max_negative_angular_fraction) {
                diagnostics.max_negative_angular_fraction = negative / total;
                diagnostics.negative_angular_temperature = T;
                diagnostics.negative_angular_g = static_cast<int>(g);
            }
        }
    }
}

Table build_serial(TableSpec const& spec, ProgressCallback const& progress)
{
    Table table = compute_partial(spec, 0, 1, progress);
    finalize(table);
    return table;
}

void write_table(std::string const& path, Table const& table)
{
    std::string const temporary = std::format("{}.tmp.{}", path, unique_token());
    {
        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        if (!file) {
            throw std::runtime_error("cannot write Compton table " + temporary);
        }
        TableDiagnostics const& d = table.diagnostics;
        file << format_magic << ' ' << format_version << '\n'
             << "key " << table.key << '\n'
             << "mace_commit " << table.mace_commit << '\n'
             << "groups " << table.spec.centers.size() << '\n'
             << "angle_bins " << table.spec.num_angle_bins << '\n'
             << "temperature_count " << table.spec.temperatures.size() << '\n'
             << spec_text(table.spec)
             << std::format("max_detailed_balance_mismatch {:.6e} at_T {:.6e} g {} gp {}\n",
                            d.max_detailed_balance_mismatch, d.mismatch_temperature,
                            d.mismatch_g, d.mismatch_gp)
             << "tapered_entries " << d.tapered_entries << '\n'
             << std::format("min_remainder_fraction {:.6e} at_T {:.6e} g {} negative_rows {}\n",
                            d.min_remainder_fraction, d.remainder_temperature, d.remainder_g,
                            d.negative_remainder_rows)
             << std::format("max_negative_angular_fraction {:.6e} at_T {:.6e} g {}\n",
                            d.max_negative_angular_fraction, d.negative_angular_temperature,
                            d.negative_angular_g)
             << std::format("build_seconds {:.3f}\n", d.build_seconds)
             << "byte_order little\n"
             << "arrays boundaries centers temperatures sigma dsigma_dT total dtotal_dT\n"
             << "END_HEADER\n";
        write_doubles(file, table.spec.boundaries);
        write_doubles(file, table.spec.centers);
        write_doubles(file, table.spec.temperatures);
        write_doubles(file, table.sigma);
        write_doubles(file, table.dsigma_dT);
        write_doubles(file, table.total);
        write_doubles(file, table.dtotal_dT);
        if (!file) {
            throw std::runtime_error("failed writing Compton table " + temporary);
        }
    }
    std::filesystem::rename(temporary, path);
}

Table read_table(std::string const& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("cannot open Compton table " + path);
    }
    Table table;
    std::size_t G = 0;
    std::size_t NT = 0;
    std::string line;
    std::getline(file, line);
    if (line != std::format("{} {}", format_magic, format_version)) {
        throw std::runtime_error("not a version-" + std::to_string(format_version) +
                                 " MACE Compton table: " + path);
    }
    while (std::getline(file, line) && line != "END_HEADER") {
        std::istringstream stream(line);
        std::vector<std::string> fields;
        for (std::string token; stream >> token;) {
            fields.push_back(token);
        }
        if (fields.empty()) {
            continue;
        }
        auto field = [&](std::size_t const k) -> std::string const& {
            if (k >= fields.size()) {
                throw std::runtime_error("malformed Compton table header line: " + line);
            }
            return fields[k];
        };
        std::string const& name = fields[0];
        if (name == "key") {
            table.key = field(1);
        } else if (name == "mace_commit") {
            table.mace_commit = field(1);
        } else if (name == "groups") {
            G = std::stoul(field(1));
        } else if (name == "angle_bins") {
            table.spec.num_angle_bins = std::stoi(field(1));
        } else if (name == "temperature_count") {
            NT = std::stoul(field(1));
        } else if (name == "solver") {
            table.spec.solver = solver_from_name(field(1));
        } else if (name == "weight") {
            table.spec.planck_cap_x = parse_double(field(3));
        } else if (name == "cutoff_ratio") {
            table.spec.cutoff_ratio = parse_double(field(1));
        } else if (name == "quadrature") {
            table.spec.cold_temperature = parse_double(field(2));
            table.spec.e_panel_order_cold = std::stoi(field(4));
            table.spec.e_panel_order_hot = std::stoi(field(6));
        } else if (name == "taper") {
            table.spec.taper_start = parse_double(field(1));
            table.spec.taper_end = parse_double(field(2));
        } else if (name == "max_detailed_balance_mismatch") {
            table.diagnostics.max_detailed_balance_mismatch = parse_double(field(1));
            table.diagnostics.mismatch_temperature = parse_double(field(3));
            table.diagnostics.mismatch_g = std::stoi(field(5));
            table.diagnostics.mismatch_gp = std::stoi(field(7));
        } else if (name == "tapered_entries") {
            table.diagnostics.tapered_entries = std::stoul(field(1));
        } else if (name == "min_remainder_fraction") {
            table.diagnostics.min_remainder_fraction = parse_double(field(1));
            table.diagnostics.remainder_temperature = parse_double(field(3));
            table.diagnostics.remainder_g = std::stoi(field(5));
            table.diagnostics.negative_remainder_rows = std::stoul(field(7));
        } else if (name == "max_negative_angular_fraction") {
            table.diagnostics.max_negative_angular_fraction = parse_double(field(1));
            table.diagnostics.negative_angular_temperature = parse_double(field(3));
            table.diagnostics.negative_angular_g = std::stoi(field(5));
        } else if (name == "build_seconds") {
            table.diagnostics.build_seconds = parse_double(field(1));
        }
    }
    if (line != "END_HEADER" || G == 0 || NT == 0 || table.spec.num_angle_bins < 1) {
        throw std::runtime_error("malformed Compton table header: " + path);
    }
    std::size_t const A = static_cast<std::size_t>(table.spec.num_angle_bins);
    read_doubles(file, table.spec.boundaries, G + 1);
    read_doubles(file, table.spec.centers, G);
    read_doubles(file, table.spec.temperatures, NT);
    read_doubles(file, table.sigma, NT * G * G * A);
    read_doubles(file, table.dsigma_dT, NT * G * G * A);
    read_doubles(file, table.total, NT * G * A);
    read_doubles(file, table.dtotal_dT, NT * G * A);
    if (key_for(table.spec, table.mace_commit) != table.key) {
        throw std::runtime_error("Compton table key does not match its contents: " + path);
    }
    return table;
}

void check_compatible(Table const& table, TableSpec const& spec)
{
    auto fail = [](std::string const& what) {
        throw std::runtime_error("Compton table does not match the requested specification (" + what + ")");
    };
    auto close = [](double const a, double const b, double const tolerance) {
        return std::abs(a - b) <= tolerance * std::max(std::abs(a), std::abs(b));
    };
    auto same_values = [&](std::vector<double> const& a, std::vector<double> const& b, char const* name) {
        if (a.size() != b.size()) {
            fail(std::string(name) + " count");
        }
        for (std::size_t k = 0; k < a.size(); ++k) {
            if (!close(a[k], b[k], 1e-10)) {
                fail(name);
            }
        }
    };
    TableSpec const& stored = table.spec;
    same_values(stored.boundaries, spec.boundaries, "boundaries");
    same_values(stored.centers, spec.centers, "centers");
    same_values(stored.temperatures, spec.temperatures, "temperatures");
    // Widths and centre separations enter the effective-rate multiplier directly.
    for (std::size_t g = 0; g + 1 < spec.boundaries.size(); ++g) {
        if (!close(stored.boundaries[g + 1] - stored.boundaries[g], spec.boundaries[g + 1] - spec.boundaries[g], 1e-8)) {
            fail("group widths");
        }
    }
    for (std::size_t g = 0; g + 1 < spec.centers.size(); ++g) {
        if (!close(stored.centers[g + 1] - stored.centers[g], spec.centers[g + 1] - spec.centers[g], 1e-8)) {
            fail("centre separations");
        }
    }
    if (stored.num_angle_bins != spec.num_angle_bins || stored.solver != spec.solver ||
        stored.e_panel_order_cold != spec.e_panel_order_cold || stored.e_panel_order_hot != spec.e_panel_order_hot ||
        !close(stored.planck_cap_x, spec.planck_cap_x, 1e-10) || !close(stored.cutoff_ratio, spec.cutoff_ratio, 1e-10) ||
        !close(stored.cold_temperature, spec.cold_temperature, 1e-10) ||
        !close(stored.taper_start, spec.taper_start, 1e-10) || !close(stored.taper_end, spec.taper_end, 1e-10)) {
        fail("settings");
    }
    std::size_t const G = spec.centers.size();
    std::size_t const size = spec.temperatures.size() * G * G * static_cast<std::size_t>(spec.num_angle_bins);
    if (table.sigma.size() != size || table.dsigma_dT.size() != size ||
        table.total.size() * G != size || table.dtotal_dT.size() * G != size) {
        fail("array sizes");
    }
}

std::string unique_token()
{
    char host[256] = {};
    ::gethostname(host, sizeof(host) - 1);
    std::random_device device;
    std::uint64_t const random = (static_cast<std::uint64_t>(device()) << 32) ^ device();
    return std::format("{}.{}.{:016x}", host, ::getpid(), random);
}

CacheLock::CacheLock(std::filesystem::path path, std::string token)
    : path_(std::move(path)), token_(std::move(token))
{
    heartbeat_ = std::thread([this] {
        auto last = std::chrono::steady_clock::now();
        while (!stop_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            if (std::chrono::steady_clock::now() - last >= std::chrono::seconds(60)) {
                if (owned()) {
                    std::error_code error;
                    std::filesystem::last_write_time(
                        path_, std::filesystem::file_time_type::clock::now(), error);
                }
                last = std::chrono::steady_clock::now();
            }
        }
    });
}

CacheLock::~CacheLock()
{
    stop_ = true;
    heartbeat_.join();
    if (owned()) {
        std::error_code error;
        std::filesystem::remove(path_, error);
    }
}

bool CacheLock::owned() const
{
    std::ifstream file(path_);
    std::string content;
    return static_cast<bool>(std::getline(file, content)) && content == token_;
}

CacheDecision acquire_table(
    std::string const& path,
    double const stale_seconds,
    std::ostream* const log,
    std::unique_ptr<CacheLock>& lock)
{
    std::filesystem::path const table_path(path);
    std::filesystem::path const lock_path(path + ".lock");
    if (table_path.has_parent_path()) {
        std::filesystem::create_directories(table_path.parent_path());
    }
    bool announced = false;
    while (true) {
        if (std::filesystem::exists(table_path)) {
            return CacheDecision::read;
        }
        std::string const token = unique_token();
        int const fd = ::open(lock_path.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0644);
        if (fd >= 0) {
            std::string const line = token + '\n';
            bool const written = ::write(fd, line.data(), line.size()) == static_cast<ssize_t>(line.size());
            ::close(fd);
            if (!written) {
                std::error_code error;
                std::filesystem::remove(lock_path, error);
                throw std::runtime_error("cannot write the Compton table lock " + lock_path.string());
            }
            lock = std::make_unique<CacheLock>(lock_path, token);
            // Another job may have published the table between the check and the lock.
            if (std::filesystem::exists(table_path)) {
                lock.reset();
                return CacheDecision::read;
            }
            return CacheDecision::build;
        }
        if (errno != EEXIST) {
            throw std::runtime_error("cannot create the Compton table lock " + lock_path.string() + ": " +
                                     std::strerror(errno));
        }
        std::error_code error;
        auto const modified = std::filesystem::last_write_time(lock_path, error);
        if (error) {
            continue; // the lock vanished: try again
        }
        double const age = std::chrono::duration<double>(
                               std::filesystem::file_time_type::clock::now() - modified)
                               .count();
        if (age > stale_seconds) {
            // Atomic takeover: only one process can rename the stale lock away.
            std::filesystem::path const tombstone(lock_path.string() + ".stale." + token);
            std::filesystem::rename(lock_path, tombstone, error);
            if (!error) {
                std::filesystem::remove(tombstone, error);
                if (log != nullptr) {
                    *log << "[mace] took over the stale Compton table lock " << lock_path << '\n';
                }
            }
            continue;
        }
        if (!announced && log != nullptr) {
            *log << "[mace] waiting for another job to build " << table_path << '\n' << std::flush;
            announced = true;
        }
        std::this_thread::sleep_for(std::chrono::seconds(10));
    }
}

Table load_or_build(
    TableSpec const& spec,
    std::string const& cache_dir,
    std::ostream* const log,
    double const stale_seconds)
{
    validate(spec);
    std::filesystem::path const directory =
        cache_dir.empty() ? std::filesystem::path(".") : std::filesystem::path(cache_dir);
    std::string const path = (directory / cache_file_name(spec)).string();
    std::unique_ptr<CacheLock> lock;
    if (acquire_table(path, stale_seconds, log, lock) == CacheDecision::read) {
        Table table = read_table(path);
        check_compatible(table, spec);
        if (log != nullptr) {
            *log << "[mace] read Compton table " << path << '\n';
        }
        return table;
    }
    if (log != nullptr) {
        *log << "[mace] building Compton table " << path << " (one process; build_compton_table with MPI is faster)\n"
             << std::flush;
    }
    Table table = build_serial(spec);
    write_table(path, table);
    return table;
}

} // namespace compton::table
