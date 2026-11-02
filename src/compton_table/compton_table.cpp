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

#include <unistd.h>

#ifndef MACE_GIT_COMMIT
#define MACE_GIT_COMMIT "unknown"
#endif

namespace compton::table {

namespace {

constexpr int format_version = 1;
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

/// Smallest s over sampled points of the box E in g, E' in g', xi in bin a.
double box_exponent(
    TableSpec const& spec,
    int const g,
    int const gp,
    double const xi_lo,
    double const xi_hi,
    double const T)
{
    constexpr int energy_points = 5;
    constexpr int xi_points = 3;
    double const mc2 = units::me_c2;
    double const kT = units::k_boltz * T;
    double best = std::numeric_limits<double>::infinity();
    auto energy = [&](int const group, int const k) {
        double const lo = spec.boundaries[static_cast<std::size_t>(group)];
        double const hi = spec.boundaries[static_cast<std::size_t>(group) + 1];
        return lo * std::pow(hi / lo, static_cast<double>(k) / (energy_points - 1));
    };
    for (int i = 0; i < energy_points; ++i) {
        double const e = energy(g, i) / mc2;
        for (int j = 0; j < energy_points; ++j) {
            double const ep = energy(gp, j) / mc2;
            for (int k = 0; k < xi_points; ++k) {
                double const xi = xi_lo + (xi_hi - xi_lo) * k / (xi_points - 1);
                best = std::min(best, minimum_kinetic_energy(e, ep, xi) * mc2 / kT);
            }
        }
    }
    return best;
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

    bool const logging = compute_logging_enabled().exchange(false);
    auto const start = std::chrono::steady_clock::now();

    auto const weight =
        std::make_shared<PlanckWeightFunction>(spec.planck_cap_x, spec.boundaries);
    ComptonMultigroupKernel const cold_kernel(spec.boundaries, weight, quadrature(spec, true));
    ComptonMultigroupKernel const hot_kernel(spec.boundaries, weight, quadrature(spec, false));
    NumberWeightedEnergyTransferMultiplier const multiplier(spec.boundaries, spec.centers);
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
        std::vector<double> const sigma =
            kernel.compute_sigma_row(*solver, spec.num_angle_bins, T, g, multiplier);
        std::vector<double> const dsigma =
            kernel.compute_dsigma_dT_row(*solver, spec.num_angle_bins, T, g, multiplier, sigma);
        for (std::size_t gp = 0; gp < G; ++gp) {
            for (std::size_t a = 0; a < A; ++a) {
                std::size_t const local = gp * A + a;
                double dw_ds = 0.0;
                double const s = suppression_exponent(
                    spec, g, static_cast<int>(gp), static_cast<int>(a), T);
                double const w = taper_weight(spec, s, dw_ds);
                std::size_t const index = table.index(i, static_cast<std::size_t>(g), gp, a);
                table.sigma[index] = w * sigma[local];
                // ds/dT = -s/T at fixed energies.
                table.dsigma_dT[index] =
                    w * dsigma[local] + sigma[local] * dw_ds * (-s / T);
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
    compute_logging_enabled() = logging;
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
    diagnostics.tapered_entries = 0;
    diagnostics.zeroed_entries = 0;
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
                    double dw_ds = 0.0;
                    double const w = taper_weight(
                        spec,
                        suppression_exponent(
                            spec,
                            static_cast<int>(g),
                            static_cast<int>(gp),
                            static_cast<int>(a),
                            T),
                        dw_ds);
                    if (w > 0.0 && w < 1.0) {
                        ++diagnostics.tapered_entries;
                    }
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
    std::string const temporary = std::format("{}.tmp.{}", path, ::getpid());
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
             << std::format("build_seconds {:.3f}\n", d.build_seconds)
             << "byte_order little\n"
             << "arrays boundaries centers temperatures sigma dsigma_dT\n"
             << "END_HEADER\n";
        write_doubles(file, table.spec.boundaries);
        write_doubles(file, table.spec.centers);
        write_doubles(file, table.spec.temperatures);
        write_doubles(file, table.sigma);
        write_doubles(file, table.dsigma_dT);
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
    if (key_for(table.spec, table.mace_commit) != table.key) {
        throw std::runtime_error("Compton table key does not match its contents: " + path);
    }
    return table;
}

} // namespace compton::table
