// Compton table checks (run under MPI with any number of ranks):
//   * the MPI table equals the serial table bit for bit;
//   * load_or_build_mpi builds the cache file once, then reads it back
//     unchanged, and leaves no lock file;
//   * write/read round trip keeps the key and every array;
//   * minimum_lorentz_factor and the taper satisfy closed-form checks.
// Exit status 0 when every check passes.

#include "compton_table/compton_table.hpp"
#include "compton_table/compton_table_mpi.hpp"
#include "utilities/units.hpp"

#include <mpi.h>

#include <chrono>
#include <fstream>
#include <cmath>
#include <filesystem>
#include <format>
#include <print>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

int failures = 0;
int rank = 0;

void check(bool const ok, std::string const& what)
{
    if (rank == 0) {
        std::println("{} {}", ok ? "PASS" : "FAIL", what);
    }
    if (!ok) {
        ++failures;
    }
}

bool same(compton::table::Table const& a, compton::table::Table const& b)
{
    return a.key == b.key && a.sigma == b.sigma && a.dsigma_dT == b.dsigma_dT;
}

} // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    using namespace compton::table;

    TableSpec spec = log_grid_spec(1e-3 * units::kev, 1e2 * units::kev, 6);
    spec.temperatures = {1e4, 3e6, 1e8};
    spec.num_angle_bins = 4;

    auto const start = std::chrono::steady_clock::now();
    Table const mpi = build_mpi(spec, MPI_COMM_WORLD, nullptr);
    double const mpi_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (rank == 0) {
        auto const serial_start = std::chrono::steady_clock::now();
        Table const serial = build_serial(spec);
        double const serial_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - serial_start).count();
        std::println("ranks {}: MPI build {:.2f} s, serial build {:.2f} s", size, mpi_seconds, serial_seconds);
        check(same(mpi, serial), std::format("MPI table ({} ranks) equals the serial table", size));
        double largest = 0.0;
        for (double const value : serial.sigma) {
            largest = std::max(largest, value);
        }
        check(largest > 0.0 && std::isfinite(largest), "table is finite and nonzero");
        std::println("max detailed-balance mismatch {:.3e} at T {:.3g} K (groups {}, {}); tapered entries {}",
                     serial.diagnostics.max_detailed_balance_mismatch,
                     serial.diagnostics.mismatch_temperature,
                     serial.diagnostics.mismatch_g,
                     serial.diagnostics.mismatch_gp,
                     serial.diagnostics.tapered_entries);
    }

    std::string directory;
    if (rank == 0) {
        directory = (std::filesystem::temp_directory_path() / std::format("mace_table_test_{}", ::getpid())).string();
        std::filesystem::remove_all(directory);
    }
    unsigned long long length = directory.size();
    MPI_Bcast(&length, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    directory.resize(length);
    MPI_Bcast(directory.data(), static_cast<int>(length), MPI_CHAR, 0, MPI_COMM_WORLD);

    Table const built = load_or_build_mpi(spec, directory, MPI_COMM_WORLD, nullptr);
    Table const read = load_or_build_mpi(spec, directory, MPI_COMM_WORLD, nullptr);
    check(same(built, mpi), "load_or_build_mpi builds the same table");
    check(same(read, built), "a second load_or_build_mpi reads it back unchanged");
    std::filesystem::path const path = std::filesystem::path(directory) / cache_file_name(spec);
    if (rank == 0) {
        check(std::filesystem::exists(path), "cache file written");
        check(!std::filesystem::exists(path.string() + ".lock"), "no lock left behind");
        Table const file = read_table(path.string());
        check(file.key == table_key(spec) && file.spec.boundaries == spec.boundaries &&
                  file.spec.temperatures == spec.temperatures && file.sigma == built.sigma &&
                  file.dsigma_dT == built.dsigma_dT,
              "read_table round trip");
        std::filesystem::remove_all(directory);
    }

    TableSpec nudged = spec;
    for (double& T : nudged.temperatures) {
        T = std::nextafter(T, 2.0 * T);
    }
    check(table_key(nudged) == table_key(spec),
          "last-bit changes of the temperature grid keep the cache key");
    nudged.temperatures[1] *= 1.0 + 1e-9;
    check(table_key(nudged) != table_key(spec), "a real change of the grid changes the cache key");

    // Cold recoil (astra's counterexample): E in [0.5, 1] m_e c^2, E' in [0.25, 0.5], xi in [-0.5625, -0.5]
    // contains e' = e / (1 + e (1 - xi)) (e = 0.75, xi = -0.53125), so no electron energy is needed: s = 0.
    {
        TableSpec recoil;
        recoil.boundaries = {0.25 * units::me_c2, 0.5 * units::me_c2, 1.0 * units::me_c2};
        recoil.centers = {0.375 * units::me_c2, 0.75 * units::me_c2};
        recoil.temperatures = {1e3, 1e4};
        recoil.num_angle_bins = 32;
        check(suppression_exponent(recoil, 1, 0, 7, 1e3) == 0.0, "cold recoil inside the box is not suppressed");
    }

    // Failures are collective: a corrupt cache file and an unwritable cache directory make every rank throw.
    auto all_throw = [&](std::string const& cache) {
        int threw = 0;
        try {
            (void) load_or_build_mpi(spec, cache, MPI_COMM_WORLD, nullptr);
        } catch (std::exception const&) {
            threw = 1;
        }
        int total = 0;
        MPI_Allreduce(&threw, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        return total == size;
    };
    std::string const corrupt = directory + "_corrupt";
    std::string const locked = directory + "_readonly";
    if (rank == 0) {
        std::filesystem::create_directories(corrupt);
        std::ofstream(std::filesystem::path(corrupt) / cache_file_name(spec)) << "not a table\n";
        std::filesystem::create_directories(locked);
        std::filesystem::permissions(locked, std::filesystem::perms::owner_read | std::filesystem::perms::owner_exec);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    check(all_throw(corrupt), "a corrupt cache file fails on every rank");
    check(all_throw(locked), "an unwritable cache directory fails on every rank");

    // A stale lock is taken over and the table is built.
    std::string const stale = directory + "_stale";
    if (rank == 0) {
        std::filesystem::create_directories(stale);
        std::filesystem::path const lock = std::filesystem::path(stale) / (cache_file_name(spec) + ".lock");
        std::ofstream(lock) << "dead.owner.0\n";
        std::filesystem::last_write_time(lock, std::filesystem::file_time_type::clock::now() - std::chrono::hours(2));
    }
    MPI_Barrier(MPI_COMM_WORLD);
    Table const taken = load_or_build_mpi(spec, stale, MPI_COMM_WORLD, nullptr, 60.0);
    check(same(taken, mpi), "a stale lock is taken over and the table built");

    if (rank == 0) {
        // The cache must reject a table whose centre separations differ (key aliasing at 12 digits).
        Table altered = read_table((std::filesystem::path(stale) / cache_file_name(spec)).string());
        bool rejected = false;
        TableSpec moved = spec;
        moved.centers[1] *= 1.0 + 1e-6;
        try {
            check_compatible(altered, moved);
        } catch (std::exception const&) {
            rejected = true;
        }
        check(rejected, "check_compatible rejects moved centres");
        std::filesystem::permissions(locked, std::filesystem::perms::owner_all);
        for (std::string const& path : {corrupt, locked, stale}) {
            std::filesystem::remove_all(path);
        }
    }

    double const e = 0.3;
    check(std::abs(minimum_lorentz_factor(e, e, -1.0) - std::sqrt(1.0 + e * e)) < 1e-14,
          "backscatter without energy change needs gamma = sqrt(1 + e^2)");
    check(minimum_lorentz_factor(e, e, 1.0) == 1.0, "forward elastic scattering needs no electron energy");
    check(std::isinf(minimum_lorentz_factor(e, 2.0 * e, 1.0)), "forward scattering cannot change the energy");
    double dw = 0.0;
    check(taper_weight(spec, 399.0, dw) == 1.0 && taper_weight(spec, 501.0, dw) == 0.0 &&
              std::abs(taper_weight(spec, 450.0, dw) - 0.5) < 1e-15 && std::abs(dw + 1.5 / 100.0) < 1e-15,
          "taper weight and slope");
    check(suppression_exponent(spec, 2, 3, 3, 1e4) < 1e-3,
          "adjacent groups are not suppressed (shared boundary)");
    check(suppression_exponent(spec, 0, 5, 0, 1e4) > 500.0,
          "a far upscatter at 1e4 K is fully suppressed");

    int total = failures;
    MPI_Allreduce(MPI_IN_PLACE, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
        std::println("{} failure(s)", failures);
    }
    MPI_Finalize();
    return total == 0 ? 0 : 1;
}
