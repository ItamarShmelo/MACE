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
