// build_compton_table: build (or find) a MACE Compton table cache file with
// MPI, for job scripts that start many serial runs on the same grid.
//
//   mpirun -np N build_compton_table --groups-file groups.txt [options]
//   mpirun -np N build_compton_table --log-grid 1e-4,1e3,32 [options]
//
// --groups-file FILE   lines "lower upper centre" [erg]; '#' starts a comment
//                      (the groups.txt the IMC Compton drivers write)
// --log-grid LO,HI,G   G log-uniform groups from LO to HI [keV], arithmetic
//                      centres
// --angle-bins N       default 32
// --solver exact|approximate
// --t-min K --t-max K --t-count N   default 1e3, 1.2e10, 32 (logarithmic)
// --cache-dir DIR      default .
// --threads N          OpenMP threads per rank (default 1 with more than one
//                      rank, else OMP_NUM_THREADS or all cores)
//
// Prints the path of the table.

#include "compton_table/compton_table.hpp"
#include "compton_table/compton_table_mpi.hpp"
#include "utilities/units.hpp"

#include <mpi.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <print>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using compton::table::SolverKind;
using compton::table::TableSpec;

double parse_number(std::string const& text, std::string_view const name)
{
    std::size_t used = 0;
    double const value = std::stod(text, &used);
    if (used != text.size() || !std::isfinite(value)) {
        throw std::invalid_argument("invalid value for " + std::string(name) + ": " + text);
    }
    return value;
}

void read_groups_file(std::string const& path, TableSpec& spec)
{
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("cannot open " + path);
    }
    std::vector<double> lower;
    std::vector<double> upper;
    for (std::string line; std::getline(file, line);) {
        if (auto const hash = line.find('#'); hash != std::string::npos) {
            line.erase(hash);
        }
        std::istringstream fields(line);
        double lo = 0.0;
        double hi = 0.0;
        double centre = 0.0;
        if (!(fields >> lo >> hi >> centre)) {
            continue;
        }
        lower.push_back(lo);
        upper.push_back(hi);
        spec.centers.push_back(centre);
    }
    if (lower.empty()) {
        throw std::runtime_error("no groups in " + path);
    }
    for (std::size_t g = 0; g + 1 < lower.size(); ++g) {
        if (upper[g] != lower[g + 1]) {
            throw std::runtime_error("groups in " + path + " are not contiguous");
        }
    }
    spec.boundaries = lower;
    spec.boundaries.push_back(upper.back());
}

} // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int status = 0;
    try {
        TableSpec spec;
        bool have_grid = false;
        double t_min = 1e3;
        double t_max = 1.2e10;
        int t_count = 32;
        int threads = 0;
        std::string cache_dir = ".";
        for (int k = 1; k < argc; ++k) {
            std::string const argument = argv[k];
            auto value = [&]() -> std::string {
                if (k + 1 >= argc) {
                    throw std::invalid_argument("missing value for " + argument);
                }
                return argv[++k];
            };
            if (argument == "--groups-file") {
                read_groups_file(value(), spec);
                have_grid = true;
            } else if (argument == "--log-grid") {
                std::string const text = value();
                std::istringstream fields(text);
                std::string lo, hi, groups;
                std::getline(fields, lo, ',');
                std::getline(fields, hi, ',');
                std::getline(fields, groups, ',');
                TableSpec grid = compton::table::log_grid_spec(
                    parse_number(lo, argument) * units::kev,
                    parse_number(hi, argument) * units::kev,
                    static_cast<int>(parse_number(groups, argument)));
                spec.boundaries = grid.boundaries;
                spec.centers = grid.centers;
                have_grid = true;
            } else if (argument == "--angle-bins") {
                spec.num_angle_bins = static_cast<int>(parse_number(value(), argument));
            } else if (argument == "--solver") {
                std::string const name = value();
                if (name == "exact") {
                    spec.solver = SolverKind::exact;
                } else if (name == "approximate") {
                    spec.solver = SolverKind::approximate;
                } else {
                    throw std::invalid_argument("--solver takes exact or approximate");
                }
            } else if (argument == "--t-min") {
                t_min = parse_number(value(), argument);
            } else if (argument == "--t-max") {
                t_max = parse_number(value(), argument);
            } else if (argument == "--t-count") {
                t_count = static_cast<int>(parse_number(value(), argument));
            } else if (argument == "--cache-dir") {
                cache_dir = value();
            } else if (argument == "--threads") {
                threads = static_cast<int>(parse_number(value(), argument));
            } else {
                throw std::invalid_argument("unknown option " + argument);
            }
        }
        if (!have_grid) {
            throw std::invalid_argument("give --groups-file or --log-grid");
        }
        if (t_count < 1 || !(t_min > 0.0) || !(t_max >= t_min)) {
            throw std::invalid_argument("invalid temperature range");
        }
        spec.temperatures.resize(static_cast<std::size_t>(t_count));
        for (int i = 0; i < t_count; ++i) {
            spec.temperatures[static_cast<std::size_t>(i)] =
                t_count == 1 ? t_min : t_min * std::pow(t_max / t_min, static_cast<double>(i) / (t_count - 1));
        }
#ifdef _OPENMP
        if (threads > 0) {
            omp_set_num_threads(threads);
        } else if (size > 1) {
            omp_set_num_threads(1);
        }
#endif
        compton::table::Table const table = compton::table::load_or_build_mpi(
            spec,
            cache_dir,
            MPI_COMM_WORLD,
            rank == 0 ? &std::cout : nullptr);
        if (rank == 0) {
            std::println("{}", (std::filesystem::path(cache_dir) / compton::table::cache_file_name(spec)).string());
        }
    } catch (std::exception const& error) {
        std::cerr << "build_compton_table (rank " << rank << "): " << error.what() << '\n';
        status = 1;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
