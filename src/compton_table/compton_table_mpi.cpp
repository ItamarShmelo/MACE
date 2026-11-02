#include "compton_table/compton_table_mpi.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <format>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace compton::table {

namespace {

constexpr std::size_t mpi_chunk = std::size_t{1} << 26;

void allreduce_sum(std::vector<double>& values, MPI_Comm const comm)
{
    for (std::size_t offset = 0; offset < values.size(); offset += mpi_chunk) {
        int const count = static_cast<int>(std::min(mpi_chunk, values.size() - offset));
        MPI_Allreduce(MPI_IN_PLACE, values.data() + offset, count, MPI_DOUBLE, MPI_SUM, comm);
    }
}

void broadcast(std::vector<double>& values, MPI_Comm const comm)
{
    unsigned long long size = values.size();
    MPI_Bcast(&size, 1, MPI_UNSIGNED_LONG_LONG, 0, comm);
    values.resize(static_cast<std::size_t>(size));
    for (std::size_t offset = 0; offset < values.size(); offset += mpi_chunk) {
        int const count = static_cast<int>(std::min(mpi_chunk, values.size() - offset));
        MPI_Bcast(values.data() + offset, count, MPI_DOUBLE, 0, comm);
    }
}

void broadcast(std::string& text, MPI_Comm const comm)
{
    unsigned long long size = text.size();
    MPI_Bcast(&size, 1, MPI_UNSIGNED_LONG_LONG, 0, comm);
    text.resize(static_cast<std::size_t>(size));
    MPI_Bcast(text.data(), static_cast<int>(size), MPI_CHAR, 0, comm);
}

/// Rank 0's outcome of a step only it performs, shared before any further
/// collective: every rank throws the same error when rank 0 failed.
void share_root_status(bool const failed, std::string message, MPI_Comm const comm)
{
    int status = failed ? 1 : 0;
    MPI_Bcast(&status, 1, MPI_INT, 0, comm);
    if (status != 0) {
        broadcast(message, comm);
        throw std::runtime_error(message);
    }
}

/// Every rank's outcome of a step all ranks perform: if any failed, all throw.
void share_all_status(bool const failed, std::string const& message, MPI_Comm const comm)
{
    int local = failed ? 1 : 0;
    int any = 0;
    MPI_Allreduce(&local, &any, 1, MPI_INT, MPI_MAX, comm);
    if (any != 0) {
        throw std::runtime_error(failed ? message : "Compton table build failed on another rank");
    }
}

std::string format_seconds(double const seconds)
{
    auto const total = static_cast<long long>(seconds + 0.5);
    if (total >= 3600) {
        return std::format("{}h{:02}m", total / 3600, (total % 3600) / 60);
    }
    return std::format("{}m{:02}s", total / 60, total % 60);
}

} // namespace

Table build_mpi(TableSpec const& spec, MPI_Comm const comm, std::ostream* const log)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    double last_report = 0.0;
    ProgressCallback progress;
    if (rank == 0 && log != nullptr) {
        progress = [&](std::size_t const done, std::size_t const mine, double const seconds) {
            if (done < mine && seconds - last_report < 60.0) {
                return;
            }
            last_report = seconds;
            double const eta = seconds * static_cast<double>(mine - done) / static_cast<double>(done);
            *log << std::format(
                        "[mace] Compton table: {}/{} items on rank 0 ({:.0f}%), {} elapsed, ETA {}\n",
                        done,
                        mine,
                        100.0 * static_cast<double>(done) / static_cast<double>(mine),
                        format_seconds(seconds),
                        format_seconds(eta))
                 << std::flush;
        };
    }
    Table table;
    bool failed = false;
    std::string message;
    try {
        table = compute_partial(spec, rank, size, progress);
    } catch (std::exception const& error) {
        failed = true;
        message = std::format("Compton table build failed on rank {}: {}", rank, error.what());
    }
    share_all_status(failed, message, comm);
    allreduce_sum(table.sigma, comm);
    allreduce_sum(table.dsigma_dT, comm);
    double seconds = table.diagnostics.build_seconds;
    MPI_Allreduce(MPI_IN_PLACE, &seconds, 1, MPI_DOUBLE, MPI_MAX, comm);
    table.diagnostics.build_seconds = seconds;
    unsigned long long tapered = table.diagnostics.tapered_entries;
    MPI_Allreduce(MPI_IN_PLACE, &tapered, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);
    table.diagnostics.tapered_entries = static_cast<std::size_t>(tapered);
    finalize(table);
    return table;
}

Table load_or_build_mpi(
    TableSpec const& spec,
    std::string const& cache_dir,
    MPI_Comm const comm,
    std::ostream* const log,
    double const stale_seconds)
{
    validate(spec);
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    std::filesystem::path const directory =
        cache_dir.empty() ? std::filesystem::path(".") : std::filesystem::path(cache_dir);
    std::string const path = (directory / cache_file_name(spec)).string();

    // Rank 0 decides (read, or build holding the lock) and shares the outcome.
    std::unique_ptr<CacheLock> lock;
    int action = 0;
    {
        bool failed = false;
        std::string message;
        if (rank == 0) {
            try {
                action = acquire_table(path, stale_seconds, log, lock) == CacheDecision::build ? 1 : 0;
            } catch (std::exception const& error) {
                failed = true;
                message = error.what();
            }
        }
        share_root_status(failed, message, comm);
    }
    MPI_Bcast(&action, 1, MPI_INT, 0, comm);

    Table table;
    if (action == 1) {
        if (rank == 0 && log != nullptr) {
            *log << "[mace] building Compton table " << path << '\n' << std::flush;
        }
        table = build_mpi(spec, comm, log);
        bool failed = false;
        std::string message;
        if (rank == 0) {
            try {
                write_table(path, table);
            } catch (std::exception const& error) {
                failed = true;
                message = error.what();
            }
            lock.reset();
        }
        share_root_status(failed, message, comm);
    } else {
        std::vector<double> diagnostics(6, 0.0);
        bool failed = false;
        std::string message;
        if (rank == 0) {
            try {
                table = read_table(path);
                check_compatible(table, spec);
                diagnostics = {table.diagnostics.max_detailed_balance_mismatch,
                               table.diagnostics.mismatch_temperature,
                               static_cast<double>(table.diagnostics.mismatch_g),
                               static_cast<double>(table.diagnostics.mismatch_gp),
                               table.diagnostics.build_seconds,
                               static_cast<double>(table.diagnostics.tapered_entries)};
            } catch (std::exception const& error) {
                failed = true;
                message = "Compton table " + path + ": " + error.what();
            }
        }
        share_root_status(failed, message, comm);
        broadcast(table.key, comm);
        broadcast(table.mace_commit, comm);
        broadcast(table.sigma, comm);
        broadcast(table.dsigma_dT, comm);
        broadcast(diagnostics, comm);
        table.diagnostics.max_detailed_balance_mismatch = diagnostics[0];
        table.diagnostics.mismatch_temperature = diagnostics[1];
        table.diagnostics.mismatch_g = static_cast<int>(diagnostics[2]);
        table.diagnostics.mismatch_gp = static_cast<int>(diagnostics[3]);
        table.diagnostics.build_seconds = diagnostics[4];
        table.diagnostics.tapered_entries = static_cast<std::size_t>(diagnostics[5]);
    }
    table.spec = spec;
    if (rank == 0 && log != nullptr) {
        TableDiagnostics const& d = table.diagnostics;
        *log << std::format(
                    "[mace] Compton table {} ({}, built in {}); max detailed-balance mismatch {:.2e}\n",
                    path,
                    action == 1 ? "built now" : "read from cache",
                    format_seconds(d.build_seconds),
                    d.max_detailed_balance_mismatch);
        if (d.max_detailed_balance_mismatch > 1e-4) {
            *log << std::format(
                        "[mace] WARNING: detailed-balance mismatch {:.2e} above 1e-4 at T = {:.4g} K, "
                        "groups {} and {} (enforced at run time)\n",
                        d.max_detailed_balance_mismatch,
                        d.mismatch_temperature,
                        d.mismatch_g,
                        d.mismatch_gp);
        }
        log->flush();
    }
    return table;
}

} // namespace compton::table
