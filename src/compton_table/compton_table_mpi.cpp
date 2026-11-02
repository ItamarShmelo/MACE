#include "compton_table/compton_table_mpi.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <format>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

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

std::string format_seconds(double const seconds)
{
    auto const total = static_cast<long long>(seconds + 0.5);
    if (total >= 3600) {
        return std::format("{}h{:02}m", total / 3600, (total % 3600) / 60);
    }
    return std::format("{}m{:02}s", total / 60, total % 60);
}

/// Touches the lock file every minute while the table is being built, so
/// other jobs do not take it over.
class LockHeartbeat {
  public:
    explicit LockHeartbeat(std::filesystem::path path)
        : path_(std::move(path)), thread_([this] { run(); })
    {
    }
    LockHeartbeat(LockHeartbeat const&) = delete;
    LockHeartbeat& operator=(LockHeartbeat const&) = delete;
    ~LockHeartbeat()
    {
        stop_ = true;
        thread_.join();
    }

  private:
    void run()
    {
        auto last = std::chrono::steady_clock::now();
        while (!stop_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            if (std::chrono::steady_clock::now() - last >= std::chrono::seconds(60)) {
                std::error_code error;
                std::filesystem::last_write_time(
                    path_, std::filesystem::file_time_type::clock::now(), error);
                last = std::chrono::steady_clock::now();
            }
        }
    }

    std::filesystem::path path_;
    std::atomic<bool> stop_{false};
    std::thread thread_;
};

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
    Table table = compute_partial(spec, rank, size, progress);
    allreduce_sum(table.sigma, comm);
    allreduce_sum(table.dsigma_dT, comm);
    double seconds = table.diagnostics.build_seconds;
    MPI_Allreduce(MPI_IN_PLACE, &seconds, 1, MPI_DOUBLE, MPI_MAX, comm);
    table.diagnostics.build_seconds = seconds;
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
    std::filesystem::path const directory = cache_dir.empty() ? std::filesystem::path(".") : std::filesystem::path(cache_dir);
    std::filesystem::path const path = directory / cache_file_name(spec);
    std::filesystem::path const lock = std::filesystem::path(path.string() + ".lock");

    // Rank 0 decides: 0 = read the table, 1 = build it (this job holds the lock).
    int action = 0;
    if (rank == 0) {
        std::filesystem::create_directories(directory);
        bool announced = false;
        while (true) {
            if (std::filesystem::exists(path)) {
                action = 0;
                break;
            }
            int const fd = ::open(lock.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0644);
            if (fd >= 0) {
                char host[256] = {};
                ::gethostname(host, sizeof(host) - 1);
                std::string const owner = std::format("{} {}\n", host, ::getpid());
                (void) ::write(fd, owner.data(), owner.size());
                ::close(fd);
                action = 1;
                break;
            }
            std::error_code error;
            auto const modified = std::filesystem::last_write_time(lock, error);
            double const age = error ? 0.0
                                     : std::chrono::duration<double>(
                                           std::filesystem::file_time_type::clock::now() - modified)
                                           .count();
            if (age > stale_seconds) {
                if (log != nullptr) {
                    *log << "[mace] taking over the stale Compton table lock " << lock << '\n';
                }
                std::filesystem::remove(lock, error);
                continue;
            }
            if (!announced && log != nullptr) {
                *log << "[mace] waiting for another job to build " << path << '\n' << std::flush;
                announced = true;
            }
            std::this_thread::sleep_for(std::chrono::seconds(10));
        }
    }
    MPI_Bcast(&action, 1, MPI_INT, 0, comm);

    Table table;
    if (action == 1) {
        if (rank == 0 && log != nullptr) {
            *log << "[mace] building Compton table " << path << '\n' << std::flush;
        }
        {
            std::unique_ptr<LockHeartbeat> heartbeat;
            if (rank == 0) {
                heartbeat = std::make_unique<LockHeartbeat>(lock);
            }
            table = build_mpi(spec, comm, log);
            if (rank == 0) {
                write_table(path.string(), table);
            }
        }
        if (rank == 0) {
            std::error_code error;
            std::filesystem::remove(lock, error);
        }
        MPI_Barrier(comm);
    } else {
        std::vector<double> diagnostics(5, 0.0);
        if (rank == 0) {
            table = read_table(path.string());
            if (table.key != table_key(spec)) {
                throw std::runtime_error("Compton table " + path.string() +
                                         " does not match the requested specification");
            }
            diagnostics = {table.diagnostics.max_detailed_balance_mismatch,
                           table.diagnostics.mismatch_temperature,
                           static_cast<double>(table.diagnostics.mismatch_g),
                           static_cast<double>(table.diagnostics.mismatch_gp),
                           table.diagnostics.build_seconds};
        } else {
            table.spec = spec;
        }
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
        table.spec = spec;
    }
    if (rank == 0 && log != nullptr) {
        TableDiagnostics const& d = table.diagnostics;
        *log << std::format(
                    "[mace] Compton table {} ({}, built in {}); max detailed-balance mismatch {:.2e}\n",
                    path.string(),
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
