#ifndef COMPTON_TABLE_MPI_HPP
#define COMPTON_TABLE_MPI_HPP

/**
 * @file compton_table_mpi.hpp
 * @brief MPI construction and caching of Compton tables.
 *
 * Work items (temperature, source group) go round-robin over the ranks of
 * the communicator; the partial tables are summed (each entry has one
 * contributor, so the result equals the serial table bit for bit).
 */

#include "compton_table/compton_table.hpp"

#include <mpi.h>

#include <ostream>
#include <string>

namespace compton::table {

/// Build on all ranks of comm; every rank returns the full table.  Rank 0
/// reports progress with an ETA to log (nullptr: silent).
Table build_mpi(TableSpec const& spec, MPI_Comm comm, std::ostream* log);

/**
 * Read cache_dir/cache_file_name(spec) when it exists; otherwise build it
 * with all ranks of comm and write it from rank 0.  A lock file per key
 * (".lock" next to the table) keeps concurrent jobs from building the same
 * table: a job that finds the lock waits for the table; a lock not touched
 * for stale_seconds is taken over.  Every rank returns the full table.
 */
Table load_or_build_mpi(
    TableSpec const& spec,
    std::string const& cache_dir,
    MPI_Comm comm,
    std::ostream* log,
    double stale_seconds = 1800.0);

} // namespace compton::table

#endif
