#ifndef COMPTON_TABLE_HPP
#define COMPTON_TABLE_HPP

/**
 * @file compton_table.hpp
 * @brief Multigroup Compton tables for transport codes.
 *
 * A table holds, at each table temperature T_i, the angle-resolved
 * number-weighted effective-rate cross sections
 *
 *     sigma_{g->g',a}(T) = (2 pi / B_g) int_g dE B(E) (E_g / E)
 *                          int_{g'} dE' int_{xi bin a} dxi K(E -> E', xi)
 *                          (E' - E) / (E_g' - E_g)          (g' != g)
 *
 * (Planck weight B, NumberWeightedEnergyTransferMultiplier; for g' = g the
 * multiplier is E_g/E, the in-group scattering rate) per free electron [cm^2],
 * and their full temperature derivatives [cm^2/K].  With these coefficients a
 * centre-form packet or group that changes group exchanges the exact energy of
 * a Planck-shaped source group (Giron, Krief & Steinberg).
 *
 * A thermal-truncation taper multiplies each entry by w(s), where s is the
 * smallest electron kinetic energy (in units of kT) that allows the
 * transition anywhere in the (g, g', angle bin) box, in either direction:
 * w = 1 for s <= taper_start, 0 for s >= taper_end, smoothstep in between;
 * the derivative includes dw/dT.
 *
 * Work items (temperature, source group) are independent, so a table can be
 * built in pieces (compute_partial) and summed (see compton_table_mpi.hpp).
 */

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

namespace compton::table {

enum class SolverKind { exact, approximate };

struct TableSpec {
    std::vector<double> boundaries;   ///< G+1 group boundaries [erg]
    std::vector<double> centers;      ///< G group centres [erg]
    std::vector<double> temperatures; ///< table temperatures [K], increasing
    int num_angle_bins = 32;
    SolverKind solver = SolverKind::exact;
    double planck_cap_x = 25.0;       ///< PlanckWeightFunction cap
    double cutoff_ratio = 1e-14;      ///< outward target-group cutoff
    /// Below this temperature the cold quadrature settings are used
    /// (e_panel_order_cold, ep_k_cut 10, E' orders 48).
    double cold_temperature = 8e4;
    int e_panel_order_cold = 128;
    int e_panel_order_hot = 64;
    double taper_start = 400.0;
    double taper_end = 500.0;
};

struct TableDiagnostics {
    /// Largest |sigma_{g'->g} / (sigma_{g->g'} f_db) - 1| over resolved pairs,
    /// f_db the centre-form detailed-balance factor (as enforced at run time).
    double max_detailed_balance_mismatch = 0.0;
    double mismatch_temperature = 0.0;
    int mismatch_g = -1;
    int mismatch_gp = -1;
    std::size_t tapered_entries = 0; ///< entries with 0 < w < 1
    std::size_t zeroed_entries = 0;  ///< nonzero entries removed by w = 0
    double build_seconds = 0.0;      ///< wall time of the build
};

struct Table {
    TableSpec spec;
    std::string key;         ///< cache key (hex)
    std::string mace_commit; ///< MACE commit the table was built with
    std::vector<double> sigma;     ///< [T][g][g'][a] [cm^2]
    std::vector<double> dsigma_dT; ///< [T][g][g'][a] [cm^2/K]
    TableDiagnostics diagnostics;

    std::size_t index(
        std::size_t i,
        std::size_t g,
        std::size_t gp,
        std::size_t a) const;
};

/// Progress of the calling rank: items done, items assigned, seconds.
using ProgressCallback =
    std::function<void(std::size_t done, std::size_t total, double seconds)>;

/// 32 temperatures, logarithmic from 1e3 K to 1.2e10 K.
std::vector<double> default_temperatures();

/// Log-uniform boundaries from E_lo to E_hi [erg] with arithmetic centres.
TableSpec log_grid_spec(double E_lo, double E_hi, int num_groups);

/// Throws std::invalid_argument when the specification is inconsistent.
void validate(TableSpec const& spec);

/// Commit of the MACE source the library was built from ("-dirty" if
/// modified, "unknown" outside a git checkout).
std::string mace_commit();

/// Cache key: a hash of every field of the specification, the format
/// version and the MACE commit.
std::string table_key(TableSpec const& spec);

/// "mace_compton_<key>.tbl"
std::string cache_file_name(TableSpec const& spec);

/// Smallest electron kinetic energy gamma - 1 (units of m_e c^2) that
/// scatters a photon of energy e into energy ep at scattering cosine xi
/// (energies in units of m_e c^2); infinity when impossible.
double minimum_kinetic_energy(double e, double ep, double xi);

/// 1 + minimum_kinetic_energy(e, ep, xi).
double minimum_lorentz_factor(double e, double ep, double xi);

/// Suppression exponent s = K_min / kT of the (g, g', angle bin) box, the
/// smaller of the two directions.
double suppression_exponent(
    TableSpec const& spec,
    int g,
    int gp,
    int a,
    double T);

/// Taper weight w(s) and dw/ds.
double taper_weight(TableSpec const& spec, double s, double& dw_ds);

/// Items k = i G + g with k % size == rank, into a zero-filled table.
Table compute_partial(
    TableSpec const& spec,
    int rank,
    int size,
    ProgressCallback const& progress = {});

/// Detailed-balance diagnostics of an assembled table.
void finalize(Table& table);

/// All items on the calling process (OpenMP inside each item).
Table build_serial(
    TableSpec const& spec,
    ProgressCallback const& progress = {});

/// Versioned cache file: a text header ending with END_HEADER, then the
/// arrays as raw little-endian doubles.  Written to a temporary file and
/// renamed.
void write_table(std::string const& path, Table const& table);

Table read_table(std::string const& path);

} // namespace compton::table

#endif
