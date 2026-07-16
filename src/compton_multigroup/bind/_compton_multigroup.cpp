#include <pybind11/numpy.h> // NOLINT(misc-include-cleaner) -- implicit numpy converters
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // NOLINT(misc-include-cleaner) -- implicit STL converters

#include "compton_multigroup/compton_multigroup_deterministic/compton_multigroup_deterministic.hpp"
#include "compton_multigroup/compton_multigroup_monte_carlo/compton_multigroup_monte_carlo.hpp"
#include "compton_multigroup/weight_function.hpp"
#include "utilities/bind_helpers.hpp"
#include "utilities/gauss_legendre.hpp"

#include <cstddef>
#include <optional>

namespace py = pybind11;
using namespace pybind11::literals;
using namespace compton;
using compton::bind::flat_to_numpy_2d;
using compton::bind::flat_to_numpy_3d;

PYBIND11_MODULE(_compton_multigroup, m) // NOLINT(misc-include-cleaner)
{
    m.doc() = "Weighted multigroup-multiangle Compton scattering matrix";

    py::module_::import("compton_matrix._compton_common");
    py::module_::import("compton_matrix._compton_differential_cross_section");

    py::class_<KernelMultiplier>(m, "KernelMultiplier");

    py::class_<ConstantMultiplier, KernelMultiplier>(m, "ConstantMultiplier")
        .def(py::init<>());

    py::class_<MGIntegrationConfig>(m, "MGIntegrationConfig")
        .def(
            py::init<
                std::optional<double>,
                std::optional<int>,
                double,
                std::optional<int>,
                double,
                double,
                std::optional<int>,
                std::optional<int>,
                std::optional<int>,
                double,
                double>(),
            "cutoff_ratio"_a = 1e-8,     // NOLINT(misc-include-cleaner)
            "xi_order"_a = std::nullopt, // NOLINT(misc-include-cleaner)
            "xi_peak_k"_a = 5.0,
            "xi_tail_order"_a = std::nullopt,
            "ep_k_cut"_a = 5.0,
            "ep_k_in"_a = 2.0,
            "ep_edge_order"_a = std::nullopt,
            "ep_interior_order"_a = std::nullopt,
            "e_panel_order"_a = std::nullopt,
            "log_e_panel_ratio"_a = 2.0,
            "e_boundary_k"_a = 5.0)
        .def_readwrite("xi_order", &MGIntegrationConfig::xi_order)
        .def_readwrite("xi_tail_order", &MGIntegrationConfig::xi_tail_order)
        .def_readwrite("cutoff_ratio", &MGIntegrationConfig::cutoff_ratio)
        .def_readwrite("xi_peak_k", &MGIntegrationConfig::xi_peak_k)
        .def_readwrite("ep_k_cut", &MGIntegrationConfig::ep_k_cut)
        .def_readwrite("ep_k_in", &MGIntegrationConfig::ep_k_in)
        .def_readwrite("ep_edge_order", &MGIntegrationConfig::ep_edge_order)
        .def_readwrite(
            "ep_interior_order",
            &MGIntegrationConfig::ep_interior_order)
        .def_readwrite("e_panel_order", &MGIntegrationConfig::e_panel_order)
        .def_readwrite(
            "log_e_panel_ratio",
            &MGIntegrationConfig::log_e_panel_ratio)
        .def_readwrite("e_boundary_k", &MGIntegrationConfig::e_boundary_k)
        .def("effective_xi_order", &MGIntegrationConfig::effective_xi_order)
        .def(
            "effective_xi_tail_order",
            &MGIntegrationConfig::effective_xi_tail_order)
        .def(
            "effective_ep_edge_order",
            &MGIntegrationConfig::effective_ep_edge_order)
        .def(
            "effective_ep_interior_order",
            &MGIntegrationConfig::effective_ep_interior_order)
        .def(
            "effective_e_panel_order",
            &MGIntegrationConfig::effective_e_panel_order);

    py::class_<ComptonMultigroupKernel>(m, "ComptonMultigroupKernel")
        .def(
            py::init<
                std::vector<double> const&,
                std::shared_ptr<WeightFunction const>,
                MGIntegrationConfig const&>(),
            "energy_group_boundaries"_a,
            "weight_function"_a,
            "config"_a = MGIntegrationConfig{})

        .def_property_readonly(
            "num_groups",
            &ComptonMultigroupKernel::num_groups)

        .def_property_readonly(
            "group_boundaries",
            [](ComptonMultigroupKernel const& self) {
                auto const& b = self.group_boundaries();
                py::array_t<double> arr(b.size());
                auto buf = arr.mutable_unchecked<1>();
                for (py::ssize_t i = 0; // NOLINT(misc-include-cleaner)
                     i < static_cast<py::ssize_t>(b.size());
                     ++i)
                    buf(i) = b[i];
                return arr;
            })

        .def(
            "compute_sigma_matrix",
            [](ComptonMultigroupKernel const& self,
               ComptonKernelSolver const& kernel,
               int num_angle_bins,
               double T,
               double Ne,
               KernelMultiplier const& multiplier) {
                return flat_to_numpy_3d(self.num_groups(), num_angle_bins, [&] {
                    return self.compute_sigma_matrix(
                        kernel,
                        num_angle_bins,
                        T,
                        Ne,
                        multiplier);
                });
            },
            "kernel"_a,
            "num_angle_bins"_a,
            "T"_a,
            "Ne"_a,
            "multiplier"_a = ConstantMultiplier())

        .def(
            "compute_dsigma_dT_matrix",
            [](ComptonMultigroupKernel const& self,
               ComptonKernelSolver const& kernel,
               int num_angle_bins,
               double T,
               double Ne,
               KernelMultiplier const& multiplier) {
                return flat_to_numpy_3d(self.num_groups(), num_angle_bins, [&] {
                    return self.compute_dsigma_dT_matrix(
                        kernel,
                        num_angle_bins,
                        T,
                        Ne,
                        multiplier);
                });
            },
            "kernel"_a,
            "num_angle_bins"_a,
            "T"_a,
            "Ne"_a,
            "multiplier"_a = ConstantMultiplier())

        .def(
            "compute_sigma_matrix",
            [](ComptonMultigroupKernel const& self,
               ComptonKernelSolver const& kernel,
               double T,
               double Ne,
               KernelMultiplier const& multiplier) {
                return flat_to_numpy_2d(self.num_groups(), [&] {
                    return self
                        .compute_sigma_matrix(kernel, 1, T, Ne, multiplier);
                });
            },
            "kernel"_a,
            "T"_a,
            "Ne"_a,
            "multiplier"_a = ConstantMultiplier())

        .def(
            "compute_dsigma_dT_matrix",
            [](ComptonMultigroupKernel const& self,
               ComptonKernelSolver const& kernel,
               double T,
               double Ne,
               KernelMultiplier const& multiplier) {
                return flat_to_numpy_2d(self.num_groups(), [&] {
                    return self
                        .compute_dsigma_dT_matrix(kernel, 1, T, Ne, multiplier);
                });
            },
            "kernel"_a,
            "T"_a,
            "Ne"_a,
            "multiplier"_a = ConstantMultiplier())

        .def(
            "compute_full_dsigma_dT_matrix",
            [](ComptonMultigroupKernel const& self,
               ComptonKernelSolver const& kernel,
               int num_angle_bins,
               double T,
               double Ne,
               KernelMultiplier const& multiplier) {
                return flat_to_numpy_3d(self.num_groups(), num_angle_bins, [&] {
                    return self.compute_full_dsigma_dT_matrix(
                        kernel,
                        num_angle_bins,
                        T,
                        Ne,
                        multiplier);
                });
            },
            "kernel"_a,
            "num_angle_bins"_a,
            "T"_a,
            "Ne"_a,
            "multiplier"_a = ConstantMultiplier())

        .def(
            "compute_full_dsigma_dT_matrix",
            [](ComptonMultigroupKernel const& self,
               ComptonKernelSolver const& kernel,
               double T,
               double Ne,
               KernelMultiplier const& multiplier) {
                return flat_to_numpy_2d(self.num_groups(), [&] {
                    return self.compute_full_dsigma_dT_matrix(
                        kernel,
                        1,
                        T,
                        Ne,
                        multiplier);
                });
            },
            "kernel"_a,
            "T"_a,
            "Ne"_a,
            "multiplier"_a = ConstantMultiplier())

        .def(
            "compute_xi_integral_sigma",
            [](ComptonMultigroupKernel const& self,
               ComptonKernelSolver const& kernel,
               double E,
               double Ep,
               int num_xi_bins,
               double T,
               double Ne,
               KernelMultiplier const& multiplier) {
                auto v = self.compute_xi_integral_impl(
                    kernel,
                    &ComptonKernelSolver::sigma_E,
                    E,
                    Ep,
                    num_xi_bins,
                    T,
                    Ne,
                    multiplier);
                py::array_t<double> arr(v.size());
                auto buf = arr.mutable_unchecked<1>();
                for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(v.size());
                     ++i)
                    buf(i) = v[i];
                return arr;
            },
            "kernel"_a,
            "E"_a,
            "Ep"_a,
            "num_xi_bins"_a,
            "T"_a,
            "Ne"_a,
            "multiplier"_a = ConstantMultiplier(),
            "Integrate sigma_E over xi bins for fixed (E, E')")

        .def(
            "compute_xi_integral_dsigma_dT",
            [](ComptonMultigroupKernel const& self,
               ComptonKernelSolver const& kernel,
               double E,
               double Ep,
               int num_xi_bins,
               double T,
               double Ne,
               KernelMultiplier const& multiplier) {
                auto v = self.compute_xi_integral_impl(
                    kernel,
                    &ComptonKernelSolver::dsigma_E_dT,
                    E,
                    Ep,
                    num_xi_bins,
                    T,
                    Ne,
                    multiplier);
                py::array_t<double> arr(v.size());
                auto buf = arr.mutable_unchecked<1>();
                for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(v.size());
                     ++i)
                    buf(i) = v[i];
                return arr;
            },
            "kernel"_a,
            "E"_a,
            "Ep"_a,
            "num_xi_bins"_a,
            "T"_a,
            "Ne"_a,
            "multiplier"_a = ConstantMultiplier(),
            "Integrate dsigma_E/dT over xi bins for fixed (E, E')")

        .def(
            "compute_Ep_xi_integral_sigma",
            [](ComptonMultigroupKernel const& self,
               ComptonKernelSolver const& kernel,
               double E,
               double Ep_lo,
               double Ep_hi,
               int num_xi_bins,
               double T,
               double Ne,
               KernelMultiplier const& multiplier) {
                auto v = self.compute_Ep_xi_integral_impl(
                    kernel,
                    &ComptonKernelSolver::sigma_E,
                    E,
                    Ep_lo,
                    Ep_hi,
                    num_xi_bins,
                    T,
                    Ne,
                    multiplier);
                py::array_t<double> arr(v.size());
                auto buf = arr.mutable_unchecked<1>();
                for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(v.size());
                     ++i)
                    buf(i) = v[i];
                return arr;
            },
            "kernel"_a,
            "E"_a,
            "Ep_lo"_a,
            "Ep_hi"_a,
            "num_xi_bins"_a,
            "T"_a,
            "Ne"_a,
            "multiplier"_a = ConstantMultiplier(),
            "Integrate sigma_E over E' range and xi bins for fixed E")

        .def(
            "compute_Ep_xi_integral_dsigma_dT",
            [](ComptonMultigroupKernel const& self,
               ComptonKernelSolver const& kernel,
               double E,
               double Ep_lo,
               double Ep_hi,
               int num_xi_bins,
               double T,
               double Ne,
               KernelMultiplier const& multiplier) {
                auto v = self.compute_Ep_xi_integral_impl(
                    kernel,
                    &ComptonKernelSolver::dsigma_E_dT,
                    E,
                    Ep_lo,
                    Ep_hi,
                    num_xi_bins,
                    T,
                    Ne,
                    multiplier);
                py::array_t<double> arr(v.size());
                auto buf = arr.mutable_unchecked<1>();
                for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(v.size());
                     ++i)
                    buf(i) = v[i];
                return arr;
            },
            "kernel"_a,
            "E"_a,
            "Ep_lo"_a,
            "Ep_hi"_a,
            "num_xi_bins"_a,
            "T"_a,
            "Ne"_a,
            "multiplier"_a = ConstantMultiplier(),
            "Integrate dsigma_E/dT over E' range and xi bins for fixed E");

    py::class_<WeightFunction, std::shared_ptr<WeightFunction>>(
        m,
        "WeightFunction");

    py::class_<
        PlanckWeightFunction,
        WeightFunction,
        std::shared_ptr<PlanckWeightFunction>>(m, "PlanckWeightFunction")
        .def(py::init<double>(), py::kw_only(), "cap_x"_a)
        .def("weight", &PlanckWeightFunction::weight, "E"_a, "T"_a)
        .def(
            "compute_denominator",
            &PlanckWeightFunction::compute_denominator,
            "E_left"_a,
            "E_right"_a,
            "T"_a)
        .def("peak_energy", &PlanckWeightFunction::peak_energy, "T"_a)
        .def("d_weight_dT", &PlanckWeightFunction::d_weight_dT, "E"_a, "T"_a)
        .def(
            "d_log_weight_dT",
            &PlanckWeightFunction::d_log_weight_dT,
            "E"_a,
            "T"_a)
        .def(
            "d_denominator_dT",
            &PlanckWeightFunction::d_denominator_dT,
            "E_left"_a,
            "E_right"_a,
            "T"_a)
        .def("cap_x", &PlanckWeightFunction::cap_x);

    py::class_<
        UniformWeightFunction,
        WeightFunction,
        std::shared_ptr<UniformWeightFunction>>(m, "UniformWeightFunction")
        .def(py::init<>())
        .def("weight", &UniformWeightFunction::weight, "E"_a, "T"_a)
        .def(
            "compute_denominator",
            &UniformWeightFunction::compute_denominator,
            "E_left"_a,
            "E_right"_a,
            "T"_a)
        .def("peak_energy", &UniformWeightFunction::peak_energy, "T"_a)
        .def("d_weight_dT", &UniformWeightFunction::d_weight_dT, "E"_a, "T"_a)
        .def(
            "d_log_weight_dT",
            &UniformWeightFunction::d_log_weight_dT,
            "E"_a,
            "T"_a)
        .def(
            "d_denominator_dT",
            &UniformWeightFunction::d_denominator_dT,
            "E_left"_a,
            "E_right"_a,
            "T"_a);

    py::class_<
        WienWeightFunction,
        WeightFunction,
        std::shared_ptr<WienWeightFunction>>(m, "WienWeightFunction")
        .def(py::init<double>(), py::kw_only(), "cap_x"_a)
        .def("weight", &WienWeightFunction::weight, "E"_a, "T"_a)
        .def(
            "compute_denominator",
            &WienWeightFunction::compute_denominator,
            "E_left"_a,
            "E_right"_a,
            "T"_a)
        .def("peak_energy", &WienWeightFunction::peak_energy, "T"_a)
        .def("d_weight_dT", &WienWeightFunction::d_weight_dT, "E"_a, "T"_a)
        .def(
            "d_log_weight_dT",
            &WienWeightFunction::d_log_weight_dT,
            "E"_a,
            "T"_a)
        .def(
            "d_denominator_dT",
            &WienWeightFunction::d_denominator_dT,
            "E_left"_a,
            "E_right"_a,
            "T"_a)
        .def("cap_x", &WienWeightFunction::cap_x);

    m.def(
        "gauss_legendre_rule",
        [](int N) {
            auto rule = compton::compute_gauss_legendre(N);
            py::array_t<double> nodes(static_cast<py::ssize_t>(
                rule.nodes.size())); // NOLINT(misc-include-cleaner)
            py::array_t<double> weights(
                static_cast<py::ssize_t>(rule.weights.size()));
            auto nodes_buf = nodes.mutable_unchecked<1>();
            auto weights_buf = weights.mutable_unchecked<1>();
            for (py::ssize_t i = 0; // NOLINT(misc-include-cleaner)
                 i < static_cast<py::ssize_t>(rule.nodes.size());
                 ++i) {
                nodes_buf(i) = rule.nodes[i];
                weights_buf(i) = rule.weights[i];
            }
            return py::make_tuple(nodes, weights);
        },
        "N"_a,
        "Compute N-point Gauss-Legendre nodes and weights on [-1, 1]");

    m.def(
        "adaptive_legendre_integrate",
        [](py::function const& integrand, // NOLINT(misc-include-cleaner)
           int base_order,
           double a,
           double b,
           double tol,
           int max_depth) {
            auto rule = compton::compute_gauss_legendre(base_order);
            return compton::adaptive_legendre_integrate(
                [&](double x) { return integrand(x).cast<double>(); },
                rule,
                a,
                b,
                tol,
                max_depth);
        },
        "integrand"_a,
        "base_order"_a,
        "a"_a,
        "b"_a,
        "tol"_a = 1e-8,
        "max_depth"_a = 15,
        "Adaptive Gauss-Legendre integration of f over [a, b]");

    m.def(
        "adaptive_log_legendre_integrate",
        [](py::function const& integrand,
           int base_order,
           double a,
           double b,
           double tol,
           int max_depth) {
            auto rule = compton::compute_gauss_legendre(base_order);
            return compton::adaptive_log_legendre_integrate(
                [&](double x) { return integrand(x).cast<double>(); },
                rule,
                a,
                b,
                tol,
                max_depth);
        },
        "integrand"_a,
        "base_order"_a,
        "a"_a,
        "b"_a,
        "tol"_a = 1e-8,
        "max_depth"_a = 15,
        "Adaptive log-space GL integration of f over [a, b] (clusters nodes "
        "near a)");

    m.def(
        "adaptive_rlog_legendre_integrate",
        [](py::function const& integrand,
           int base_order,
           double a,
           double b,
           double tol,
           int max_depth) {
            auto rule = compton::compute_gauss_legendre(base_order);
            return compton::adaptive_rlog_legendre_integrate(
                [&](double x) { return integrand(x).cast<double>(); },
                rule,
                a,
                b,
                tol,
                max_depth);
        },
        "integrand"_a,
        "base_order"_a,
        "a"_a,
        "b"_a,
        "tol"_a = 1e-8,
        "max_depth"_a = 15,
        "Adaptive reflected-log GL integration of f over [a, b] (clusters "
        "nodes near b)");

    m.def(
        "cold_recoil_lo",
        [](double E, double xi_lo) {
            return compton::compute_ridge_bounds(E, xi_lo, 1.0, 0.0).cold_lo;
        },
        "E"_a,
        "xi_lo"_a,
        "Lower edge of the cold Compton recoil band [erg] (T=0 limit)");

    m.def(
        "cold_recoil_hi",
        [](double E, double xi_hi) {
            return compton::compute_ridge_bounds(E, -1.0, xi_hi, 0.0).cold_hi;
        },
        "E"_a,
        "xi_hi"_a,
        "Upper edge of the cold Compton recoil band [erg] (T=0 limit)");

    m.def(
        "ridge_thermal_width",
        [](double E, double xi, double T) {
            return compton::ridge_thermal_width(E, xi, T);
        },
        "E"_a,
        "xi"_a,
        "T"_a,
        "Local thermal width of the Compton ridge in E' [erg]");

    py::class_<compton::RidgeBounds>(m, "RidgeBounds")
        .def_readonly("cold_lo", &compton::RidgeBounds::cold_lo)
        .def_readonly("cold_hi", &compton::RidgeBounds::cold_hi)
        .def_readonly("sigma_lo", &compton::RidgeBounds::sigma_lo)
        .def_readonly("sigma_hi", &compton::RidgeBounds::sigma_hi);

    m.def(
        "compute_ridge_bounds",
        [](double E, double xi_lo, double xi_hi, double T) {
            return compton::compute_ridge_bounds(E, xi_lo, xi_hi, T);
        },
        "E"_a,
        "xi_lo"_a,
        "xi_hi"_a,
        "T"_a,
        "Ridge bounds with cold endpoints and thermal widths [erg]");

    m.def(
        "endpoint_localized_xi",
        &compton::endpoint_localized_xi,
        "gamma"_a,
        "gamma_p"_a,
        "tau"_a,
        "Test whether the endpoint-localized reflected-log condition is "
        "met");

    m.attr("XI_ENDPOINT_EPS") = compton::XI_ENDPOINT_EPS;
    m.attr("XI_CUSP_TAU") = compton::XI_CUSP_TAU;

    // --- Monte Carlo ---
    py::class_<MCIntegrationConfig>(m, "MCIntegrationConfig")
        .def(
            py::init<std::size_t, int, bool>(), // NOLINT(misc-include-cleaner)
            "num_samples"_a = 1'000'000,
            "seed"_a = -1,
            "discard_out_of_grid"_a = true)
        .def_readwrite("num_samples", &MCIntegrationConfig::num_samples)
        .def_readwrite("seed", &MCIntegrationConfig::seed)
        .def_readwrite(
            "discard_out_of_grid",
            &MCIntegrationConfig::discard_out_of_grid);

    py::class_<ComptonMonteCarloKernel>(m, "ComptonMonteCarloKernel")
        .def(
            py::init<
                std::vector<double> const&,
                std::shared_ptr<WeightFunction const>,
                MCIntegrationConfig const&>(),
            "energy_group_boundaries"_a,
            "weight_function"_a,
            "config"_a = MCIntegrationConfig{})

        .def_property_readonly(
            "num_groups",
            &ComptonMonteCarloKernel::num_groups)

        .def_property_readonly(
            "group_centers",
            [](ComptonMonteCarloKernel const& self) {
                auto const& c = self.group_centers();
                py::array_t<double> arr(c.size());
                auto buf = arr.mutable_unchecked<1>();
                for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(c.size());
                     ++i)
                    buf(i) = c[i];
                return arr;
            })

        .def_property_readonly(
            "group_boundaries",
            [](ComptonMonteCarloKernel const& self) {
                auto const& b = self.group_boundaries();
                py::array_t<double> arr(b.size());
                auto buf = arr.mutable_unchecked<1>();
                for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(b.size());
                     ++i)
                    buf(i) = b[i];
                return arr;
            })

        .def(
            "compute_sigma_matrix",
            [](ComptonMonteCarloKernel const& self,
               int num_angle_bins,
               double T,
               double Ne,
               KernelMultiplier const& multiplier) {
                return flat_to_numpy_3d(self.num_groups(), num_angle_bins, [&] {
                    return self.compute_sigma_matrix(
                        num_angle_bins,
                        T,
                        Ne,
                        multiplier);
                });
            },
            "num_angle_bins"_a,
            "T"_a,
            "Ne"_a,
            "multiplier"_a = ConstantMultiplier())

        .def(
            "compute_sigma_matrix",
            [](ComptonMonteCarloKernel const& self,
               double T,
               double Ne,
               KernelMultiplier const& multiplier) {
                return flat_to_numpy_2d(self.num_groups(), [&] {
                    return self.compute_sigma_matrix(T, Ne, multiplier);
                });
            },
            "T"_a,
            "Ne"_a,
            "multiplier"_a = ConstantMultiplier())

        .def(
            "compute_dsigma_dT_matrix",
            [](ComptonMonteCarloKernel const& self,
               int num_angle_bins,
               double T,
               double Ne,
               KernelMultiplier const& multiplier) {
                return flat_to_numpy_3d(self.num_groups(), num_angle_bins, [&] {
                    return self.compute_dsigma_dT_matrix(
                        num_angle_bins,
                        T,
                        Ne,
                        multiplier);
                });
            },
            "num_angle_bins"_a,
            "T"_a,
            "Ne"_a,
            "multiplier"_a = ConstantMultiplier())

        .def(
            "compute_dsigma_dT_matrix",
            [](ComptonMonteCarloKernel const& self,
               double T,
               double Ne,
               KernelMultiplier const& multiplier) {
                return flat_to_numpy_2d(self.num_groups(), [&] {
                    return self.compute_dsigma_dT_matrix(T, Ne, multiplier);
                });
            },
            "T"_a,
            "Ne"_a,
            "multiplier"_a = ConstantMultiplier())

        .def(
            "compute_full_dsigma_dT_matrix",
            [](ComptonMonteCarloKernel const& self,
               int num_angle_bins,
               double T,
               double Ne,
               KernelMultiplier const& multiplier) {
                return flat_to_numpy_3d(self.num_groups(), num_angle_bins, [&] {
                    return self.compute_full_dsigma_dT_matrix(
                        num_angle_bins,
                        T,
                        Ne,
                        multiplier);
                });
            },
            "num_angle_bins"_a,
            "T"_a,
            "Ne"_a,
            "multiplier"_a = ConstantMultiplier())

        .def(
            "compute_full_dsigma_dT_matrix",
            [](ComptonMonteCarloKernel const& self,
               double T,
               double Ne,
               KernelMultiplier const& multiplier) {
                return flat_to_numpy_2d(self.num_groups(), [&] {
                    return self.compute_full_dsigma_dT_matrix(T, Ne, multiplier);
                });
            },
            "T"_a,
            "Ne"_a,
            "multiplier"_a = ConstantMultiplier());
}
