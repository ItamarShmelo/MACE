#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "utilities/bind_helpers.hpp"
#include "compton_common/compton_common.hpp"
#include "compton_multigroup/compton_multigroup_deterministic/compton_multigroup_deterministic.hpp"
#include "compton_multigroup/compton_multigroup_monte_carlo/compton_multigroup_monte_carlo.hpp"
#include "utilities/gauss_legendre.hpp"
#include "compton_multigroup/weight_function.hpp"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace compton;
using compton::bind::flat_to_numpy_2d;
using compton::bind::flat_to_numpy_3d;

PYBIND11_MODULE(_compton_multigroup, m) {
    m.doc() = "Weighted multigroup-multiangle Compton scattering matrix";

    py::module_::import("_compton_common");
    py::module_::import("_compton_differential_cross_section");

    py::class_<KernelMultiplier>(m, "KernelMultiplier");

    py::class_<ConstantMultiplier, KernelMultiplier>(m, "ConstantMultiplier")
        .def(py::init<>());

    py::enum_<FlatEpDensityMode>(m, "FlatEpDensityMode")
        .value("log_proportional", FlatEpDensityMode::log_proportional)
        .value("linear_proportional", FlatEpDensityMode::linear_proportional)
        .value("points_per_decade", FlatEpDensityMode::points_per_decade);

    py::class_<FlatEpConfig>(m, "FlatEpConfig")
        .def(py::init<double, int, int, FlatEpDensityMode, bool, bool>(),
             "density"_a = 64.0,
             "min_points"_a = 8,
             "max_points"_a = 1024,
             "mode"_a = FlatEpDensityMode::points_per_decade,
             "flat_E"_a = true,
             "flat_mu"_a = true)
        .def_readwrite("density",    &FlatEpConfig::density)
        .def_readwrite("min_points", &FlatEpConfig::min_points)
        .def_readwrite("max_points", &FlatEpConfig::max_points)
        .def_readwrite("mode",       &FlatEpConfig::mode)
        .def_readwrite("flat_E",     &FlatEpConfig::flat_E)
        .def_readwrite("flat_mu",    &FlatEpConfig::flat_mu);

    py::class_<MGIntegrationConfig>(m, "MGIntegrationConfig")
        .def(py::init<int, double, double, int, int,
                       std::optional<int>, std::optional<int>,
                       std::optional<int>, double,
                       std::optional<FlatEpConfig>>(),
             "base_order"_a = 24,
             "integration_tolerance"_a = 1e-3,
             "cutoff_ratio"_a = 1e-8,
             "peak_max_depth"_a = 5,
             "cold_temperature_order"_a = 48,
             "tail_order"_a = std::nullopt,
             "far_order"_a = std::nullopt,
             "mu_order"_a = std::nullopt,
             "mu_peak_k"_a = 10.0,
             "flat_ep"_a = std::nullopt)
        .def_readwrite("base_order",              &MGIntegrationConfig::base_order)
        .def_readwrite("cold_temperature_order",  &MGIntegrationConfig::cold_temperature_order)
        .def_readwrite("peak_max_depth",          &MGIntegrationConfig::peak_max_depth)
        .def_readwrite("tail_order",              &MGIntegrationConfig::tail_order)
        .def_readwrite("far_order",               &MGIntegrationConfig::far_order)
        .def_readwrite("mu_order",                &MGIntegrationConfig::mu_order)
        .def_readwrite("integration_tolerance",   &MGIntegrationConfig::integration_tolerance)
        .def_readwrite("cutoff_ratio",            &MGIntegrationConfig::cutoff_ratio)
        .def_readwrite("mu_peak_k",              &MGIntegrationConfig::mu_peak_k)
        .def_readwrite("flat_ep",                 &MGIntegrationConfig::flat_ep)
        .def("effective_tail_order", &MGIntegrationConfig::effective_tail_order)
        .def("effective_far_order",  &MGIntegrationConfig::effective_far_order)
        .def("effective_mu_order",   &MGIntegrationConfig::effective_mu_order)
        .def_static("cold_adaptive", &MGIntegrationConfig::cold_adaptive,
            "High-accuracy adaptive config for T < 0.1 keV (bo=192, pd=9, mu=512)")
        .def_static("warm_flat", &MGIntegrationConfig::warm_flat,
            "High-accuracy flat E' config for T >= 0.1 keV (bo=96, d=512, mu=96)");

    py::class_<ComptonMultigroupKernel>(m, "ComptonMultigroupKernel")
        .def(py::init<std::vector<double> const&,
                       std::shared_ptr<WeightFunction const>,
                       MGIntegrationConfig const&>(),
             "energy_group_boundaries"_a,
             "weight_function"_a,
             "config"_a = MGIntegrationConfig{})

        .def_property_readonly("num_groups", &ComptonMultigroupKernel::num_groups)

        .def_property_readonly("group_centers", [](ComptonMultigroupKernel const& self) {
            auto const& c = self.group_centers();
            py::array_t<double> arr(c.size());
            auto buf = arr.mutable_unchecked<1>();
            for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(c.size()); ++i)
                buf(i) = c[i];
            return arr;
        })

        .def_property_readonly("group_boundaries", [](ComptonMultigroupKernel const& self) {
            auto const& b = self.group_boundaries();
            py::array_t<double> arr(b.size());
            auto buf = arr.mutable_unchecked<1>();
            for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(b.size()); ++i)
                buf(i) = b[i];
            return arr;
        })

        .def("compute_sigma_matrix",
            [](ComptonMultigroupKernel const& self,
               ComptonKernelSolver const& kernel,
               int num_angle_bins, double T, double Ne,
               KernelMultiplier const& multiplier) {
                return flat_to_numpy_3d(self.num_groups(), num_angle_bins, [&] {
                    return self.compute_sigma_matrix(kernel, num_angle_bins, T, Ne, multiplier);
                });
            },
            "kernel"_a, "num_angle_bins"_a, "T"_a, "Ne"_a,
            "multiplier"_a = ConstantMultiplier())

        .def("compute_dsigma_dT_matrix",
            [](ComptonMultigroupKernel const& self,
               ComptonKernelSolver const& kernel,
               int num_angle_bins, double T, double Ne,
               KernelMultiplier const& multiplier) {
                return flat_to_numpy_3d(self.num_groups(), num_angle_bins, [&] {
                    return self.compute_dsigma_dT_matrix(kernel, num_angle_bins, T, Ne, multiplier);
                });
            },
            "kernel"_a, "num_angle_bins"_a, "T"_a, "Ne"_a,
            "multiplier"_a = ConstantMultiplier())

        .def("compute_sigma_matrix",
            [](ComptonMultigroupKernel const& self,
               ComptonKernelSolver const& kernel,
               double T, double Ne,
               KernelMultiplier const& multiplier) {
                return flat_to_numpy_2d(self.num_groups(), [&] {
                    return self.compute_sigma_matrix(kernel, 1, T, Ne, multiplier);
                });
            },
            "kernel"_a, "T"_a, "Ne"_a,
            "multiplier"_a = ConstantMultiplier())

        .def("compute_dsigma_dT_matrix",
            [](ComptonMultigroupKernel const& self,
               ComptonKernelSolver const& kernel,
               double T, double Ne,
               KernelMultiplier const& multiplier) {
                return flat_to_numpy_2d(self.num_groups(), [&] {
                    return self.compute_dsigma_dT_matrix(kernel, 1, T, Ne, multiplier);
                });
            },
            "kernel"_a, "T"_a, "Ne"_a,
            "multiplier"_a = ConstantMultiplier());

    py::class_<WeightFunction, std::shared_ptr<WeightFunction>>(m, "WeightFunction");

    py::class_<PlanckWeightFunction, WeightFunction,
                std::shared_ptr<PlanckWeightFunction>>(m, "PlanckWeightFunction")
        .def(py::init<double>(), py::kw_only(), "cap_x"_a)
        .def("weight", &PlanckWeightFunction::weight, "E"_a, "T"_a)
        .def("compute_denominator", &PlanckWeightFunction::compute_denominator,
             "E_left"_a, "E_right"_a, "T"_a);

    py::class_<UniformWeightFunction, WeightFunction,
                std::shared_ptr<UniformWeightFunction>>(m, "UniformWeightFunction")
        .def(py::init<>())
        .def("weight", &UniformWeightFunction::weight, "E"_a, "T"_a)
        .def("compute_denominator", &UniformWeightFunction::compute_denominator,
             "E_left"_a, "E_right"_a, "T"_a);

    py::class_<WienWeightFunction, WeightFunction,
                std::shared_ptr<WienWeightFunction>>(m, "WienWeightFunction")
        .def(py::init<double>(), py::kw_only(), "cap_x"_a)
        .def("weight", &WienWeightFunction::weight, "E"_a, "T"_a)
        .def("compute_denominator", &WienWeightFunction::compute_denominator,
             "E_left"_a, "E_right"_a, "T"_a);

    m.def("gauss_legendre_rule", [](int N) {
        auto rule = compton::compute_gauss_legendre(N);
        py::array_t<double> nodes(rule.nodes.size());
        py::array_t<double> weights(rule.weights.size());
        auto nodes_buf = nodes.mutable_unchecked<1>();
        auto weights_buf = weights.mutable_unchecked<1>();
        for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(rule.nodes.size()); ++i) {
            nodes_buf(i) = rule.nodes[i];
            weights_buf(i) = rule.weights[i];
        }
        return py::make_tuple(nodes, weights);
    }, "N"_a, "Compute N-point Gauss-Legendre nodes and weights on [-1, 1]");

    m.def("adaptive_legendre_integrate", [](py::function integrand,
                                            int base_order,
                                            double a, double b,
                                            double tol, int max_depth) {
        auto rule = compton::compute_gauss_legendre(base_order);
        return compton::adaptive_legendre_integrate(
            [&](double x) { return integrand(x).cast<double>(); },
            rule, a, b, tol, max_depth);
    }, "integrand"_a, "base_order"_a, "a"_a, "b"_a,
       "tol"_a = 1e-8, "max_depth"_a = 15,
       "Adaptive Gauss-Legendre integration of f over [a, b]");

    m.def("adaptive_log_legendre_integrate", [](py::function integrand,
                                                int base_order,
                                                double a, double b,
                                                double tol, int max_depth) {
        auto rule = compton::compute_gauss_legendre(base_order);
        return compton::adaptive_log_legendre_integrate(
            [&](double x) { return integrand(x).cast<double>(); },
            rule, a, b, tol, max_depth);
    }, "integrand"_a, "base_order"_a, "a"_a, "b"_a,
       "tol"_a = 1e-8, "max_depth"_a = 15,
       "Adaptive log-space GL integration of f over [a, b] (clusters nodes near a)");

    m.def("adaptive_rlog_legendre_integrate", [](py::function integrand,
                                                 int base_order,
                                                 double a, double b,
                                                 double tol, int max_depth) {
        auto rule = compton::compute_gauss_legendre(base_order);
        return compton::adaptive_rlog_legendre_integrate(
            [&](double x) { return integrand(x).cast<double>(); },
            rule, a, b, tol, max_depth);
    }, "integrand"_a, "base_order"_a, "a"_a, "b"_a,
       "tol"_a = 1e-8, "max_depth"_a = 15,
       "Adaptive reflected-log GL integration of f over [a, b] (clusters nodes near b)");

    m.def("peak_limits", [](double E, double mu_lo, double mu_hi, double T) {
        return compton::peak_limits(E, mu_lo, mu_hi, T);
    }, "E"_a, "mu_lo"_a, "mu_hi"_a, "T"_a,
       "Thermally broadened peak E' limits [erg]: returns (lo, hi)");

    m.def("cold_recoil_lo", [](double E, double mu_lo) {
        return compton::peak_limits(E, mu_lo, 1.0, 0.0).first;
    }, "E"_a, "mu_lo"_a,
       "Lower edge of the cold Compton recoil band [erg] (T=0 limit)");

    m.def("cold_recoil_hi", [](double E, double mu_hi) {
        return compton::peak_limits(E, -1.0, mu_hi, 0.0).second;
    }, "E"_a, "mu_hi"_a,
       "Upper edge of the cold Compton recoil band [erg] (T=0 limit)");

    // --- Monte Carlo ---
    py::class_<MCIntegrationConfig>(m, "MCIntegrationConfig")
        .def(py::init<std::size_t, int, bool>(),
             "num_samples"_a = 1'000'000,
             "seed"_a = -1,
             "discard_out_of_grid"_a = true)
        .def_readwrite("num_samples",        &MCIntegrationConfig::num_samples)
        .def_readwrite("seed",               &MCIntegrationConfig::seed)
        .def_readwrite("discard_out_of_grid", &MCIntegrationConfig::discard_out_of_grid);

    py::class_<ComptonMonteCarloKernel>(m, "ComptonMonteCarloKernel")
        .def(py::init<std::vector<double> const&,
                       std::shared_ptr<WeightFunction const>,
                       MCIntegrationConfig const&>(),
             "energy_group_boundaries"_a,
             "weight_function"_a,
             "config"_a = MCIntegrationConfig{})

        .def_property_readonly("num_groups",
            &ComptonMonteCarloKernel::num_groups)

        .def_property_readonly("group_centers",
            [](ComptonMonteCarloKernel const& self) {
                auto const& c = self.group_centers();
                py::array_t<double> arr(c.size());
                auto buf = arr.mutable_unchecked<1>();
                for (py::ssize_t i = 0;
                     i < static_cast<py::ssize_t>(c.size()); ++i)
                    buf(i) = c[i];
                return arr;
            })

        .def_property_readonly("group_boundaries",
            [](ComptonMonteCarloKernel const& self) {
                auto const& b = self.group_boundaries();
                py::array_t<double> arr(b.size());
                auto buf = arr.mutable_unchecked<1>();
                for (py::ssize_t i = 0;
                     i < static_cast<py::ssize_t>(b.size()); ++i)
                    buf(i) = b[i];
                return arr;
            })

        .def("compute_sigma_matrix",
            [](ComptonMonteCarloKernel const& self,
               int num_angle_bins, double T, double Ne,
               KernelMultiplier const& multiplier) {
                return flat_to_numpy_3d(self.num_groups(), num_angle_bins, [&] {
                    return self.compute_sigma_matrix(num_angle_bins, T, Ne, multiplier);
                });
            },
            "num_angle_bins"_a, "T"_a, "Ne"_a,
            "multiplier"_a = ConstantMultiplier())

        .def("compute_sigma_matrix",
            [](ComptonMonteCarloKernel const& self,
               double T, double Ne,
               KernelMultiplier const& multiplier) {
                return flat_to_numpy_2d(self.num_groups(), [&] {
                    return self.compute_sigma_matrix(T, Ne, multiplier);
                });
            },
            "T"_a, "Ne"_a,
            "multiplier"_a = ConstantMultiplier())

        .def("compute_dsigma_dT_matrix",
            [](ComptonMonteCarloKernel const& self,
               int num_angle_bins, double T, double Ne,
               KernelMultiplier const& multiplier) {
                return flat_to_numpy_3d(self.num_groups(), num_angle_bins, [&] {
                    return self.compute_dsigma_dT_matrix(num_angle_bins, T, Ne, multiplier);
                });
            },
            "num_angle_bins"_a, "T"_a, "Ne"_a,
            "multiplier"_a = ConstantMultiplier())

        .def("compute_dsigma_dT_matrix",
            [](ComptonMonteCarloKernel const& self,
               double T, double Ne,
               KernelMultiplier const& multiplier) {
                return flat_to_numpy_2d(self.num_groups(), [&] {
                    return self.compute_dsigma_dT_matrix(T, Ne, multiplier);
                });
            },
            "T"_a, "Ne"_a,
            "multiplier"_a = ConstantMultiplier());
}
