#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "compton_multigroup/compton_multigroup.hpp"
#include "compton_common/compton_common.hpp"
#include "compton_multigroup/gauss_legendre.hpp"
#include "compton_multigroup/weight_function.hpp"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace compton;

namespace {

template<typename KernelT>
using MatrixMethod = std::vector<double> (ComptonMultigroupKernel::*)(
    KernelT const&, int, double, double, KernelMultiplier const&) const;

template<typename KernelT>
py::array_t<double> wrap_3d(
    ComptonMultigroupKernel const& self,
    KernelT const& kernel,
    int const num_angle_bins,
    double const T,
    double const Ne,
    KernelMultiplier const& multiplier,
    MatrixMethod<KernelT> method)
{
    std::vector<double> flat = (self.*method)(kernel, num_angle_bins, T, Ne, multiplier);
    int const G = self.num_groups();
    py::array_t<double> arr({G, G, num_angle_bins});
    auto buf = arr.mutable_unchecked<3>();
    std::size_t idx = 0;
    for (int g = 0; g < G; ++g)
        for (int gp = 0; gp < G; ++gp)
            for (int a = 0; a < num_angle_bins; ++a)
                buf(g, gp, a) = flat[idx++];
    return arr;
}

template<typename KernelT>
py::array_t<double> wrap_2d(
    ComptonMultigroupKernel const& self,
    KernelT const& kernel,
    double const T,
    double const Ne,
    KernelMultiplier const& multiplier,
    MatrixMethod<KernelT> method)
{
    std::vector<double> flat = (self.*method)(kernel, 1, T, Ne, multiplier);
    int const G = self.num_groups();
    py::array_t<double> arr({G, G});
    auto buf = arr.mutable_unchecked<2>();
    std::size_t idx = 0;
    for (int g = 0; g < G; ++g)
        for (int gp = 0; gp < G; ++gp)
            buf(g, gp) = flat[idx++];
    return arr;
}

} // anonymous namespace

PYBIND11_MODULE(_compton_multigroup, m) {
    m.doc() = "Weighted multigroup-multiangle Compton scattering matrix";

    py::module_::import("_compton_common");
    py::module_::import("_compton_kernel_solver");

    py::class_<KernelMultiplier>(m, "KernelMultiplier");

    py::class_<ConstantMultiplier, KernelMultiplier>(m, "ConstantMultiplier")
        .def(py::init<>());

    py::class_<MGIntegrationConfig>(m, "MGIntegrationConfig")
        .def(py::init<int, double, double, int,
                       std::optional<int>, std::optional<int>>(),
             "base_order"_a = 24,
             "integration_tolerance"_a = 1e-3,
             "cutoff_ratio"_a = 1e-8,
             "peak_max_depth"_a = 5,
             "tail_order"_a = std::nullopt,
             "far_order"_a = std::nullopt)
        .def_readwrite("base_order",            &MGIntegrationConfig::base_order)
        .def_readwrite("peak_max_depth",        &MGIntegrationConfig::peak_max_depth)
        .def_readwrite("tail_order",            &MGIntegrationConfig::tail_order)
        .def_readwrite("far_order",             &MGIntegrationConfig::far_order)
        .def_readwrite("integration_tolerance", &MGIntegrationConfig::integration_tolerance)
        .def_readwrite("cutoff_ratio",          &MGIntegrationConfig::cutoff_ratio)
        .def("effective_tail_order", &MGIntegrationConfig::effective_tail_order)
        .def("effective_far_order",  &MGIntegrationConfig::effective_far_order);

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

        // ── Multiangle: solver kernel ─────────────────────────────────────
        .def("compute_sigma_matrix",
            [](ComptonMultigroupKernel const& self,
               ComptonKernelSolver const& kernel,
               int num_angle_bins, double T, double Ne,
               KernelMultiplier const& multiplier) {
                return wrap_3d(self, kernel, num_angle_bins, T, Ne, multiplier,
                    &ComptonMultigroupKernel::compute_sigma_matrix<ComptonKernelSolver>);
            },
            "kernel"_a, "num_angle_bins"_a, "T"_a, "Ne"_a,
            "multiplier"_a = ConstantMultiplier())

        .def("compute_dsigma_dT_matrix",
            [](ComptonMultigroupKernel const& self,
               ComptonKernelSolver const& kernel,
               int num_angle_bins, double T, double Ne,
               KernelMultiplier const& multiplier) {
                return wrap_3d(self, kernel, num_angle_bins, T, Ne, multiplier,
                    &ComptonMultigroupKernel::compute_dsigma_dT_matrix<ComptonKernelSolver>);
            },
            "kernel"_a, "num_angle_bins"_a, "T"_a, "Ne"_a,
            "multiplier"_a = ConstantMultiplier())

        // ── Angle-integrated: solver kernel ──────────────────────────────
        .def("compute_sigma_matrix",
            [](ComptonMultigroupKernel const& self,
               ComptonKernelSolver const& kernel,
               double T, double Ne,
               KernelMultiplier const& multiplier) {
                return wrap_2d(self, kernel, T, Ne, multiplier,
                    &ComptonMultigroupKernel::compute_sigma_matrix<ComptonKernelSolver>);
            },
            "kernel"_a, "T"_a, "Ne"_a,
            "multiplier"_a = ConstantMultiplier())

        .def("compute_dsigma_dT_matrix",
            [](ComptonMultigroupKernel const& self,
               ComptonKernelSolver const& kernel,
               double T, double Ne,
               KernelMultiplier const& multiplier) {
                return wrap_2d(self, kernel, T, Ne, multiplier,
                    &ComptonMultigroupKernel::compute_dsigma_dT_matrix<ComptonKernelSolver>);
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
}
