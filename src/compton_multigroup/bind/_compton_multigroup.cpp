#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "compton_multigroup/compton_multigroup.hpp"
#include "compton_multigroup/gauss_legendre.hpp"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace compton;

namespace {

template<typename KernelT>
py::array_t<double> wrap_3d(
    ComptonMultigroupKernel const& self,
    KernelT const& kernel,
    int const num_angle_bins,
    double const T,
    double const Ne,
    std::vector<double> (ComptonMultigroupKernel::*method)(
        KernelT const&, int, double, double) const)
{
    std::vector<double> flat = (self.*method)(kernel, num_angle_bins, T, Ne);
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
    std::vector<double> (ComptonMultigroupKernel::*method)(
        KernelT const&, int, double, double) const)
{
    std::vector<double> flat = (self.*method)(kernel, 1, T, Ne);
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
    m.doc() = "Planck-weighted multigroup-multiangle Compton scattering matrix";

    py::module_::import("_compton_common");

    py::class_<ComptonMultigroupKernel>(m, "ComptonMultigroupKernel")
        .def(py::init<std::vector<double> const&, int, int, int, double>(),
             "energy_group_boundaries"_a,
             "quad_order_E"_a  = 8,
             "quad_order_Ep"_a = 8,
             "quad_order_mu"_a = 8,
             "planck_cap_x"_a  = 25.0)

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

        .def("compute_denominator", &ComptonMultigroupKernel::compute_denominator,
             "g"_a, "T"_a)

        // ── Multiangle: quadrature kernel ────────────────────────────────
        .def("compute_sigma_matrix",
            [](ComptonMultigroupKernel const& self,
               ComptonKernelQuadrature const& kernel,
               int num_angle_bins, double T, double Ne) {
                return wrap_3d(self, kernel, num_angle_bins, T, Ne,
                    &ComptonMultigroupKernel::compute_sigma_matrix<ComptonKernelQuadrature>);
            },
            "kernel"_a, "num_angle_bins"_a, "T"_a, "Ne"_a)

        .def("compute_dsigma_dT_matrix",
            [](ComptonMultigroupKernel const& self,
               ComptonKernelQuadrature const& kernel,
               int num_angle_bins, double T, double Ne) {
                return wrap_3d(self, kernel, num_angle_bins, T, Ne,
                    &ComptonMultigroupKernel::compute_dsigma_dT_matrix<ComptonKernelQuadrature>);
            },
            "kernel"_a, "num_angle_bins"_a, "T"_a, "Ne"_a)

        // ── Multiangle: series kernel ────────────────────────────────────
        .def("compute_sigma_matrix",
            [](ComptonMultigroupKernel const& self,
               ComptonKernelSeries const& kernel,
               int num_angle_bins, double T, double Ne) {
                return wrap_3d(self, kernel, num_angle_bins, T, Ne,
                    &ComptonMultigroupKernel::compute_sigma_matrix<ComptonKernelSeries>);
            },
            "kernel"_a, "num_angle_bins"_a, "T"_a, "Ne"_a)

        .def("compute_dsigma_dT_matrix",
            [](ComptonMultigroupKernel const& self,
               ComptonKernelSeries const& kernel,
               int num_angle_bins, double T, double Ne) {
                return wrap_3d(self, kernel, num_angle_bins, T, Ne,
                    &ComptonMultigroupKernel::compute_dsigma_dT_matrix<ComptonKernelSeries>);
            },
            "kernel"_a, "num_angle_bins"_a, "T"_a, "Ne"_a)

        // ── Angle-integrated: quadrature kernel ──────────────────────────
        .def("compute_sigma_matrix",
            [](ComptonMultigroupKernel const& self,
               ComptonKernelQuadrature const& kernel,
               double T, double Ne) {
                return wrap_2d(self, kernel, T, Ne,
                    &ComptonMultigroupKernel::compute_sigma_matrix<ComptonKernelQuadrature>);
            },
            "kernel"_a, "T"_a, "Ne"_a)

        .def("compute_dsigma_dT_matrix",
            [](ComptonMultigroupKernel const& self,
               ComptonKernelQuadrature const& kernel,
               double T, double Ne) {
                return wrap_2d(self, kernel, T, Ne,
                    &ComptonMultigroupKernel::compute_dsigma_dT_matrix<ComptonKernelQuadrature>);
            },
            "kernel"_a, "T"_a, "Ne"_a)

        // ── Angle-integrated: series kernel ──────────────────────────────
        .def("compute_sigma_matrix",
            [](ComptonMultigroupKernel const& self,
               ComptonKernelSeries const& kernel,
               double T, double Ne) {
                return wrap_2d(self, kernel, T, Ne,
                    &ComptonMultigroupKernel::compute_sigma_matrix<ComptonKernelSeries>);
            },
            "kernel"_a, "T"_a, "Ne"_a)

        .def("compute_dsigma_dT_matrix",
            [](ComptonMultigroupKernel const& self,
               ComptonKernelSeries const& kernel,
               double T, double Ne) {
                return wrap_2d(self, kernel, T, Ne,
                    &ComptonMultigroupKernel::compute_dsigma_dT_matrix<ComptonKernelSeries>);
            },
            "kernel"_a, "T"_a, "Ne"_a);

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
}
