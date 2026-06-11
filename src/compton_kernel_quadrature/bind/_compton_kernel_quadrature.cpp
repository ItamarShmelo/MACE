#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "compton_common/bind_helpers.hpp"
#include "compton_kernel_quadrature/compton_kernel_quadrature.hpp"
#include "compton_kernel_quadrature/gauss_laguerre.hpp"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace compton;

PYBIND11_MODULE(_compton_kernel_quadrature, m) {
    m.doc() = "Compton scattering kernel via direct Gauss-Laguerre quadrature";

    py::module_::import("_compton_common");

    py::enum_<QuadratureForm>(m, "QuadratureForm")
        .value("PostIBP", QuadratureForm::PostIntegrationByParts)
        .value("PreIBP", QuadratureForm::PreIntegrationByParts);

    py::class_<ComptonKernelQuadrature>(m, "ComptonKernelQuadrature")
        .def(py::init<int, QuadratureForm>(),
             "NL"_a = 64,
             "form"_a = QuadratureForm::PostIntegrationByParts)
        .def("sigma_E", &ComptonKernelQuadrature::sigma_E,
             "E"_a, "E_prime"_a, "xi"_a,
             "T"_a, "Ne"_a)
        .def("sigma_E_vec", [](ComptonKernelQuadrature const& self,
                               double E,
                               py::array_t<double, py::array::c_style | py::array::forcecast> E_prime_arr,
                               double xi, double T, double Ne) {
            return compton::bind::vectorize_sigma(
                self, E, E_prime_arr, xi, T, Ne,
                &ComptonKernelQuadrature::sigma_E);
        }, "E"_a, "E_prime_arr"_a, "xi"_a,
           "T"_a, "Ne"_a)
        .def("dsigma_E_dT", &ComptonKernelQuadrature::dsigma_E_dT,
             "E"_a, "E_prime"_a, "xi"_a,
             "T"_a, "Ne"_a)
        .def("dsigma_E_dT_vec", [](ComptonKernelQuadrature const& self,
                                    double E,
                                    py::array_t<double, py::array::c_style | py::array::forcecast> E_prime_arr,
                                    double xi, double T, double Ne) {
            return compton::bind::vectorize_sigma(
                self, E, E_prime_arr, xi, T, Ne,
                &ComptonKernelQuadrature::dsigma_E_dT);
        }, "E"_a, "E_prime_arr"_a, "xi"_a,
           "T"_a, "Ne"_a);

    m.def("scaled_K2", &scaled_K2, "x"_a,
          "Returns kve(2, x) = exp(x) * K_2(x)");

    m.def("scaled_K1", &scaled_K1, "x"_a,
          "Returns kve(1, x) = exp(x) * K_1(x)");

    m.def("kappa_ratio", &kappa_ratio, "tau"_a,
          "Returns K_1(1/tau) / K_2(1/tau)");

    m.def("gauss_laguerre_rule", [](int N) {
        auto rule = compton::compute_gauss_laguerre(N);

        py::array_t<double> nodes(rule.nodes.size());
        py::array_t<double> weights(rule.weights.size());

        auto nodes_buf = nodes.mutable_unchecked<1>();
        auto weights_buf = weights.mutable_unchecked<1>();

        for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(rule.nodes.size()); ++i) {
            nodes_buf(i) = rule.nodes[i];
            weights_buf(i) = rule.weights[i];
        }

        return py::make_tuple(nodes, weights);
    }, "N"_a, "Compute N-point Gauss-Laguerre nodes and weights");
}
