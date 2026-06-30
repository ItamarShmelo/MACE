#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "utilities/gauss_laguerre.hpp"
#include "utilities/gauss_legendre.hpp"

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(_utilities, m) // NOLINT(misc-include-cleaner)
{
    m.doc() = "Quadrature utilities: Gauss-Legendre and Gauss-Laguerre rules";

    m.def(
        "gauss_legendre_rule",
        [](int N) {
            auto rule = compton::compute_gauss_legendre(N);
            py::array_t<double> nodes(static_cast<py::ssize_t>(rule.nodes.size())); // NOLINT(misc-include-cleaner)
            py::array_t<double> weights(static_cast<py::ssize_t>(rule.weights.size()));
            auto nodes_buf = nodes.mutable_unchecked<1>();
            auto weights_buf = weights.mutable_unchecked<1>();
            for (py::ssize_t i = 0;
                 i < static_cast<py::ssize_t>(rule.nodes.size());
                 ++i) {
                nodes_buf(i) = rule.nodes[i];
                weights_buf(i) = rule.weights[i];
            }
            return py::make_tuple(nodes, weights); // NOLINT(misc-include-cleaner)
        },
        "N"_a, // NOLINT(misc-include-cleaner)
        "Compute N-point Gauss-Legendre nodes and weights on [-1, 1]");

    m.def(
        "gauss_laguerre_rule",
        [](int N) {
            auto rule = compton::compute_gauss_laguerre(N);
            py::array_t<double> nodes(static_cast<py::ssize_t>(rule.nodes.size()));
            py::array_t<double> weights(static_cast<py::ssize_t>(rule.weights.size()));
            auto nodes_buf = nodes.mutable_unchecked<1>();
            auto weights_buf = weights.mutable_unchecked<1>();
            for (py::ssize_t i = 0;
                 i < static_cast<py::ssize_t>(rule.nodes.size());
                 ++i) {
                nodes_buf(i) = rule.nodes[i];
                weights_buf(i) = rule.weights[i];
            }
            return py::make_tuple(nodes, weights);
        },
        "N"_a,
        "Compute N-point Gauss-Laguerre nodes and weights");
}
