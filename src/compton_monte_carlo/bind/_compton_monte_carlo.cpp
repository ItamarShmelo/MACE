#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "compton_common/bind_helpers.hpp"
#include "compton_monte_carlo/compton_monte_carlo.hpp"

using namespace pybind11::literals;
using namespace compton;
using compton::bind::flat_to_numpy_2d;
using compton::bind::flat_to_numpy_3d;

PYBIND11_MODULE(_compton_monte_carlo, m) {
    m.doc() = "Monte Carlo multigroup-multiangle Compton scattering matrix";

    py::module_::import("_compton_multigroup");

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
