#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // NOLINT(misc-include-cleaner) -- implicit STL converters

#include "kernel_multipliers/kernel_multipliers.hpp"

#include <vector>

namespace py = pybind11;
using namespace pybind11::literals;
using namespace compton;

PYBIND11_MODULE(_compton_kernel_multipliers, m) // NOLINT(misc-include-cleaner)
{
    m.doc() = "Concrete kernel multipliers for multigroup Compton integrals";

    py::module_::import("compton_matrix._compton_multigroup");

    py::class_<EnergyTransferMultiplier, KernelMultiplier>(
        m,
        "EnergyTransferMultiplier")
        .def(
            py::init<std::vector<double>, std::vector<double>>(),
            "energy_group_boundaries"_a, // NOLINT(misc-include-cleaner)
            "energy_group_centers"_a);

    py::class_<NumberWeightedEnergyTransferMultiplier, KernelMultiplier>(
        m,
        "NumberWeightedEnergyTransferMultiplier")
        .def(
            py::init<std::vector<double>, std::vector<double>>(),
            "energy_group_boundaries"_a,
            "energy_group_centers"_a);

    py::class_<NumberWeightMultiplier, KernelMultiplier>(
        m,
        "NumberWeightMultiplier")
        .def(
            py::init<std::vector<double>, std::vector<double>>(),
            "energy_group_boundaries"_a,
            "energy_group_centers"_a);

    py::class_<EpMultiplier, KernelMultiplier>(
        m,
        "EpMultiplier")
        .def(py::init<>());

    py::class_<EpOverEMultiplier, KernelMultiplier>(
        m,
        "EpOverEMultiplier")
        .def(py::init<>());

    py::class_<EpOverETimesXiMultiplier, KernelMultiplier>(
        m,
        "EpOverETimesXiMultiplier")
        .def(py::init<>());

    py::class_<EMinusEpMultiplier, KernelMultiplier>(
        m,
        "EMinusEpMultiplier")
        .def(py::init<>());

    py::class_<EnergyMomentMultiplier, KernelMultiplier>(
        m,
        "EnergyMomentMultiplier")
        .def(py::init<int>(), "n"_a);

    py::class_<LegendreMultiplier0123, KernelMultiplier>(
        m,
        "LegendreMultiplier0123")
        .def(py::init<int>(), "n"_a);
}
