#include <pybind11/pybind11.h>

#include "compton_common/compton_common.hpp"

namespace py = pybind11;
using namespace compton;

PYBIND11_MODULE(_compton_common, m) {
    m.doc() = "Shared types for Compton scattering kernel modules";

    py::class_<SigmaResult>(m, "SigmaResult")
        .def_readonly("value", &SigmaResult::value)
        .def_readonly("estimated_abs_error", &SigmaResult::estimated_abs_error)
        .def_readonly("estimated_rel_error", &SigmaResult::estimated_rel_error);
}
