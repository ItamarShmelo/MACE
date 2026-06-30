#include <pybind11/pybind11.h>

#include "compton_common/compton_common.hpp"

namespace py = pybind11;
using namespace compton;

PYBIND11_MODULE(_compton_common, m)
{
    m.doc() = "Shared types for Compton scattering kernel modules";

    py::class_<ComptonResult>(m, "ComptonResult")
        .def_readonly("value", &ComptonResult::value)
        .def_readonly("terms_used", &ComptonResult::terms_used)
        .def_readonly(
            "estimated_abs_error",
            &ComptonResult::estimated_abs_error)
        .def_readonly(
            "estimated_rel_error",
            &ComptonResult::estimated_rel_error);
}
