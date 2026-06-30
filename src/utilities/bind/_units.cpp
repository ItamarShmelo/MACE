#include "utilities/units.hpp"
#include <pybind11/pybind11.h>

PYBIND11_MODULE(_units, m)
{
    m.doc() = "Physical constants in CGS units";

    m.attr("me") = units::me;
    m.attr("clight") = units::clight;
    m.attr("me_c2") = units::me_c2;

    m.attr("k_boltz") = units::k_boltz;

    m.attr("sigma_sb") = units::sigma_sb;
    m.attr("arad") = units::arad;

    m.attr("sigma_thomson") = units::sigma_thomson;

    m.attr("ev") = units::ev;
    m.attr("kev") = units::kev;
    m.attr("ev_kelvin") = units::ev_kelvin;
    m.attr("kev_kelvin") = units::kev_kelvin;

    m.attr("Navogadro") = units::Navogadro;
    m.attr("planck_constant") = units::planck_constant;

    m.attr("r_e2") = units::r_e2;

    m.attr("barn") = units::barn;
    m.attr("mbarn") = units::mbarn;
}
