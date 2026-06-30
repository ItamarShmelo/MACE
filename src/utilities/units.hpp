#pragma once
#include <numbers>

// All constants in CGS units
struct units {
    static constexpr double me = 9.109383713928e-28;   // electron mass [g]
    static constexpr double clight = 2.99792458000e10; // speed of light [cm/s]
    static constexpr double me_c2 =
        me * clight * clight; // electron rest energy [erg]

    static constexpr double k_boltz =
        1.380649e-16; // Boltzmann constant [erg/K]

    static constexpr double sigma_sb =
        5.670374419e-5; // Stefan-Boltzmann [erg cm^-2 s^-1 K^-4]
    static constexpr double arad =
        4.0 * sigma_sb / clight; // radiation constant [erg cm^-3 K^-4]

    static constexpr double sigma_thomson =
        6.652458732160e-25; // Thomson cross section [cm^2]

    static constexpr double ev = 1.602176634e-12;         // 1 eV [erg]
    static constexpr double kev = 1e3 * ev;               // 1 keV [erg]
    static constexpr double ev_kelvin = ev / k_boltz;     // 1 eV in Kelvin
    static constexpr double kev_kelvin = ev_kelvin * 1e3; // 1 keV in Kelvin

    static constexpr double Navogadro = 6.02214076e23;
    static constexpr double planck_constant =
        6.62607015e-27; // Planck constant [erg s]

    static constexpr double r_e2 =
        sigma_thomson / (8.0 * std::numbers::pi /
                         3.0); // classical electron radius squared [cm^2]

    static constexpr double barn = 1e-24;        // barn [cm^2]
    static constexpr double mbarn = 1e-3 * barn; // millibarn [cm^2]
};
