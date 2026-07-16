#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // NOLINT(misc-include-cleaner) -- implicit STL converters

#include "compton_multigroup/compton_multigroup_deterministic/compton_multigroup_deterministic.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <utility>
#include <vector>

namespace py = pybind11;
using namespace pybind11::literals;
using namespace compton;

class EnergyTransferMultiplier : public KernelMultiplier {
  public:
    EnergyTransferMultiplier(
        std::vector<double> boundaries,
        std::vector<double> centers)
        : boundaries_(std::move(boundaries)),
          centers_(std::move(centers))
    {
        if (boundaries_.size() != centers_.size() + 1) {
            throw std::invalid_argument(
                "boundaries.size() must equal centers.size() + 1");
        }
        for (std::size_t i = 0; i + 1 < boundaries_.size(); ++i) {
            if (boundaries_[i] >= boundaries_[i + 1]) {
                throw std::invalid_argument(
                    "boundaries must be strictly increasing");
            }
        }
    }

    double
    operator()(double E, double Ep, double /*xi*/)
        const override
    {
        auto it_g = std::upper_bound(boundaries_.begin(), boundaries_.end(), E);
        auto it_gp =
            std::upper_bound(boundaries_.begin(), boundaries_.end(), Ep);
        int const g =
            static_cast<int>(std::distance(boundaries_.begin(), it_g)) - 1;
        int const gp =
            static_cast<int>(std::distance(boundaries_.begin(), it_gp)) - 1;
        if (g == gp) {
            return 1.0;
        }
        return (Ep - E) / (centers_[gp] - centers_[g]);
    }

  private:
    std::vector<double> boundaries_;
    std::vector<double> centers_;
};

class EpMultiplier : public KernelMultiplier {
  public:
    double
    operator()(double /*E*/, double Ep, double /*xi*/)
        const override
    {
        return Ep;
    }
};

class EpOverEMultiplier : public KernelMultiplier {
  public:
    double
    operator()(double E, double Ep, double /*xi*/)
        const override
    {
        return Ep / E;
    }
};

class EpOverETimesXiMultiplier : public KernelMultiplier {
  public:
    double
    operator()(double E, double Ep, double xi)
        const override
    {
        return (Ep / E) * xi;
    }
};

class EMinusEpMultiplier : public KernelMultiplier {
  public:
    double
    operator()(double E, double Ep, double /*xi*/)
        const override
    {
        return E - Ep;
    }
};

class EnergyMomentMultiplier : public KernelMultiplier {
  public:
    explicit EnergyMomentMultiplier(int n) : n_(n)
    {
        if (n < 0)
            throw std::invalid_argument("n must be >= 0");
    }

    double
    operator()(double E, double Ep, double /*xi*/)
        const override
    {
        double const x = (Ep - E) / E;
        return std::pow(x, n_);
    }

  private:
    int n_;
};

class LegendreMultiplier0123 : public KernelMultiplier {
  public:
    explicit LegendreMultiplier0123(int n) : n_(n)
    {
        if (n < 0 || n > 3)
            throw std::invalid_argument("n must be 0, 1, 2, or 3");
    }

    double
    operator()(double /*E*/, double /*Ep*/, double xi)
        const override
    {
        switch (n_) {
        case 0: return 1.0;
        case 1: return xi;
        case 2: return 0.5 * (3.0 * xi * xi - 1.0);
        case 3: return 0.5 * (5.0 * xi * xi * xi - 3.0 * xi);
        default: return 0.0;
        }
    }

  private:
    int n_;
};

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
