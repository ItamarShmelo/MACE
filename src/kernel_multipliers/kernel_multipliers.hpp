#ifndef KERNEL_MULTIPLIERS_HPP
#define KERNEL_MULTIPLIERS_HPP

/**
 * Concrete kernel multipliers for the multigroup Compton integrals, shared by
 * the Python bindings and the C++ library (table builder).
 */

#include "compton_multigroup/compton_multigroup_deterministic/compton_multigroup_deterministic.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <utility>
#include <vector>

namespace compton {

/**
 * Group of energy E for the boundaries b_0 < ... < b_G, clamped to [0, G-1]:
 * an energy that rounds onto or past the outer edges belongs to the end group
 * instead of indexing outside the centers.
 */
inline int group_index(std::vector<double> const& boundaries, double const E)
{
    auto const it = std::upper_bound(boundaries.begin(), boundaries.end(), E);
    int const g = static_cast<int>(std::distance(boundaries.begin(), it)) - 1;
    return std::clamp(g, 0, static_cast<int>(boundaries.size()) - 2);
}

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
        int const g = group_index(boundaries_, E);
        int const gp = group_index(boundaries_, Ep);
        if (g == gp) {
            return 1.0;
        }
        return (Ep - E) / (centers_[gp] - centers_[g]);
    }

  private:
    std::vector<double> boundaries_;
    std::vector<double> centers_;
};

/**
 * Group lookup shared by the number-weighted multipliers.
 *
 * With the energy weight B(E) and its denominator B_g, the factor E_g/E turns
 * the group integral into (E_g/B_g) int (B(E)/E) ..., the photon-number
 * weight B(E)/E normalized to the model photon count B_g/E_g of the group.
 */
class GroupCenters {
  public:
    GroupCenters(
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

    int group_of(double const E) const
    {
        return group_index(boundaries_, E);
    }

    double center(int const g) const
    {
        return centers_[static_cast<std::size_t>(g)];
    }

  private:
    std::vector<double> boundaries_;
    std::vector<double> centers_;
};

/**
 * Number-weighted photon transfer between group centers,
 *
 *     (E_g/E) (E' - E) / (E_g' - E_g)   for g != g',
 *     E_g/E                             for g == g'.
 *
 * With the energy weight and denominator B_g, the off-diagonal integral is
 * E_g [sigma_in - sigma_out] / (E_g' - E_g), where sigma_in and sigma_out use
 * the multipliers E'/E and 1.
 */
class NumberWeightedEnergyTransferMultiplier : public KernelMultiplier {
  public:
    NumberWeightedEnergyTransferMultiplier(
        std::vector<double> boundaries,
        std::vector<double> centers)
        : groups_(std::move(boundaries), std::move(centers))
    {
    }

    double
    operator()(double E, double Ep, double /*xi*/)
        const override
    {
        int const g = groups_.group_of(E);
        int const gp = groups_.group_of(Ep);
        double const Eg = groups_.center(g);
        if (g == gp) {
            return Eg / E;
        }
        return (Eg / E) * (Ep - E) / (groups_.center(gp) - Eg);
    }

  private:
    GroupCenters groups_;
};

/// Number weight E_g/E of the incident photon, without an energy split.
class NumberWeightMultiplier : public KernelMultiplier {
  public:
    NumberWeightMultiplier(
        std::vector<double> boundaries,
        std::vector<double> centers)
        : groups_(std::move(boundaries), std::move(centers))
    {
    }

    double
    operator()(double E, double /*Ep*/, double /*xi*/)
        const override
    {
        return groups_.center(groups_.group_of(E)) / E;
    }

  private:
    GroupCenters groups_;
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

} // namespace compton

#endif
