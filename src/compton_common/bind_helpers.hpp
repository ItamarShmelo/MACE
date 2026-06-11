#ifndef COMPTON_BIND_HELPERS_HPP
#define COMPTON_BIND_HELPERS_HPP
/**
 * @file bind_helpers.hpp
 * @brief Pybind11 utilities shared across binding translation units.
 *
 * Provides flat-vector-to-numpy reshaping helpers used by both the
 * multigroup and Monte Carlo pybind11 modules.
 */

#include <pybind11/numpy.h>

#include <cstddef>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace compton::bind {

/**
 * @brief Reshape a flat G*G*A vector into a 3-D numpy array [G, G, A].
 *
 * @tparam Fn  Callable returning std::vector<double>.
 * @param G               Number of energy groups.
 * @param num_angle_bins  Number of angle bins (A).
 * @param fn              Callable that produces the flat data.
 */
template<typename Fn>
py::array_t<double> flat_to_numpy_3d(
    int const G,
    int const num_angle_bins,
    Fn&& fn)
{
    std::vector<double> flat = std::forward<Fn>(fn)();
    py::array_t<double> arr({G, G, num_angle_bins});
    auto buf = arr.mutable_unchecked<3>();
    std::size_t idx = 0;
    for (int g = 0; g < G; ++g)
        for (int gp = 0; gp < G; ++gp)
            for (int a = 0; a < num_angle_bins; ++a)
                buf(g, gp, a) = flat[idx++];
    return arr;
}

/**
 * @brief Reshape a flat G*G vector into a 2-D numpy array [G, G].
 *
 * @tparam Fn  Callable returning std::vector<double>.
 * @param G   Number of energy groups.
 * @param fn  Callable that produces the flat data.
 */
template<typename Fn>
py::array_t<double> flat_to_numpy_2d(
    int const G,
    Fn&& fn)
{
    std::vector<double> flat = std::forward<Fn>(fn)();
    py::array_t<double> arr({G, G});
    auto buf = arr.mutable_unchecked<2>();
    std::size_t idx = 0;
    for (int g = 0; g < G; ++g)
        for (int gp = 0; gp < G; ++gp)
            buf(g, gp) = flat[idx++];
    return arr;
}

} // namespace compton::bind

#endif
