#ifndef COMPTON_BIND_HELPERS_HPP
#define COMPTON_BIND_HELPERS_HPP
/**
 * @file bind_helpers.hpp
 * @brief Pybind11 utilities shared across binding translation units.
 *
 * Provides flat-vector-to-numpy reshaping helpers used by both the
 * multigroup and Monte Carlo pybind11 modules, plus a vectorized-sigma
 * helper shared by the kernel binding modules.
 */

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "compton_common/compton_common.hpp"

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

/**
 * @brief Vectorize a scalar kernel method over an array of E' values.
 *
 * Calls (self.*fn)(E, E'[i], xi, T, Ne) for each element, collecting
 * SigmaResult::value and SigmaResult::estimated_abs_error into two
 * numpy arrays returned as a tuple.
 *
 * @tparam Class           Kernel class (e.g. ComptonKernelQuadrature).
 * @tparam MemberFunction  Pointer-to-member returning SigmaResult.
 */
template<typename Class, typename MemberFunction>
py::tuple vectorize_sigma(
    Class const& self,
    double E,
    py::array_t<double, py::array::c_style | py::array::forcecast> E_prime_arr,
    double xi,
    double T,
    double Ne,
    MemberFunction fn)
{
    auto in = E_prime_arr.unchecked<1>();
    py::ssize_t const n = in.shape(0);

    py::array_t<double> values(n);
    py::array_t<double> errors(n);
    auto out_values = values.mutable_unchecked<1>();
    auto out_errors = errors.mutable_unchecked<1>();

    for (py::ssize_t i = 0; i < n; ++i) {
        SigmaResult r = (self.*fn)(E, in(i), xi, T, Ne);
        out_values(i) = r.value;
        out_errors(i) = r.estimated_abs_error;
    }
    return py::make_tuple(values, errors);
}

} // namespace compton::bind

#endif
