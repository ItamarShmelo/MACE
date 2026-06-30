#ifndef COMPUTE_LOGGER_HPP
#define COMPUTE_LOGGER_HPP

#include <chrono>
#include <format>
#include <fstream>
#include <print>
#include <string>
#include <string_view>
#include <unistd.h>

namespace compton {

class ComputeLogger {
  public:
    ComputeLogger(std::string_view tag, std::string_view params)
        : tag_(tag),
          wall_t0_(std::chrono::steady_clock::now()),
          file_(
              std::format("compton_multigroup_{}.log", ::getpid()),
              std::ios::app)
    {
        if (!file_)
            return;
        auto const now = std::chrono::system_clock::now();
        std::println(file_, "{:%H:%M:%S} [compton] {}: {}", now, tag_, params);
        file_.flush();
    }

    void done()
    {
        if (!file_)
            return;
        auto const t1 = std::chrono::steady_clock::now();
        double const elapsed =
            std::chrono::duration<double>(t1 - wall_t0_).count();
        auto const now = std::chrono::system_clock::now();
        std::println(
            file_,
            "{:%H:%M:%S} [compton] {}: done in {:g} s",
            now,
            tag_,
            elapsed);
    }

  private:
    std::string tag_;
    std::chrono::steady_clock::time_point wall_t0_;
    std::ofstream file_;
};

} // namespace compton

#endif
