#ifndef COMPUTE_LOGGER_HPP
#define COMPUTE_LOGGER_HPP

#include <atomic>
#include <chrono>
#include <format>
#include <fstream>
#include <string>
#include <string_view>
#include <unistd.h>

namespace compton {

/// Switch for the per-call log files compton_multigroup_<pid>.log (on by
/// default; library users such as the table builder turn it off).
inline std::atomic<bool>& compute_logging_enabled()
{
    static std::atomic<bool> enabled{true};
    return enabled;
}

class ComputeLogger {
  public:
    ComputeLogger(std::string_view tag, std::string_view params)
        : tag_(tag),
          wall_t0_(std::chrono::steady_clock::now()),
          log_path_(std::format("compton_multigroup_{}.log", ::getpid())),
          enabled_(compute_logging_enabled().load())
    {
        if (!enabled_) {
            return;
        }
        std::ofstream file(log_path_, std::ios::app);
        if (!file) {
            return;
        }
        auto const now = std::chrono::system_clock::now();
        file << std::format("{:%H:%M:%S} [compton] {}: {}", now, tag_, params) << '\n';
    }

    void done()
    {
        if (!enabled_) {
            return;
        }
        std::ofstream file(log_path_, std::ios::app);
        if (!file) {
            return;
        }
        auto const t1 = std::chrono::steady_clock::now();
        double const elapsed =
            std::chrono::duration<double>(t1 - wall_t0_).count();
        auto const now = std::chrono::system_clock::now();
        file << std::format(
            "{:%H:%M:%S} [compton] {}: done in {:g} s",
            now,
            tag_,
            elapsed) << '\n';
    }

  private:
    std::string tag_;
    std::chrono::steady_clock::time_point wall_t0_;
    std::string log_path_;
    bool enabled_;
};

} // namespace compton

#endif
