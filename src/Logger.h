#ifndef SRC_LOGGER_H_
#define SRC_LOGGER_H_

#include <cstdint>
#include <string>

namespace leveldb {
    class DB;
}

namespace vcsmc {

class Logger {
public:
    enum Level : int32_t {
        kFatal = 1,
        kError = 2,
        kWarning = 3,
        kInfo = 4,
        kDebug = 5
    };

    static void Initialize(leveldb::DB* database, int32_t echoLogLevel);
    static void Log(Level level, const char* format, ...);
    static std::string Timestamp();
    static void Teardown();
};

}  // namespace vcsmc

#define LOG_DEBUG(...) ::vcsmc::Logger::Log(::vcsmc::Logger::kDebug, __VA_ARGS__)
#define LOG_INFO(...) ::vcsmc::Logger::Log(::vcsmc::Logger::kInfo, __VA_ARGS__)
#define LOG_WARN(...) ::vcsmc::Logger::Log(::vcsmc::Logger::kWarning, __VA_ARGS__)
#define LOG_ERROR(...) ::vcsmc::Logger::Log(::vcsmc::Logger::kError, __VA_ARGS__)
#define LOG_FATAL(...) ::vcsmc::Logger::Log(::vcsmc::Logger::kFatal, __VA_ARGS__)

#endif  // SRC_LOGGER_H_

