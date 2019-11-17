#include "Logger.h"

#include <leveldb/db.h>

#include <cassert>
#include <chrono>
#include <cstring>
#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>

namespace {
    static leveldb::DB* db = nullptr;
    static const char logLevels[] = { 'X', 'F', 'E', 'W', 'I', 'D' };
    static int32_t minEchoLevel;
}


namespace vcsmc {

// static
void Logger::Initialize(leveldb::DB* database, int32_t echoLogLevel) {
    db = database;
    minEchoLevel = echoLogLevel;
    LOG_INFO("Logging system initialized.");
}

// static
void Logger::Log(Logger::Level level, const char* format, ...) {
    char buf[4096];

    uint64_t epochMillis = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    // There's a temptation to pad this epoch key out to a few more digits but this is ridiculous as it seems we won't
    // need another digit in the milliseconds epoch count for the useful life of this code. But should humanity survive
    // for however long that is and somehow this code is still relevant adding a most-significant digit to these numbers
    // will break the sorting.
    int keyLength = snprintf(buf, 4096, "log:%" PRIx64 ":%c", epochMillis, logLevels[level]);

    va_list args;
    va_start(args, format);
    int valueLength = vsnprintf(buf + keyLength + 1, 4096 - 1 - keyLength, format, args);
    va_end(args);

    leveldb::Status status = db->Put(leveldb::WriteOptions(), leveldb::Slice(buf, keyLength),
            leveldb::Slice(buf + keyLength + 1, valueLength));

    // Because we are logging to a database, if database writes are failing this is a fatal error and we don't want
    // to log it, so we crash.
    if (!status.ok()) {
        fprintf(stderr, "Logging system failed to write to database, crashing.\n");
        assert(false);
    }

    if (level <= minEchoLevel) {
        printf("%s %s\n", buf, buf + keyLength + 1);
    }
}

// static
void Logger::Teardown() {
    LOG_INFO("Logging system terminated.");
    db = nullptr;
}

}
