// epfg - Evolutionary Programming across a Frame Group - uses Evolutionary
// Programming to optimize a population of algorithms to fit a provided
// set of target images which are presumed to be similar.

#include "Logger.h"

#include <gflags/gflags.h>
#include <leveldb/cache.h>
#include <leveldb/db.h>

DEFINE_string(db_path, "data/", "Path to file database directory.");
DEFINE_bool(new_db, false, "Create a new database at the provided db_path. Will clobber an existing database if it "
        "already exists at that path.");
DEFINE_int32(db_cache_mb, 512, "Size of database cache in megabytes.");

DEFINE_int32(echo_log_level, vcsmc::Logger::kWarning, "minimum log importance to echo to terminal");

DEFINE_int32(http_listen_port, 8001, "HTTP port to listen to for incoming web requests.");
DEFINE_int32(http_listen_threads, 4, "Number of threads to listen to for HTTP requests.");

int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, false);

    // Initialize Database
    leveldb::Options dbOptions;
    dbOptions.create_if_missing = FLAGS_new_db;
    dbOptions.error_if_exists = false;
    if (FLAGS_db_cache_mb > 0) {
        dbOptions.block_cache = leveldb::NewLRUCache(FLAGS_db_cache_mb * 1024 * 1024);
    }

    leveldb::DB* database = nullptr;
    leveldb::Status dbStatus = leveldb::DB::Open(dbOptions, FLAGS_db_path, &database);
    if (!dbStatus.ok()) {
        fprintf(stderr, "error opening database at %s\n", FLAGS_db_path.c_str());
        return -1;
    }

    vcsmc::Logger::Initialize(database, FLAGS_echo_log_level);

    vcsmc::HttpEndpoint httpEndpoint(FLAGS_http_listen_port, FLAGS_http_listen_threads, database);
    httpEndpoint.startServerThread();

    // BLOCK

    httpEndpoint.shutdown();

    vcsmc::Logger::Teardown();

    delete database;
    database = nullptr;

    return 0;
}
