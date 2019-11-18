// epfg - Evolutionary Programming across a Frame Group - uses Evolutionary
// Programming to optimize a population of algorithms to fit a provided
// set of target images which are presumed to be similar.

#include "HttpEndpoint.h"
#include "Logger.h"

#include "gflags/gflags.h"
#include "leveldb/cache.h"
#include "leveldb/db.h"

#include <pthread.h>
#include <semaphore.h>
#include <signal.h>

DEFINE_string(db_path, "data/", "Path to file database directory.");
DEFINE_bool(new_db, false, "Create a new database at the provided db_path. Will clobber an existing database if it "
        "already exists at that path.");
DEFINE_int32(db_cache_mb, 512, "Size of database cache in megabytes.");

DEFINE_int32(echo_log_level, vcsmc::Logger::kWarning, "minimum log importance to echo to terminal");

DEFINE_int32(http_listen_port, 8001, "HTTP port to listen to for incoming web requests.");
DEFINE_int32(http_listen_threads, 4, "Number of threads to listen to for HTTP requests.");

DEFINE_string(html_path, "../html", "Path to the included HTML files to serve.");

int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, false);

    // Set thread masks to ignore termination signals, so we can catch them on the main thread and gracefully exit.
    sigset_t signals;
    sigemptyset(&signals);
    sigaddset(&signals, SIGHUP);
    sigaddset(&signals, SIGINT);
    sigaddset(&signals, SIGTERM);

    if (pthread_sigmask(SIG_BLOCK, &signals, nullptr) != 0) {
        fprintf(stderr, "error setting pthread thread mask to ignore SIGINT.\n");
        return -1;
    }

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

    vcsmc::HttpEndpoint httpEndpoint(FLAGS_http_listen_port, FLAGS_http_listen_threads, database, FLAGS_html_path);
    httpEndpoint.startServerThread();

    int signal = 0;
    // Block until termination signal sent.
    int status = sigwait(&signals, &signal);
    if (status == 0) {
        LOG_INFO("got termination signal %d", signal);
    } else {
        LOG_ERROR("got error from sigwait %d", status);
    }

    httpEndpoint.shutdown();

    vcsmc::Logger::Teardown();

    delete database;
    database = nullptr;

    return 0;
}
