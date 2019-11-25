// epfg - Evolutionary Programming across a Frame Group - uses Evolutionary
// Programming to optimize a population of algorithms to fit a provided
// set of target images which are presumed to be similar.

#include "HttpEndpoint.h"
#include "Logger.h"
#include "Workflow.h"

#include "gflags/gflags.h"
#include "leveldb/cache.h"
#include "leveldb/db.h"

#include <memory>
#include <pthread.h>
#include <semaphore.h>
#include <signal.h>

// Database options.
DEFINE_string(db_path, "data/", "Path to file database directory.");
DEFINE_int32(db_cache_mb, 512, "Size of database cache in megabytes.");

// Logging options.
DEFINE_int32(echo_log_level, vcsmc::Logger::kWarning, "minimum log importance to echo to terminal");

// Http Server options.
DEFINE_int32(http_listen_port, 8001, "HTTP port to listen to for incoming web requests.");
DEFINE_int32(http_listen_threads, 4, "Number of threads to listen to for HTTP requests.");
DEFINE_string(html_path, "../html", "Path to the included HTML files to serve.");

// Media options.
DEFINE_string(movie_path, "", "Path to the movie file to ingest as target to encode against. If the database already "
        "contains a media file this is ignored.");

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
    dbOptions.create_if_missing = true;
    dbOptions.error_if_exists = false;
    if (FLAGS_db_cache_mb > 0) {
        dbOptions.block_cache = leveldb::NewLRUCache(FLAGS_db_cache_mb * 1024 * 1024);
    }

    leveldb::DB* db = nullptr;
    leveldb::Status dbStatus = leveldb::DB::Open(dbOptions, FLAGS_db_path, &db);
    if (!dbStatus.ok()) {
        fprintf(stderr, "error opening database at %s\n", FLAGS_db_path.c_str());
        return -1;
    }

    vcsmc::Logger::Initialize(db, FLAGS_echo_log_level);

    // If there's a media path write it into the database.
    if (FLAGS_movie_path != "") {
        LOG_INFO("saving movie path %s into database", FLAGS_movie_path.c_str());
        db->Put(leveldb::WriteOptions(), "FLAGS_movie_path", FLAGS_movie_path);
    }

    vcsmc::HttpEndpoint httpEndpoint(FLAGS_http_listen_port, FLAGS_http_listen_threads, db, FLAGS_html_path);
    httpEndpoint.startServerThread();

    vcsmc::Workflow workflow(db);
    workflow.runThread();

    int signal = 0;
    // Block until termination signal sent.
    int status = sigwait(&signals, &signal);
    if (status == 0) {
        LOG_INFO("got termination signal %d", signal);
    } else {
        LOG_ERROR("got error from sigwait %d", status);
    }

    workflow.shutdown();
    httpEndpoint.shutdown();
    vcsmc::Logger::Teardown();
    delete db;
    db = nullptr;

    return 0;
}
