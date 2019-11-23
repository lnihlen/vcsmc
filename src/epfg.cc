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

    vcsmc::HttpEndpoint httpEndpoint(FLAGS_http_listen_port, FLAGS_http_listen_threads, db, FLAGS_html_path);
    httpEndpoint.startServerThread();

    vcsmc::Workflow workflow(db);
    workflow.runThread();

    // TODO - move to thread.

    // Time to figure out what to do, by consulting the database for the most recent state (if any).
    std::unique_ptr<leveldb::Iterator> it(db->NewIterator(leveldb::ReadOptions()));
    constexpr std::string_view kStatePrefix = "state:";
    it->Seek(kStatePrefix.data());
    // State keys are structured as "state:<64-bit hex timestamp>:" with string value provided below, or there is no
    // State key, which means that the database is new and should now be initialized.
    if (!it->Valid() || it->key().ToString().substr(0, kStatePrefix.size()) != kStatePrefix) {
        LOG_INFO("found empty state table, initializing");
        db->Put(leveldb::WriteOptions(), kStatePrefix.data() + vcsmc::Logger::Timestamp(), "Initial Empty");
        // Seek again, to find this key so that we can assume a valid key for rest of the state machine.
        it->Seek(kStatePrefix.data());
    }

    LOG_INFO("starting epfg with state %s -> %s", it->key().ToString().data(), it->value().ToString().data());
    if (it->value() == "Initial Empty") {
    } else if (it->value() == "Movie Ingest") {
    } else if (it->value() == "Encode") {
    } else if (it->value() == "Idle") {
    } else {
        LOG_WARN("unknown state %s, idling", it->value().ToString().data());
    }

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
