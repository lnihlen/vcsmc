#include "HttpEndpoint.h"

#include "Logger.h"

#include "leveldb/db.h"
#include "pistache/endpoint.h"
#include "pistache/router.h"

#include <string>

namespace vcsmc {

class HttpEndpoint::HttpHandler {
public:
    HttpHandler(int listenPort, int numThreads, leveldb::DB* database, const std::string& htmlPath) :
        m_listenPort(listenPort),
        m_numThreads(numThreads),
        m_db(database),
        m_htmlPath(htmlPath) { }

    void setupRoutes() {
        Pistache::Address address(Pistache::Ipv4::any(), Pistache::Port(m_listenPort));
        m_server.reset(new Pistache::Http::Endpoint(address));
        auto opts = Pistache::Http::Endpoint::options()
            .threads(m_numThreads)
            .maxRequestSize(kMaxRequestSize)
            .maxResponseSize(kMaxResponseSize);
        m_server->init(opts);

        // Static file serving.
        Pistache::Rest::Routes::Get(m_router,"/", Pistache::Rest::Routes::bind(
            &HttpEndpoint::HttpHandler::serveFile, this));
        Pistache::Rest::Routes::Get(m_router, "/index.html", Pistache::Rest::Routes::bind(
            &HttpEndpoint::HttpHandler::serveFile, this));

        // The "from" should be a string with the number of microseconds from unix epoch in hex, or zero.
        // Will return the first key with value >= from.
        Pistache::Rest::Routes::Get(m_router, "/log/:from", Pistache::Rest::Routes::bind(
            &HttpEndpoint::HttpHandler::getLog, this));
    }

    void startServerThread() {
        LOG_INFO("opening http port on %d with %d threads", m_listenPort, m_numThreads);
        m_server->setHandler(m_router.handler());
        m_server->serveThreaded();
    }

    void shutdown() {
        m_server->shutdown();
    }

private:
    static constexpr size_t kMaxRequestSize = 4096;
    static constexpr size_t kMaxResponseSize = 4096;
    // Pistache seems to throw an exception if we don't budget a bit of size for headers, etc in total response size.
    static constexpr size_t kMaxResponseContentSize = kMaxResponseSize - 128;

    void serveFile(const Pistache::Rest::Request& request, Pistache::Http::ResponseWriter response) {
        std::string resource = m_htmlPath + (request.resource() == "/" ? "/index.html" : request.resource());
        LOG_DEBUG("serving file %s", resource.c_str());
        Pistache::Http::serveFile(response, resource);
    }

    void getLog(const Pistache::Rest::Request& request, Pistache::Http::ResponseWriter response) {
        auto timeString = request.param(":from").as<std::string>();
        std::string logKey = "log:";
        if (timeString.size() == 16) {
            logKey += timeString;
        }
        std::unique_ptr<leveldb::Iterator> it(m_db->NewIterator(leveldb::ReadOptions()));
        auto isLogKey = [](auto& it) {
            return it->Valid() && it->key().ToString().substr(0, 4) == "log:";
        };
        it->Seek(logKey);
        if (isLogKey(it)) {
            std::string logResponse = it->key().ToString() + "\t" + it->value().ToString();
            it->Next();
            size_t count = 0;
            while (isLogKey(it) && count < 10000) {
                std::string nextLog = it->key().ToString() + "\t" + it->value().ToString();
                if (logResponse.size() + nextLog.size() + 1 >= kMaxResponseContentSize) break;
                logResponse += "\n" + nextLog;
                it->Next();
                ++count;
            }
            response.send(Pistache::Http::Code::Ok, logResponse, MIME(Text, Plain));
        } else {
            // Send back empty response to indicate no new log entries.
            response.send(Pistache::Http::Code::Ok);
        }
    }

    int m_listenPort;
    int m_numThreads;
    leveldb::DB* m_db;
    std::string m_htmlPath;
    std::shared_ptr<Pistache::Http::Endpoint> m_server;
    Pistache::Rest::Router m_router;
};

HttpEndpoint::HttpEndpoint(int listenPort, int numThreads, leveldb::DB* database, const std::string& htmlPath) :
    m_handler(new HttpHandler(listenPort, numThreads, database, htmlPath)) {
}

HttpEndpoint::~HttpEndpoint() {
}

void HttpEndpoint::startServerThread() {
    m_handler->setupRoutes();
    m_handler->startServerThread();
}

void HttpEndpoint::shutdown() {
    m_handler->shutdown();
}

}  // namespace vcsmc
