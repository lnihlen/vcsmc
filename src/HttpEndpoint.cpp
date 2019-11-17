#include "HttpEndpoint.h"

#include "Logger.h"

#include "leveldb/db.h"
#include "pistache/endpoint.h"
#include "pistache/router.h"

namespace vcsmc {

class HttpEndpoint::HttpHandler {
public:
    HttpHandler(int listenPort, int numThreads, leveldb::DB* database) :
        m_listenPort(listenPort),
        m_numThreads(numThreads),
        m_db(database) { }

    void setupRoutes() {
        Pistache::Address address(Pistache::Ipv4::any(), Pistache::Port(m_listenPort));
        m_server.reset(new Pistache::Http::Endpoint(address));
        auto opts = Pistache::Http::Endpoint::options().threads(m_numThreads);
        m_server->init(opts);

        Pistache::Rest::Routes::Get(m_router, "/log/:from", Pistache::Rest::Routes::bind(
            &HttpEndpoint::HttpHandler::getLog, this));
    }

    void startServerThread() {
        LOG_INFO("opening http port on %d with %d threads", m_listenPort, m_numThreads);
        m_server->setHandler(m_router.handler());
        m_server->serve();
    }

    void shutdown() {
        m_server->shutdown();
    }

private:
    void getLog(const Pistache::Rest::Request& request, Pistache::Http::ResponseWriter response) {
        auto timeString = request.param(":from").as<std::string>();
        LOG_INFO("got %", timeString.c_str());
        response.send(Pistache::Http::Code::Ok);
    }

    int m_listenPort;
    int m_numThreads;
    leveldb::DB* m_db;
    std::shared_ptr<Pistache::Http::Endpoint> m_server;
    Pistache::Rest::Router m_router;
};

HttpEndpoint::HttpEndpoint(int listenPort, int numThreads, leveldb::DB* database) :
    m_handler(new HttpHandler(listenPort, numThreads, database)) {
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

