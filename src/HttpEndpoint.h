#ifndef SRC_HTTP_ENDPOINT_H_
#define SRC_HTTP_ENDPOINT_H_

#include <memory>

namespace leveldb {
    class DB;
}

namespace vcsmc {

class HttpEndpoint {
public:
    HttpEndpoint(int listenPort, int numThreads, leveldb::DB* database);
    ~HttpEndpoint();

    void startServerThread();
    void shutdown();

private:
    class HttpHandler;
    std::unique_ptr<HttpHandler> m_handler;
};

}  // namespace vcsmc

#endif  // SRC_HTTP_ENDPOINT_H_
