#include "HttpEndpoint.h"

#include "constants.h"
#include "image_file.h"
#include "Logger.h"

#include "SourceFrame_generated.h"
#include "TargetFrame_generated.h"

#include "leveldb/db.h"
#include "pistache/endpoint.h"
#include "pistache/router.h"

#include <png.h>

#include <filesystem>
#include <inttypes.h>
#include <string>

namespace fs = std::filesystem;

namespace vcsmc {

class HttpEndpoint::HttpHandler {
public:
    HttpHandler(int listenPort, int numThreads, leveldb::DB* database, const std::string& htmlPath,
            const std::string& cachePath) :
        m_listenPort(listenPort),
        m_numThreads(numThreads),
        m_db(database),
        m_htmlPath(htmlPath),
        m_cachePath(cachePath) { }

    void setupRoutes() {
        Pistache::Address address(Pistache::Ipv4::any(), Pistache::Port(m_listenPort));
        m_server.reset(new Pistache::Http::Endpoint(address));
        auto opts = Pistache::Http::Endpoint::options()
            .threads(m_numThreads)
            .maxRequestSize(kMaxRequestSize)
            .maxResponseSize(kMaxResponseSize);
        m_server->init(opts);

        // Static file serving.
        Pistache::Rest::Routes::Get(m_router, "/", Pistache::Rest::Routes::bind(
            &HttpEndpoint::HttpHandler::serveFile, this));
        Pistache::Rest::Routes::Get(m_router, "/index.html", Pistache::Rest::Routes::bind(
            &HttpEndpoint::HttpHandler::serveFile, this));

        // Image serving.
        // Number should be an 8 character hex string. Hash should be a 16 character hex string and should match the
        // hash supplied in the metadata. This is to avoid caching issues in browsers when fitting different videos.
        Pistache::Rest::Routes::Get(m_router, "/img/source/:number/:hash", Pistache::Rest::Routes::bind(
            &HttpEndpoint::HttpHandler::getImageSource, this));
        Pistache::Rest::Routes::Get(m_router, "/img/target/:number/:hash", Pistache::Rest::Routes::bind(
            &HttpEndpoint::HttpHandler::getImageTarget, this));

        // The "from" should be a string with the number of microseconds from unix epoch in hex, or zero.
        // Will return the first key with value >= from.
        Pistache::Rest::Routes::Get(m_router, "/log/:from", Pistache::Rest::Routes::bind(
            &HttpEndpoint::HttpHandler::getLog, this));

        // Returns some JSON with the source frame metadata contained. Number should be an 8 character hex string.
        Pistache::Rest::Routes::Get(m_router, "/source/:number", Pistache::Rest::Routes::bind(
            &HttpEndpoint::HttpHandler::getSource, this));
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
        LOG_INFO("serving file %s", resource.c_str());
        Pistache::Http::serveFile(response, resource);
    }

    void getImageSource(const Pistache::Rest::Request& request, Pistache::Http::ResponseWriter response) {
        auto frameNumber = request.param(":number").as<std::string>();
        auto hash = request.param(":hash").as<std::string>();
        std::string frameKey = "sourceFrame:" + frameNumber;
        std::unique_ptr<leveldb::Iterator> it(m_db->NewIterator(leveldb::ReadOptions()));
        it->Seek(frameKey);
        if (it->Valid() && it->key().ToString() == frameKey) {
            // First look up image in database. Match hash to one provided.
            const Data::SourceFrame* sourceFrame = Data::GetSourceFrame(it->value().data());
            if (!sourceFrame) {
                LOG_WARN("failed to retrieve source frame data for frame %s, hash %s", frameNumber.c_str(),
                    hash.c_str());
                response.send(Pistache::Http::Code::Internal_Server_Error);
            } else {
                uint64_t hashValue = strtoull(hash.c_str(), nullptr, 16);
                if (sourceFrame->imageHash() != hashValue) {
                    LOG_WARN("source frame %s requested hash %s doesn't match stored hash %016" PRIx64,
                        frameNumber.c_str(), hash.c_str(), hashValue);
                    response.send(Pistache::Http::Code::Not_Found);
                } else {
                    fs::path imageFilePath = m_cachePath + "sourceImage-" + frameNumber + "-" + hash + ".png";
                    bool imageOK = fs::exists(imageFilePath);
                    if (!imageOK) {
                        imageOK = SaveImage(sourceFrame->imageRGB()->data(), kTargetFrameWidthPixels,
                            kFrameHeightPixels, imageFilePath);
                    }
                    if (imageOK) {
                        Pistache::Http::serveFile(response, imageFilePath);
                    } else {
                        LOG_WARN("error saving image file to %s", imageFilePath.c_str());
                        response.send(Pistache::Http::Code::Internal_Server_Error);
                    }
                }
            }
            // Next look up path in cache, to see if image already extracted, if so, serve it.
            // If not, save image then serve it.
        } else {
            response.send(Pistache::Http::Code::Not_Found);
        }
    }

    void getImageTarget(const Pistache::Rest::Request& request, Pistache::Http::ResponseWriter response) {
        response.send(Pistache::Http::Code::Ok);
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

    void getSource(const Pistache::Rest::Request& request, Pistache::Http::ResponseWriter response) {
        auto frameNumber = request.param(":number").as<std::string>();
        std::string frameKey = "sourceFrame:" + frameNumber;
        std::unique_ptr<leveldb::Iterator> it(m_db->NewIterator(leveldb::ReadOptions()));
        it->Seek(frameKey);
        if (it->Valid() && it->key().ToString() == frameKey) {
            const Data::SourceFrame* sourceFrame = Data::GetSourceFrame(it->value().data());
            std::array<char, 32> buf;
            std::string frameJSON = "{ \"frameNumber\":";
            snprintf(buf.data(), sizeof(buf), "%d", sourceFrame->frameNumber());
            frameJSON += buf.data() + std::string(", \"isKeyFrame\":");
            if (sourceFrame->isKeyFrame()) {
                frameJSON += "true";
            } else {
                frameJSON += "false";
            }
            frameJSON += ", \"frameTime\":";
            snprintf(buf.data(), sizeof(buf), "%" PRId64, sourceFrame->frameTime());
            frameJSON += buf.data() + std::string(", \"imageHash\":");
            snprintf(buf.data(), sizeof(buf), "%" PRIx64, sourceFrame->imageHash());
            frameJSON += buf.data() + std::string(" }");
            response.send(Pistache::Http::Code::Ok, frameJSON, MIME(Application, Json));
        } else {
            response.send(Pistache::Http::Code::Not_Found);
        }
    }

    int m_listenPort;
    int m_numThreads;
    leveldb::DB* m_db;
    std::string m_htmlPath;
    std::string m_cachePath;
    std::shared_ptr<Pistache::Http::Endpoint> m_server;
    Pistache::Rest::Router m_router;
};

HttpEndpoint::HttpEndpoint(int listenPort, int numThreads, leveldb::DB* database, const std::string& htmlPath,
    const std::string& cachePath) :
    m_handler(new HttpHandler(listenPort, numThreads, database, htmlPath, cachePath)) {
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

