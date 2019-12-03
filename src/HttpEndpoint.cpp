#include "HttpEndpoint.h"

#include "constants.h"
#include "image_file.h"
#include "Logger.h"

#include "Duration_generated.h"
#include "FrameGroup_generated.h"
#include "SourceFrame_generated.h"

#include "leveldb/db.h"
#include "pistache/endpoint.h"
#include "pistache/router.h"

#include <png.h>

#include <filesystem>
#include <inttypes.h>
#include <stdio.h>
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
        Pistache::Rest::Routes::Get(m_router, "/third_party/dygraph.min.js", Pistache::Rest::Routes::bind(
            &HttpEndpoint::HttpHandler::serveFile, this));

        // Returns a JSON array of up to kMaxResponseSize characters with individual Duration elements packed within,
        // each with timestamp >= from. From is a 16-character hex string, microseconds from unix epoch.
        Pistache::Rest::Routes::Get(m_router, "/duration/:from", Pistache::Rest::Routes::bind(
            &HttpEndpoint::HttpHandler::getDuration, this));

        // Returns some JSON with the frame group data. Number should be an 8 character hex string.
        Pistache::Rest::Routes::Get(m_router, "/group/:number", Pistache::Rest::Routes::bind(
            &HttpEndpoint::HttpHandler::getGroup, this));

        // Image serving.
        // Number should be an 8 character hex string. Hash should be a 16 character hex string and should match the
        // hash supplied in the metadata. This is to avoid caching issues in browsers when fitting different videos.
        Pistache::Rest::Routes::Get(m_router, "/img/source/:hash", Pistache::Rest::Routes::bind(
            &HttpEndpoint::HttpHandler::getImageSource, this));
        Pistache::Rest::Routes::Get(m_router, "/img/quantized/:hash", Pistache::Rest::Routes::bind(
            &HttpEndpoint::HttpHandler::getImageTarget, this));

        // The "from" should be a string with the number of microseconds from unix epoch in hex, or zero.
        // Will return the first key with value >= from.
        Pistache::Rest::Routes::Get(m_router, "/log/:from", Pistache::Rest::Routes::bind(
            &HttpEndpoint::HttpHandler::getLog, this));

        // Returns a 64-bit hash of the quantized image associated with the provided source image hash.
        Pistache::Rest::Routes::Get(m_router, "/quantizeMap/:hash", Pistache::Rest::Routes::bind(
            &HttpEndpoint::HttpHandler::getQuantizeMap, this));
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
        Pistache::Http::serveFile(response, resource);
    }

    void getDuration(const Pistache::Rest::Request& request, Pistache::Http::ResponseWriter response) {
        auto timeString = request.param(":from").as<std::string>();
        std::string durationKey = "duration:";
        if (timeString.size() == 16) {
            durationKey += timeString;
        }
        std::unique_ptr<leveldb::Iterator> it(m_db->NewIterator(leveldb::ReadOptions()));
        auto isDurationKey = [](auto& it) {
            return it->Valid() && it->key().ToString().substr(0, 9) == "duration:";
        };
        it->Seek(durationKey);
        if (isDurationKey(it)) {
            std::string json = "[ ";
            std::string object;
            while (json.size() + object.size() < kMaxResponseContentSize - 2) {
                if (json.size() > 2) json += ", ";
                json += object;
                const Data::Duration* duration = Data::GetDuration(it->value().data());
                if (!duration) {
                    LOG_WARN("error deserializing duration at key %s", it->key().data());
                    response.send(Pistache::Http::Code::Internal_Server_Error);
                    return;
                }
                object = "{ \"type\":";
                std::array<char, 32> buf;
                snprintf(buf.data(), sizeof(buf), "%d", duration->type());
                object += buf.data() + std::string(", \"startTime\":");
                snprintf(buf.data(), sizeof(buf), "%" PRId64, duration->startTime());
                object += buf.data() + std::string(", \"duration\":");
                snprintf(buf.data(), sizeof(buf), "%" PRId64, duration->duration());
                object += buf.data() + std::string(" }");
                it->Next();
                if (!isDurationKey(it)) break;
            }
            json += " ]";
            response.send(Pistache::Http::Code::Ok, json, MIME(Application, Json));
        } else {
            // Send back empty array to indicate done state.
            response.send(Pistache::Http::Code::Ok, "[ ]", MIME(Application, Json));
        }
    }

    // TODO: convert to sending JSON array, like duration.
    void getGroup(const Pistache::Rest::Request& request, Pistache::Http::ResponseWriter response) {
        auto number = request.param(":number").as<std::string>();
        std::string groupKey = "frameGroup:" + number;
        std::unique_ptr<leveldb::Iterator> it(m_db->NewIterator(leveldb::ReadOptions()));
        it->Seek(groupKey);
        if ( it->Valid() && it->key().ToString() == groupKey) {
            const Data::FrameGroup* frameGroup = Data::GetFrameGroup(it->value().data());
            std::array<char, 32> buf;
            std::string json = "{ \"firstFrame\":";
            snprintf(buf.data(), sizeof(buf), "%d", frameGroup->firstFrame());
            json += buf.data() + std::string(", \"lastFrame\":");
            snprintf(buf.data(), sizeof(buf), "%d", frameGroup->lastFrame());
            json += buf.data() + std::string(", \"imageHashes\":[");
            auto& hashes = *frameGroup->imageHashes();
            bool appending = false;
            for (auto hash : hashes) {
                if (appending) {
                    json += ", ";
                }
                snprintf(buf.data(), sizeof(buf), "\"%016" PRIx64 "\"", hash);
                json += buf.data();
                appending = true;
            }
            json += "] }";
            response.send(Pistache::Http::Code::Ok, json, MIME(Application, Json));
        } else {
            LOG_WARN("frame group number %s not found.", number.c_str());
            response.send(Pistache::Http::Code::Not_Found);
        }
    }

    void getImageSource(const Pistache::Rest::Request& request, Pistache::Http::ResponseWriter response) {
        auto hash = request.param(":hash").as<std::string>();
        std::string frameKey = "sourceImage:" + hash;
        std::unique_ptr<leveldb::Iterator> it(m_db->NewIterator(leveldb::ReadOptions()));
        it->Seek(frameKey);
        if (it->Valid() && it->key().ToString() == frameKey) {
            fs::path imageFilePath = m_cachePath + "sourceImage-" + hash + ".png";
            bool imageOK = fs::exists(imageFilePath);
            if (!imageOK) {
                imageOK = SaveImage(reinterpret_cast<const uint8_t*>(it->value().data()), kTargetFrameWidthPixels,
                    kFrameHeightPixels, imageFilePath);
            }
            if (imageOK) {
                Pistache::Http::serveFile(response, imageFilePath);
            } else {
                LOG_WARN("error saving image file to %s", imageFilePath.c_str());
                response.send(Pistache::Http::Code::Internal_Server_Error);
            }
        } else {
            LOG_WARN("source image hash %s not found.", hash.c_str());
            response.send(Pistache::Http::Code::Not_Found);
        }
    }

    void getImageTarget(const Pistache::Rest::Request& request, Pistache::Http::ResponseWriter response) {
        auto hash = request.param(":hash").as<std::string>();
        std::string quantKey = "quantImage:" + hash;
        std::unique_ptr<leveldb::Iterator> it(m_db->NewIterator(leveldb::ReadOptions()));
        it->Seek(quantKey);
        if (it->Valid() && it->key().ToString() == quantKey) {
            fs::path imageFilePath = m_cachePath + "quantImage-" + hash + ".png";
            bool imageOK = fs::exists(imageFilePath);
            if (!imageOK) {
                imageOK = SaveAtariPaletteImage(reinterpret_cast<const uint8_t*>(it->value().data()), imageFilePath);
            }
            if (imageOK) {
                Pistache::Http::serveFile(response, imageFilePath);
            } else {
                LOG_WARN("error saving palette image file to %s", imageFilePath.c_str());
                response.send(Pistache::Http::Code::Internal_Server_Error);
            }
        } else {
            LOG_WARN("quantized imaged hash %s not found.", quantKey.c_str());
            response.send(Pistache::Http::Code::Not_Found);
        }
    }

    // TODO: convert to sending JSON array, just like duration.
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

    void getQuantizeMap(const Pistache::Rest::Request& request, Pistache::Http::ResponseWriter response) {
        auto sourceHash = request.param(":hash").as<std::string>();
        std::string mapKey = "quantizeMap:" + sourceHash;
        std::unique_ptr<leveldb::Iterator> it(m_db->NewIterator(leveldb::ReadOptions()));
        it->Seek(mapKey);
        if (it->Valid() && it->key().ToString() == mapKey) {
            response.send(Pistache::Http::Code::Ok, it->value().ToString(), MIME(Text, Plain));
        } else {
            response.send(Pistache::Http::Code::Not_Found);
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
            frameJSON += buf.data() + std::string(", \"sourceHash\":\"");
            snprintf(buf.data(), sizeof(buf), "%016" PRIx64, sourceFrame->sourceHash());
            frameJSON += buf.data() + std::string("\" }");
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

