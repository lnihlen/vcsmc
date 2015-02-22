#include "parts/net.h"

namespace vcsmc {

namespace parts {

Net::Net(const std::string& name)
    : name_(name) {
}

void Net::AddInput(std::shared_ptr<Part> part) {
  inputs_.push_back(part);
}

void Net::AddOutput(std::shared_ptr<Part> part) {
  outputs_.push_back(part);
}

}  // namespace parts

}  // namespace vcsmc
