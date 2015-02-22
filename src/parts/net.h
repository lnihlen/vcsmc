#ifndef SRC_PARTS_NET_H_
#define SRC_PARTS_NET_H_

#include <memory>
#include <string>
#include <vector>

namespace vcsmc {

namespace parts {

class Part;

class Net {
 public:
  Net(const std::string& name);
  ~Net() {}

  // Inputs and outputs are from the prespective of the Net, so Part outputs
  // should be connected to Net inputs, and the converse.
  void AddInput(std::shared_ptr<Part> part);
  void AddOutput(std::shared_ptr<Part> part);

  uint32 NumberOfInputs() const { return inputs_.size(); }
  uint32 NumberOfOutputs() const { return outputs_.size(); }

  const std::string& name() const { return name_; }


 private:
  std::string name_;
  std::vector<std::shared_ptr<Part>> inputs_;
  std::vector<std::shared_ptr<Part>> outputs_;
};

}  // namespace parts

}  // namespace vcsmc

#endif  // SRC_PARTS_NET_H_
