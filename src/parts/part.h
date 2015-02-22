#ifndef SRC_PARTS_PART_H_
#define SRC_PARTS_PART_H_

#include "types.h"

#include <memory>
#include <string>
#include <vector>

namespace vcsmc {

namespace parts {

class Net;

// The Part abstract base class defines the essential building block component
// for gate-level (ish) simulation of the TIA. Parts are instantiated via the
// YAML schematic descriptions and are connected via Nets. They can have an
// arbitrary number of inputs and outputs, can self-validate and report on
// certain kinds of errors, and are ultimately responsible for the generation
// of the OpenCL code for GPU-accelerated simulation of the TIA at scale.
class Part {
 public:
  Part(const std::string& name);
  virtual ~Part() {}

  const std::string& name() const { return name_; }

  // Compute the value of all outputs based on current state of input Nets. If
  // enough input Nets are unknown state this will fail and return false.
  virtual bool Sim() = 0;

  // Returns the number of inputs required to be connected to this Part.
  virtual uint32 NumberOfInputs() const = 0;

  // Returns the number of outputs this Part provides.
  virtual uint32 NumberOfOutputs() const = 0;

 protected:
  std::string name_;
  typedef std::vector<std::shared_ptr<Net>> Nets;
  Nets inputs_;
  Nets outputs_;
};

}  // namespace parts

}  // namespace vcsmc

#endif  // SRC_PARTS_PART_H_
