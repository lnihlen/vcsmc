#ifndef SRC_PARTS_XOR_H_
#define SRC_PARTS_XOR_H_

#include "parts/part.h"

namespace vcsmc {

namespace parts {

class Xor : public Part {
 public:
  Xor(const std::string& name);
  virtual ~Xor() {}

  virtual bool Sim() override;
  virtual uint32 NumberOfInputs() const override;
  virtual uint32 NumberOfOutputs() const override;
};

}  // namespace parts

}  // namespace vcsmc

#endif  // SRC_PARTS_XOR_H_
