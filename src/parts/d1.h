#ifndef SRC_PARTS_D1_H_
#define SRC_PARTS_D1_H_

#include "parts/part.h"

namespace vcsmc {

namespace parts {

class D1 : public Part {
 public:
  D1(const std::string& name);
  virtual ~D1() {}

  virtual bool Sim() override;
  virtual uint32 NumberOfInputs() const override;
  virtual uint32 NumberOfOutputs() const override;
};

}  // namespace parts

}  // namespace vcsmc

#endif  // SRC_PARTS_D1_H_
