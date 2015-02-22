#ifndef SRC_PARTS_D2_H_
#define SRC_PARTS_D2_H_

#include "parts/part.h"

namespace vcsmc {

namespace parts {

class D2 : public Part {
 public:
  D2(const std::string& name);
  virtual ~D2() {}

  virtual bool Sim() override;
  virtual uint32 NumberOfInputs() const override;
  virtual uint32 NumberOfOutputs() const override;
};

}  // namespace parts

}  // namespace vcsmc

#endif  // SRC_PARTS_D2_H_
