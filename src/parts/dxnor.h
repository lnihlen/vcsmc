#ifndef SRC_PARTS_DXNOR_H_
#define SRC_PARTS_DXNOR_H_

#include "parts/part.h"

namespace vcsmc {

namespace parts {

class DXNor : public Part {
 public:
  DXNor(const std::string& name);
  virtual ~DXNor() {}

  virtual bool Sim() override;
  virtual uint32 NumberOfInputs() const override;
  virtual uint32 NumberOfOutputs() const override;
};

}  // namespace parts

}  // namespace vcsmc

#endif  // SRC_PARTS_DXNOR_H_
