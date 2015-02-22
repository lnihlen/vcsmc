#ifndef SRC_PARTS_DGNOR_H_
#define SRC_PARTS_DGNOR_H_

#include "parts/part.h"

namespace vcsmc {

namespace parts {

class DGNor : public Part {
 public:
  DGNor(const std::string& name);
  virtual ~DGNor() {}

  virtual bool Sim() override;
  virtual uint32 NumberOfInputs() const override;
  virtual uint32 NumberOfOutputs() const override;
};

}  // namespace parts

}  // namespace vcsmc

#endif  // SRC_PARTS_DGNOR_H_
