#ifndef SRC_PARTS_DGNOT_H_
#define SRC_PARTS_DGNOT_H_

#include "parts/part.h"

namespace vcsmc {

namespace parts {

class DGNot : public Part {
 public:
  DGNot(const std::string& name);
  virtual ~DGNot() {}

  virtual bool Sim() override;
  virtual uint32 NumberOfInputs() const override;
  virtual uint32 NumberOfOutputs() const override;
};

}  // namespace parts

}  // namespace vcsmc

#endif  // SRC_PARTS_DGNOT_H_
