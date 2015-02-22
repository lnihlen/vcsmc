#ifndef SRC_PARTS_NOT_H_
#define SRC_PARTS_NOT_H_

#include "parts/part.h"

namespace vcsmc {

namespace parts {

class Not : public Part {
 public:
  Not(const std::string& name);
  virtual ~Not() {}

  virtual bool Sim() override;
  virtual uint32 NumberOfInputs() const override;
  virtual uint32 NumberOfOutputs() const override;
};

}  // namespace parts

}  // namespace vcsmc

#endif  // SRC_PARTS_NOT_H_
