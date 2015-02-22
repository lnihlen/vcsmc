#ifndef SRC_PARTS_AND_H_
#define SRC_PARTS_AND_H_

#include "parts/part.h"

namespace vcsmc {

namespace parts {

class And : public Part {
 public:
  And(const std::string& name);
  virtual ~And() {}

  virtual bool Sim() override;
  virtual uint32 NumberOfInputs() const override;
  virtual uint32 NumberOfOutputs() const override;
};

}  // namespace parts

}  // namespace vcsmc

#endif  // SRC_PARTS_AND_H_
