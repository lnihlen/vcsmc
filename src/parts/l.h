#ifndef SRC_PARTS_L_H_
#define SRC_PARTS_L_H_

#include "parts/part.h"

namespace vcsmc {

namespace parts {

class L : public Part {
 public:
  L(const std::string& name);
  virtual ~L() {}

  virtual bool Sim() override;
  virtual uint32 NumberOfInputs() const override;
  virtual uint32 NumberOfOutputs() const override;
};

}  // namespace parts

}  // namespace vcsmc

#endif  // SRC_PARTS_L_H_
