#ifndef SRC_PARTS_F1_H_
#define SRC_PARTS_F1_H_

#include "parts/part.h"

namespace vcsmc {

namespace parts {

class F1 : public Part {
 public:
  F1(const std::string& name);
  virtual ~F1() {}

  virtual bool Sim() override;
  virtual uint32 NumberOfInputs() const override;
  virtual uint32 NumberOfOutputs() const override;
};

}  // namespace parts

}  // namespace vcsmc

#endif  // SRC_PARTS_F1_H_
