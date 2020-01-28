//===- ConvertKernelFuncToSPIRV.cpp - MLIR GPU lowering passes -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/Serialization.h"
#include "mlir/ADT/TypeSwitch.h"
#include "mlir/Dialect/SPIRV/SPIRVBinaryUtils.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/SPIRVTypes.h"
#include "mlir/IR/Builders.h"

#include "mlir/Conversion/GPUToSPIRV/ConvertGPUToSPIRVPass.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

using namespace mlir;

namespace {
class GpuKernelToSPIRVPass
    : public OperationPass<GpuKernelToSPIRVPass, spirv::ModuleOp> {
public:
  GpuKernelToSPIRVPass() = default;

  void runOnOperation() override {
    auto spirvModule = getOperation();
    /*

      module.setAttr(kCubinAnnotation, cubinAttr);
    else
      signalPassFailure();
      */
  }

private:
};
} // anonymous namespace

/*
OwnedCubin GpuKernelToSPIRVPass::convertModuleToSPIRV(spirv &llvmModule,
                                                      Location loc,
                                                      StringRef name) {
  std::unique_ptr<llvm::TargetMachine> targetMachine;
  {
    std::string error;
    // TODO(herhut): Make triple configurable.
    constexpr const char *cudaTriple = "nvptx64-nvidia-cuda";
    llvm::Triple triple(cudaTriple);
    const llvm::Target *target =
        llvm::TargetRegistry::lookupTarget("", triple, error);
    if (target == nullptr) {
      emitError(loc, "cannot initialize target triple");
      return {};
    }
    targetMachine.reset(
        target->createTargetMachine(triple.str(), "sm_35", "+ptx60", {}, {}));
  }

  // Set the data layout of the llvm module to match what the ptx target needs.
  llvmModule.setDataLayout(targetMachine->createDataLayout());

  auto ptx = translateModuleToPtx(llvmModule, *targetMachine);

  return cubinGenerator(ptx, loc, name);
}

StringAttr GpuKernelToSPIRVPass::translateGPUModuleToCubinAnnotation(
    llvm::Module &llvmModule, Location loc, StringRef name) {
  auto cubin = convertModuleToCubin(llvmModule, loc, name);
  if (!cubin)
    return {};
  return StringAttr::get({cubin->data(), cubin->size()}, loc->getContext());
}
*/

std::unique_ptr<OpPassBase<spirv::ModuleOp>>
mlir::createConvertGPUKernelToSPIRVPass() {
  return std::make_unique<GpuKernelToSPIRVPass>();
}

static PassRegistration<GpuKernelToSPIRVPass>
    pass("test-kernel-to-spirv",
         "Convert all kernel functions to CUDA cubin blobs");
