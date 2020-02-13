//===- mlir-vulkan-runner.cpp - MLIR Vulkan Execution Driver --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a command line utility that executes an MLIR file on the Vulkan by
// translating MLIR GPU module to SPIR-V and host part to LLVM IR before
// JIT-compiling and executing the latter.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUToSPIRV/ConvertGPUToSPIRVPass.h"
#include "mlir/Conversion/GPUToVulkan/ConvertGPUToVulkanPass.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRVPass.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/SPIRV/Passes.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/JitRunner.h"

using namespace mlir;

static LogicalResult runMLIRPasses(ModuleOp m) {
  PassManager pm(m.getContext());
  applyPassManagerCLOptions(pm);

  pm.addPass(createGpuKernelOutliningPass());
  pm.addPass(createLegalizeStdOpsForSPIRVLoweringPass());
  pm.addPass(createConvertGPUToSPIRVPass());
  OpPassManager &modulePM = pm.nest<spirv::ModuleOp>();
  modulePM.addPass(spirv::createLowerABIAttributesPass());
  pm.addPass(createConvertGpuLaunchFuncToVulkanCallsPass());
  pm.addPass(createLowerToLLVMPass());
  return pm.run(m);
}

int main(int argc, char **argv) {
  registerPassManagerCLOptions();
  return mlir::JitRunnerMain(argc, argv, &runMLIRPasses);
}
