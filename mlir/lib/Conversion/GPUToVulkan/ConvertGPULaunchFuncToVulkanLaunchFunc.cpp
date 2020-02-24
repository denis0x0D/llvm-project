//===- TODO ConvertLaunchFuncToVulkanCalls.cpp - MLIR Vulkan conversion passes -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TODO!!
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUToVulkan/ConvertGPUToVulkanPass.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Serialization.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

static constexpr const char *kSPIRVBinary = "SPIRV_BIN";
static constexpr const char *kSPIRVEntryPoint = "SPIRVEntryPoint";
static constexpr const char *kVulkanLaunch = "vulkanLaunch";

namespace {

class ConvertGpuLaunchFuncToVulkanLaunchFunc
    : public ModulePass<ConvertGpuLaunchFuncToVulkanLaunchFunc> {
private:
  /// Creates a SPIR-V binary shader from the given `module` using
  /// `spirv::serialize` function.
  LogicalResult createBinaryShader(ModuleOp module,
                                   std::vector<char> &binaryShader);

  void convertGpuLaunchFunc(mlir::gpu::LaunchFuncOp launchOp);
  // TDOD: comment
  bool isSupportedType(Type type) { return true; }
  // TODO: comment.
  LogicalResult declareVulkanLaunchFunc(Location loc,
                                        mlir::gpu::LaunchFuncOp launchOp);

public:
  void runOnModule() override;
};

} // anonymous namespace

void ConvertGpuLaunchFuncToVulkanLaunchFunc::runOnModule() {
  getModule().walk(
      [this](mlir::gpu::LaunchFuncOp op) { convertGpuLaunchFunc(op); });

  // Erase `gpu::GPUModuleOp` and `spirv::Module` operations.
  for (auto gpuModule :
       llvm::make_early_inc_range(getModule().getOps<gpu::GPUModuleOp>()))
    gpuModule.erase();

  for (auto spirvModule :
       llvm::make_early_inc_range(getModule().getOps<spirv::ModuleOp>()))
    spirvModule.erase();
}

LogicalResult ConvertGpuLaunchFuncToVulkanLaunchFunc::declareVulkanLaunchFunc(
    Location loc, mlir::gpu::LaunchFuncOp launchOp) {
  OpBuilder builder(getModule().getBody()->getTerminator());
  SmallVector<Type, 8> inputTypes;
  for (auto operand : launchOp.getOperands()) {
    if (!isSupportedType(operand.getType()))
      return failure();
    inputTypes.push_back(operand.getType());
  }
  builder.create<FuncOp>(loc, kVulkanLaunch,
                         FunctionType::get(inputTypes, {}, loc->getContext()),
                         ArrayRef<NamedAttribute>{});
  return success();
}

LogicalResult ConvertGpuLaunchFuncToVulkanLaunchFunc::createBinaryShader(
    ModuleOp module, std::vector<char> &binaryShader) {
  bool done = false;
  SmallVector<uint32_t, 0> binary;
  for (auto spirvModule : module.getOps<spirv::ModuleOp>()) {
    if (done)
      return spirvModule.emitError("should only contain one 'spv.module' op");
    done = true;

    if (failed(spirv::serialize(spirvModule, binary)))
      return failure();
  }
  binaryShader.resize(binary.size() * sizeof(uint32_t));
  std::memcpy(binaryShader.data(), reinterpret_cast<char *>(binary.data()),
              binaryShader.size());
  return success();
}

void ConvertGpuLaunchFuncToVulkanLaunchFunc::convertGpuLaunchFunc(
    mlir::gpu::LaunchFuncOp launchOp) {
  ModuleOp module = getModule();
  OpBuilder builder(launchOp);
  Location loc = launchOp.getLoc();

  // Serialize `spirv::Module` into binary form.
  std::vector<char> binary;
  if (failed(ConvertGpuLaunchFuncToVulkanLaunchFunc::createBinaryShader(
          module, binary)))
    return signalPassFailure();

  if (failed(declareVulkanLaunchFunc(loc, launchOp)))
    return signalPassFailure();

  // Create vulkan launch call op.
  auto vulkanLaunchCallOp = builder.create<mlir::CallOp>(
      loc, ArrayRef<Type>{}, builder.getSymbolRefAttr(kVulkanLaunch),
      launchOp.getOperands());

  // Set an attribute with SPIRV binary shader data.
  vulkanLaunchCallOp.setAttr(
      kSPIRVBinary,
      StringAttr::get({binary.data(), binary.size()}, loc->getContext()));

  // Set an attribute with entry point name.
  vulkanLaunchCallOp.setAttr(
      kSPIRVEntryPoint, StringAttr::get(launchOp.kernel(), loc->getContext()));

  launchOp.erase();
}

std::unique_ptr<mlir::OpPassBase<mlir::ModuleOp>>
mlir::createConvertGpuLaunchFuncToVulkanLaunchFuncPass() {
  return std::make_unique<ConvertGpuLaunchFuncToVulkanLaunchFunc>();
}

static PassRegistration<ConvertGpuLaunchFuncToVulkanLaunchFunc>
    pass("gpu-launch-to-vulkan-launch",
         "Convert gpu_launch func to vulkan launch func");
