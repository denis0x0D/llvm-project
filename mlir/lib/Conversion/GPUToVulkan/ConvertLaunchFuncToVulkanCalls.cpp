//===- ConvertLaunchFuncToVulkanCalls.cpp - MLIR Vulkan conversion passes -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert gpu.launch_func op into a sequence of
// Vulkan runtime calls. The Vulkan runtime API surface is huge so currently we
// don't expose separate external functions in IR for each of them, instead we
// expose a few external functions to wrapper libraries which manages Vulkan
// runtime.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUToVulkan/ConvertGPUToVulkanPass.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallString.h"

using namespace mlir;

static constexpr const char *kSetBinaryShader = "setBinaryShader";
static constexpr const char *kSetEntryPoint = "setEntryPoint";
static constexpr const char *kSetNumWorkGroups = "setNumWorkGroups";
static constexpr const char *kRunOnVulkan = "runOnVulkan";
static constexpr const char *kSPIRVBinary = "SPIRV_BIN";
static constexpr const char *kVulkanLaunch = "vulkanLaunch";
static constexpr const char *kSPIRVEntryPoint = "SPIRVEntryPoint";
static constexpr const char *kBindResource = "bindResource";
static constexpr const char *kInitVulkan = "initVulkan";
static constexpr const char *kDeinitVulkan = "deinitVulkan";

namespace {

/// A pass to convert gpu.launch_func operation into a sequence of Vulkan
/// runtime calls.
///
/// * setBinaryShader      -- sets the binary shader data
/// * setEntryPoint        -- sets the entry point name
/// * setNumWorkGroups     -- sets the number of a local workgroups
/// * runOnVulkan          -- runs vulkan runtime
///
class VulkanLaunchFuncToVulkanCallsPass
    : public ModulePass<VulkanLaunchFuncToVulkanCallsPass> {
private:
  LLVM::LLVMDialect *getLLVMDialect() { return llvmDialect; }

  llvm::LLVMContext &getLLVMContext() {
    return getLLVMDialect()->getLLVMContext();
  }

  void initializeCachedTypes() {
    llvmDialect = getContext().getRegisteredDialect<LLVM::LLVMDialect>();
    llvmFloatType = LLVM::LLVMType::getFloatTy(llvmDialect);
    llvmVoidType = LLVM::LLVMType::getVoidTy(llvmDialect);
    llvmPointerType = LLVM::LLVMType::getInt8PtrTy(llvmDialect);
    llvmInt32Type = LLVM::LLVMType::getInt32Ty(llvmDialect);
    llvmInt64Type = LLVM::LLVMType::getInt64Ty(llvmDialect);
  }

  LLVM::LLVMType getFloatType() { return llvmFloatType; }
  LLVM::LLVMType getVoidType() { return llvmVoidType; }
  LLVM::LLVMType getPointerType() { return llvmPointerType; }
  LLVM::LLVMType getInt32Type() { return llvmInt32Type; }
  LLVM::LLVMType getInt64Type() { return llvmInt64Type; }

  /// Creates a LLVM global for the given `name`.
  Value createEntryPointNameConstant(StringRef name, Location loc,
                                     OpBuilder &builder);

  /// Declares all needed runtime functions.
  void declareVulkanFunctions(Location loc);

  /// Checks whether the given LLVM::CallOp is a Vulkan launch call op.
  bool isVulkanLaunchCallOp(LLVM::CallOp callOp) {
    return (callOp.callee() && callOp.callee().getValue() == kVulkanLaunch);
  }

  /// Translates the given `launcOp` op to the sequence of Vulkan runtime
  /// calls
  void translateVulkanLaunchCall(LLVM::CallOp vulkanLaunchCallOp);

  void bindResources(LLVM::CallOp vulkanLaunchCallOp);

public:
  void runOnModule() override;

private:
  LLVM::LLVMDialect *llvmDialect;
  LLVM::LLVMType llvmVoidType;
  LLVM::LLVMType llvmPointerType;
  LLVM::LLVMType llvmInt32Type;
  LLVM::LLVMType llvmInt64Type;
  LLVM::LLVMType llvmFloatType;
};

} // anonymous namespace

void VulkanLaunchFuncToVulkanCallsPass::runOnModule() {
  initializeCachedTypes();
  getModule().walk([this](LLVM::CallOp op) {
    if (isVulkanLaunchCallOp(op))
      translateVulkanLaunchCall(op);
  });
}

void VulkanLaunchFuncToVulkanCallsPass::bindResources(
    LLVM::CallOp vulkanLaunchCallOp) {
  if (vulkanLaunchCallOp.getNumOperands() == 6)
    return;

  OpBuilder builder(vulkanLaunchCallOp);
  Location loc = vulkanLaunchCallOp.getLoc();
  Value descriptorSet = builder.create<LLVM::ConstantOp>(
      loc, getInt32Type(), builder.getI32IntegerAttr(0));

  // TODO: Use operand adaptor pattern.
  const int32_t offset = 6;
  // According current calling convention for memRef.
  // TODO: Move to constexpr.
  const int32_t resourceOperandCount = 5;
  const int32_t sizeIndex = 3;
  const int32_t resourceCount =
      (vulkanLaunchCallOp.getNumOperands() - offset) / resourceOperandCount;

  for (auto resourceIdx : llvm::seq(0, resourceCount)) {
    Value descriptorBinding = builder.create<LLVM::ConstantOp>(
        loc, getInt32Type(), builder.getI32IntegerAttr(resourceIdx));
    auto resourcePtr = vulkanLaunchCallOp.getOperand(
        offset + resourceIdx * resourceOperandCount);
    auto resourceSize = vulkanLaunchCallOp.getOperand(
        offset + resourceIdx * resourceOperandCount + sizeIndex);

    builder.create<LLVM::CallOp>(loc, ArrayRef<Type>{getVoidType()},
                                 builder.getSymbolRefAttr(kBindResource),
                                 ArrayRef<Value>{descriptorSet,
                                                 descriptorBinding, resourcePtr,
                                                 resourceSize});
  }
}

void VulkanLaunchFuncToVulkanCallsPass::declareVulkanFunctions(Location loc) {
  ModuleOp module = getModule();
  OpBuilder builder(module.getBody()->getTerminator());

  if (!module.lookupSymbol(kSetEntryPoint)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kSetEntryPoint,
        LLVM::LLVMType::getFunctionTy(getVoidType(), {getPointerType()},
                                      /*isVarArg=*/false));
  }

  if (!module.lookupSymbol(kSetNumWorkGroups)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kSetNumWorkGroups,
        LLVM::LLVMType::getFunctionTy(
            getVoidType(), {getInt64Type(), getInt64Type(), getInt64Type()},
            /*isVarArg=*/false));
  }

  if (!module.lookupSymbol(kSetBinaryShader)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kSetBinaryShader,
        LLVM::LLVMType::getFunctionTy(getVoidType(),
                                      {getPointerType(), getInt32Type()},
                                      /*isVarArg=*/false));
  }

  if (!module.lookupSymbol(kRunOnVulkan)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kRunOnVulkan,
        LLVM::LLVMType::getFunctionTy(getVoidType(), {},
                                      /*isVarArg=*/false));
  }

  if (!module.lookupSymbol(kBindResource)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kBindResource,
        LLVM::LLVMType::getFunctionTy(getVoidType(),
                                      {getInt32Type(), getInt32Type(),
                                       getFloatType().getPointerTo(),
                                       getInt64Type()},
                                      /*isVarArg=*/false));
  }

  if (!module.lookupSymbol(kInitVulkan)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kInitVulkan,
        LLVM::LLVMType::getFunctionTy(getPointerType(), {},
                                      /*isVarArg=*/false));
  }
  if (!module.lookupSymbol(kDeinitVulkan)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kDeinitVulkan,
        LLVM::LLVMType::getFunctionTy(getVoidType(), {getPointerType()},
                                      /*isVarArg=*/false));
  }
}

Value VulkanLaunchFuncToVulkanCallsPass::createEntryPointNameConstant(
    StringRef name, Location loc, OpBuilder &builder) {
  SmallString<16> shaderName(name.begin(), name.end());
  // Append `\0` to follow C style string given that LLVM::createGlobalString()
  // won't handle this directly for us.
  shaderName.push_back('\0');

  std::string entryPointGlobalName = (name + "_spv_entry_point_name").str();
  return LLVM::createGlobalString(loc, builder, entryPointGlobalName,
                                  shaderName, LLVM::Linkage::Internal,
                                  getLLVMDialect());
}

void VulkanLaunchFuncToVulkanCallsPass::translateVulkanLaunchCall(
    LLVM::CallOp vulkanLaunchCallOp) {
  OpBuilder builder(vulkanLaunchCallOp);
  Location loc = vulkanLaunchCallOp.getLoc();

  auto spirvBinaryShaderAttr =
      vulkanLaunchCallOp.getAttrOfType<StringAttr>(kSPIRVBinary);
  if (!spirvBinaryShaderAttr) {
    // TODO: emit error
    signalPassFailure();
  }

  auto entryPointNameAttr =
      vulkanLaunchCallOp.getAttrOfType<StringAttr>(kSPIRVEntryPoint);
  if (!entryPointNameAttr) {
    // TODO: emit error
    signalPassFailure();
  }

  // Create LLVM global with SPIR-V binary data, so we can pass a pointer with
  // that data to runtime call.
  Value ptrToSPIRVBinary = LLVM::createGlobalString(
      loc, builder, kSPIRVBinary, spirvBinaryShaderAttr.getValue(),
      LLVM::Linkage::Internal, getLLVMDialect());

  // Create LLVM constant for the size of SPIR-V binary shader.
  Value binarySize = builder.create<LLVM::ConstantOp>(
      loc, getInt32Type(),
      builder.getI32IntegerAttr(spirvBinaryShaderAttr.getValue().size()));

  bindResources(vulkanLaunchCallOp);

  // Create call to `setBinaryShader` runtime function with the given pointer to
  // SPIR-V binary and binary size.
  builder.create<LLVM::CallOp>(loc, ArrayRef<Type>{getVoidType()},
                               builder.getSymbolRefAttr(kSetBinaryShader),
                               ArrayRef<Value>{ptrToSPIRVBinary, binarySize});
  // Create LLVM global with entry point name.
  Value entryPointName =
      createEntryPointNameConstant(entryPointNameAttr.getValue(), loc, builder);
  // Create call to `setEntryPoint` runtime function with the given pointer to
  // entry point name.
  builder.create<LLVM::CallOp>(loc, ArrayRef<Type>{getVoidType()},
                               builder.getSymbolRefAttr(kSetEntryPoint),
                               ArrayRef<Value>{entryPointName});

  // Create number of local workgroup for each dimension.
  // local workgroup.
  builder.create<LLVM::CallOp>(
      loc, ArrayRef<Type>{getVoidType()},
      builder.getSymbolRefAttr(kSetNumWorkGroups),
      ArrayRef<Value>{vulkanLaunchCallOp.getOperand(0),
                      vulkanLaunchCallOp.getOperand(1),
                      vulkanLaunchCallOp.getOperand(2)});

  // Create call to `runOnVulkan` runtime function.
  builder.create<LLVM::CallOp>(loc, ArrayRef<Type>{getVoidType()},
                               builder.getSymbolRefAttr(kRunOnVulkan),
                               ArrayRef<Value>{});

  // Declare runtime functions.
  declareVulkanFunctions(loc);

  // erase op.
  vulkanLaunchCallOp.erase();
}

std::unique_ptr<mlir::OpPassBase<mlir::ModuleOp>>
mlir::createConvertVulkanLaunchFuncToVulkanCallsPass() {
  return std::make_unique<VulkanLaunchFuncToVulkanCallsPass>();
}

static PassRegistration<VulkanLaunchFuncToVulkanCallsPass>
    pass("launch-func-to-vulkan",
         "Convert vulkan launch_func op to Vulkan runtime calls");
