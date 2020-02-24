//===- vulkan-runtime-wrappers.cpp - MLIR Vulkan runner wrapper library ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements C runtime wrappers around the VulkanRuntime.
//
//===----------------------------------------------------------------------===//

#include <mutex>
#include <numeric>

#include "VulkanRuntime.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/raw_ostream.h"

namespace {

// TODO(denis0x0D): This static machinery should be replaced by `initVulkan` and
// `deinitVulkan` to be more explicit and to avoid static initialization and
// destruction.
class VulkanRuntimeManager;
static llvm::ManagedStatic<VulkanRuntimeManager> vkRuntimeManager;

class VulkanRuntimeManager {
  public:
    VulkanRuntimeManager() = default;
    VulkanRuntimeManager(const VulkanRuntimeManager &) = delete;
    VulkanRuntimeManager operator=(const VulkanRuntimeManager &) = delete;
    ~VulkanRuntimeManager() = default;

    void setResourceData(DescriptorSetIndex setIndex, BindingIndex bindIndex,
                         const VulkanHostMemoryBuffer &memBuffer) {
      std::lock_guard<std::mutex> lock(mutex);
      vulkanRuntime.setResourceData(setIndex, bindIndex, memBuffer);
    }

    void setEntryPoint(const char *entryPoint) {
      std::lock_guard<std::mutex> lock(mutex);
      vulkanRuntime.setEntryPoint(entryPoint);
    }

    void setNumWorkGroups(NumWorkGroups numWorkGroups) {
      std::lock_guard<std::mutex> lock(mutex);
      vulkanRuntime.setNumWorkGroups(numWorkGroups);
    }

    void setShaderModule(uint8_t *shader, uint32_t size) {
      std::lock_guard<std::mutex> lock(mutex);
      vulkanRuntime.setShaderModule(shader, size);
    }

    void runOnVulkan() {
      std::lock_guard<std::mutex> lock(mutex);
      if (failed(vulkanRuntime.initRuntime()) || failed(vulkanRuntime.run()) ||
          failed(vulkanRuntime.updateHostMemoryBuffers()) ||
          failed(vulkanRuntime.destroy())) {
        llvm::errs() << "runOnVulkan failed";
      }
    }

  private:
    VulkanRuntime vulkanRuntime;
    std::mutex mutex;
};

} // namespace

extern "C" {

void *initVulkan() { return new VulkanRuntimeManager(); }

void deinitVulkan(void *vkManager) {
  if (vkManager)
    delete reinterpret_cast<VulkanRuntimeManager *>(vkManager);
}

/// Binds the given memref to the given descriptor set and descriptor index.
void bindResource(DescriptorSetIndex setIndex, BindingIndex bindIndex,
                  float *ptr, int64_t size) {
  VulkanHostMemoryBuffer memBuffer{ptr,
                                   static_cast<uint32_t>(size * sizeof(float))};
  vkRuntimeManager->setResourceData(setIndex, bindIndex, memBuffer);
}

void runOnVulkan() { vkRuntimeManager->runOnVulkan(); }

/// Fills the given memref with the given value.
void fillResource1DFloat(float *allocated, float *aligned, int64_t offset,
                         int64_t size, int64_t stride, float value) {
  if (!allocated)
    llvm::errs() << "passed nullptr";
  std::fill_n(allocated, size, value);
}

void setEntryPoint(const char *entryPoint) {
  vkRuntimeManager->setEntryPoint(entryPoint);
}

void setNumWorkGroups(uint32_t x, uint32_t y, uint32_t z) {
  vkRuntimeManager->setNumWorkGroups({x, y, z});
}

void setBinaryShader(uint8_t *shader, uint32_t size) {
  vkRuntimeManager->setShaderModule(shader, size);
}
}
