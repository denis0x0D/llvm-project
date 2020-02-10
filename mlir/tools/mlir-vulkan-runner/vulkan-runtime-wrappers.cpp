//===- vulkan-runtime-wrappers.cpp - MLIR Vulkan runner wrapper library ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <mutex>
#include <numeric>

#include "llvm/Support/raw_ostream.h"
#include "VulkanRuntime.h"

/// This class represents a bridge between VulkanRuntime and C style runtime
/// wrappers. It's designed to handle single kernel and the main purpose to
/// test a progressive lowering to SPIR-V.
class VulkanRuntimeManager {
  public:
    VulkanRuntimeManager(const VulkanRuntimeManager &) = delete;
    VulkanRuntimeManager operator=(const VulkanRuntimeManager &) = delete;
    ~VulkanRuntimeManager() = default;

    static VulkanRuntimeManager *instance() {
      static VulkanRuntimeManager *runtimeManager = new VulkanRuntimeManager;
      return runtimeManager;
    }

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
      vulkanRuntime.initRuntime();
      vulkanRuntime.run();
      vulkanRuntime.updateHostMemoryBuffers();
      vulkanRuntime.destroy();
    }

  private:
    VulkanRuntimeManager() = default;
    VulkanRuntime vulkanRuntime;
    std::mutex mutex;
};

template <typename T, int N>
struct MemRefT {
  T *basePtr;
  T *data;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

extern "C" {
void setResourceData(const DescriptorSetIndex setIndex, BindingIndex bindIndex,
                     const MemRefT<float, 1> *memRef, float value) {
  std::fill_n(memRef->data, memRef->sizes[0], value);
  VulkanHostMemoryBuffer memBuffer{
      memRef->data, static_cast<uint32_t>(memRef->sizes[0] * sizeof(float))};
  VulkanRuntimeManager::instance()->setResourceData(setIndex, bindIndex,
                                                    memBuffer);
}

void setEntryPoint(const char *entryPoint) {
  VulkanRuntimeManager::instance()->setEntryPoint(entryPoint);
}

void setNumWorkGroups(uint32_t x, uint32_t y, uint32_t z) {
  VulkanRuntimeManager::instance()->setNumWorkGroups({x, y, z});
}

void setBinaryShader(uint8_t *shader, uint32_t size) {
  VulkanRuntimeManager::instance()->setShaderModule(shader, size);
}

void runOnVulkan() { 
  VulkanRuntimeManager::instance()->runOnVulkan(); 
}
}
