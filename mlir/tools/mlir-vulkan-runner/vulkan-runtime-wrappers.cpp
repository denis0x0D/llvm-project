//===- vulkan-runtime-wrappers.cpp - MLIR Vulkan runner wrapper library ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <numeric>
#include <iostream>

#include "llvm/Support/raw_ostream.h"
#include "VulkanRuntime.h"

class VulkanRuntimeManager {
  public:
    static VulkanRuntimeManager *instance() {
      static VulkanRuntimeManager *runtimeManager = new VulkanRuntimeManager;
      return runtimeManager;
    }

    void setResourceData(DescriptorSetIndex setIndex, BindingIndex bindIndex,
                         const VulkanHostMemoryBuffer &memBuffer) {
      vulkanRuntime.setResourceData(setIndex, bindIndex, memBuffer);
    }

    void setEntryPoint(llvm::StringRef entryPoint) {
      vulkanRuntime.setEntryPoint(entryPoint);
    }

    void setNumWorkGroups(NumWorkGroups numWorkGroups) {
      vulkanRuntime.setNumWorkGroups(numWorkGroups);
    }

    void setShaderModule(uint8_t *shader, uint32_t size) {
      vulkanRuntime.setShaderModule(shader, size);
    }

    void runOnVulkan() {
      vulkanRuntime.initRuntime();
      vulkanRuntime.run();
      vulkanRuntime.updateHostMemoryBuffers();
      vulkanRuntime.destroy();
    }

  private:
    VulkanRuntimeManager() {}
    VulkanRuntime vulkanRuntime;
};

template <typename T, int N>
struct MemRef {
  T *basePtr;
  T *data;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

extern "C" {
void setResourceData(const DescriptorSetIndex setIndex, BindingIndex bindIndex,
                     const MemRef<float, 1> *memRef) {
  std::fill_n(memRef->data, memRef->sizes[0], 1.0);
  VulkanHostMemoryBuffer memBuffer{
      memRef->data, static_cast<uint32_t>(memRef->sizes[0] * sizeof(float))};
  VulkanRuntimeManager::instance()->setResourceData(setIndex, bindIndex,
                                                    memBuffer);
}

void printResourceData(const MemRef<float, 1> *memRef) {
  for (int i = 0; i < memRef->sizes[0]; ++i){
    std::cout << memRef->data[i] << " ";
  }
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
