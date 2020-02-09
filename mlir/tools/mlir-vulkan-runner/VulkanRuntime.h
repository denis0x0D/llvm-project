//===- VulkanRuntime.cpp - MLIR Vulkan runtime ------------------*- C++ -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file provides a library for running a module on a Vulkan device.
// Implements a Vulkan runtime to run a spirv::ModuleOp. It also defines a few
// utility functions to extract information from a spirv::ModuleOp.
//
//===----------------------------------------------------------------------===//
#ifndef VULKAN_RUNTIME_H
#define VULKAN_RUNTIME_H

#include "mlir/Analysis/Passes.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Serialization.h"
#include "mlir/IR/Module.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/StringExtras.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/ToolOutputFile.h"

#include <vulkan/vulkan.h>

using namespace mlir;

using DescriptorSetIndex = uint32_t;
using BindingIndex = uint32_t;

/// Struct containing information regarding to a device memory buffer.
struct VulkanDeviceMemoryBuffer {
  BindingIndex bindingIndex{0};
  VkDescriptorType descriptorType{VK_DESCRIPTOR_TYPE_MAX_ENUM};
  VkDescriptorBufferInfo bufferInfo{VK_NULL_HANDLE};
  VkBuffer buffer{VK_NULL_HANDLE};
  VkDeviceMemory deviceMemory{VK_NULL_HANDLE};
};

/// Struct containing information regarding to a host memory buffer.
struct VulkanHostMemoryBuffer {
  /// Pointer to a host memory.
  void *ptr{nullptr};
  /// Size of a host memory in bytes.
  uint32_t size{0};
};

/// Struct containing the number of local workgroups to dispatch for each
/// dimension.
struct NumWorkGroups {
  uint32_t x{1};
  uint32_t y{1};
  uint32_t z{1};
};

/// Struct containing information regarding a descriptor set.
struct DescriptorSetInfo {
  /// Index of a descriptor set in descriptor sets.
  DescriptorSetIndex descriptorSet{0};
  /// Number of desriptors in a set.
  uint32_t descriptorSize{0};
  /// Type of a descriptor set.
  VkDescriptorType descriptorType{VK_DESCRIPTOR_TYPE_MAX_ENUM};
};

/// VulkanHostMemoryBuffer mapped into a descriptor set and a binding.
using ResourceData =
    llvm::DenseMap<DescriptorSetIndex,
                   llvm::DenseMap<BindingIndex, VulkanHostMemoryBuffer>>;

/// StorageClass mapped into a descriptor set and a binding.
using ResourceStorageClassData =
    llvm::DenseMap<DescriptorSetIndex,
                   llvm::DenseMap<BindingIndex, mlir::spirv::StorageClass>>;

inline void emitVulkanError(const llvm::Twine &message, VkResult error) {
  llvm::errs()
      << message.concat(" failed with error code ").concat(llvm::Twine{error});
}

#define RETURN_ON_VULKAN_ERROR(result, msg)                                    \
  if ((result) != VK_SUCCESS) {                                                \
    emitVulkanError(msg, (result));                                            \
    return failure();                                                          \
  }

/// Vulkan runtime.
/// The purpose of this class is to run SPIR-V computation shader on Vulkan
/// device.
/// Before the run, user must provide and set resource data with descriptors,
/// spir-v shader, number of work groups and entry point. After the creation of
/// VulkanRuntime, special methods must be called in the following
/// sequence: initRuntime(), run(), updateHostMemoryBuffers(), destroy();
/// each method in the sequence returns succes or failure depends on the Vulkan
/// result code.
class VulkanRuntime {
public:
  explicit VulkanRuntime() = default;
  VulkanRuntime(const VulkanRuntime &) = delete;
  VulkanRuntime &operator=(const VulkanRuntime &) = delete;

  /// Sets needed data for Vulkan runtime.
  void setResourceData(const ResourceData &resData);
  void setResourceData(const DescriptorSetIndex desIndex,
                       const BindingIndex bindIndex,
                       const VulkanHostMemoryBuffer &hostMemBuffer);
  void setShaderModule(llvm::ArrayRef<uint32_t> binaryRef);
  void setNumWorkGroups(const NumWorkGroups &nWorkGroups);
  void setResourceStorageClassData(const ResourceStorageClassData &stClassData);
  void setEntryPoint(llvm::StringRef entryPointName);

  /// Runtime initialization.
  LogicalResult initRuntime();

  /// Runs runtime.
  LogicalResult run();

  /// Updates host memory buffers.
  LogicalResult updateHostMemoryBuffers();

  /// Destroys all created vulkan objects and resources.
  LogicalResult destroy();

private:
  //===--------------------------------------------------------------------===//
  // Pipeline creation methods.
  //===--------------------------------------------------------------------===//

  LogicalResult createInstance();
  LogicalResult createDevice();
  LogicalResult getBestComputeQueue(const VkPhysicalDevice &physicalDevice);
  LogicalResult createMemoryBuffers();
  LogicalResult createShaderModule();
  void initDescriptorSetLayoutBindingMap();
  LogicalResult createDescriptorSetLayout();
  LogicalResult createPipelineLayout();
  LogicalResult createComputePipeline();
  LogicalResult createDescriptorPool();
  LogicalResult allocateDescriptorSets();
  LogicalResult setWriteDescriptors();
  LogicalResult createCommandPool();
  LogicalResult createComputeCommandBuffer();
  LogicalResult submitCommandBuffersToQueue();

  //===--------------------------------------------------------------------===//
  // Helper methods.
  //===--------------------------------------------------------------------===//

  /// Maps storage class to a descriptor type.
  LogicalResult
  mapStorageClassToDescriptorType(spirv::StorageClass storageClass,
                                  VkDescriptorType &descriptorType);

  /// Maps storage class to buffer usage flags.
  LogicalResult
  mapStorageClassToBufferUsageFlag(spirv::StorageClass storageClass,
                                   VkBufferUsageFlagBits &bufferUsage);

  LogicalResult countDeviceMemorySize();

  //===--------------------------------------------------------------------===//
  // Vulkan objects.
  //===--------------------------------------------------------------------===//

  VkInstance instance;
  VkDevice device;
  VkQueue queue;

  /// Specifies VulkanDeviceMemoryBuffers divided into sets.
  llvm::DenseMap<DescriptorSetIndex,
                 llvm::SmallVector<VulkanDeviceMemoryBuffer, 1>>
      deviceMemoryBufferMap;

  /// Specifies shader module.
  VkShaderModule shaderModule;

  /// Specifies layout bindings.
  llvm::DenseMap<DescriptorSetIndex,
                 llvm::SmallVector<VkDescriptorSetLayoutBinding, 1>>
      descriptorSetLayoutBindingMap;

  /// Specifies layouts of descriptor sets.
  llvm::SmallVector<VkDescriptorSetLayout, 1> descriptorSetLayouts;
  VkPipelineLayout pipelineLayout;

  /// Specifies descriptor sets.
  llvm::SmallVector<VkDescriptorSet, 1> descriptorSets;

  /// Specifies a pool of descriptor set info, each descriptor set must have
  /// information such as type, index and amount of bindings.
  llvm::SmallVector<DescriptorSetInfo, 1> descriptorSetInfoPool;
  VkDescriptorPool descriptorPool;

  /// Computation pipeline.
  VkPipeline pipeline;
  VkCommandPool commandPool;
  llvm::SmallVector<VkCommandBuffer, 1> commandBuffers;

  //===--------------------------------------------------------------------===//
  // Vulkan memory context.
  //===--------------------------------------------------------------------===//

  uint32_t queueFamilyIndex{0};
  uint32_t memoryTypeIndex{VK_MAX_MEMORY_TYPES};
  VkDeviceSize memorySize{0};

  //===--------------------------------------------------------------------===//
  // Vulkan execution context.
  //===--------------------------------------------------------------------===//

  NumWorkGroups numWorkGroups;
  std::string entryPoint;
  llvm::SmallVector<uint32_t, 0> binary;

  //===--------------------------------------------------------------------===//
  // Vulkan resource data and storage classes.
  //===--------------------------------------------------------------------===//

  ResourceData resourceData;
  ResourceStorageClassData resourceStorageClassData;
};
#endif
