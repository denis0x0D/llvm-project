if(MLIR_VULKAN_RUNNER_ENABLED)
  add_llvm_library(MLIRGPUtoVulkanTransforms
    ConvertLaunchFuncToVulkanCalls.cpp
    )

  target_link_libraries(MLIRGPUtoVulkanTransforms
    MLIRGPU
    MLIRIR
    MLIRLLVMIR
    MLIRPass
    MLIRSPIRV
    MLIRSPIRVSerialization
    MLIRStandardOps
    MLIRSupport
    MLIRTransforms
    )
endif()
