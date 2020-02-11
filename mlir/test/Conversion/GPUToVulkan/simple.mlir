// RUN: mlir-opt %s -launch-func-to-vulkan | FileCheck %s

// CHECK: llvm.mlir.global internal constant @kernel_1_spv_entry_point_name
// CHECK: llvm.mlir.global internal constant @SPIRV_BIN
// CHECK: llvm.call @setBinaryShader(%{{.*}}, %{{.*}}) : (!llvm<"i8*">, !llvm.i32) -> !llvm.void
// CHECK: llvm.call @setEntryPoint(%{{.*}}) : (!llvm<"i8*">) -> !llvm.void
// CHECK: llvm.call @setNumWorkGroups(%{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.i32, !llvm.i32, !llvm.i32) -> !llvm.void
// CHECK: llvm.call @runOnVulkan() : () -> !llvm.void

module attributes {gpu.container_module} {
  spv.module "Logical" "GLSL450" {
    spv.globalVariable @kernel_1_arg_0 bind(0, 0) : !spv.ptr<!spv.struct<f32 [0]>, StorageBuffer>
    spv.globalVariable @kernel_1_arg_1 bind(0, 1) : !spv.ptr<!spv.struct<!spv.array<12 x f32 [4]> [0]>, StorageBuffer>
    func @kernel_1() attributes {workgroup_attributions = 0 : i64} {
      %0 = spv._address_of @kernel_1_arg_1 : !spv.ptr<!spv.struct<!spv.array<12 x f32 [4]> [0]>, StorageBuffer>
      %1 = spv._address_of @kernel_1_arg_0 : !spv.ptr<!spv.struct<f32 [0]>, StorageBuffer>
      %2 = spv.constant 0 : i32
      %3 = spv.AccessChain %1[%2] : !spv.ptr<!spv.struct<f32 [0]>, StorageBuffer>
      %4 = spv.Load "StorageBuffer" %3 : f32
      spv.Return
    }
    spv.EntryPoint "GLCompute" @kernel_1
    spv.ExecutionMode @kernel_1 "LocalSize", 1, 1, 1
  } attributes {capabilities = ["Shader"], extensions = ["SPV_KHR_storage_buffer_storage_class"]}
  gpu.module @kernels {
    gpu.func @kernel_1(%arg0: f32, %arg1: memref<12xf32>) kernel {
      gpu.return
    }
  }
  func @foo() {
    %0 = "op"() : () -> f32
    %1 = "op"() : () -> memref<12xf32>
    %c1 = constant 1 : index
    "gpu.launch_func"(%c1, %c1, %c1, %c1, %c1, %c1, %0, %1) {kernel = "kernel_1", kernel_module = @kernels} : (index, index, index, index, index, index, f32, memref<12xf32>) -> ()
    return
  }
}
