// RUN: mlir-vulkan-runner %s --shared-libs=%vulkan_wrapper_library_dir/libvulkan-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s

// CHECK: [3.3,  3.3,  3.3,  3.3,  3.3,  3.3,  3.3,  3.3]
module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @kernel_add(%arg0 : memref<8xf32>, %arg1 : memref<8xf32>, %arg2 : memref<8xf32>) attributes {gpu.kernel} {
      %0 = "gpu.block_id"() {dimension = "x"} : () -> index
      %1 = load %arg0[%0] : memref<8xf32>
      %2 = load %arg1[%0] : memref<8xf32>
      %3 = addf %1, %2 : f32
      store %3, %arg2[%0] : memref<8xf32>
      gpu.return
    }
  }

  func @main() {
    %arg0 = alloc() : memref<8xf32>
    %arg1 = alloc() : memref<8xf32>
    %arg2 = alloc() : memref<8xf32>
    %0 = constant 0 : i32
    %1 = constant 1 : i32
    %2 = constant 2 : i32
    %value0 = constant 0.0 : f32
    %value1 = constant 1.1 : f32
    %value2 = constant 2.2 : f32
    call @setResourceData(%0, %0, %arg0, %value1) : (i32, i32, memref<8xf32>, f32) -> ()
    call @setResourceData(%0, %1, %arg1, %value2) : (i32, i32, memref<8xf32>, f32) -> ()
    call @setResourceData(%0, %2, %arg2, %value0) : (i32, i32, memref<8xf32>, f32) -> ()

    %cst1 = constant 1 : index
    %cst8 = constant 8 : index
    "gpu.launch_func"(%cst8, %cst1, %cst1, %cst1, %cst1, %cst1, %arg0, %arg1, %arg2) { kernel = "kernel_add", kernel_module = @kernels }
        : (index, index, index, index, index, index, memref<8xf32>, memref<8xf32>, memref<8xf32>) -> ()

    %4 = memref_cast %arg2 : memref<8xf32> to memref<?xf32>
    call @print_memref_1d_f32(%4) : (memref<?xf32>) -> ()
    return
  }
  func @setResourceData(%0 : i32, %1 : i32, %arg2 : memref<8xf32>, %value : f32)
  func @print_memref_1d_f32(memref<?xf32>)
}
