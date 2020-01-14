module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @kernel_1(%arg0 : f32, %arg1 : memref<12xf32>) attributes {gpu.kernel} {
      gpu.return
    }
  }
}
