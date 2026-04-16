# Lightweight Human Pose Estimation on AMD (ROCm & MIGraphX)

This repository is an optimized version of the Lightweight OpenPose project, specifically tailored for the AMD ROCm ecosystem. While the original work focused on CPU inference, this version achieves 200+ FPS on AMD Strix Halo hardware by utilizing MIGraphX and advanced memory optimization techniques.

## Key Improvements (AMD Optimization)

Compared to the original repository, the following changes have been implemented:

    ROCm Port: Migrated from PyTorch 0.4.1/Python 3.6 to the latest PyTorch with ROCm support and Python 3.10+.

    MIGraphX Backend: The model is compiled for AMD’s graph inference engine, achieving a baseline of 150 FPS.

    Zero-Copy Memory Opt: Eliminated a bottleneck where data was being copied between the CPU and GPU six times. Intermediate results (Heatmaps & PAFs) now remain in GPU memory.

    Kernel Exhaustive Search: Identification of the fastest mathematical kernels specifically for the Strix Halo architecture.

    Seamless Quantization: Implemented a unified codebase for FP32, FP16, BF16, and INT8 (both static and dynamic).

## Benchmark Results (Strix Halo)

Following optimization, FP16 proved to be the optimal choice for this hardware:
|Optimization Phase |Backend |Precision|Throughput (FPS)|Latency (ms)|
|------------------|--------|---------|----------------|------------|
|Initial Port      |PyTorch |FP32     |8               |125.0       |
|Compiled          |PyTorch |FP16     |91.8            |10.89       |
|MIGraphX          |MIGraphX|FP16     |150             |6.6         |
|Final (Memory Opt)|MIGraphX|FP16     |220+            |~4.5        |

Accuracy (COCO Keypoints val2017)
|Model Version     |AP @ 0.5:0.95|AP @ 0.5|AR @ 0.5:0.95|
|------------------|-------------|--------|-------------|
|Original Paper    |0.400        |-       |-            |
|Our Refinement 2  |0.458        |0.698   |0.513        |

2. Profile of fp16 with 1ref step

-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                          ProfilerStep*         0.00%       0.000us         0.00%       0.000us       0.000us      20.963ms       107.10%      20.963ms       2.096ms            10  
                    mlir_convolution_broadcast_add_relu         0.00%       0.000us         0.00%       0.000us       0.000us      15.227ms        77.79%      15.227ms      78.085us           195  
                mlir_convolution_broadcast_add_relu_add         0.00%       0.000us         0.00%       0.000us       0.000us       2.101ms        10.73%       2.101ms      84.022us            25  
                                       mlir_convolution         0.00%       0.000us         0.00%       0.000us       0.000us     749.703us         3.83%     749.703us      24.990us            30  
                                          ProfilerStep*         3.52%     946.228us       100.00%      26.909ms       5.382ms       0.000us         0.00%     502.377us     100.475us             5  
                                               aten::to         0.05%      12.949us         8.95%       2.409ms     240.887us       0.000us         0.00%     502.377us      50.238us            10  
                                         aten::_to_copy         0.11%      30.589us         8.90%       2.396ms     479.183us       0.000us         0.00%     502.377us     100.475us             5  
                                            aten::copy_         0.13%      35.419us         8.67%       2.332ms     466.436us     502.377us         2.57%     502.377us     100.475us             5  
                           Memcpy DtoH (Device -> Host)         0.00%       0.000us         0.00%       0.000us       0.000us     502.377us         2.57%     502.377us     100.475us             5  
                      _migraphxgpudevicelauncherIZZN...         0.00%       0.000us         0.00%       0.000us       0.000us     389.653us         1.99%     389.653us      25.977us            15  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 26.910ms
Self CUDA time total: 19.575ms


## Installation and Setup

1. Prepare COCO Dataset:\
Extract COCO 2017 into the <COCO_HOME> folder.

2. Environment:
```
    python3 -m venv venv
    source venv/bin/activate
    # Install ROCm PyTorch
    pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
```
3. Run Inference:
```
    python demo.py --checkpoint-path <CHECKPOINT> --mode fp16 --compile
```
## Deep Dive: Lessons Learned

Mixed Precision: We found that mixed_fp16 can actually be slower due to data conversion overhead on this hardware. Native FP16 is twice as fast as FP32 with negligible loss in accuracy.

PCIe Bottleneck: Profiling revealed that sending intermediate heatmaps back to the CPU slowed the system down by 50 FPS.

Quantization: Strix Halo hardware shows similar performance for INT8 and FP16; FP16 was chosen for its superior stability and precision.

## Citations and References

This project is based on:

    Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose
    Daniil Osokin, 2018. arXiv:1811.12004

Original Project: osokin/lightweight-human-pose-estimation.pytorch