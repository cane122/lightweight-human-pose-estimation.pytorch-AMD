# Lightweight Human Pose Estimation on AMD (ROCm & MIGraphX)

This repository is an optimized version of the Lightweight OpenPose project, specifically tailored for the AMD ROCm ecosystem. By leveraging the AMD Strix Halo APU’s unified memory and MIGraphX inference engine, this implementation achieves real-time multi-person tracking suitable for edge deployment in high-traffic venues.

## Key Improvements (AMD Optimization)

Compared to the original repository, the following changes have been implemented:

1. ROCm 7.2+ Port: Migrated to Python 3.10+ and the latest PyTorch ROCm binaries.

2. MIGraphX Backend: Model compiled specifically for RDNA 3.5, jumping from ~8 FPS (Native) to ~205 FPS.

3. Zero-Copy Memory Opt: Reduced 6x redundant PCIe copies between CPU/GPU to 2x. Intermediate Heatmaps & PAFs remain in unified GPU memory.

4. Kernel Exhaustive Search: Identification of the fastest mathematical kernels specifically for the Strix Halo architecture.

5. Seamless Quantization: Implemented a unified codebase for FP32, FP16, BF16, and INT8 (both static and dynamic).

## Benchmark Results (Strix Halo)

Following extensive optimization, FP16 with 1 refinement stage emerged as the "sweet spot" for balancing throughput and precision on the Strix Halo architecture.

|Optimization Phase |Backend |Precision|Throughput (FPS)|Latency (ms)| 
|------------------|--------|---------|----------------|------------|
|Initial Port      |PyTorch |FP16     |8.02               |124.66       |
|MIGraphX          |MIGraphX|FP16     |148.31            |6.74       |
|Improved model perf|MIGraphX|FP16     |200            |~4.5        |
|Final (Exhaustive kernel)|MIGraphX|FP16     |205.18            |~4.5        |

### Key Observations:

Backend Impact: The initial PyTorch port using native kernels was severely bottlenecked (INT8 peaked at only 28.58 FPS). Migrating to MIGraphX provided an immediate 18x performance boost.

Quantization Choice: While FP32 performed respectably at 82.07 FPS, FP16 reached 148.31 FPS. Testing with INT8 and BF16 showed no significant gains in throughput over FP16, leading us to standardize on FP16 for the best balance of speed and stability.

IR & Memory Optimization: Profiling the Intermediate Representation (IR) revealed six redundant PCIe copy operations. By optimizing the data flow to retain heatmaps and Part Affinity Fields (PAFs) in GPU memory, we significantly reduced overhead.

Kernel Tuning: We utilized MIOpen exhaustive search to tune convolution algorithms. By pre-selecting optimal tiling sizes and memory layouts for RDNA 3.5 Compute Units, we shaved an additional 1.5ms off the tail latency.

## Accuracy Analysis

Accuracy testing demonstrated that FP16 maintains parity with FP32, whereas INT8 suffers from significant precision degradation.

|Model Variant|Precision|AP @ 0.5:0.95|AR @ 0.5:0.95|Notes                       |
|-------------|---------|-------------|-------------|----------------------------|
|Refinement 1 |FP32     |0.436        |0.490        |Baseline                    |
|             |FP16     |0.428        |0.482        |< 1% drop from FP32         |
|             |INT8     |0.274        |0.313        |Significant precision loss  |
|Refinement 2 |FP32     |0.458        |0.513        |                            |
|             |FP16     |0.453        |0.508        |Optimal performance/accuracy|
|             |INT8     |0.296        |0.336        |                            |
|Refinement 3 |FP32     |0.461        |0.518        |Peak accuracy               |
|             |FP16     |0.456        |0.510        |                            |



Power Efficiency Final
|Precision         |Avg. Power (Watts)|Throughput (FPS)|Efficiency (FPS/Watt)|
|------------------|-------------|--------|-------------|
|FP16 (1 Stage) |	50.04W	|205.18	|4.10
|FP16 (2 Stages)|	43.25W|	207.02|	4.78


### Performance Profiling

Below is the execution profile for FP16 (1 refinement step) processing five 968 × 544 images. The profile highlights the dominance of fused MLIR kernels (convolution + ReLU) and the minimized impact of Device-to-Host (DtoH) copies.

-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  

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


## Installation and Setup

1. Prepare COCO Dataset:\
Download from cocodataset.org [https://cocodataset.org/#home\]
Extract COCO 2017 into the <COCO_HOME> folder.

2. Environment:
```
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements/requirements.txt
    https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html
    # Export ROCm Library Path (Crucial for MIGraphX/ROCm 7.2.0)
    export PYTHONPATH=$PYTHONPATH:/opt/rocm-7.2.0/lib
```
3. Example validation script:
```
    python3 val.py \
    --checkpoint-path models/int8_2ref.pth \
    --labels coco/annotations/person_keypoints_val2017.json \
    --images-folder coco/val2017 \
    --quantization int8 \
    --num-refinement-stages 2 \
    --output-name detections_int8.json
```

## Citations and References

This project is based on:

    Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose
    Daniil Osokin, 2018. arXiv:1811.12004

Original Project: osokin/lightweight-human-pose-estimation.pytorch