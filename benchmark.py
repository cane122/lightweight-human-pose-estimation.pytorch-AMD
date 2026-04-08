import torch
import time
import subprocess
import numpy as np
import json
import cv2
from datasets.coco import CocoValDataset
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state
import torch.profiler
import csv 

FINAL_RESULTS = []
base_height = 544
base_width = 968

def calibrate_model(model, args):
    print("--- Starting Calibration (Static PTQ) ---")
    dataset = CocoValDataset(args.labels, args.images_folder)
    with torch.no_grad():
        for i in range(min(135, len(dataset))):
            img = dataset[i]['img']
            img_resized = cv2.resize(img, (base_width, base_height))
            img_mean = 128
            img_scale = 1/256
            normalized_img = (img.astype(np.float32) - img_mean) * img_scale
            input_tensor = torch.from_numpy(normalized_img).permute(2, 0, 1).unsqueeze(0).float()
            model(input_tensor)
            if i % 10 == 0:
                print(f"Calibrating image {i}/135...")
    print("--- Calibration Complete ---")

import torch
from torchao.quantization import quantize_, Int8WeightOnlyConfig, Int8DynamicActivationInt8WeightConfig

def load_model(args, device):
    # Initialize base model
    net = PoseEstimationWithMobileNet(num_refinement_stages=args.num_refinement_stages)
    net.eval()
    
    # Load standard weights first
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    load_state(net, checkpoint)
    net = net.to(device)

    print(f"--- Applying torchcao quantization: {args.quantization} ---")

    if args.quantization == 'int8':
        # apply int8 weight-only quantization
        # This is fast and doesn't require a calibration dataset
        quantize_(net, Int8WeightOnlyConfig())
        print("Applied torchcao int8_weight_only quantization.")

    elif args.quantization == 'int8_dynamic':
        # More advanced: dynamic quantization for activations + weights
        quantize_(net, Int8DynamicActivationInt8WeightConfig())
        print("Applied torchcao int8_dynamic_activation_int8_weight quantization.")

    elif args.quantization == 'fp16':
        net.half()
        print("Running in FP16 (Half Precision).")

    elif args.quantization == 'bf16':
        net.to(torch.bfloat16)
        print("Running in BF16 (BFloat16).")
    
    # Optional: Compile for extra speedup (Torch 2.0+)
    # net = torch.compile(net, mode="reduce-overhead")
    
    return net.eval()

def get_gpu_power():
    try:
        res = subprocess.check_output(['rocm-smi', '--showpower', '--json']).decode('utf-8')
        data = json.loads(res)
        power_str = data['card0']['Current Socket Graphics Package Power (W)']
        return float(power_str)
    except Exception:
        return 0.0

def benchmark(args, iterations=100, warm_up=20, compile_model=True, profiler=False):
 
    device = torch.device('cuda')
    if args.quantization == 'fp16':
        input_dtype = torch.float16
    elif args.quantization == 'bf16':
        input_dtype = torch.bfloat16
    else:
        # Default for int8 weight-only and fp32 is float32
        input_dtype = torch.float32

    print(f"\n--- Benchmarking: {args.quantization.upper()} on {device} ---")
    net = load_model(args, device)

    if compile_model and device.type == 'cuda':
        print("Compiling model with torch.compile...")
        net = torch.compile(net, mode="reduce-overhead")

    dummy_input = torch.randn(1, 3, base_width, base_height, dtype=input_dtype).to(device)

    print(f"Warming up...")
    with torch.no_grad():
        for _ in range(warm_up):
            _ = net(dummy_input)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()

    if profiler:
        print("\n--- Running Profiler ---")
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=5, active=5),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/pose_estimation'),
            record_shapes=True,
            with_stack=True
        ) as prof:
            with torch.no_grad():
                for _ in range(11): # Matches schedule (1+5+5)
                    _ = net(dummy_input)
                    prof.step()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    latencies = []
    powers = []
    
    start_benchmark = time.perf_counter()
    with torch.no_grad():
        for i in range(iterations):
            iter_start = time.perf_counter()
            
            _ = net(dummy_input)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            latencies.append(time.perf_counter() - iter_start)
            
            if i % 10 == 0 and device.type == 'cuda':
                powers.append(get_gpu_power())

    total_bench_time = time.perf_counter() - start_benchmark
    
    avg_latency = np.mean(latencies) * 1000
    fps = 1.0 / np.mean(latencies)
    avg_power = np.mean(powers) if powers else 0
    res = {
        "Stages": args.num_refinement_stages,
        "Mode": args.quantization,
        "Latency (ms)": f"{avg_latency:.2f}",
        "Throughput (FPS)": f"{fps:.2f}",
        "Power (W)": f"{avg_power:.2f}" if device.type == 'cuda' else "N/A", 
        "Compiled": compile_model if device.type == 'cuda' else "N/A"
    }
    FINAL_RESULTS.append(res)
    print(f"DONE: {fps:.2f} FPS")
    print(f"Results for {args.quantization}:")
    print(f"  - Avg Latency: {avg_latency:.2f} ms")
    print(f"  - Throughput:  {fps:.2f} FPS")
    if device.type == 'cuda':
        print(f"  - Avg Power:   {avg_power:.2f} W")
        print(f"  - Efficiency:  {fps/avg_power if avg_power > 0 else 0:.2f} FPS/Watt")
    else:
        print(f"  - Note: Power reading skipped for CPU mode.")

import argparse

def create_args(mode, refinement_stages):

    ckpt = "models/checkpoint_iter_370000.pth"

    return argparse.Namespace(
        quantization=mode,
        num_refinement_stages=refinement_stages,
        checkpoint_path=ckpt,
        labels='coco/annotations/person_keypoints_val2017.json', 
        images_folder='coco/val2017/'
    )

if __name__ == '__main__':
    target_refinements = [1, 2]
    target_modes = ['fp32', 'fp16', 'bf16', 'int8', 'mixed_fp32', 'mixed_fp16']
    target_modes = ['int8']
    
    for ref in target_refinements:
        print(f"\n{'='*20}")
        print(f" TESTING REFINEMENT STAGES: {ref} ")
        print(f"{'='*20}")
        
        for mode in target_modes:
            try:
                args = create_args(mode, ref)
                benchmark(args, iterations=50, compile_model=True, profiler=True)
            except Exception as e:
                print(f"FAILED [Mode: {mode}, Ref: {ref}]: {e}")

    print("\n" + "="*50)
    print(f"{'STAGES':<8} | {'MODE':<10} | {'LATENCY':<10} | {'FPS':<10}")
    print("-" * 50)
    for r in FINAL_RESULTS:
        print(f"{r['Stages']:<8} | {r['Mode']:<10} | {r['Latency (ms)']:<10} | {r['Throughput (FPS)']:<10} | {r['Power (W)']:<10}")
    
    with open('benchmark_results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FINAL_RESULTS[0].keys())
        writer.writeheader()
        writer.writerows(FINAL_RESULTS)
    print(f"\nResults saved to benchmark_results.csv")