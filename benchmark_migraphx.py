import torch
import time
import subprocess
import numpy as np
import json
import cv2
import torch.profiler
import csv
import argparse
import os

import migraphx
from datasets.coco import CocoValDataset
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state

FINAL_RESULTS = []
base_height = 544
base_width = 968

def load_model(args, device):
    onnx_path = "pose_model1.onnx"

    compiled_model_path = f"pose_model1_{args.quantization}.mxr"
    if os.path.exists(compiled_model_path):
        print(f"--- Loading pre-compiled model from {compiled_model_path} ---")
        return migraphx.load(compiled_model_path)

    print(f"--- Compiled model not found. Compiling {onnx_path} (Exhaustive Tune) ---")
    model = migraphx.parse_onnx(onnx_path)
    
    target = migraphx.get_target("gpu")

    if args.quantization == 'fp16':
        migraphx.quantize_fp16(model)
    elif args.quantization == 'int8':
        migraphx.quantize_int8(model, target, [])
    elif args.quantization == 'bf16':
        migraphx.quantize_bf16(model)
    
    model.compile(target, exhaustive_tune=True, fast_math=False, offload_copy=True)
    
    print(f"--- Saving compiled model to {compiled_model_path} ---")
    migraphx.save(model, compiled_model_path)

    return model

def get_gpu_power():
    try:
        res = subprocess.check_output(['rocm-smi', '--showpower', '--json']).decode('utf-8')
        data = json.loads(res)
        power_str = data['card0']['Current Socket Graphics Package Power (W)']
        return float(power_str)
    except Exception:
        return 0.0

def run_inference(model, tensor_input):
    if isinstance(model, migraphx.program):
        n_input = tensor_input.cpu().numpy()
        param_type = model.get_parameter_shapes()['input'].type()
        
        if param_type == 'half_type':
            n_input = n_input.astype(np.float16)
            
        return model.run({'input': n_input})
    else:
        return model(tensor_input)

def benchmark(args, iterations=100, warm_up=20, compile_model=True, profiler=False):
    input_dtype = torch.float32
    device = torch.device('cuda')

    print(f"\n--- Benchmarking: {args.quantization.upper()} on {device} ---")
    net = load_model(args, device)

    dummy_input = torch.randn(1, 3, base_height, base_width, dtype=input_dtype).to("cuda")

    print(f"Warming up...")
    with torch.inference_mode():
        for _ in range(warm_up):
            _ = run_inference(net, dummy_input)
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
            with torch.inference_mode():
                for _ in range(11):  # Matches schedule (1+5+5)
                    _ = run_inference(net, dummy_input)
                    prof.step()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    param_name = net.get_parameter_names()[0] # Usually 'input'
    print(f"DEBUG: Model parameter name: {param_name}")

    input_arg = migraphx.argument(dummy_input.detach().cpu().numpy())
    run_params = {"input": input_arg}
    
    latencies = []
    powers = []
    start_benchmark = time.perf_counter()
    for i in range(iterations):
        iter_start = time.perf_counter()
        net.run(run_params)
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - iter_start)
        if i % 10 == 0:
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
    print(f" - Avg Latency: {avg_latency:.2f} ms")
    print(f" - Throughput: {fps:.2f} FPS")
    if device.type == 'cuda':
        print(f" - Avg Power: {avg_power:.2f} W")
        print(f" - Efficiency: {fps/avg_power if avg_power > 0 else 0:.2f} FPS/Watt")
    else:
        print(f" - Note: Power reading skipped for CPU mode.")


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
    target_modes = ['fp16'] 
    target_refinements = [1]
    for ref in target_refinements:
        print(f"\n{'='*20}")
        print(f" TESTING REFINEMENT STAGES: {ref} ")
        print(f"{'='*20}")
        for mode in target_modes:
            try:
                args = create_args(mode, ref)
                benchmark(args, iterations=50, compile_model=False, profiler=True)
            except Exception as e:
                print(f"FAILED [Mode: {mode}, Ref: {ref}]: {e}")

    print("\n" + "="*50)
    print(f"{'STAGES':<8} | {'MODE':<10} | {'LATENCY':<10} | {'FPS':<10}")
    print("-" * 50)
    for r in FINAL_RESULTS:
        print(f"{r['Stages']:<8} | {r['Mode']:<10} | {r['Latency (ms)']:<10} | {r['Throughput (FPS)']:<10} | {r.get('Power (W)', 'N/A'):<10}")

    if FINAL_RESULTS:
        with open('benchmark_results.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FINAL_RESULTS[0].keys())
            writer.writeheader()
            writer.writerows(FINAL_RESULTS)
        print(f"\nResults saved to benchmark_results.csv")