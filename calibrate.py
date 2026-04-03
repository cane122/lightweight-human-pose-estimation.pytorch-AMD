import torch
import numpy as np
import copy
from modules.load_state import load_state
from models.with_mobilenet import PoseEstimationWithMobileNet

def calibrate_and_compare(args):
    device = torch.device('cpu') 
    
    # 1. Load the original FP32 model
    net_fp32 = PoseEstimationWithMobileNet(num_refinement_stages=args.num_refinement_stages)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    load_state(net_fp32, checkpoint)
    net_fp32.eval()

    net_int8 = copy.deepcopy(net_fp32)
    net_int8.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    if hasattr(net_int8, 'fuse_model'):
        net_int8.fuse_model()

    torch.quantization.prepare(net_int8, inplace=True)

    print("--- Starting Calibration ---")
    from datasets.coco import CocoValDataset
    dataset = CocoValDataset(args.labels, args.images_folder)
    
    with torch.no_grad():
        for i in range(min(135, len(dataset))):
            img = dataset[i]['img']
            img_mean = 128
            img_scale = 1/256
            normalized_img = (img.astype(np.float32) - img_mean) * img_scale
            input_tensor = torch.from_numpy(normalized_img).permute(2, 0, 1).unsqueeze(0).float()
            net_int8(input_tensor)
            if i % 10 == 0: print(f"Calibrating: {i}/135")

    torch.quantization.convert(net_int8, inplace=True)
    print("--- Calibration Complete ---")

    test_img_raw = dataset[0]['img']
    test_img_norm = (test_img_raw.astype(np.float32) - 128) * (1/256)
    test_img = torch.from_numpy(test_img_norm).permute(2, 0, 1).unsqueeze(0).float()

    with torch.no_grad():
        out_fp32 = net_fp32(test_img)[-2]
        out_int8 = net_int8(test_img)[-2]

    diff = torch.mean((out_fp32 - out_int8)**2).item()
    print(f"\n[Validation] Prediction Difference (MSE): {diff:.8f}")
    
    if diff < 0.001:
        print("Result: PASS - INT8 predictions are very close to FP32.")
    else:
        print("Result: CAUTION - Significant variance detected. Consider more calibration data.")
    
    if args.output_path:
        save_path = args.output_path
    else:
        save_path = args.checkpoint_path.replace('.pth', f'_int8_s{args.num_refinement_stages}.pth')

    torch.save(net_int8.state_dict(), save_path)
    print(f"Saved calibrated model to: {save_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-path', type=str, required=True)
    parser.add_argument('--labels', type=str, required=True)
    parser.add_argument('--images-folder', type=str, required=True)
    parser.add_argument('--num-refinement-stages', type=int, default=1)
    parser.add_argument('--output-path', type=str, default=None, help='Custom path for the quantized model')
    args = parser.parse_args()
    calibrate_and_compare(args)