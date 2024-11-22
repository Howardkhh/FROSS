"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
import torch
from pathlib import Path
from glob import glob
import onnx 
import tensorrt as trt

from egtr.model.rtdetr.rtdetr import RtDetrConfig, RtDetr
from egtr.model.rtdetr_egtr import EgtrHead

def main(args, ):
    """main
    """
    rtdetr_onnx_path = args.artifact_path / 'rt-detr.onnx'
    rtdetr_engine_path = args.artifact_path / 'rt-detr.engine'
    egtr_onnx_path = args.artifact_path / 'egtr-head.onnx'
    egtr_engine_path = args.artifact_path / 'egtr-head.engine'
    
    config = RtDetrConfig.from_pretrained(args.artifact_path)
    config.deploy = True
    rtdetr_model = RtDetr(config)
    egtr_model = EgtrHead(config)
    ckpt_path = sorted(
        glob(f"{args.artifact_path}/checkpoints/epoch=*.ckpt"),
        key=lambda x: int(x.split("epoch=")[1].split("-")[0]),
    )[-1]
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]

    rtdetr_state_dict = {}
    egtr_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model.obj_det_model."):
            rtdetr_state_dict[k[20:]] = v
        elif k.startswith("model.egtr_head."):
            egtr_state_dict[k[16:]] = v
    
    rtdetr_model.load_state_dict(rtdetr_state_dict)
    egtr_model.load_state_dict(egtr_state_dict)
    rtdetr_model.eval()
    egtr_model.eval()

    data = torch.rand(1, 3, 640, 640)
    rtdetr_output = rtdetr_model(data)
    _ = egtr_model(rtdetr_output[2], rtdetr_output[0], rtdetr_output[3], rtdetr_output[4])

    print('Exporting RT-DETR model to onnx...')
    torch.onnx.export(
        rtdetr_model, 
        data, 
        rtdetr_onnx_path,
        input_names=['images'],
        output_names=['logits', 'pred_boxes', 'last_hidden_state', 'decoder_attention_queries', 'decoder_attention_keys'],
        dynamic_axes=None,
        opset_version=17, 
        verbose=False,
        do_constant_folding=True,
    )

    print("Exporting EGTR model to onnx...")
    torch.onnx.export(
        egtr_model,
        (rtdetr_output[2], rtdetr_output[0], rtdetr_output[3], rtdetr_output[4]),
        egtr_onnx_path,
        input_names=['last_hidden_state', 'logits', 'decoder_attention_queries', 'decoder_attention_keys'],
        output_names=['pred_rel', 'pred_connectivity', 'rel_gate'],
        dynamic_axes=None,
        opset_version=17,
        verbose=False,
        do_constant_folding=True,
    )

    print('Checking exported onnx model...')
    rtdetr_onnx_model = onnx.load(rtdetr_onnx_path)
    onnx.checker.check_model(rtdetr_onnx_model)
    egtr_onnx_model = onnx.load(egtr_onnx_path)
    onnx.checker.check_model(egtr_onnx_model)
    print('Exported onnx model successfully...')

    print('Converting RT-DETR onnx model to TensorRT engine...')
    trt_logger = trt.Logger(trt.Logger.WARNING)

    builder = trt.Builder(trt_logger)
    config = builder.create_builder_config()
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt_logger)

    with open(rtdetr_onnx_path, 'rb') as model_file:
        if not parser.parse(model_file.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]

    for input in inputs:
        print(f"RT-DETR model's input: {input.name}, {input.shape}, {input.dtype}")
    for output in outputs:
        print(f"RT-DETR model's output: {output.name}, {output.shape}, {output.dtype}") 

    # Build the TensorRT engine
    config.set_flag(trt.BuilderFlag.FP16)

    engine_bytes = builder.build_serialized_network(network, config) 
    with open(rtdetr_engine_path, "wb") as f:
        f.write(engine_bytes)

    builder = trt.Builder(trt_logger)
    config = builder.create_builder_config()
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt_logger)

    with open(egtr_onnx_path, 'rb') as model_file:
        if not parser.parse(model_file.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]

    for input in inputs:
        print(f"EGTR model's input: {input.name}, {input.shape}, {input.dtype}")
    for output in outputs:
        print(f"EGTR model's output: {output.name}, {output.shape}, {output.dtype}") 

    # Build the TensorRT engine
    config.set_flag(trt.BuilderFlag.FP16)

    engine_bytes = builder.build_serialized_network(network, config) 
    with open(egtr_engine_path, "wb") as f:
        f.write(engine_bytes)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact_path', required=True, type=Path)

    args = parser.parse_args()

    main(args)
