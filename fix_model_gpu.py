import torch
from pathlib import Path

# 모델 경로 설정
input_model_path = "/home/airobot/opensource/yolov5/best.torchscript"  # 기존 TorchScript 모델 파일 경로
output_model_path = "/home/airobot/opensource/yolov5/best_gpu.torchscript"  # 수정된 GPU 모델 저장 경로

# GPU 디바이스 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 모델 불러오기
print("Loading model...")
model = torch.jit.load(input_model_path, map_location=device)
model.eval()  # 평가 모드 설정

# 모델의 모든 상수 및 파라미터를 GPU로 이동
def move_constants_to_gpu(module, device):
    for param in module.parameters():
        param.data = param.data.to(device)
    for buffer in module.buffers():
        buffer.data = buffer.data.to(device)
    for name, sub_module in module._modules.items():
        if sub_module is not None:
            move_constants_to_gpu(sub_module, device)

# 모델과 내부 Tensor를 GPU로 이동
print("Moving model to GPU...")
move_constants_to_gpu(model, device)
model.to(device)

# GPU 상수 확인 (디버깅용)
for name, param in model.named_parameters():
    print(f"Parameter {name} is on {param.device}")
for name, buffer in model.named_buffers():
    print(f"Buffer {name} is on {buffer.device}")

# GPU에서 모델 다시 저장
print("Saving GPU model...")
model.save(output_model_path)
print(f"Model saved to {output_model_path}")

