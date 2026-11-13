import torch
from models.yolo import Model  # YOLOv5 모델 정의 불러오기

# 1. YOLOv5 모델 로드
model = Model(cfg='yolov5s.yaml')  # yolov5s.yaml 설정 기반
model.to('cuda')  # 모델을 GPU로 이동
model.eval()  # 평가 모드 설정

# 2. 더미 입력 생성 (GPU)
dummy_input = torch.randn(1, 3, 640, 640).to('cuda')

# 3. TorchScript 변환 및 저장
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save('best_gpu.torchscript')  # GPU용 TorchScript 모델 저장

print("TorchScript 모델이 GPU에서 성공적으로 저장되었습니다.")

