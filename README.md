# ai
YOLO 기반 그림 분석 및 LLM 기반 심리 리포트 생성 모듈

## Repository Structure

```
ai/
├── configs/                  # 데이터셋 설정 파일 (YOLO yaml)
│   ├── house.yaml
│   ├── tree.yaml
│   └── person.yaml
├── models/                   # YOLOv8m 학습 결과
│   ├── house/
│   │   ├── weights/best.pt
│   │   ├── results.csv
│   │   ├── results.png
│   │   ├── confusion_matrix.png
│   │   ├── confusion_matrix_normalized.png
│   │   └── Box*_curve.png
│   ├── tree/
│   │   └── (동일 구성)
│   └── person/
│       └── (동일 구성)
├── colab/
├── configs/
├── data/
└── src/
```
