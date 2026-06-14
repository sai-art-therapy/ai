# AI

YOLO 기반 HTP(House-Tree-Person) 그림 분석 모델

## Project Overview

본 프로젝트는 HTP(House-Tree-Person) 그림 검사 이미지를 분석하기 위한 객체 탐지 모델을 개발한다.

AI Hub HTP 데이터셋을 기반으로 House, Tree, Person 영역을 분리하여 각각 독립적인 YOLOv8m 모델을 학습하였으며, 탐지 결과는 백엔드 서버를 통해 심리 리포트 생성 과정에 활용된다.

본 저장소는 학습된 모델 가중치와 학습 결과(성능 지표, Confusion Matrix, 학습 곡선 등)를 관리하기 위한 저장소이다.

---

## Dataset

AI Hub HTP(House-Tree-Person) 데이터셋을 사용하여 학습을 진행하였다.

기존 통합 데이터셋을 House, Tree, Person 3개 영역으로 분리하여 각각 독립적인 객체 탐지 모델 학습이 가능하도록 재구성하였다.

| Model  | Classes |
| ------ | ------: |
| House  |      14 |
| Tree   |      14 |
| Person |      20 |

### Person Classes (20)

male_person, female_person, head, face, eye, nose, mouth, ear, hair, neck, upper_body, arm, hand, leg, foot, button, pocket, sneakers, female_shoes, male_shoes

---

## Data Preparation

* AI Hub HTP 데이터셋 분석
* House / Tree / Person 데이터셋 분리
* YOLO 포맷 데이터셋 구성
* Person 클래스 재구성

  * male_person / female_person 분리
  * sneakers / male_shoes / female_shoes 분리

---

## Model Training

각 데이터셋에 대해 독립적인 YOLOv8m 모델을 학습하였다.

| Item        | Value            |
| ----------- | ---------------- |
| Base Model  | YOLOv8m          |
| Framework   | Ultralytics YOLO |
| Optimizer   | AdamW            |
| Image Size  | 640              |
| Epochs      | 50               |
| Environment | Google Colab Pro |

### Model Strategy

House, Tree, Person을 하나의 통합 모델로 학습하는 대신 각 영역별 모델을 독립적으로 학습하여 세부 요소 탐지에 활용하였다.

---

## Repository Structure

```text
ai/
├── colab/
│   └── htp_dataset_preprocessing.ipynb
├── configs/                  # 데이터셋 설정 파일 (YOLO yaml)
│   ├── house.yaml
│   ├── tree.yaml
│   └── person.yaml
├── models/                   # YOLOv8m 학습 결과
│   ├── house/
│   │   ├── best.pt
│   │   ├── results.csv
│   │   ├── results.png
│   │   └── ...
│   ├── tree/
│   │   ├── best.pt
│   │   ├── results.csv
│   │   ├── results.png
│   │   └── ...
│   └── person/
│       ├── best.pt
│       ├── results.csv
│       ├── results.png
│       └── ...
└── README.md
```

`colab/htp_dataset_preprocessing.ipynb` contains the preprocessing workflow for converting AI Hub HTP JSON annotations into YOLO format and generating House, Tree, and Person datasets.

---

## Results

각 모델의 학습 결과 및 성능 지표는 `models` 디렉토리에서 확인할 수 있다.

포함 항목:

* Trained Weights (`best.pt`)
* Training Curves
* Precision / Recall Curves
* Confusion Matrix
* Training Metrics (`results.csv`)

### Models

* House Model
* Tree Model
* Person Model

