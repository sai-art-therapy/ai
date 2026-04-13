"""
YOLOv8 Fine-tuning 학습 스크립트

사용법:
    python src/train.py                          # 기본 설정으로 학습
    python src/train.py --config configs/train_config.yaml
    python src/train.py --model yolov8s.pt --epochs 50 --batch 8
    python src/train.py --resume runs/train/exp/weights/last.pt
"""

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 Fine-tuning")

    # 설정 파일
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="학습 설정 YAML 파일 경로",
    )

    # 개별 오버라이드 인수 (설정 파일보다 우선)
    parser.add_argument("--model", type=str, help="사전학습 가중치 경로 (e.g. yolov8n.pt)")
    parser.add_argument("--data", type=str, help="데이터셋 YAML 경로")
    parser.add_argument("--epochs", type=int, help="학습 에폭 수")
    parser.add_argument("--batch", type=int, help="배치 크기")
    parser.add_argument("--imgsz", type=int, help="입력 이미지 크기")
    parser.add_argument("--device", type=str, help="학습 장치 (0, 1, cpu)")
    parser.add_argument("--project", type=str, help="결과 저장 루트 디렉토리")
    parser.add_argument("--name", type=str, help="실험 이름")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="중단된 학습 재개할 last.pt 경로",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. 설정 파일 로드
    config = load_config(args.config)
    print(f"[INFO] 설정 파일 로드: {args.config}")

    # 2. CLI 인수로 설정 오버라이드
    overrides = {k: v for k, v in vars(args).items() if v is not None and k != "config"}
    config.update(overrides)

    # 3. 모델 로드
    if args.resume:
        print(f"[INFO] 학습 재개: {args.resume}")
        model = YOLO(args.resume)
        results = model.train(resume=True)
    else:
        model_path = config.get("model", "yolov8n.pt")
        print(f"[INFO] 모델 로드: {model_path}")
        model = YOLO(model_path)

        # task 설정 (detect/segment/classify)
        task = config.pop("task", "detect")

        # 4. 파인튜닝 시작
        print("[INFO] 학습 시작...")
        results = model.train(
            data=config.get("data", "data/dataset.yaml"),
            epochs=config.get("epochs", 100),
            imgsz=config.get("imgsz", 640),
            batch=config.get("batch", 16),
            workers=config.get("workers", 8),
            device=config.get("device", 0),
            optimizer=config.get("optimizer", "AdamW"),
            lr0=config.get("lr0", 0.001),
            lrf=config.get("lrf", 0.01),
            momentum=config.get("momentum", 0.937),
            weight_decay=config.get("weight_decay", 0.0005),
            warmup_epochs=config.get("warmup_epochs", 3.0),
            warmup_momentum=config.get("warmup_momentum", 0.8),
            warmup_bias_lr=config.get("warmup_bias_lr", 0.1),
            box=config.get("box", 7.5),
            cls=config.get("cls", 0.5),
            dfl=config.get("dfl", 1.5),
            hsv_h=config.get("hsv_h", 0.015),
            hsv_s=config.get("hsv_s", 0.7),
            hsv_v=config.get("hsv_v", 0.4),
            degrees=config.get("degrees", 0.0),
            translate=config.get("translate", 0.1),
            scale=config.get("scale", 0.5),
            shear=config.get("shear", 0.0),
            perspective=config.get("perspective", 0.0),
            flipud=config.get("flipud", 0.0),
            fliplr=config.get("fliplr", 0.5),
            mosaic=config.get("mosaic", 1.0),
            mixup=config.get("mixup", 0.0),
            copy_paste=config.get("copy_paste", 0.0),
            project=config.get("project", "runs/train"),
            name=config.get("name", "exp"),
            save=config.get("save", True),
            save_period=config.get("save_period", -1),
            cache=config.get("cache", False),
            exist_ok=config.get("exist_ok", False),
            pretrained=config.get("pretrained", True),
            plots=config.get("plots", True),
            verbose=config.get("verbose", True),
            seed=config.get("seed", 0),
            patience=config.get("patience", 50),
        )

    # 5. 결과 출력
    save_dir = Path(results.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    print("\n[완료] 학습 종료!")
    print(f"  결과 저장 위치: {save_dir}")
    print(f"  최적 가중치:    {best_weights}")


if __name__ == "__main__":
    main()
