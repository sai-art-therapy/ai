"""
YOLOv8 모델 평가(Evaluation) 스크립트

mAP50, mAP50-95, Precision, Recall 등을 계산합니다.

사용법:
    python src/evaluate.py --weights runs/train/exp/weights/best.pt --data data/dataset.yaml
    python src/evaluate.py --weights best.pt --data data/dataset.yaml --split test
    python src/evaluate.py --weights best.pt --data data/dataset.yaml --conf 0.001 --iou 0.6
"""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 Evaluation")

    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="평가할 모델 가중치 경로",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/dataset.yaml",
        help="데이터셋 YAML 경로",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="평가 데이터 분할 (val 권장)",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="입력 이미지 크기")
    parser.add_argument(
        "--conf", type=float, default=0.001, help="신뢰도 임계값 (mAP 계산 시 낮게 설정)"
    )
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IoU 임계값")
    parser.add_argument("--batch", type=int, default=16, help="배치 크기")
    parser.add_argument("--device", type=str, default="", help="장치 (0, cpu 등)")
    parser.add_argument("--project", type=str, default="runs/val", help="결과 저장 루트")
    parser.add_argument("--name", type=str, default="exp", help="실험 이름")
    parser.add_argument(
        "--save-json", action="store_true", help="COCO JSON 형식으로 결과 저장"
    )
    parser.add_argument(
        "--save-txt", action="store_true", help="결과를 txt 파일로 저장"
    )
    parser.add_argument(
        "--plots", action="store_true", default=True, help="결과 플롯 저장"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True, help="클래스별 결과 출력"
    )

    return parser.parse_args()


def print_metrics(metrics) -> None:
    print("\n" + "=" * 50)
    print("  평가 결과 (Detection Metrics)")
    print("=" * 50)
    print(f"  mAP50:     {metrics.box.map50:.4f}")
    print(f"  mAP50-95:  {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall:    {metrics.box.mr:.4f}")
    print("=" * 50)

    # 클래스별 결과
    if hasattr(metrics.box, "ap_class_index") and metrics.box.ap_class_index is not None:
        print("\n  클래스별 mAP50:")
        for i, cls_idx in enumerate(metrics.box.ap_class_index):
            ap50 = metrics.box.ap50[i] if hasattr(metrics.box, "ap50") else 0.0
            print(f"    [{int(cls_idx)}] {ap50:.4f}")
    print()


def save_metrics_json(metrics, save_path: Path) -> None:
    result = {
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[INFO] 결과 JSON 저장: {save_path}")


def main() -> None:
    args = parse_args()

    print(f"[INFO] 모델 로드: {args.weights}")
    model = YOLO(args.weights)

    print(f"[INFO] 평가 시작 (split={args.split})...")
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        batch=args.batch,
        device=args.device if args.device else None,
        project=args.project,
        name=args.name,
        save_json=args.save_json,
        save_txt=args.save_txt,
        plots=args.plots,
        verbose=args.verbose,
    )

    print_metrics(metrics)

    # 결과 JSON 저장
    json_path = Path(args.project) / args.name / "metrics.json"
    save_metrics_json(metrics, json_path)

    print(f"[완료] 결과 저장 위치: {Path(args.project) / args.name}")


if __name__ == "__main__":
    main()
