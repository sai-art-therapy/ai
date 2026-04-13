"""
YOLOv8 추론(Inference) 스크립트

사용법:
    # 이미지 한 장
    python src/predict.py --weights runs/train/exp/weights/best.pt --source image.jpg

    # 디렉토리 전체
    python src/predict.py --weights best.pt --source data/images/test/

    # 웹캠 (장치 id 0)
    python src/predict.py --weights best.pt --source 0

    # 동영상 파일
    python src/predict.py --weights best.pt --source video.mp4

    # 결과를 파일로 저장하지 않고 화면에만 표시
    python src/predict.py --weights best.pt --source image.jpg --show --no-save
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 Inference")

    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="모델 가중치 경로 (e.g. runs/train/exp/weights/best.pt)",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="추론 대상: 이미지/영상 경로, 디렉토리, 웹캠 id(0)",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="입력 이미지 크기")
    parser.add_argument(
        "--conf", type=float, default=0.25, help="객체 신뢰도 임계값 (0~1)"
    )
    parser.add_argument(
        "--iou", type=float, default=0.45, help="NMS IoU 임계값 (0~1)"
    )
    parser.add_argument(
        "--max-det", type=int, default=300, help="이미지당 최대 검출 수"
    )
    parser.add_argument("--device", type=str, default="", help="장치 (0, cpu 등)")
    parser.add_argument(
        "--show", action="store_true", help="결과를 화면에 표시"
    )
    parser.add_argument(
        "--no-save", action="store_true", help="결과 파일 저장 안 함"
    )
    parser.add_argument(
        "--save-txt", action="store_true", help="결과를 YOLO 형식 txt로 저장"
    )
    parser.add_argument(
        "--save-conf", action="store_true", help="txt에 신뢰도 점수 포함"
    )
    parser.add_argument(
        "--save-crop", action="store_true", help="검출 객체 크롭 이미지 저장"
    )
    parser.add_argument("--project", type=str, default="runs/predict", help="결과 저장 루트")
    parser.add_argument("--name", type=str, default="exp", help="실험 이름")
    parser.add_argument(
        "--exist-ok", action="store_true", help="기존 결과 폴더 덮어쓰기"
    )
    parser.add_argument(
        "--line-width", type=int, default=2, help="바운딩 박스 선 두께"
    )
    parser.add_argument(
        "--hide-labels", action="store_true", help="라벨 텍스트 숨김"
    )
    parser.add_argument(
        "--hide-conf", action="store_true", help="신뢰도 숫자 숨김"
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[INFO] 모델 로드: {args.weights}")
    model = YOLO(args.weights)

    print(f"[INFO] 추론 대상: {args.source}")
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device if args.device else None,
        show=args.show,
        save=not args.no_save,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        save_crop=args.save_crop,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        line_width=args.line_width,
        hide_labels=args.hide_labels,
        hide_conf=args.hide_conf,
        verbose=True,
    )

    # 결과 요약 출력
    total_det = sum(len(r.boxes) for r in results if r.boxes is not None)
    save_dir = Path(args.project) / args.name
    print(f"\n[완료] 총 검출 수: {total_det}")
    if not args.no_save:
        print(f"  결과 저장 위치: {save_dir}")


if __name__ == "__main__":
    main()
