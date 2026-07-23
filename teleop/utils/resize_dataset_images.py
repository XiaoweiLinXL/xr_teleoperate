"""
Copy a dataset to a new directory with images resized to 342x256 (4:3, shortest edge=256).
JSON files are copied as-is; images are downsampled with INTER_AREA.

    python3 resize_dataset_images.py \
        --src ~/xr_teleoperate/teleop/utils/data/Load_the_bottle_water_to_the_shelf \
        --dst ~/xr_teleoperate/teleop/utils/data/Load_the_bottle_water_to_the_shelf_256
"""

import argparse
import glob
import os
import shutil
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed

TARGET_W, TARGET_H = 342, 256


def process_episode(ep_src: str, ep_dst: str) -> tuple[int, int]:
    os.makedirs(os.path.join(ep_dst, "colors"), exist_ok=True)
    os.makedirs(os.path.join(ep_dst, "depths"), exist_ok=True)
    os.makedirs(os.path.join(ep_dst, "audios"), exist_ok=True)

    # Copy JSON
    shutil.copy2(os.path.join(ep_src, "data.json"), os.path.join(ep_dst, "data.json"))

    # Resize color images
    ok = err = 0
    for img_path in glob.glob(os.path.join(ep_src, "colors", "*.jpg")):
        dst_path = os.path.join(ep_dst, "colors", os.path.basename(img_path))
        img = cv2.imread(img_path)
        if img is None:
            err += 1
            continue
        resized = cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
        cv2.imwrite(dst_path, resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
        ok += 1
    return ok, err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Original dataset directory")
    parser.add_argument("--dst", required=True, help="Output directory for resized dataset")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    src = os.path.expanduser(args.src)
    dst = os.path.expanduser(args.dst)
    os.makedirs(dst, exist_ok=True)

    episodes = sorted(d for d in os.listdir(src) if d.startswith("episode_"))
    print(f"Found {len(episodes)} episodes — copying to {dst} at {TARGET_W}x{TARGET_H}")

    total_ok = total_err = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_episode, os.path.join(src, ep), os.path.join(dst, ep)): ep
            for ep in episodes
        }
        done = 0
        for f in as_completed(futures):
            ok, err = f.result()
            total_ok += ok
            total_err += err
            done += 1
            print(f"  [{done}/{len(episodes)}] {futures[f]}  ({ok} imgs, {err} errors)")

    print(f"\nDone. {total_ok} images resized, {total_err} errors.")


if __name__ == "__main__":
    main()
