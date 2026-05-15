import argparse
import logging
import re
from pathlib import Path
import itertools

import cv2
import numpy as np
import torch

log = logging.getLogger("nudorms.lightglue_pto")

def parse_pto_images(pto_path: Path):
    """Parse 'i' lines from PTO file to get image indices and filenames."""
    images = {}
    with open(pto_path, "r") as f:
        for line in f:
            if line.startswith("i "):
                # e.g., i w3840 h2160 f0 v50 n"path/to/img.jpg"
                match_n = re.search(r'n"([^"]+)"', line)
                if not match_n:
                    match_n = re.search(r'n(\S+)', line)
                
                if match_n:
                    img_path = match_n.group(1)
                    idx = len(images)
                    images[idx] = img_path
    return images

def extract_and_match(img_path1: str, img_path2: str, device: str):
    """
    Extract ALIKED features and match with LightGlue.
    (Requires kornia or LightGlue installed in the Python env)
    """
    try:
        from kornia.feature import LightGlue, ALIKED
        from kornia import image_to_tensor, color
    except ImportError:
        log.error("Kornia is required for LightGlue matching in Python. Install with: pip install kornia")
        return []

    extractor = ALIKED(max_num_keypoints=2048).to(device).eval()
    matcher = LightGlue("aliked").to(device).eval()

    def load_img(p):
        img = cv2.imread(p)
        if img is None: return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = image_to_tensor(img, keepdim=False).float() / 255.0
        return color.rgb_to_grayscale(t).to(device)

    t1, t2 = load_img(img_path1), load_img(img_path2)
    if t1 is None or t2 is None: return []

    with torch.inference_mode():
        feat1 = extractor(t1)
        feat2 = extractor(t2)
        
        # Kornia extractor returns list of dicts for batched inputs
        la1 = {"image0": {"keypoints": feat1[0], "descriptors": feat1[2]}}
        la2 = {"image0": {"keypoints": feat2[0], "descriptors": feat2[2]}}
        
        # Format for kornia LightGlue
        matches = matcher(la1["image0"], la2["image0"])
        
        # matches is a dict with 'matches' and 'scores'
        idx1 = matches['matches'][:, 0].cpu().numpy()
        idx2 = matches['matches'][:, 1].cpu().numpy()
        
        kp1 = feat1[0][idx1].cpu().numpy()
        kp2 = feat2[0][idx2].cpu().numpy()

    return np.concatenate([kp1, kp2], axis=1) # shape: (N, 4) -> x1, y1, x2, y2

def main():
    parser = argparse.ArgumentParser(description="Match images in a PTO file using LightGlue")
    parser.add_argument("--pto", type=Path, required=True, help="Path to the Hugin .pto file")
    args = parser.parse_args()
    
    if not args.pto.exists():
        log.error(f"PTO file not found: {args.pto}")
        return
        
    log.info(f"Loading {args.pto} for LightGlue feature matching...")
    images = parse_pto_images(args.pto)
    
    if len(images) < 2:
        log.warning("Not enough images in PTO file to match.")
        return

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    log.info(f"matching on device: {device}")

    # Read the original PTO lines
    with open(args.pto, "r") as f:
        pto_content = f.read()

    # Generate pairwise matches (neighboring frames + self loop closure)
    indices = list(images.keys())
    pairs_to_match = []
    for i in range(len(indices)):
        pairs_to_match.append((indices[i], indices[(i + 1) % len(indices)]))
        if len(indices) > 3:
            pairs_to_match.append((indices[i], indices[(i + 2) % len(indices)]))

    pairs_to_match = list(set([tuple(sorted(p)) for p in pairs_to_match if p[0] != p[1]]))
    
    control_points = []
    
    for (i, j) in pairs_to_match:
        log.info(f"Matching image {i} with {j}...")
        matches = extract_and_match(images[i], images[j], device)
        
        for match in matches:
            x1, y1, x2, y2 = match
            # Hugin control point syntax: c n<img1> N<img2> x<x1> y<y1> X<x2> Y<y2> t0
            c_line = f"c n{i} n{j} x{x1:.4f} y{y1:.4f} X{x2:.4f} Y{y2:.4f} t0\n"
            control_points.append(c_line)

    if control_points:
        with open(args.pto, "w") as f:
            f.write(pto_content)
            f.write("\n# LightGlue Control Points\n")
            f.writelines(control_points)
        log.info(f"Appended {len(control_points)} control points to PTO.")
    else:
        log.warning("No matches found.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
