from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class AlignmentResult:
    aligned: np.ndarray
    homography: np.ndarray
    num_matches: int
    inliers: int


def _to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def align_image_to_template(
    image: np.ndarray,
    template: np.ndarray,
    max_features: int = 8000,
    keep_top_k_matches: int = 400,
    ransac_thresh: float = 5.0,
) -> AlignmentResult:
    gray_img = _to_gray(image)
    gray_tpl = _to_gray(template)

    orb = cv2.ORB_create(nfeatures=max_features)
    kp1, des1 = orb.detectAndCompute(gray_img, None)
    kp2, des2 = orb.detectAndCompute(gray_tpl, None)

    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        H = np.eye(3, dtype=np.float32)
        aligned = cv2.resize(image, (template.shape[1], template.shape[0]))
        return AlignmentResult(aligned=aligned, homography=H, num_matches=0, inliers=0)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[: min(len(matches), keep_top_k_matches)]

    if len(matches) < 8:
        H = np.eye(3, dtype=np.float32)
        aligned = cv2.resize(image, (template.shape[1], template.shape[0]))
        return AlignmentResult(aligned=aligned, homography=H, num_matches=len(matches), inliers=0)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
    if H is None:
        H = np.eye(3, dtype=np.float32)
        aligned = cv2.resize(image, (template.shape[1], template.shape[0]))
        return AlignmentResult(aligned=aligned, homography=H, num_matches=len(matches), inliers=0)

    aligned = cv2.warpPerspective(image, H, (template.shape[1], template.shape[0]))
    inliers = int(mask.sum()) if mask is not None else 0
    return AlignmentResult(aligned=aligned, homography=H, num_matches=len(matches), inliers=inliers)
