"""
Offline check: runs the same detection + geometry pipeline as main.py on a saved
screenshot (no screen capture) and writes an image with the overlay lines drawn on it.

Usage: python test_render.py
Output goes to test_output/.
"""
import os
import cv2
import numpy as np
from vision import SnookerVision
from geometry import SnookerGeometry

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "test_output")

def sort_corners(approx):
    # Same ordering as main.py: TL, TR, BR, BL
    pts = sorted(approx.reshape(4, 2), key=lambda x: x[1])
    top = sorted(pts[:2], key=lambda x: x[0])
    bottom = sorted(pts[2:], key=lambda x: x[0], reverse=True)
    return top + bottom

def pt(p):
    return int(round(p[0])), int(round(p[1]))

def dashed_line(img, p1, p2, color, thickness, dash=14, gap=10):
    p1, p2 = np.array(p1, float), np.array(p2, float)
    length = np.linalg.norm(p2 - p1)
    if length == 0:
        return
    unit = (p2 - p1) / length
    d = 0.0
    while d < length:
        a = p1 + unit * d
        b = p1 + unit * min(d + dash, length)
        cv2.line(img, pt(a), pt(b), color, thickness, cv2.LINE_AA)
        d += dash + gap

def render(image_name, click, portrait, out_name):
    img = cv2.imread(os.path.join(HERE, image_name))
    vision = SnookerVision()
    geo = SnookerGeometry()
    geo.set_orientation(portrait)

    is_main, approx = vision.detect_main_view(img)
    if not is_main or len(approx) != 4:
        print(f"{image_name}: table not found")
        return
    corners = sort_corners(approx)
    geo.update_homography(corners)

    cue = vision.detect_cue_ball(img)
    target = vision.find_ball_near(img, click, search_radius=40)
    if cue is None or target is None:
        print(f"{image_name}: cue={cue}, target={target}")
        return
    cue = cue[:2]

    found = vision.find_balls(img, geo.playing_area(), geo.ball_radius_px)
    balls = [b[:3] for b in found]
    plan = geo.plan_shot(cue, target, balls)
    out = img.copy()

    # Detected table corners (green), pockets (magenta), detected balls with colour labels
    for c in corners:
        cv2.circle(out, pt(c), 10, (0, 255, 0), 3, cv2.LINE_AA)
    for p in geo.pockets:
        cv2.circle(out, pt(geo.ball_to_camera(p)), 14, (255, 0, 255), 2, cv2.LINE_AA)
    for x, y, r, colour in found:
        cv2.circle(out, pt((x, y)), int(r) + 3, (255, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(out, colour, (int(x - r), int(y + r + 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(out, colour, (int(x - r), int(y + r + 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    if plan is None:
        cv2.putText(out, "No pottable pocket", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)
    else:
        # One-ball-wide corridors, red when blocked
        band = out.copy()
        for key, color in (("aim", (255, 255, 255)), ("pot", (0, 255, 255))):
            poly = plan[key + "_corridor"]
            if poly is not None:
                cv2.fillPoly(band, [np.int32([pt(p) for p in poly])],
                             (0, 0, 255) if plan[key + "_blocked"] else color)
        out = cv2.addWeighted(band, 0.4, out, 0.6, 0)

        ghost = plan["aim"][1]
        dashed_line(out, cue, ghost, (255, 255, 255), 3)                              # aim line
        cv2.line(out, pt(target), pt(plan["pocket"]), (0, 255, 255), 3, cv2.LINE_AA)  # pot line
        dashed_line(out, ghost, plan["cue_path"][1], (255, 255, 0), 3, dash=4, gap=8) # cue ball path
        cv2.circle(out, pt(ghost), 6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(out, pt(plan["pocket"]), 20, (0, 255, 255), 3, cv2.LINE_AA)
        for b in plan["blockers"]:
            cv2.circle(out, pt(b), 28, (0, 0, 255), 4, cv2.LINE_AA)
        status = (f"BLOCKED by {len(plan['blockers'])} ball(s)" if plan["blocked"]
                  else f"Clear, cut {plan['cut_deg']:.0f} deg")
        cv2.putText(out, status, (30, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                    (0, 0, 255) if plan["blocked"] else (0, 255, 0), 4)
        print(f"{image_name}: {len(balls)} balls, cue={pt(cue)} target={pt(target)} "
              f"pocket={pt(plan['pocket'])} cut={plan['cut_deg']:.0f} blocked={plan['blocked']} "
              f"blockers={[pt(b) for b in plan['blockers']]}")

    cv2.circle(out, pt(cue), 22, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.circle(out, pt(target), 22, (0, 255, 255), 3, cv2.LINE_AA)
    cv2.drawMarker(out, pt(click), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)  # raw click

    os.makedirs(OUT_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(OUT_DIR, out_name), out)

if __name__ == "__main__":
    # Main broadcast view: click slightly off the blue
    render("梯形桌面截屏.png", click=(1215, 505), portrait=True, out_name="主视角_白球打蓝球.png")
    # Main view: the black has a clear line from the cue ball
    render("梯形桌面截屏.png", click=(904, 870), portrait=True, out_name="主视角_白球打黑球.png")
    # Overhead view (landscape): click slightly off the pink
    render("斯诺克.png", click=(540, 495), portrait=False, out_name="俯视_白球打粉球.png")
