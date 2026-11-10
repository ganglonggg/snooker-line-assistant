"""
Offline tests for live-stream behaviour, built from the saved screenshots (no screen capture).

1. Simulated stream: camera cuts, a player covering balls, a ball rolling away, a pot.
2. Robustness: ball detection and colour labels on smaller / darker / brighter / blurrier copies.

Usage: python test_stream.py
"""
import json
import os
import cv2
import numpy as np
from analyzer import SnookerAnalyzer
from vision import SnookerVision, TableLock
from geometry import SnookerGeometry
from tracking import BallTracker

HERE = os.path.dirname(os.path.abspath(__file__))
S = 1.12  # scale of the main-view screenshot labels (read off a 2000px-wide preview of a 2242px image)

def _load_ground_truth():
    """Ball positions and colours shared with the browser extension's tests."""
    with open(os.path.join(HERE, "test_data", "ground_truth.json"), encoding="utf-8") as f:
        data = json.load(f)
    return {img["image"]: (img["portrait"], [(b["colour"], b["x"], b["y"]) for b in img["balls"]], 1.0)
            for img in data["images"]}

GROUND_TRUTH = _load_ground_truth()

failures = []

def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)

def remove_ball(img, x, y, r):
    """Paints cloth over a ball and its shadow, as if it had been potted."""
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.circle(mask, (int(x), int(y + r * 0.3)), int(r * 1.7), 255, -1)
    return cv2.inpaint(img, mask, 7, cv2.INPAINT_TELEA)

def move_ball(img, x, y, r, nx, ny):
    k = int(r * 1.3)
    patch = img[int(y) - k:int(y) + k, int(x) - k:int(x) + k].copy()
    out = remove_ball(img, x, y, r)
    disc = np.zeros(patch.shape[:2], np.uint8)
    cv2.circle(disc, (k, k), int(r * 1.05), 255, -1)
    region = out[int(ny) - k:int(ny) + k, int(nx) - k:int(nx) + k]
    region[disc > 0] = patch[disc > 0]
    return out

class Stream:
    """Feeds frames to the analyzer at 20 fps of simulated time."""
    def __init__(self):
        self.a = SnookerAnalyzer()
        self.t = 0.0

    def play(self, frame, seconds):
        for _ in range(int(seconds * 20)):
            self.a.process(frame, self.t)
            self.t += 0.05
        return self.a

def stream_test():
    print("Simulated stream (main view: 梯形桌面截屏.png)")
    main = cv2.imread(os.path.join(HERE, "梯形桌面截屏.png"))
    h, w = main.shape[:2]
    blue = (1091 * S, 458 * S)
    # Camera on a player / part of the table: crop and enlarge a corner of the table
    closeup = cv2.resize(main[int(h * 0.55):, int(w * 0.35):int(w * 0.8)], (w, h))
    # Overhead shot from another camera
    overhead = cv2.resize(cv2.imread(os.path.join(HERE, "斯诺克.png")), (w, h))
    # Non-table frame (stream page while loading)
    page = cv2.resize(cv2.imread(os.path.join(HERE, "huya.png")), (w, h))

    s = Stream()
    a = s.play(main, 0.2)
    check("not drawing before the table is learned", a.state == "locking", a.state)
    a = s.play(main, 1.5)
    check("table learned and balls settled", a.state == "settled", a.state)
    check("19 balls tracked", len(a.tracker.balls()) == 19, f"{len(a.tracker.balls())}")
    cue = a.cue_ball
    check("cue ball found", cue is not None and np.hypot(*(cue.pos - (1113 * S, 889 * S))) < 15)

    t = a.select_target((blue[0] - 10, blue[1] + 8))
    check("click selects the blue", t is not None and t.colour == "blue", t.colour if t else None)
    a = s.play(main, 0.1)
    check("blue shot is blocked", a.plan is not None and a.plan["blocked"],
          f"{len(a.plan['blockers'])} blockers" if a.plan else "no plan")

    for name, frame in (("player close-up / part of table", closeup), ("overhead camera", overhead),
                        ("non-table frame", page)):
        a = s.play(frame, 0.6)
        check(f"cut to {name}: lines hidden", a.state == "other_view" and a.plan is None, a.state)
    a = s.play(main, 0.1)
    check("cut back: main view recognised immediately", a.state == "settled", a.state)
    check("cut back: blue still selected and plan shown",
          a.target is not None and a.target.colour == "blue" and a.plan is not None)

    # A player leans over and covers part of the red cluster
    covered = main.copy()
    cv2.rectangle(covered, (int(1060 * S), int(700 * S)), (int(1320 * S), int(830 * S)), (30, 30, 30), -1)
    a = s.play(covered, 1.0)
    check("player covering reds: still settled", a.state == "settled", a.state)
    check("covered reds remembered", len(a.tracker.balls()) == 19, f"{len(a.tracker.balls())}")
    check("covered reds still block the shot", a.plan is not None and a.plan["blocked"])
    a = s.play(main, 0.5)

    # The blue rolls away in steps
    r = a.target.r
    moved = main
    states = []
    for i in range(1, 6):
        frame = move_ball(main, blue[0], blue[1], r, blue[0] + 40 * i, blue[1] - 25 * i)
        a = s.play(frame, 0.2)
        states.append(a.state)
        moved = frame
    check("while the blue rolls: lines hidden", "moving" in states and a.plan is None, ",".join(states))
    check("blue moved away: selection cleared", a.target is None)
    a = s.play(moved, 1.0)
    check("balls settle again", a.state == "settled", a.state)

    # The cue ball comes to rest touching a red, then rolls away: the cue ball marker must
    # follow the white, not stay on the red it was touching
    s2 = Stream()
    cue0 = np.array([1113 * S, 889 * S])
    red = np.array([1227 * S, 972 * S])
    rr = 18.5
    touching = red + (cue0 - red) / np.linalg.norm(cue0 - red) * 2 * rr
    a2 = s2.play(move_ball(main, cue0[0], cue0[1], rr, touching[0], touching[1]), 2.0)
    c = a2.cue_ball
    check("cue ball resting against a red: found", c is not None and np.linalg.norm(c.pos - touching) < 12,
          None if c is None else f"at {c.pos.round()}")
    for i in (1, 2, 3):
        p = touching + (cue0 - touching) * i / 4
        a2 = s2.play(move_ball(main, cue0[0], cue0[1], rr, p[0], p[1]), 0.2)
    a2 = s2.play(main, 1.5)
    c = a2.cue_ball
    check("cue ball rolls away from the red: marker follows the white",
          c is not None and np.linalg.norm(c.pos - cue0) < 12, None if c is None else f"at {c.pos.round()}")
    near_red = [t for t in a2.tracker.balls() if np.linalg.norm(t.pos - red) < 12]
    check("the red it touched is still a red", len(near_red) == 1 and near_red[0].colour == "red",
          ",".join(t.colour for t in near_red))

    # Pot the pink: it should be forgotten
    pink = (549 * S, 898 * S)
    potted = remove_ball(moved, pink[0], pink[1], r * 1.1)
    a = s.play(potted, 1.0)
    colours = [t.colour for t in a.tracker.balls()]
    check("potted pink forgotten", "pink" not in colours and len(colours) == 18, f"{len(colours)} balls")

def robustness_test():
    print("Robustness: detection + colour labels on degraded copies")
    variants = {
        "original": lambda im: im,
        "half size": lambda im: cv2.resize(im, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA),
        "darker (x0.6)": lambda im: cv2.convertScaleAbs(im, alpha=0.6),
        "brighter (x1.3)": lambda im: cv2.convertScaleAbs(im, alpha=1.3),
        "blurry": lambda im: cv2.GaussianBlur(im, (7, 7), 2.5),
    }
    for image, (portrait, gt, scale) in GROUND_TRUTH.items():
        base = cv2.imread(os.path.join(HERE, image))
        for vname, fn in variants.items():
            im = fn(base)
            k = im.shape[1] / base.shape[1]
            v, g, lock = SnookerVision(), SnookerGeometry(), TableLock()
            g.set_orientation(portrait)
            ok, approx = v.detect_main_view(im)
            if not ok or approx is None or len(approx) != 4:
                check(f"{image} {vname}: table found", False)
                continue
            g.update_homography(lock.sort_corners(approx))
            found_balls = v.find_balls(im, g.playing_area(), g.ball_radius_px)
            balls = [b[:3] for b in found_balls]
            labels = [b[3] for b in found_balls]
            found = right = 0
            wrong = []
            for colour, x, y in gt:
                x, y = x * scale * k, y * scale * k
                if not balls:
                    break
                i = min(range(len(balls)), key=lambda i: np.hypot(balls[i][0] - x, balls[i][1] - y))
                if np.hypot(balls[i][0] - x, balls[i][1] - y) < 12 * k + 4:
                    found += 1
                    if labels[i] == colour:
                        right += 1
                    else:
                        wrong.append(f"{colour}->{labels[i]}")
            extra = len(balls) - found
            check(f"{image} {vname}: {found}/{len(gt)} found, {right}/{len(gt)} colours right, {extra} extra",
                  found == len(gt) and right == len(gt) and extra == 0, ", ".join(wrong))

def pale_pink_test():
    """A pale, glossy pink (strong lighting) must not be taken for the cue ball."""
    print("Pale pink vs cue ball (test_data/直播实测_主视角_球员遮挡.png)")
    base = cv2.imread(os.path.join(HERE, "test_data", "直播实测_主视角_球员遮挡.png"))
    pink, cue = (519, 356), (683, 549)
    for wash in (0.25, 0.35, 0.45, 0.55, 0.65, 0.75):
        im = base.copy()
        disc = np.zeros(im.shape[:2], np.uint8)
        cv2.circle(disc, pink, 8, 255, -1)
        im[disc > 0] = (im[disc > 0] * (1 - wash) + 255 * wash).astype(np.uint8)
        v, g = SnookerVision(), SnookerGeometry()
        ok, approx = v.detect_main_view(im)
        g.update_homography(TableLock.sort_corners(approx))
        found = v.find_balls(im, g.playing_area(), g.ball_radius_px)
        at = lambda p: [b[3] for b in found if np.hypot(b[0] - p[0], b[1] - p[1]) < 8]
        check(f"pink washed {int(wash * 100)}% towards white: still pink, cue ball still white",
              at(pink) == ["pink"] and at(cue) == ["white"], f"pink -> {at(pink)}, cue -> {at(cue)}")

def pocket_leather_test():
    """Cream pocket leather at a pocket mouth must not become a ball (it was taken for the cue ball)."""
    print("Pocket leather (test_data/直播实测_主视角_球员遮挡.png)")
    base = cv2.imread(os.path.join(HERE, "test_data", "直播实测_主视角_球员遮挡.png"))
    for name, bgr in (("cream", (200, 225, 235)), ("off-white", (215, 225, 230))):
        im = base.copy()
        cv2.circle(im, (326, 311), 8, bgr, -1)  # top-left pocket mouth
        s = Stream()
        a = s.play(im, 2.0)
        colours = {t.colour: t.pos for t in a.tracker.balls()}
        near_pocket = [t.colour for t in a.tracker.balls() if np.hypot(*(t.pos - (326, 311))) < 10]
        cue = a.cue_ball
        check(f"{name} leather at a pocket: not a ball, cue ball and pink unaffected",
              not near_pocket and cue is not None and np.hypot(*(cue.pos - (683, 549))) < 8
              and "pink" in colours and np.hypot(*(colours["pink"] - (519, 356))) < 8,
              f"at pocket: {near_pocket}, cue: {None if cue is None else cue.pos.round()}")

def pocket_mouth_test():
    """The pot line is aimed into the opening, on the cloth, not up on the cushion jaw."""
    print("Pocket mouth")
    g = SnookerGeometry()
    g.update_homography(np.array([[120, 60], [880, 70], [960, 680], [50, 700]], np.float32))
    centre = np.array([g.target_w / 2, g.target_h / 2])
    for i in range(6):
        jaw = np.asarray(g.pockets[i], float)
        mouth = g.pocket_mouth(i)
        check(f"pocket {i} opening is past the jaw",
              np.linalg.norm(mouth - centre) > np.linalg.norm(jaw - centre) + 5)
    plan = g.plan_shot(np.array([400.0, 500.0]), np.array([700.0, 220.0]), [])
    mouth = g.pocket_mouth(plan["pocket_index"])
    check("pot line ends at the opening on the cloth",
          np.linalg.norm(plan["pocket"] - g.to_camera_view(mouth)) < 1.0,
          f"pocket {plan['pocket_index']}")
    lifted = g.ball_to_camera(g.pockets[plan["pocket_index"]])
    check("pot line is not the point lifted onto the cushion",
          np.linalg.norm(plan["pocket"] - lifted) > 2.0)

def pocket_choice_test():
    """Pocket choice by difficulty (distance, cut, middle-pocket entry), kept steady between frames."""
    print("Pocket choice (top-down coordinates)")
    g = SnookerGeometry()
    g.update_homography(g.dst_pts)  # camera == top-down, so positions below are on the flat table
    P = lambda *xy: np.array(xy, dtype=float)
    # Corner pocket needs a thinner cut but is 4x further away than the middle pocket
    plan = g.plan_shot(P(300, 1000), P(200, 800), [])
    check("close middle pocket beats a far corner with a thinner cut", plan["pocket_index"] == 4,
          f"pocket {plan['pocket_index']}, cut {plan['cut_deg']:.0f}")
    # Ball running along the side cushion past the middle pocket: far too steep to drop there
    plan = g.plan_shot(P(70, 150), P(60, 400), [])
    check("steep angle into a middle pocket is not offered", plan is None or plan["pocket_index"] != 4,
          "none" if plan is None else f"pocket {plan['pocket_index']}")
    # Two corner pockets of similar difficulty: the one chosen last time is kept
    cue, target = P(355, 900), P(355, 500)
    first = g.plan_shot(cue, target, [])
    other = 1 - first["pocket_index"]
    again = g.plan_shot(cue, target, [], prefer=other)
    check("near-equal pockets: the previous choice is kept", again["pocket_index"] == other,
          f"first {first['pocket_index']}, kept {again['pocket_index']}")

def jitter_test():
    """Detection noise on still balls must not flicker the drawing or count as movement."""
    print("Jitter on still balls")
    rng = np.random.default_rng(0)
    balls = [(200, 300, "red"), (260, 340, "red"), (400, 500, "blue"), (520, 610, "black"),
             (330, 700, "white"), (600, 250, "pink")]
    tr = BallTracker()
    t, moving_after_settle, versions = 0.0, 0, []
    for i in range(50):  # 10 s of detections every 0.2 s, each off by up to 0.3 radius per axis
        r = 10.0
        dets = [(x + rng.uniform(-0.3, 0.3) * r, y + rng.uniform(-0.3, 0.3) * r, r, c) for x, y, c in balls]
        if i % 7 == 3:  # now and then one detection is off by 0.8 radius for a single frame
            x, y, rr, c = dets[i % len(dets)]
            dets[i % len(dets)] = (x + 0.8 * r, y, rr, c)
        tr.update(dets, t, lambda x, y, rr: False)
        if t > 1.5 and not tr.settled(t):
            moving_after_settle += 1
        versions.append(tr.version)
        t += 0.2
    check("jittering still balls never count as moving", moving_after_settle == 0, f"{moving_after_settle} moving updates")
    check("jitter doesn't replan every update", versions[-1] - versions[10] < 20, f"{versions[-1] - versions[10]} replans in 8 s")

def duplicate_test():
    """One ball must never turn into two (a phantom right next to the cue ball blocked its own path)."""
    print("Duplicate balls")
    never_cloth = lambda x, y, r: False  # the old spot is never bare: worst case for phantoms
    tr = BallTracker()
    for i, c in enumerate(["white"] * 3 + ["brown", "white", "white", "brown", "brown", "white"]):
        tr.update([(683, 549, 8, c), (520, 569, 7, "black")], i * 0.2, never_cloth)
    check("colour label flickering on one ball: still one ball", len(tr.balls()) == 2,
          ", ".join(t.colour for t in tr.balls()))
    tr, x = BallTracker(), 683.0
    for i in range(12):
        if 4 <= i < 8:
            x += 14  # rolls ~1.75 radii between detections, then stops
        tr.update([(x, 549, 8, "white"), (520, 569, 7, "black")], i * 0.2, never_cloth)
    check("ball rolling to a stop leaves no phantom behind", len(tr.balls()) == 2,
          ", ".join(f"{t.colour}@{int(t.pos[0])}" for t in tr.balls()))
    s = Stream()
    a = s.play(cv2.imread(os.path.join(HERE, "test_data", "直播实测_主视角_球员遮挡.png")), 2.0)
    g = a.geometry
    cue, black = np.array([683.0, 549.0]), np.array([520.0, 569.0])
    plan = g.plan_shot(cue, black, [cue, cue + (4, 5), black])
    check("a double detection of the cue ball doesn't block its own path", not plan["blocked"])

def cue_centre_test():
    """The cue ball's centre is the whole ball's, not just its brightly lit top."""
    print("Cue ball centre")
    for image, portrait, truth in (("梯形桌面截屏.png", True, (1246.6, 995.7)),
                                   ("test_data/直播实测_主视角_球员遮挡.png", True, (683.0, 549.0))):
        im = cv2.imread(os.path.join(HERE, image))
        v, g = SnookerVision(), SnookerGeometry()
        g.set_orientation(portrait)
        ok, approx = v.detect_main_view(im)
        g.update_homography(TableLock.sort_corners(approx))
        whites = [b for b in v.find_balls(im, g.playing_area(), g.ball_radius_px) if b[3] == "white"]
        err = None if not whites else float(np.hypot(whites[0][0] - truth[0], whites[0][1] - truth[1]))
        check(f"{image}: cue ball centre within 2 px", err is not None and err < 2.0,
              "not found" if err is None else f"{err:.1f} px off")

if __name__ == "__main__":
    cue_centre_test()
    duplicate_test()
    pocket_mouth_test()
    pocket_choice_test()
    jitter_test()
    pocket_leather_test()
    stream_test()
    robustness_test()
    pale_pink_test()
    print(f"\n{'All passed' if not failures else f'{len(failures)} failed'}")
