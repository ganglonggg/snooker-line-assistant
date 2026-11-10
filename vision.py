import cv2
import numpy as np

def expand_polygon(poly, d):
    """Convex polygon grown outwards by d pixels (each side pushed out, neighbours intersected)."""
    pts = np.array(poly, dtype=float)
    n = len(pts)
    area2 = sum(pts[i, 0] * pts[(i + 1) % n, 1] - pts[(i + 1) % n, 0] * pts[i, 1] for i in range(n))
    sign = 1.0 if area2 > 0 else -1.0
    lines = []
    for i in range(n):
        p, q = pts[i], pts[(i + 1) % n]
        e = q - p
        normal = sign * np.array([e[1], -e[0]]) / max(np.hypot(*e), 1e-9)  # points outwards
        lines.append((p + normal * d, e))
    out = []
    for i in range(n):
        (p1, d1), (p2, d2) = lines[i - 1], lines[i]
        denom = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(denom) < 1e-9:
            out.append(p2)
            continue
        t = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / denom
        out.append(p1 + t * d1)
    return out

class SnookerVision:
    def __init__(self):
        # HSV ranges for snooker table green
        self.green_lower = np.array([35, 40, 40])
        self.green_upper = np.array([85, 255, 255])

        # HSV ranges for white ball (cue ball is slightly yellowish, so allow some saturation)
        self.white_lower = np.array([0, 0, 170])
        self.white_upper = np.array([180, 90, 255])

        # Filled mask of the playing surface from the last detect_main_view call
        self.table_mask = None

    def detect_main_view(self, frame):
        """
        Detects if the current frame is the main trapezoidal view of the snooker table.
        Returns (is_main_view, approx_polygon).
        """
        self.table_mask = None
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.green_lower, self.green_upper)

        # Calculate percentage of green area. The table can be small when the capture
        # box covers a whole web page around the player, so the bar is fairly low.
        green_ratio = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
        if green_ratio < self.MIN_TABLE_AREA:
            return False, None

        # Find contours of the green area
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, None

        # Get the largest contour (likely the table)
        table_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(table_contour) < frame.shape[0] * frame.shape[1] * self.MIN_TABLE_AREA:
            return False, None

        # Balls are holes in the green contour, so use the convex hull as the table area
        hull = cv2.convexHull(table_contour)
        self.table_mask = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(self.table_mask, [hull], -1, 255, -1)

        quad = self.fit_quad(hull, frame.shape)
        if quad is None:
            return False, None
        # The table area is the corner quadrilateral, which also covers a corner hidden by a player
        self.table_mask = np.zeros(mask.shape, np.uint8)
        cv2.fillPoly(self.table_mask, [quad], 255)
        return True, quad

    MIN_TABLE_AREA = 0.08  # fraction of the captured frame

    @staticmethod
    def fit_quad(hull, shape):
        """
        Four table corners from the table outline, even when a player or a caption hides one:
        take the four longest sides of the simplified outline and intersect neighbouring
        sides, so a hidden corner is rebuilt from the visible parts of its two edges.
        Returns a (4, 1, 2) int array like approxPolyDP, or None.
        """
        # Fine simplification: the real table edges stay long straight sides, while the
        # outline of a person in front of the table breaks into several short ones
        peri = cv2.arcLength(hull, True)
        poly = cv2.approxPolyDP(hull, 0.01 * peri, True).reshape(-1, 2).astype(float)
        if len(poly) < 4:
            return None
        if len(poly) == 4:
            return poly.reshape(4, 1, 2).astype(np.int32)
        if len(poly) > 12:
            return None  # not table-shaped

        n = len(poly)
        sides = [(np.linalg.norm(poly[(i + 1) % n] - poly[i]), i) for i in range(n)]
        keep = sorted(i for _, i in sorted(sides, reverse=True)[:4])
        lines = [(poly[i], poly[(i + 1) % n]) for i in keep]

        corners = []
        for k in range(4):
            (p1, p2), (p3, p4) = lines[k], lines[(k + 1) % 4]
            d1, d2 = p2 - p1, p4 - p3
            denom = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(denom) < 1e-6:
                return None
            t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / denom
            corners.append(p1 + t * d1)
        corners = np.array(corners)

        # The rebuilt corners must stay near the frame and form a convex quadrilateral
        h, w = shape[:2]
        if (corners[:, 0] < -0.1 * w).any() or (corners[:, 0] > 1.1 * w).any() or \
                (corners[:, 1] < -0.1 * h).any() or (corners[:, 1] > 1.1 * h).any():
            return None
        if not cv2.isContourConvex(corners.reshape(4, 1, 2).astype(np.float32)):
            return None
        return np.round(corners).reshape(4, 1, 2).astype(np.int32)

    def detect_cue_ball(self, frame):
        """
        Detects the white cue ball inside the table area.
        Picks the largest roughly-circular bright, low-saturation blob, which rejects
        captions outside the table, the baulk line, and specular highlights on coloured balls.
        Returns (x, y, r) or None. Call detect_main_view first.
        """
        if self.table_mask is None:
            return None
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.white_lower, self.white_upper)
        mask = cv2.bitwise_and(mask, self.table_mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for c in contours:
            area = cv2.contourArea(c)
            if area < 20:
                continue
            (x, y), r = cv2.minEnclosingCircle(c)
            if area / (np.pi * r * r) < 0.45:
                continue
            if best is None or area > best[0]:
                best = (area, x, y, r)

        if best is None:
            return None
        return best[1], best[2], best[3]

    def find_ball_near(self, frame, point, search_radius=30):
        """
        Snaps a click to the centre of the nearest ball (any non-green blob on the table).
        Returns (x, y) or None if nothing ball-like is near the click.
        """
        if self.table_mask is None:
            return None
        h, w = frame.shape[:2]
        px, py = int(point[0]), int(point[1])
        x0, y0 = max(px - search_radius, 0), max(py - search_radius, 0)
        x1, y1 = min(px + search_radius, w), min(py + search_radius, h)
        if x1 <= x0 or y1 <= y0:
            return None

        hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        not_green = cv2.bitwise_not(cv2.inRange(hsv, self.green_lower, self.green_upper))
        not_green = cv2.bitwise_and(not_green, self.table_mask[y0:y1, x0:x1])
        not_green = cv2.morphologyEx(not_green, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        contours, _ = cv2.findContours(not_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for c in contours:
            if cv2.contourArea(c) < 15:
                continue
            (cx, cy), _ = cv2.minEnclosingCircle(c)
            d = np.hypot(cx + x0 - px, cy + y0 - py)
            if best is None or d < best[0]:
                best = (d, cx + x0, cy + y0)

        if best is None:
            return None
        return best[1], best[2]

    def detect_balls(self, frame, inner_poly, radius_at):
        """
        Finds every ball on the playing surface.
        inner_poly: camera-view polygon of the playing area without cushions.
        radius_at(x, y): expected ball radius in pixels at that camera position
        (balls shrink towards the far end in the broadcast view).
        Returns a list of (x, y, r).
        """
        h, w = frame.shape[:2]
        radii = [radius_at(x, y) for x, y in inner_poly]
        r_min, r_max = int(min(radii) * 0.6), int(max(radii) * 1.4)
        if r_min < 3:
            return []

        # Edges away from the table (web page text, crowd, captions) can't be balls and make
        # circle detection slow, so they are blanked out of the edge map below. (Cropping to the
        # table instead would be faster still, but shifts the borderline circle votes.)
        play = np.zeros((h, w), np.uint8)
        cv2.fillPoly(play, [np.int32(inner_poly)], 255)
        near_play = np.zeros((h, w), np.uint8)
        cv2.fillPoly(near_play, [np.int32(expand_polygon(inner_poly, r_max + 2))], 255)

        # Balls differ from the cloth around them; the green ball is close to the cloth in hue,
        # so compare against the local cloth colour in Lab instead of thresholding "not green".
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        small = cv2.resize(lab, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
        k = max(int(w / 4 / 37) | 1, 5)  # median window a few ball widths wide
        cloth = cv2.resize(cv2.medianBlur(small, k), (w, h), interpolation=cv2.INTER_LINEAR)
        diff = np.linalg.norm(lab.astype(np.float32) - cloth.astype(np.float32), axis=2)
        diff = np.clip(diff * 3, 0, 255).astype(np.uint8)
        diff[near_play == 0] = 0

        circles = cv2.HoughCircles(
            cv2.GaussianBlur(diff, (5, 5), 1.5), cv2.HOUGH_GRADIENT_ALT, dp=1.5,
            minDist=r_min * 1.5, param1=150, param2=0.6, minRadius=r_min, maxRadius=r_max
        )
        if circles is None:
            return []

        balls = []
        for x, y, r in circles[0]:
            if not (0 <= int(y) < h and 0 <= int(x) < w) or play[int(y), int(x)] == 0:
                continue
            x, y = float(x), float(y)
            # Reject circles that aren't ball-sized for this spot (gaps between clustered reds, chalk marks)
            if not 0.7 <= r / radius_at(x, y) <= 1.35:
                continue
            balls.append((x, y, float(r)))
        return balls

    CLOTH_BRIGHTNESS = 145.0   # typical median V of the cloth; frames are scaled to this
    MIN_DETECT_RADIUS = 11.0   # balls smaller than this (px) are detected on an upscaled frame

    def find_balls(self, frame, inner_poly, radius_at, exclude=None):
        """
        Full ball pass for one frame: detection + colour labels, robust to the stream's
        brightness and resolution. Returns a list of (x, y, r, colour) in frame pixels.
        Uses self.table_mask (set by detect_main_view, or the locked view's mask).
        exclude(x, y): spots that can't hold a ball (pocket mouths); dropped before colours are
        assigned, so a look-alike there can't take the one-of-each colour slot from a real ball.
        """
        if self.table_mask is None:
            return []
        table_mask = self.table_mask

        # Brightness: scale the frame so the cloth always has about the same brightness
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cloth = cv2.inRange(hsv, self.green_lower, self.green_upper) & table_mask
        cloth_hue = cloth_sat = None
        if np.count_nonzero(cloth) > 100:
            cloth_hue = float(np.median(hsv[..., 0][cloth > 0]))
            cloth_sat = float(np.median(hsv[..., 1][cloth > 0]))
            gain = np.clip(self.CLOTH_BRIGHTNESS / max(np.median(hsv[..., 2][cloth > 0]), 1), 0.6, 2.0)
            if abs(gain - 1) > 0.05:
                frame = cv2.convertScaleAbs(frame, alpha=gain)

        # Resolution: small balls are unreliable for circle detection, so upscale first
        smallest = min(radius_at(x, y) for x, y in inner_poly)
        k = float(np.clip(self.MIN_DETECT_RADIUS / max(smallest, 1), 1.0, 3.0))
        if k > 1.05:
            frame = cv2.resize(frame, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
            self.table_mask = cv2.resize(table_mask, (frame.shape[1], frame.shape[0]),
                                         interpolation=cv2.INTER_NEAREST)
        else:
            k = 1.0
        try:
            poly = [(x * k, y * k) for x, y in inner_poly]
            balls = self.detect_balls(frame, poly, lambda x, y: radius_at(x / k, y / k) * k)
            cue = self.detect_cue_ball(frame)
            if cue is not None and all(np.hypot(b[0] - cue[0], b[1] - cue[1]) > cue[2] for b in balls):
                # The bright patch is only the lit top of the ball: fit the whole outline instead
                fitted = self.refine_circle(frame, cue[0], cue[1], radius_at(cue[0] / k, cue[1] / k) * k)
                balls.append(fitted if fitted is not None else tuple(float(v) for v in cue))
            if exclude is not None:
                balls = [b for b in balls if not exclude(b[0] / k, b[1] / k)]
            labels = self.classify_balls(frame, balls, cloth_hue)
            if "green" not in labels and cloth_hue is not None:
                green = self.find_green_ball(frame, poly, lambda x, y: radius_at(x / k, y / k) * k,
                                             cloth_hue, cloth_sat, balls)
                if green is not None and (exclude is None or not exclude(green[0] / k, green[1] / k)):
                    balls.append(green)
                    labels.append("green")
            occluded = self.occluder_mask(frame, smallest * k)
        finally:
            self.table_mask = table_mask
        out = []
        for (x, y, r), label in zip(balls, labels):
            if label == "cloth":
                continue
            # An unrecognisable "ball" inside a player's body or arm reaching over the table
            if label == "unknown" and occluded[min(int(y), occluded.shape[0] - 1), min(int(x), occluded.shape[1] - 1)]:
                continue
            out.append((x / k, y / k, r / k, label))
        return out

    def occluder_mask(self, frame, ball_r):
        """
        Areas covered by something reaching in over the table edge (a player, an arm):
        big non-cloth regions that touch the edge of the table area. Thin cues are
        opened away first so a cue resting on the cue ball doesn't join it to the player.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        non_cloth = cv2.bitwise_not(cv2.inRange(hsv, self.green_lower, self.green_upper)) & self.table_mask
        size = max(int(ball_r * 0.8) | 1, 3)
        non_cloth = cv2.morphologyEx(non_cloth, cv2.MORPH_OPEN,
                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
        edge = self.table_mask & cv2.bitwise_not(cv2.erode(self.table_mask, np.ones((7, 7), np.uint8)))
        n, comp, stats, _ = cv2.connectedComponentsWithStats(non_cloth)
        occluded = np.zeros(non_cloth.shape, bool)
        min_area = 6 * np.pi * ball_r ** 2
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                continue
            region = comp == i
            if np.any(region & (edge > 0)):
                occluded |= region
        return occluded

    def find_green_ball(self, frame, inner_poly, radius_at, cloth_hue, cloth_sat, known):
        """
        Second look for the green ball, whose edge against the cloth is faint (blur, small
        size, compression). Depending on the broadcast the ball is either clearly bluer than
        the cloth, or only a little bluer but greyer; look for a ball-sized blob of either.
        """
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue, sat = hsv[..., 0].astype(np.float32), hsv[..., 1].astype(np.float32)
        bluer = hue >= cloth_hue + 8
        greyer = (hue >= cloth_hue + 4) & (sat <= 0.9 * cloth_sat)
        mask = ((bluer | greyer) & (hue <= 95) & (sat >= 60)
                & (hsv[..., 2] >= 45)).astype(np.uint8) * 255
        play = np.zeros((h, w), np.uint8)
        cv2.fillPoly(play, [np.int32(inner_poly)], 255)
        mask = cv2.morphologyEx(mask & play, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for c in contours:
            (x, y), r = cv2.minEnclosingCircle(c)
            r_exp = radius_at(x, y)
            area = cv2.contourArea(c)
            # Roughly ball-sized (the shaded half of the ball may be lost) and roundish
            if not (0.4 * r_exp <= r <= 1.4 * r_exp) or area < 0.2 * np.pi * r_exp ** 2:
                continue
            if area / (np.pi * r * r) < 0.45:
                continue
            if any(np.hypot(x - b[0], y - b[1]) < b[2] for b in known):
                continue
            if best is None or area > best[0]:
                best = (area, float(x), float(y), float(r_exp))
        return None if best is None else best[1:]

    def refine_circle(self, frame, x, y, r_exp):
        """
        Fits the whole outline of the ball near (x, y): a ball's lower half is in shade, so a
        bright-patch detection sits too high and too small. Looks for a ball-sized circle in the
        colour difference from the surrounding cloth (upscaled first if the ball is small).
        Returns (x, y, r) or None.
        """
        h, w = frame.shape[:2]
        R = int(2.5 * r_exp) + 2
        x0, y0 = max(int(x) - R, 0), max(int(y) - R, 0)
        x1, y1 = min(int(x) + R, w), min(int(y) + R, h)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        roi = frame[y0:y1, x0:x1]
        k = max(1.0, 14.0 / r_exp)
        if k > 1:
            roi = cv2.resize(roi, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
        rk = r_exp * k
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
        cloth = cv2.inRange(hsv, self.green_lower, self.green_upper) > 0
        if cloth.sum() < 10:
            return None
        diff = np.clip(np.linalg.norm(lab - np.median(lab[cloth], axis=0), axis=2) * 3, 0, 255).astype(np.uint8)
        circles = cv2.HoughCircles(cv2.GaussianBlur(diff, (5, 5), 1.5), cv2.HOUGH_GRADIENT_ALT, 1.5, rk,
                                   param1=150, param2=0.4, minRadius=int(rk * 0.75), maxRadius=int(np.ceil(rk * 1.25)))
        if circles is None:
            return None
        cx, cy = (x - x0) * k, (y - y0) * k
        bx, by, br = min(circles[0], key=lambda c: np.hypot(c[0] - cx, c[1] - cy))
        if np.hypot(bx - cx, by - cy) > 0.8 * rk:
            return None
        return float(bx / k + x0), float(by / k + y0), float(br / k)

    def is_cloth_at(self, frame, x, y, r):
        """True if the spot (a ball-sized disc) is almost entirely bare cloth."""
        h, w = frame.shape[:2]
        rr = max(int(r * 0.8), 2)
        x0, y0 = max(int(x) - rr, 0), max(int(y) - rr, 0)
        x1, y1 = min(int(x) + rr + 1, w), min(int(y) + rr + 1, h)
        if x1 <= x0 or y1 <= y0:
            return False
        hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, self.green_lower, self.green_upper)
        return np.count_nonzero(green) / green.size > 0.85

    # Typical red ball in Lab, used as the reference when too few reddish balls are left to measure one
    DEFAULT_RED_LAB = np.array([90.0, 178.0, 165.0])

    def ball_color_features(self, frame, x, y, r):
        """Median HSV and Lab of the ball's inner disc, ignoring specular highlights."""
        h, w = frame.shape[:2]
        rr = max(int(r * 0.6), 2)
        x0, y0 = max(int(x) - rr, 0), max(int(y) - rr, 0)
        x1, y1 = min(int(x) + rr + 1, w), min(int(y) + rr + 1, h)
        patch = frame[y0:y1, x0:x1]
        if patch.size == 0:
            return None
        disc = np.zeros(patch.shape[:2], np.uint8)
        cv2.circle(disc, (int(x) - x0, int(y) - y0), rr, 255, -1)
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[disc > 0]
        lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)[disc > 0]
        keep = ~((hsv[:, 2] > 220) & (hsv[:, 1] < 70))
        if keep.sum() > 5:
            hsv, lab = hsv[keep], lab[keep]
        return np.median(hsv, axis=0), np.median(lab, axis=0).astype(np.float32)

    def classify_balls(self, frame, balls, cloth_hue=None):
        """
        Labels each ball: red, yellow, green, brown, blue, pink, black, white (or unknown).
        Red / pink / brown look alike in absolute terms and vary with the broadcast's lighting,
        so they are judged relative to the median red-ish ball in the same frame
        (brown: less red; pink: bluer and brighter). Snooker has only one of each colour,
        so duplicates keep the strongest candidate and the rest fall back to red / unknown.
        With cloth_hue given, circles the same hue as the cloth (cloth, shadows between
        balls, a hand mostly showing cloth) are labelled "cloth" and are not balls; the
        green ball is told apart from the cloth by being noticeably bluer.
        """
        feats = [self.ball_color_features(frame, *b) for b in balls]
        labels, scores = [], []
        reddish = []
        for i, f in enumerate(feats):
            if f is None:
                labels.append("unknown"); scores.append(0.0); continue
            (hh, ss, vv), (L, A, B) = f
            if cloth_hue is not None and vv < 45 and ss > 230 and abs(hh - cloth_hue) < 12:
                # Deep-green darkness is the shadow between clustered balls; the black ball is far less saturated
                labels.append("cloth"); scores.append(0.0)
            elif vv < 45:
                labels.append("black"); scores.append(45 - vv)
            elif L > 150 and ss < 120:
                # Bright and pale: the cue ball, or a glossy pink washed out by strong lighting.
                # The white is neutral-to-cream (Lab a* 110-124; an over-exposed white reads 128 but
                # has no colour at all), while a pink keeps a red tint however pale it gets.
                if A < 126 or (A < 130 and ss < 12):
                    # If several qualify, the least red one is the white
                    labels.append("white"); scores.append((130 - A) * 4 + (L - ss) / 10)
                else:
                    labels.append("reddish"); scores.append(0.0); reddish.append(i)
            elif ss < 60 and vv < 150:
                # Dark grey: no snooker ball is grey (clothing, a cue, a shadowed hand)
                labels.append("cloth"); scores.append(0.0)
            elif 95 <= hh <= 130:
                labels.append("blue"); scores.append(ss)
            elif cloth_hue is not None and abs(hh - cloth_hue) < 8:
                labels.append("cloth"); scores.append(0.0)
            elif 50 <= hh < 95:
                labels.append("green"); scores.append(ss)
            elif 16 <= hh < 40 and vv > 170:
                labels.append("yellow"); scores.append(vv)
            elif hh < 40 or hh >= 140:
                # Includes dull yellow-orange / olive: that's the brown, possibly blurred into the
                # cloth (the yellow is always bright, and cloth-hued shadows were caught above)
                labels.append("reddish"); scores.append(0.0); reddish.append(i)
            else:
                labels.append("unknown"); scores.append(0.0)

        ref = self.DEFAULT_RED_LAB
        if len(reddish) >= 4:  # with fewer, pink/brown could dominate the median
            ref = np.median(np.array([feats[i][1] for i in reddish]), axis=0)
        for i in reddish:
            L, A, B = feats[i][1]
            pink_score = (ref[2] - B) + 0.3 * (L - ref[0])
            brown_score = ref[1] - A
            if ref[2] - B > 10 and L > ref[0]:
                labels[i], scores[i] = "pink", pink_score
            elif brown_score > 10:
                labels[i], scores[i] = "brown", brown_score
            else:
                labels[i], scores[i] = "red", 0.0

        # Only one ball of each colour exists
        for colour in ("white", "yellow", "green", "brown", "blue", "pink", "black"):
            idx = [i for i, l in enumerate(labels) if l == colour]
            if len(idx) > 1:
                best = max(idx, key=lambda i: scores[i])
                for i in idx:
                    if i != best:
                        labels[i] = "red" if colour in ("pink", "brown") else "unknown"
        return labels

class TableLock:
    """
    Remembers the camera views that show the whole table (a broadcast cuts between
    several, and each match/venue frames the table differently). A view is learned
    when the same full-table outline holds steady for a few frames; its corners are
    the median over those frames, so drawn lines don't jitter with per-frame noise.
    Later frames are matched to a known view by overlap of the table area, which
    also works while a player leans over the table and hides a corner.
    Close-ups of people or part of the table never match and draw nothing.
    """
    def __init__(self, frames_needed=8, iou_threshold=0.8, max_views=4):
        self.frames_needed = frames_needed
        self.iou_threshold = iou_threshold
        self.max_views = max_views
        self.reset()

    def reset(self):
        self.views = []           # list of {"corners": [...], "mask": small bool mask, "full_mask": uint8}
        self.active = None        # index of the view in the current frame, or None
        self._candidates = []

    @property
    def locked(self):
        return bool(self.views)

    @property
    def corners(self):
        return self.views[self.active]["corners"] if self.active is not None else None

    @property
    def mask(self):
        return self.views[self.active]["full_mask"] if self.active is not None else None

    @staticmethod
    def _small(mask):
        return cv2.resize(mask, (160, 90), interpolation=cv2.INTER_NEAREST) > 0

    def _add_view(self, corners, full_mask):
        # Use the corner quadrilateral as the table area: the outline seen in this frame
        # may be missing a corner hidden behind a player
        full_mask = np.zeros(full_mask.shape, np.uint8)
        cv2.fillPoly(full_mask, [np.int32(np.round(corners))], 255)
        if len(self.views) >= self.max_views:
            self.views.pop(0)  # forget the oldest view
        self.views.append({"corners": corners, "full_mask": full_mask, "mask": self._small(full_mask)})
        self.active = len(self.views) - 1

    def lock_manual(self, corners, frame_shape):
        mask = np.zeros(frame_shape[:2], np.uint8)
        corners = [np.array(c, dtype=np.float32) for c in corners]
        cv2.fillPoly(mask, [np.int32(corners)], 255)
        self.reset()
        self._add_view(corners, mask)

    @staticmethod
    def sort_corners(approx):
        pts = sorted(approx.reshape(4, 2), key=lambda p: p[1])
        top = sorted(pts[:2], key=lambda p: p[0])
        bottom = sorted(pts[2:], key=lambda p: p[0], reverse=True)
        return [np.array(p, dtype=np.float32) for p in top + bottom]

    def _plausible_full_table(self, corners, shape, portrait):
        h, w = shape[:2]
        tl, tr, br, bl = corners
        if portrait:
            # Broadcast views look down the table: a trapezoid narrowing towards the far end
            if np.linalg.norm(tr - tl) > 0.9 * np.linalg.norm(br - bl):
                return False
            # A table cut off by the frame edge (zoomed-in shot) has fake corners
            margin = 0.01 * max(h, w)
            for x, y in corners:
                if x < margin or y < margin or x > w - margin or y > h - margin:
                    return False
        return True

    def observe(self, is_table, approx, table_mask, portrait):
        """Feed one frame's detection. Returns True if this frame shows a known full-table view."""
        self.active = None
        if not is_table or table_mask is None:
            self._candidates = []
            return False

        # 1. A view we already know?
        small = self._small(table_mask)
        best, best_iou = None, 0.0
        for i, v in enumerate(self.views):
            if v["full_mask"].shape != table_mask.shape:
                continue
            union = np.count_nonzero(v["mask"] | small)
            iou = np.count_nonzero(v["mask"] & small) / union if union else 0.0
            if iou > best_iou:
                best, best_iou = i, iou
        if best is not None and best_iou >= self.iou_threshold:
            self.active = best
            self._candidates = []
            return True

        # 2. Learn a new view once the same full-table outline holds steady
        if approx is None or len(approx) != 4:
            self._candidates = []
            return False
        corners = self.sort_corners(approx)
        if not self._plausible_full_table(corners, table_mask.shape, portrait):
            self._candidates = []
            return False
        diag = np.hypot(*table_mask.shape[:2])
        if self._candidates and max(
                np.linalg.norm(a - b) for a, b in zip(corners, self._candidates[-1])) > 0.015 * diag:
            self._candidates = []  # camera still moving, start over
        self._candidates.append(corners)
        if len(self._candidates) < self.frames_needed:
            return False
        self._add_view(list(np.median(np.array(self._candidates), axis=0)), table_mask.copy())
        self._candidates = []
        return True

if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    vision = SnookerVision()
    for name in ["梯形桌面截屏.png", "斯诺克.png", "huya.png"]:
        test_img = cv2.imread(os.path.join(here, name))
        if test_img is None:
            print(f"{name}: test image not found.")
            continue
        is_main, _ = vision.detect_main_view(test_img)
        cue_ball = vision.detect_cue_ball(test_img)
        print(f"{name}: main view={is_main}, cue ball={cue_ball}")
