import cv2
import numpy as np

class SnookerGeometry:
    def __init__(self):
        # Standard snooker table dimensions (playing area) in mm
        self.TABLE_WIDTH = 3569
        self.TABLE_HEIGHT = 1778
        # The table corners found in the picture are the outer edge of the green cushion tops;
        # the cushion nose (edge of the playing area, where the pocket mouths are) is this far in
        self.CUSHION = 45
        
        # Target top-down coordinates (scaled for display/calculation)
        self.scale = 0.4
        self.portrait = True
        self._update_dst()

        self.M = None
        self.M_inv = None

    def _update_dst(self):
        # The main broadcast camera looks down the length of the table, so in camera
        # order TL->TR is a SHORT side (portrait). Overhead shots are usually landscape.
        long_side = int(self.TABLE_WIDTH * self.scale)
        short_side = int(self.TABLE_HEIGHT * self.scale)
        if self.portrait:
            self.target_w, self.target_h = short_side, long_side
        else:
            self.target_w, self.target_h = long_side, short_side
        self.dst_pts = np.array([
            [0, 0],
            [self.target_w, 0],
            [self.target_w, self.target_h],
            [0, self.target_h]
        ], dtype=np.float32)
        w, h = self.target_w, self.target_h
        # Cushion-nose corners and the middle of each long-side nose. These are the jaws,
        # still on the cloth: the hole a ball falls into is further out (see pocket_mouth).
        m = self.CUSHION * self.scale
        if self.portrait:
            middles = [[m, h / 2], [w - m, h / 2]]
        else:
            middles = [[w / 2, m], [w / 2, h - m]]
        self.pockets = [np.array(p, dtype=np.float32) for p in
                        [[m, m], [w - m, m], [w - m, h - m], [m, h - m]] + middles]

    # A short step past the jaw (mm) so the line enters the opening. Larger values leave
    # the calibrated table and the perspective mapping throws the pocket off to one side.
    CORNER_MOUTH_MM = 20   # corner pockets, along the bisector into the hole
    MIDDLE_MOUTH_MM = 20   # middle pockets, straight out through the side cushion

    def set_orientation(self, portrait, src_pts=None):
        """Switches between portrait (main broadcast view) and landscape (overhead view)."""
        self.portrait = portrait
        self._update_dst()
        if src_pts is not None:
            self.update_homography(src_pts)

    def best_pocket(self, cue_pos, target_pos, max_cut_deg=75):
        """
        Picks the pocket that needs the thinnest cut (smallest angle between the
        cue->target direction and the target->pocket direction).
        Returns None if every pocket needs a cut thicker than max_cut_deg.
        """
        shot_vec = target_pos - cue_pos
        shot_len = np.linalg.norm(shot_vec)
        if shot_len == 0:
            return None
        best = None
        for pocket in self.pockets:
            pot_vec = pocket - target_pos
            pot_len = np.linalg.norm(pot_vec)
            if pot_len == 0:
                continue
            cos = np.dot(shot_vec, pot_vec) / (shot_len * pot_len)
            cut = np.degrees(np.arccos(np.clip(cos, -1, 1)))
            if cut <= max_cut_deg and (best is None or cut < best[0]):
                best = (cut, pocket)
        return None if best is None else best[1]

    @property
    def ball_radius(self):
        # Standard snooker ball diameter is 52.5mm (top-down units)
        return 52.5 / 2 * self.scale

    def playing_area(self):
        """Camera-view polygon of the playing area (inside the cushion noses)."""
        m = self.CUSHION * self.scale
        w, h = self.target_w, self.target_h
        return [self.to_camera_view(p) for p in [(m, m), (w - m, m), (w - m, h - m), (m, h - m)]]

    def ray_to_cushion(self, start, direction):
        """
        End point of a ball rolling from start along direction until it touches a cushion
        (its centre stays one radius inside the playing area). Top-down coordinates.
        """
        m = self.CUSHION * self.scale + self.ball_radius
        lo = np.array([m, m])
        hi = np.array([self.target_w - m, self.target_h - m])
        t_max = np.inf
        for axis in (0, 1):
            if direction[axis] > 0:
                t_max = min(t_max, (hi[axis] - start[axis]) / direction[axis])
            elif direction[axis] < 0:
                t_max = min(t_max, (lo[axis] - start[axis]) / direction[axis])
        if not np.isfinite(t_max):
            return start
        return start + direction * max(t_max, 0)

    def at_pocket_mouth(self, x, y):
        """
        True if a camera-view point sits right on a pocket mouth. No resting ball can be centred
        there, but the cream pocket leather of some venues looks just like a ball, so such
        detections are discarded. A ball against both cushions in a corner is ~1.41 radii from
        the corner mouth, and one against a cushion is 1 radius from a middle mouth.
        """
        p = self.to_top_down((x, y))
        for i, pocket in enumerate(self.pockets):
            limit = (1.3 if i < 4 else 0.9) * self.ball_radius   # pockets 0-3 are the corners
            if np.linalg.norm(p - pocket) < limit:
                return True
        return False

    def ball_drop_px(self, x, y):
        """
        Screen distance (downwards) from a ball's visible centre to where it touches the cloth.
        The table mapping is for the cloth plane, while a ball's centre is one radius above it.
        Seen from a camera at elevation e, that radius shows as r*cos(e) pixels; sin(e) is how much
        the table is foreshortened along its length here (no drop at all in an overhead shot).
        """
        c = self.to_top_down((x, y))
        o = self.to_camera_view(c)
        jx = np.linalg.norm(self.to_camera_view(c + [1.0, 0.0]) - o)
        jy = np.linalg.norm(self.to_camera_view(c + [0.0, 1.0]) - o)
        sin_e = min(jx, jy) / max(jx, jy, 1e-9)
        return self.ball_radius_px(x, y) * float(np.sqrt(max(0.0, 1.0 - sin_e ** 2)))

    def ball_to_top_down(self, point):
        """Top-down position of a ball from its visible centre (via its contact point on the cloth)."""
        return self.to_top_down((point[0], point[1] + self.ball_drop_px(point[0], point[1])))

    def ball_to_camera(self, td):
        """Screen position of a ball centre at a top-down spot (the cloth point lifted by one radius)."""
        c = self.to_camera_view(td)
        return np.array([c[0], c[1] - self.ball_drop_px(c[0], c[1])], dtype=np.float32)

    def ball_radius_px(self, x, y):
        """Expected on-screen ball radius at a camera-view position."""
        c = self.to_top_down((x, y))
        a = self.to_camera_view(c - [self.ball_radius, 0])
        b = self.to_camera_view(c + [self.ball_radius, 0])
        return float(np.linalg.norm(a - b) / 2)

    def blockers(self, start, end, balls):
        """
        Balls (top-down centres) that a ball rolling from start to end would hit.
        A moving ball sweeps a corridor one ball wide, so it touches any ball whose
        centre comes within one diameter of its path.
        """
        seg = end - start
        seg_len2 = float(np.dot(seg, seg))
        hits = []
        for i, b in enumerate(balls):
            t = 0.0 if seg_len2 == 0 else np.clip(np.dot(b - start, seg) / seg_len2, 0, 1)
            if np.linalg.norm(start + seg * t - b) < 2 * self.ball_radius:
                hits.append(i)
        return hits

    def pocket_mouth(self, index):
        """Top-down centre of the opening a ball is potted into, past the cushion jaw."""
        jaw = np.asarray(self.pockets[index], dtype=float)
        w, h = float(self.target_w), float(self.target_h)
        if index < 4:
            outward = np.array([-1.0 if jaw[0] < w / 2 else 1.0, -1.0 if jaw[1] < h / 2 else 1.0])
            mm = self.CORNER_MOUTH_MM
        elif self.portrait:
            outward = np.array([-1.0 if jaw[0] < w / 2 else 1.0, 0.0])
            mm = self.MIDDLE_MOUTH_MM
        else:
            outward = np.array([0.0, -1.0 if jaw[1] < h / 2 else 1.0])
            mm = self.MIDDLE_MOUTH_MM
        mouth = jaw + outward / np.linalg.norm(outward) * (mm * self.scale)
        # Stay inside the outer cushion edge. Past it, the table mapping is an
        # extrapolation and the pocket jumps away from the hole.
        margin = 8 * self.scale
        mouth[0] = np.clip(mouth[0], margin, self.target_w - margin)
        mouth[1] = np.clip(mouth[1], margin, self.target_h - margin)
        return mouth

    def corridor(self, start, end, end_on_cloth=False):
        """Camera-view polygon of the one-ball-wide strip swept from start to end.
        end_on_cloth keeps the far end on the bed (a pocket opening) instead of lifting
        it to ball-centre height, which would park it on the cushion above the hole.
        """
        d = end - start
        n = np.linalg.norm(d)
        if n == 0:
            return None
        side = np.array([-d[1], d[0]]) / n * self.ball_radius
        pts = [start + side, end + side, end - side, start - side]
        out = []
        for i, p in enumerate(pts):
            on_cloth = end_on_cloth and i in (1, 2)
            out.append(self.to_camera_view(p) if on_cloth else self.ball_to_camera(p))
        return out

    # A middle pocket only takes balls arriving within this angle of straight in; steeper ones hit the jaw
    MIDDLE_POCKET_MAX_ENTRY_DEG = 45

    def pocket_difficulty(self, cue, target, index, max_cut_deg=75):
        """
        How hard it is to pot target (top-down) into pocket `index` from cue, or None if it can't go.
        Aiming error grows with both distances and with the cut (1 / cos(cut)); a middle pocket's
        opening also narrows as the ball comes in at an angle, and too steep an angle can't drop.
        Returns (difficulty, cut_deg).
        """
        pocket = self.pocket_mouth(index)
        shot, pot = target - cue, pocket - target
        d1, d2 = np.linalg.norm(shot), np.linalg.norm(pot)
        if d1 == 0 or d2 == 0:
            return None
        cut = float(np.degrees(np.arccos(np.clip(np.dot(shot, pot) / (d1 * d2), -1, 1))))
        if cut > max_cut_deg:
            return None
        width = 1.0
        if index >= 4:  # middle pockets: compare with the straight-in direction (across the table)
            straight_in = np.array([-1.0, 0.0] if pocket[0] < self.target_w / 2 else [1.0, 0.0]) if self.portrait \
                else np.array([0.0, -1.0] if pocket[1] < self.target_h / 2 else [0.0, 1.0])
            entry = float(np.degrees(np.arccos(np.clip(np.dot(pot / d2, straight_in), -1, 1))))
            if entry > self.MIDDLE_POCKET_MAX_ENTRY_DEG:
                return None
            width = np.cos(np.radians(entry))
        difficulty = (d1 + self.ball_radius) * d2 / (np.cos(np.radians(cut)) * width)
        return float(difficulty), cut

    def plan_shot(self, cue_cam, target_cam, balls_cam, prefer=None, max_cut_deg=75):
        """
        Chooses a pocket and checks both ball paths for obstructions.
        Inputs and outputs are CAMERA-view coordinates; balls_cam may include the
        cue and object ball (they are excluded automatically).
        Picks the easiest pot whose paths are clear (see pocket_difficulty); if every candidate
        is blocked, returns the easiest with blocked=True and the blocking balls.
        prefer: the pocket index chosen last time; kept unless another is clearly easier, so the
        drawing doesn't flip between two similar pockets from frame to frame.
        Returns None if no pocket can be reached.
        """
        td_cue = self.ball_to_top_down(cue_cam)
        td_target = self.ball_to_top_down(target_cam)
        others = []
        for b in balls_cam:
            td = self.ball_to_top_down(b[:2])
            # Skip the cue ball and the object ball themselves (and any double detection of them).
            # Checked on screen too: along the table's length the picture is so foreshortened that a
            # few pixels of detection noise is more than a ball's width on the table.
            near_td = min(np.linalg.norm(td - td_cue), np.linalg.norm(td - td_target)) <= self.ball_radius * 1.5
            near_px = any(np.hypot(*(np.asarray(b[:2]) - np.asarray(c))) <= 1.5 * self.ball_radius_px(*c)
                          for c in (cue_cam, target_cam))
            if not (near_td or near_px):
                others.append((td, b[:2]))
        others_td = [o[0] for o in others]

        plans = []
        for index in range(len(self.pockets)):
            rated = self.pocket_difficulty(td_cue, td_target, index, max_cut_deg)
            if rated is None:
                continue
            difficulty, cut = rated
            mouth = self.pocket_mouth(index)
            res = self.calculate_trajectories(td_cue, td_target, mouth)
            if res is None:
                continue
            ghost = res["impact_point"]
            aim_hits = self.blockers(td_cue, ghost, others_td)
            pot_hits = self.blockers(td_target, mouth, others_td)
            hit_ids = sorted(set(aim_hits) | set(pot_hits))
            # The hole is in the bed. Lifting the end to ball-centre height puts it on the
            # cushion above the opening, so the pot line is drawn on the cloth, into the hole.
            mouth_cam = self.to_camera_view(mouth)
            plans.append({
                "pocket_index": index,
                "difficulty": difficulty,
                "cut_deg": cut,
                "pocket": mouth_cam,
                "aim": (cue_cam, self.ball_to_camera(ghost)),
                "pot": (target_cam, mouth_cam),
                "cue_path": (self.ball_to_camera(res["cue_path"][0]), self.ball_to_camera(res["cue_path"][1])),
                "aim_corridor": self.corridor(td_cue, ghost),
                "pot_corridor": self.corridor(td_target, mouth, end_on_cloth=True),
                "blockers": [others[i][1] for i in hit_ids],
                "blocked": bool(hit_ids),
                "aim_blocked": bool(aim_hits),
                "pot_blocked": bool(pot_hits),
            })
        if not plans:
            return None
        plans.sort(key=lambda p: (p["blocked"], p["difficulty"]))
        best = plans[0]
        for p in plans:
            if p["pocket_index"] == prefer and p["blocked"] == best["blocked"] \
                    and p["difficulty"] <= 1.3 * best["difficulty"]:
                return p
        return best

    def update_homography(self, src_pts):
        """
        Updates the homography matrix based on 4 source points (corners of the table).
        src_pts should be a list/array of 4 [x, y] coordinates in order:
        Top-Left, Top-Right, Bottom-Right, Bottom-Left.
        """
        if len(src_pts) != 4:
            return False
            
        src_pts = np.array(src_pts, dtype=np.float32)
        self.M = cv2.getPerspectiveTransform(src_pts, self.dst_pts)
        self.M_inv = cv2.getPerspectiveTransform(self.dst_pts, src_pts)
        return True

    def to_top_down(self, point):
        """Converts a point from camera view to top-down view."""
        if self.M is None: return None
        pt = np.array([[[point[0], point[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.M)
        return transformed[0][0]

    def to_camera_view(self, point):
        """Converts a point from top-down view to camera view."""
        if self.M_inv is None: return None
        pt = np.array([[[point[0], point[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.M_inv)
        return transformed[0][0]

    def calculate_trajectories(self, cue_pos, target_pos, pocket_pos):
        """
        Calculates the aiming line, pot line, and post-collision cue ball trajectory.
        All inputs and outputs are in TOP-DOWN coordinates.
        """
        # 1. Pot line: from target ball to pocket
        pot_vec = pocket_pos - target_pos
        pot_dist = np.linalg.norm(pot_vec)
        if pot_dist == 0: return None
        pot_unit = pot_vec / pot_dist
        
        # 2. Impact point for cue ball
        # The cue ball must be at a distance of 2*ball_radius from the target ball's center
        # along the line connecting the pocket and the target ball.
        # Standard snooker ball diameter is 52.5mm.
        ball_radius = 52.5 / 2 * self.scale
        impact_pos = target_pos - pot_unit * (ball_radius * 2)
        
        # 3. Aiming line: from cue ball current pos to impact point
        aim_vec = impact_pos - cue_pos
        
        # 4. Separation angle (90 degree rule for stun shot)
        # Post-collision cue ball travels along the tangent line
        collision_vec = target_pos - impact_pos # Direction of target ball after hit
        collision_unit = collision_vec / np.linalg.norm(collision_vec)
        
        # Normal vector to collision direction
        # (Assuming the cue ball was moving towards the impact point)
        # The separation angle is 90 degrees if it's a stun shot (no roll).
        # Tangent direction: perpendicular to the collision_unit
        tangent_vec = np.array([-collision_unit[1], collision_unit[0]])
        
        # Determine which side of the tangent to go (away from the hit)
        # We check the dot product with the cue ball's incoming direction
        incoming_vec = impact_pos - cue_pos
        if np.dot(tangent_vec, incoming_vec) < 0:
            tangent_vec = -tangent_vec
            
        return {
            "pot_line": (target_pos, pocket_pos),
            "aim_line": (cue_pos, impact_pos),
            "impact_point": impact_pos,
            "cue_path": (impact_pos, self.ray_to_cushion(impact_pos, tangent_vec))
        }

if __name__ == "__main__":
    # Corners of the main-view table from 梯形桌面截屏.png (TL, TR, BR, BL)
    geo = SnookerGeometry()
    geo.update_homography([[652, 47], [1801, 49], [1989, 1122], [482, 1140]])

    cue = np.array([1247, 993])     # cue ball
    target = np.array([1222, 513])  # blue

    td_cue = geo.to_top_down(cue)
    td_target = geo.to_top_down(target)
    pocket = geo.best_pocket(td_cue, td_target)
    print("Chosen pocket (top-down):", pocket, "-> camera:", geo.to_camera_view(pocket))

    res = geo.calculate_trajectories(td_cue, td_target, pocket)
    print("Trajectories calculated in top-down view:")
    for k, v in res.items():
        print(f"{k}: {v}")
