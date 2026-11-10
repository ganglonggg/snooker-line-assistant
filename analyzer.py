"""
Per-frame pipeline for a live stream, independent of the GUI so it can be tested
by feeding it frames directly. All coordinates are in captured-frame pixels.

States:
  locking     no full-table view learned yet (waiting for one to hold steady)
  other_view  camera is on a player, a close-up, or part of the table: draw nothing
  moving      balls are rolling (or just appeared): draw nothing until they settle
  settled     balls are still: plan and draw the selected shot
"""
import numpy as np
from vision import SnookerVision, TableLock
from geometry import SnookerGeometry
from tracking import BallTracker

class SnookerAnalyzer:
    def __init__(self, detect_interval=0.2):
        self.detect_interval = detect_interval  # seconds between full ball detections
        self.vision = SnookerVision()
        self.geometry = SnookerGeometry()
        self.lock = TableLock()
        self.tracker = BallTracker()
        self.reset()

    def reset(self):
        """Forget the table and all balls (window moved/resized, new stream...)."""
        self.lock.reset()
        self.tracker.reset()
        self.geometry.M = None
        self.geometry.M_inv = None
        self.clear_target()
        self.state = "locking"
        self.plan = None
        self.no_pocket = False
        self._last_detect = None
        self._frame_shape = None
        self._view = None
        self._carry_target = None  # (top-down position, colour) to re-find after a camera switch

    def clear_target(self):
        self._pocket_pref = None   # pocket chosen for the current target, kept while it stays sensible
        self.target_id = None
        self._target_start = None
        self.plan = None
        self._plan_key = None
        self.no_pocket = False

    # --- user actions -------------------------------------------------------

    def calibrate_manual(self, corners):
        """corners: TL, TR, BR, BL in frame pixels."""
        if self._frame_shape is None:
            return
        self.lock.lock_manual(corners, self._frame_shape)
        self.geometry.update_homography(self.lock.corners)
        self._view = id(self.lock.views[self.lock.active])
        self.tracker.reset()
        self.clear_target()

    def toggle_orientation(self):
        self.geometry.set_orientation(not self.geometry.portrait, self.lock.corners)
        self._plan_key = None

    def select_target(self, point):
        """Selects the ball nearest to a click. Returns the track or None."""
        track = self.tracker.nearest(point)
        cue = self.tracker.cue_ball()
        if track is None or (cue is not None and track.id == cue.id):
            return None
        self.target_id = track.id
        self._target_start = track.pos.copy()
        self._plan_key = None
        self._pocket_pref = None
        return track

    # --- per-frame ----------------------------------------------------------

    @property
    def target(self):
        return self.tracker.tracks.get(self.target_id) if self.target_id is not None else None

    @property
    def cue_ball(self):
        return self.tracker.cue_ball()

    def process(self, frame, now):
        if self._frame_shape is not None and frame.shape != self._frame_shape:
            self.reset()
        self._frame_shape = frame.shape

        is_table, approx = self.vision.detect_main_view(frame)
        is_main = self.lock.observe(is_table, approx, self.vision.table_mask, self.geometry.portrait)
        if is_main:
            view = id(self.lock.views[self.lock.active])
            if view != self._view:
                self._switch_view(view)

        if not is_main:
            self.state = "other_view" if self.lock.locked else "locking"
            self.plan = None
            self._plan_key = None
            return

        # Detect inside the locked table area, so a player leaning over doesn't reshape it
        self.vision.table_mask = self.lock.mask
        if self._last_detect is None or now - self._last_detect >= self.detect_interval:
            self._last_detect = now
            self._detect(frame, now)

        # After a camera switch, re-find the selected ball by table position and colour
        if self._carry_target is not None and self.tracker.balls():
            self._refind_target()

        # The selected ball was potted or knocked away: drop the selection
        target = self.target
        if self.target_id is not None and (
                target is None or np.linalg.norm(target.pos - self._target_start) > target.r):
            self.clear_target()
            target = None

        if not self.tracker.settled(now):
            self.state = "moving"
            self.plan = None
            self._plan_key = None
            return
        self.state = "settled"

        cue = self.cue_ball
        if target is None or cue is None:
            self.plan = None
            return
        key = (self.tracker.version, self.target_id, cue.id, self.geometry.portrait)
        if key != self._plan_key:
            self._plan_key = key
            others = [t.pos for t in self.tracker.balls()]
            self.plan = self.geometry.plan_shot(cue.pos, target.pos, others, prefer=self._pocket_pref)
            self.no_pocket = self.plan is None
            if self.plan is not None:
                self._pocket_pref = self.plan["pocket_index"]

    def _switch_view(self, view):
        """A different camera angle: ball positions on screen all change."""
        target = self.target
        if target is not None and self.geometry.M is not None:
            self._carry_target = (self.geometry.to_top_down(target.pos), target.colour)
        self._view = view
        self.geometry.update_homography(self.lock.corners)
        self.tracker.reset()
        self.target_id = None
        self._target_start = None
        self._plan_key = None
        self._last_detect = None

    def _refind_target(self):
        pos, colour = self._carry_target
        self._carry_target = None
        best = None
        for t in self.tracker.balls():
            if t.colour != colour:
                continue
            d = np.linalg.norm(self.geometry.to_top_down(t.pos) - pos)
            if d < 2 * self.geometry.ball_radius and (best is None or d < best[0]):
                best = (d, t)
        if best is not None:
            self.target_id = best[1].id
            self._target_start = best[1].pos.copy()

    def _detect(self, frame, now):
        geo = self.geometry
        detections = self.vision.find_balls(frame, geo.playing_area(), geo.ball_radius_px,
                                            exclude=geo.at_pocket_mouth)
        self.tracker.update(detections, now,
                            lambda x, y, r: self.vision.is_cloth_at(frame, x, y, r))
