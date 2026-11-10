"""
Keeps track of balls across frames of a live stream.

- Balls keep a stable id and a colour decided by majority vote over recent frames; a
  detection only continues a track of a compatible colour (a white is never a red).
- A ball that vanishes is only forgotten when the cloth shows through where it was
  (potted or moved); if something covers it (a player leaning over), its last
  position is kept.
- "Settled" means no ball has moved for settle_time seconds; lines are only drawn then.
"""
from collections import Counter, deque
import numpy as np

# Colours that one ball can plausibly flicker between from frame to frame
COLOUR_FAMILY = {"red": "reddish", "pink": "reddish", "brown": "reddish"}

def compatible(a, b):
    """Can a track labelled a be the same ball as a detection labelled b?"""
    if a == "unknown" or b == "unknown":
        return True
    return COLOUR_FAMILY.get(a, a) == COLOUR_FAMILY.get(b, b)

class Track:
    def __init__(self, track_id, pos, r, colour, now):
        self.id = track_id
        self.pos = np.array(pos, dtype=float)
        self.shown = self.pos.copy()  # position the last shot plan was computed from
        self.suspect = False          # last detection was a medium jump: move if it happens again
        self.r = r
        self.votes = Counter({colour: 1})
        self.recent = deque([colour], maxlen=15)  # latest labels; the colour follows these
        self.hits = 1
        self.last_seen = now
        self.visible = True

    @property
    def colour(self):
        # Majority of the recent labels, ignoring "unknown" unless that's all we have
        for colour, _ in Counter(self.recent).most_common():
            if colour != "unknown":
                return colour
        return "unknown"

    @property
    def confirmed(self):
        return self.hits >= 2

class BallTracker:
    def __init__(self, move_tol=0.6, settle_time=0.6, match_radius=1.5):
        # Movement (in ball radii) between detections that counts as "moving". Smaller shifts are
        # detection noise on a still ball: they're smoothed out instead of making the lines jump.
        self.move_tol = move_tol
        self.settle_time = settle_time    # seconds without movement before balls count as settled
        self.match_radius = match_radius  # max jump (in ball radii) to treat a detection as the same ball
        self.reset()

    def reset(self):
        self.tracks = {}
        self.next_id = 0
        self.last_motion = None
        self.version = 0  # bumps whenever ball positions change, for caching shot plans

    def update(self, detections, now, is_cloth_at):
        """
        detections: list of (x, y, r, colour).
        is_cloth_at(x, y, r): True if bare cloth is visible at that spot.
        """
        moved = False
        changed = False
        unmatched_dets = set(range(len(detections)))
        unmatched_tracks = set(self.tracks)

        # Greedy nearest-first matching. A detection right on top of a track is the same ball even
        # if its colour label flickered (a warm-lit white read as brown for a frame, say).
        pairs = []
        for tid, t in self.tracks.items():
            for di, (x, y, r, colour) in enumerate(detections):
                d = np.hypot(x - t.pos[0], y - t.pos[1])
                same_spot = d < self.move_tol * max(r, t.r)
                if d < self.match_radius * max(r, t.r) and (compatible(t.colour, colour) or same_spot):
                    pairs.append((d, tid, di))
        for d, tid, di in sorted(pairs):
            if tid not in unmatched_tracks or di not in unmatched_dets:
                continue
            t = self.tracks[tid]
            x, y, r, colour = detections[di]
            # A real move shows up in two detections running (a struck ball keeps going); a one-off
            # jump is a detection glitch
            if d > self.move_tol * r and t.suspect:
                moved |= self._move(t, x, y)
                changed = True
            elif d > self.move_tol * r:
                t.suspect = True
            else:
                t.suspect = False
                # Jitter on a still ball: average it out; replan only once it adds up
                t.pos += 0.2 * (np.array([x, y]) - t.pos)
                if np.hypot(*(t.pos - t.shown)) > 0.2 * r:
                    t.shown = t.pos.copy()
                    changed = True
            changed |= self._seen(t, r, colour, now)
            unmatched_tracks.discard(tid)
            unmatched_dets.discard(di)

        # Second pass: a ball that vanished from its spot while a new one appeared close by has
        # simply rolled there (too far for the first pass, e.g. the last slow roll before stopping).
        # Without this the old spot would be kept as a "covered" ball: a phantom next to the real one.
        pairs = []
        for tid in unmatched_tracks:
            t = self.tracks[tid]
            for di in unmatched_dets:
                x, y, r, colour = detections[di]
                d = np.hypot(x - t.pos[0], y - t.pos[1])
                if d < 3 * max(r, t.r) and compatible(t.colour, colour):
                    pairs.append((d, tid, di))
        for d, tid, di in sorted(pairs):
            if tid not in unmatched_tracks or di not in unmatched_dets:
                continue
            t = self.tracks[tid]
            x, y, r, colour = detections[di]
            moved |= self._move(t, x, y)
            self._seen(t, r, colour, now)
            changed = True
            unmatched_tracks.discard(tid)
            unmatched_dets.discard(di)

        # Balls not seen this time: gone if the cloth shows through, otherwise covered
        for tid in list(unmatched_tracks):
            t = self.tracks[tid]
            if is_cloth_at(t.pos[0], t.pos[1], t.r):
                if t.confirmed:
                    moved = True
                del self.tracks[tid]
                changed = True
            else:
                t.visible = False

        # New balls; they only count once seen twice (see Track.confirmed)
        for di in unmatched_dets:
            x, y, r, colour = detections[di]
            self.tracks[self.next_id] = Track(self.next_id, (x, y), r, colour, now)
            self.next_id += 1
            changed = True

        # A new ball that was confirmed this round means something rolled into view
        for t in self.tracks.values():
            if t.hits == 2 and t.last_seen == now and self.last_motion is not None:
                moved = True

        if moved or self.last_motion is None:
            self.last_motion = now
        if changed:
            self.version += 1

    @staticmethod
    def _move(t, x, y):
        """Moves a track to a new spot; returns True if that counts as a ball moving."""
        t.pos = np.array([x, y], dtype=float)
        t.shown = t.pos.copy()
        t.suspect = False
        return t.confirmed

    @staticmethod
    def _seen(t, r, colour, now):
        """Records a sighting; returns True if the track had been hidden (so the drawing changes)."""
        t.r = r
        t.votes[colour] += 1
        t.recent.append(colour)
        t.hits += 1
        t.last_seen = now
        was_hidden = not t.visible
        t.visible = True
        return was_hidden

    def settled(self, now):
        return self.last_motion is not None and now - self.last_motion >= self.settle_time and bool(self.tracks)

    def balls(self):
        """Confirmed balls (including covered ones at their last known spot)."""
        return [t for t in self.tracks.values() if t.confirmed]

    def cue_ball(self):
        whites = [t for t in self.balls() if t.colour == "white"]
        if not whites:
            return None
        # Prefer the one actually in view, then the one most consistently seen as white lately
        return max(whites, key=lambda t: (t.visible, t.recent.count("white")))

    def nearest(self, pos, max_radii=2.0):
        best = None
        for t in self.balls():
            d = np.hypot(pos[0] - t.pos[0], pos[1] - t.pos[1])
            if d < max_radii * t.r and (best is None or d < best[0]):
                best = (d, t)
        return None if best is None else best[1]
