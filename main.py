import os
import sys
import time
from collections import Counter
import traceback
import ctypes
import ctypes.util
import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtCore import Qt, QTimer, QPointF, QSettings, QRect
from PyQt6.QtGui import QPainter, QColor, QPen, QPolygonF, QPixmap, QImage
import mss
from analyzer import SnookerAnalyzer

RESIZE_HANDLE = 20  # px, bottom-right corner area used for resizing

# Overlay colours for each detected ball colour
BALL_COLOURS = {
    "red": QColor(255, 60, 60), "yellow": QColor(255, 230, 0), "green": QColor(0, 200, 90),
    "brown": QColor(160, 90, 40), "blue": QColor(60, 140, 255), "pink": QColor(255, 130, 200),
    "black": QColor(40, 40, 40), "white": QColor(255, 255, 255), "unknown": QColor(150, 150, 150),
}
STATE_TEXT = {
    "locking": ("Locking table... (show the main camera view)", QColor(255, 165, 0)),
    "other_view": ("Other camera view - lines hidden", QColor(255, 80, 80)),
    "moving": ("Balls moving - waiting for them to settle", QColor(255, 165, 0)),
    "settled": ("Balls settled", QColor(0, 255, 0)),
}

def setup_macos_window(widget):
    """
    macOS window tweaks via the Objective-C runtime:
    - sharingType = NSWindowSharingNone, so screen grabs (mss) see the video underneath
      instead of our own overlay lines and text;
    - collectionBehavior += canJoinAllSpaces | fullScreenAuxiliary, so the overlay also shows
      on other desktops and on top of a player in full-screen mode (which gets its own desktop).
    """
    if sys.platform != "darwin":
        return False
    objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
    objc.sel_registerName.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]

    def send(obj, sel, *args, restype=ctypes.c_void_p, argtypes=()):
        fn = ctypes.CFUNCTYPE(restype, ctypes.c_void_p, ctypes.c_void_p, *argtypes)(("objc_msgSend", objc))
        return fn(obj, objc.sel_registerName(sel.encode()), *args)

    ns_window = send(int(widget.winId()), "window")
    if not ns_window:
        return False
    send(ns_window, "setSharingType:", 0, restype=None, argtypes=(ctypes.c_ulong,))
    CAN_JOIN_ALL_SPACES, FULL_SCREEN_AUXILIARY = 1 << 0, 1 << 8
    behavior = send(ns_window, "collectionBehavior", restype=ctypes.c_ulong)
    send(ns_window, "setCollectionBehavior:", behavior | CAN_JOIN_ALL_SPACES | FULL_SCREEN_AUXILIARY,
         restype=None, argtypes=(ctypes.c_ulong,))
    return send(ns_window, "collectionBehavior", restype=ctypes.c_ulong)

class OverlayWindow(QMainWindow):
    def __init__(self, debug=False):
        super().__init__()
        self.debug = debug          # print a status line every second (no image content)
        self.last_debug_print = 0
        self.last_frame = None
        self.setWindowTitle("Snooker Vision Assistant")

        # Set window to be transparent and stay on top
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Initial size and position (can be adjusted by user)
        # Last position and size the user set, or a default
        self.settings = QSettings("SnookerAssistant", "Overlay")
        saved = self.settings.value("geometry")
        if isinstance(saved, QRect) and saved.width() >= 200 and saved.height() >= 150:
            self.setGeometry(saved)
        else:
            self.setGeometry(100, 100, 1280, 720)

        self.sct = mss.MSS()
        self.analyzer = SnookerAnalyzer()

        self.frame_scale = 1.0  # captured-frame pixels per widget pixel
        self.last_geometry = None
        self.show_balls = True

        # Calibration state
        self.calibration_pts = []
        self.is_calibrating = False

        # Window move / resize state
        self.drag_offset = None
        self.resizing = False

        # Last error, shown in the overlay
        self.last_error = None
        self.last_error_print = 0

        # Timer for screenshotting and analysis
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(50) # 20 FPS

    def showEvent(self, event):
        super().showEvent(event)
        if not getattr(self, "_macos_setup", False):
            self._macos_setup = setup_macos_window(self)

    def w(self, pt):
        """Frame pixels -> widget QPointF."""
        return QPointF(pt[0] / self.frame_scale, pt[1] / self.frame_scale)

    def to_frame(self, pt):
        return (pt[0] * self.frame_scale, pt[1] * self.frame_scale)

    def update_frame(self):
        geo = self.geometry()
        monitor = {"top": geo.y(), "left": geo.x(), "width": geo.width(), "height": geo.height()}

        try:
            # Moving or resizing the window changes what's underneath: start over
            if self.last_geometry is not None and geo != self.last_geometry:
                self.analyzer.reset()
            self.last_geometry = geo

            sct_img = self.sct.grab(monitor)
            frame = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
            # On some setups the grab is in physical (Retina) pixels, not widget pixels
            self.frame_scale = frame.shape[1] / max(geo.width(), 1)

            if not self.is_calibrating and not self.drag_offset and not self.resizing:
                self.analyzer.process(frame, time.time())
            self.last_frame = frame
            if self.debug and time.time() - self.last_debug_print >= 1:
                self.last_debug_print = time.time()
                self.print_status(frame)

            self.last_error = None
            self.update()
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            # Print the full traceback at most every 5 s so the console isn't flooded at 20 FPS
            now = time.time()
            if now - self.last_error_print > 5:
                self.last_error_print = now
                traceback.print_exc()
            self.update()

    def print_status(self, frame):
        a = self.analyzer
        balls = a.tracker.balls()
        colours = Counter(t.colour for t in balls)
        hidden = sum(1 for t in balls if not t.visible)
        line = (f"[{time.strftime('%H:%M:%S')}] frame={frame.shape[1]}x{frame.shape[0]} state={a.state} "
                f"views={len(a.lock.views)} active={a.lock.active} balls={len(balls)} covered={hidden} "
                f"colours={dict(colours)}")
        if a.target is not None:
            line += f" target={a.target.colour}"
        if a.plan is not None:
            line += f" plan: cut={a.plan['cut_deg']:.0f} blocked={a.plan['blocked']} blockers={len(a.plan['blockers'])}"
        elif a.no_pocket:
            line += " plan: no pocket"
        print(line, flush=True)

    def save_snapshot(self):
        """S key: saves what the overlay sees and what it draws, for checking and for tests."""
        if self.last_frame is None:
            return
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(os.path.join(SNAPSHOT_DIR, f"{stamp}_frame.png"), self.last_frame)
        frame = cv2.resize(self.last_frame, (self.width(), self.height()))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        composite = QPixmap.fromImage(QImage(rgb.data, rgb.shape[1], rgb.shape[0], 3 * rgb.shape[1],
                                             QImage.Format.Format_RGB888))
        painter = QPainter(composite)
        painter.drawPixmap(0, 0, self.grab())
        painter.end()
        composite.save(os.path.join(SNAPSHOT_DIR, f"{stamp}_overlay.png"))
        self.snapshot_msg = (f"Saved snapshot {stamp}", time.time())
        print(f"[snapshot] {stamp}  {self.analyzer.state}", flush=True)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        a = self.analyzer

        # Draw border and resize handle
        painter.setPen(QPen(QColor(0, 255, 0, 150), 2))
        painter.drawRect(0, 0, self.width()-1, self.height()-1)
        w, h = self.width(), self.height()
        painter.drawLine(w - RESIZE_HANDLE, h - 2, w - 2, h - RESIZE_HANDLE)
        painter.drawLine(w - RESIZE_HANDLE // 2, h - 2, w - 2, h - RESIZE_HANDLE // 2)

        # Info text, on a dark backdrop so it stays readable over the broadcast's own graphics
        painter.fillRect(4, 4, min(self.width() - 8, 900), 104, QColor(0, 0, 0, 150))
        painter.setPen(QColor(255, 255, 255))
        orientation = "Portrait" if a.geometry.portrait else "Landscape"
        painter.drawText(10, 20, "Snooker Assistant [Click: Target | R: Reset Target | C: Calibrate | A: Re-lock table | "
                                 f"O: Orientation ({orientation}) | B: Show balls | S: Snapshot | Right-drag: Move | Corner: Resize | Esc: Quit]")

        text, colour = STATE_TEXT[a.state]
        if a.lock.locked and a.state != "locking":
            text += f"  |  {len(a.tracker.balls())} balls"
        painter.setPen(colour)
        painter.drawText(10, 40, f"Status: {text}")

        if a.state in ("settled", "moving"):
            # Detected balls with their colour label
            if self.show_balls:
                for t in a.tracker.balls():
                    pen = QPen(BALL_COLOURS.get(t.colour, BALL_COLOURS["unknown"]), 2,
                               Qt.PenStyle.SolidLine if t.visible else Qt.PenStyle.DotLine)
                    painter.setPen(pen)
                    painter.drawEllipse(self.w(t.pos), t.r / self.frame_scale + 4, t.r / self.frame_scale + 4)

            cue, target = a.cue_ball, a.target
            if cue is not None:
                painter.setPen(QPen(QColor(255, 255, 255), 3))
                painter.drawEllipse(self.w(cue.pos), cue.r / self.frame_scale + 7, cue.r / self.frame_scale + 7)
            if target is not None:
                painter.setPen(QPen(QColor(255, 255, 0), 3))
                painter.drawEllipse(self.w(target.pos), target.r / self.frame_scale + 7, target.r / self.frame_scale + 7)
                painter.drawText(10, 60, f"Target: {target.colour}")

        if a.no_pocket and a.state == "settled":
            painter.setPen(QColor(255, 165, 0))
            painter.drawText(10, 80, "No pottable pocket for this ball (cut too thin)")
        if self.last_error:
            painter.setPen(QColor(255, 80, 80))
            painter.drawText(10, 100, f"Error: {self.last_error}")
        msg = getattr(self, "snapshot_msg", None)
        if msg and time.time() - msg[1] < 2:
            painter.setPen(QColor(0, 255, 255))
            painter.drawText(10, 100, msg[0])

        # Draw Trajectories
        t = a.plan if a.state == "settled" else None
        if t:
            blocked = t["blocked"]

            # One-ball-wide corridors the balls sweep; red when another ball is in the way
            painter.setPen(Qt.PenStyle.NoPen)
            for key, color in (("aim", QColor(255, 255, 255, 50)), ("pot", QColor(255, 255, 0, 50))):
                poly = t[key + "_corridor"]
                if poly is not None:
                    painter.setBrush(QColor(255, 0, 0, 90) if t[key + "_blocked"] else color)
                    painter.drawPolygon(QPolygonF([self.w(p) for p in poly]))
            painter.setBrush(Qt.BrushStyle.NoBrush)

            # Aim line (White)
            painter.setPen(QPen(QColor(255, 255, 255, 200), 2, Qt.PenStyle.DashLine))
            painter.drawLine(self.w(t["aim"][0]), self.w(t["aim"][1]))

            # Pot line (Yellow)
            painter.setPen(QPen(QColor(255, 255, 0, 200), 2))
            painter.drawLine(self.w(t["pot"][0]), self.w(t["pot"][1]))

            # Post-impact cue path (Cyan)
            painter.setPen(QPen(QColor(0, 255, 255, 200), 2, Qt.PenStyle.DotLine))
            painter.drawLine(self.w(t["cue_path"][0]), self.w(t["cue_path"][1]))

            # Balls in the way
            painter.setPen(QPen(QColor(255, 0, 0), 3))
            for b in t["blockers"]:
                painter.drawEllipse(self.w(b), 14, 14)

            painter.setPen(QColor(255, 60, 60) if blocked else QColor(0, 255, 0))
            msg = (f"BLOCKED: {len(t['blockers'])} ball(s) in the way" if blocked
                   else f"Clear shot, cut {t['cut_deg']:.0f} deg")
            painter.drawText(10, 80, msg)

        # Draw Calibration Points
        if self.is_calibrating:
            painter.setPen(QColor(255, 165, 0))
            painter.drawText(10, 60, f"CALIBRATING: Click 4 Corners (TL, TR, BR, BL) - Points: {len(self.calibration_pts)}")
            for pt in self.calibration_pts:
                painter.drawEllipse(pt, 5, 5)

    def in_resize_handle(self, pos):
        return pos.x() >= self.width() - RESIZE_HANDLE and pos.y() >= self.height() - RESIZE_HANDLE

    def mousePressEvent(self, event):
        pos = event.position()
        if event.button() == Qt.MouseButton.RightButton:
            self.drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            return
        if event.button() == Qt.MouseButton.LeftButton and self.in_resize_handle(pos):
            self.resizing = True
            return

        if self.is_calibrating:
            self.calibration_pts.append(pos)
            if len(self.calibration_pts) == 4:
                self.analyzer.calibrate_manual([self.to_frame((p.x(), p.y())) for p in self.calibration_pts])
                self.is_calibrating = False
                self.calibration_pts = []
        elif event.button() == Qt.MouseButton.LeftButton:
            self.analyzer.select_target(self.to_frame((pos.x(), pos.y())))

        self.update()

    def mouseMoveEvent(self, event):
        if self.drag_offset is not None:
            self.move(event.globalPosition().toPoint() - self.drag_offset)
        elif self.resizing:
            pos = event.position()
            self.resize(max(int(pos.x()), 200), max(int(pos.y()), 150))

    def mouseReleaseEvent(self, event):
        if self.drag_offset is not None or self.resizing:
            self.settings.setValue("geometry", self.geometry())
        self.drag_offset = None
        self.resizing = False

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_C:
            self.is_calibrating = True
            self.calibration_pts = []
        elif event.key() == Qt.Key.Key_R:
            self.analyzer.clear_target()
        elif event.key() == Qt.Key.Key_A:
            self.analyzer.reset()
        elif event.key() == Qt.Key.Key_O:
            self.analyzer.toggle_orientation()
        elif event.key() == Qt.Key.Key_B:
            self.show_balls = not self.show_balls
        elif event.key() == Qt.Key.Key_S:
            self.save_snapshot()
        elif event.key() == Qt.Key.Key_Escape:
            self.close()
        self.update()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OverlayWindow(debug="--debug" in sys.argv)
    window.show()
    sys.exit(app.exec())
