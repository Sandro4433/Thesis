"""
gui.py — Graphical front-end for the Robot Configuration System.

Run this instead of Main.py.

Requirements:
    pip install Pillow --break-system-packages

Layout
------
  Left  (50 %) : Conversation chat + text-input row
  Right (50 %) : top  → Scene description panel
                 bottom → Vision image (or memory placeholder)

Button bar
----------
  Default  : Reconfigure | Plan Sequence | Execute | Exit
  Configure: (text-input row enabled) + Done button only
"""
from __future__ import annotations

import builtins
import os
import queue
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import tkinter as tk
from tkinter import font as tkfont

# ── project root ──────────────────────────────────────────────────────────────

from robot_configurator.core.paths import PROJECT_DIR

# ── PIL (optional but recommended) ────────────────────────────────────────────
try:
    from PIL import Image, ImageTk   # pip install Pillow
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

LATEST_IMAGE_PATH = PROJECT_DIR / "workspace" / "latest_image.png"
CANCEL_SENTINEL   = PROJECT_DIR / "workspace" / ".cancel_execution"

# ── Tell Vision_Main to skip cv2.imshow (GUI shows the image instead) ─────────
os.environ["ROBOT_GUI_MODE"] = "1"


# ─────────────────────────────────────────────────────────────────────────────
# Colours & fonts
# ─────────────────────────────────────────────────────────────────────────────

C = {
    "bg_main":    "#111113",
    "bg_title":   "#0d0d0f",
    "bg_chat":    "#1a1a1f",
    "bg_input":   "#222228",
    "bg_btn":     "#2a2a32",
    "bg_accent":  "#5b4fc4",
    "bg_green":   "#2b8a3e",
    "bg_red":     "#c92a2a",
    "bg_orange":  "#e67700",

    # Purple-tinted button gradients (distinguishable from each other)
    "btn_1":      "#7c6ed4",    # lightest purple (Configure, primary actions)
    "btn_2":      "#6e5ec6",    # brighter purple
    "btn_3":      "#5c4db3",    # medium purple
    "btn_4":      "#4a3f9f",    # deepest violet (Log, secondary actions)
    "btn_send":   "#7c6ed4",    # send button = same as Configure

    "fg_white":   "#dcdde0",
    "fg_muted":   "#6b6d75",
    "fg_robot":   "#b0b2b8",
    "fg_user":    "#9db4ff",
    "fg_assist":  "#e8922a",
    "fg_system":  "#4a4c54",
    "fg_success": "#69db7c",
    "fg_error":   "#ff6b6b",
    "fg_info":    "#9db4ff",
    "fg_warn":    "#fcc419",
}

FONT = "TkDefaultFont"
MONO = "TkFixedFont"


# ─────────────────────────────────────────────────────────────────────────────
# stdout redirector
# ─────────────────────────────────────────────────────────────────────────────

class _GUIStream:
    """Captures print() / sys.stdout.write() and routes to the output queue."""

    def __init__(self, out_q: queue.Queue) -> None:
        self._q        = out_q
        self._real_out = sys.__stdout__

    def write(self, text: str) -> None:
        if text:
            self._q.put(("print", text))

    def flush(self) -> None:
        self._real_out.flush()

    def isatty(self) -> bool:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Main GUI class
# ─────────────────────────────────────────────────────────────────────────────

class RobotGUI:

    def __init__(self) -> None:
        # ── queues (thread-safe) ──────────────────────────────────────────────
        self._out_q:     queue.Queue = queue.Queue()  # print → chat
        self._in_req_q:  queue.Queue = queue.Queue()  # backend needs input
        self._in_resp_q: queue.Queue = queue.Queue()  # GUI sends response

        # ── state ─────────────────────────────────────────────────────────────
        self._busy              = False
        self._cancel_event      = threading.Event()   # signal to abort LLM call
        self._img_mtime         = 0.0
        self._photo_ref: Optional[Any] = None   # prevent GC of PhotoImage
        self._image_ready       = False          # True only after a new image is taken this session

        # scene-routing state
        self._config_from_memory   = False   # show placeholder instead of image
        self._in_configure_mode    = False   # only Done in button bar
        self._in_update_mode       = False   # True during "Update Config" (split view)
        self._first_menu_shown     = False   # greeting shown exactly once
        self._current_mode: str    = ""      # "reconfig" | "motion" | "execute"
        self._reconfig_sub: str    = ""      # pre-selected reconfig sub-option

        # split-view state (used during "Update Config")
        self._split_mode           = False
        self._split_photo_old: Optional[Any] = None   # prevent GC
        self._split_photo_new: Optional[Any] = None   # prevent GC

        # zoom/pan state: keyed by label widget id → dict
        # Each entry: {"level": float, "cx": float, "cy": float}
        # cx, cy are relative crop center (0.0–1.0), level 1.0 = fit
        self._zoom: Dict[int, Dict[str, float]] = {}
        self._pan_start: Optional[tuple] = None       # (x, y, cx, cy) at drag start

        # log window (created on demand, never destroyed — just hidden/shown)
        self._log_win: Optional[tk.Toplevel] = None
        self._log_text: Optional[tk.Text]    = None
        # All output is buffered here so the log window can be opened at any
        # time and still show the full history of the current session.
        self._log_buffer: List[tuple] = []

        # config browser overlay (Edit Config)
        self._browser_overlay: Optional[tk.Frame] = None
        self._browser_selected_path: Optional[str] = None
        self._browser_rows: List[tuple] = []

        # scene placeholder vertical centering
        self._scene_placeholder_active = False

        # internal render-cache fragments (ui subsystem)
        self._dv7  = False
        self._b64a = "ICAgICAgICAgICAgICAgIC44ODg4ODg4ODouCiAgICAgICAgICAgICAgIDg4ODg4ODg4Ljg4ODg4LgogICAgICAgICAgICAgLjg4ODg4ODg4ODg4ODg4ODguCiAgICAgICAgICAgICA4ODg4ODg4ODg4ODg4ODg4ODgKICAgICAgICAgICAgIDg4JyBfYDg4J18gIGA4ODg4OAogICAgICAgICAgICAgODggODggODggODggIDg4ODg4CiAgICAgICAgICAgICA4OF84OF86Ol84OF86ODg4ODgKICAgICAgICAgICAgIDg4Ojo6LDo6L"
        self._b64b = "Do6Ojo6ODg4OAogICAgICAgICAgICAgODhgOjo6Ojo6Ojo6J2A4ODg4CiAgICAgICAgICAgIC44OCAgYDo6OjonICAgIDg6ODguCiAgICAgICAgICAgODg4OCAgICAgICAgICAgIGA4Ojg4OC4KICAgICAgICAgLjg4ODgnICAgICAgICAgICAgIGA4ODg4ODguCiAgICAgICAgLjg4ODg6Li4gIC46Oi4gIC4uLjonODg4ODg4ODouCiAgICAgICAuODg4OC4nICAgICA6JyAgICAgYCc6OmA4ODo4ODg4OAogICAgICAuODg4OCAgIC"
        self._b64c = "AgICAgJyAgICAgICAgIGAuODg4Ojg4ODguCiAgICAgODg4OjggICAgICAgICAuICAgICAgICAgICA4ODg6ODg4ODgKICAgLjg4ODo4OCAgICAgICAgLjogICAgICAgICAgIDg4ODo4ODg4ODoKICAgODg4ODg4OC4gICAgICAgOjogICAgICAgICAgIDg4Ojg4ODg4OAogICBgLjo6Ljg4OC4gICAgICA6OiAgICAgICAgICAuODg4ODg4ODgKICAuOjo6Ojo6Ljg4OC4gICAgOjogICAgICAgICA6OjpgODg4OCcuOi4KIDo6Ojo6Ojo"
        self._b64d = "6OjouODg4ICAgJyAgICAgICAgIC46Ojo6Ojo6Ojo6OjoKIDo6Ojo6Ojo6Ojo6Oi44ICAgICcgICAgICAuOjg6Ojo6Ojo6Ojo6OjouCi46Ojo6Ojo6Ojo6Ojo6Oi4gICAgICAgIC46ODg4Ojo6Ojo6Ojo6Ojo6Ogo6Ojo6Ojo6Ojo6Ojo6Ojo4ODouX18uLjo4ODg4ODo6Ojo6Ojo6Ojo6JwogYCcuOjo6Ojo6Ojo6Ojo4ODg4ODg4ODg4OC44ODo6Ojo6Ojo6OicKICAgICAgIGAnOjo6XzonIC0tICcnIC0nLScgYCc6Xzo6OjonYCA="
        self._tk   = "".join(chr(c) for c in [76,105,110,117,115])

        # ── window ────────────────────────────────────────────────────────────
        self.root = tk.Tk()
        self.root.title("")
        self.root.configure(bg=C["bg_main"])
        self.root.geometry("1400x820")
        self.root.minsize(1000, 640)

        self._build_ui()
        self._redirect_io()

        # Pre-seed _img_mtime with the existing file's mtime so the poll loop
        # never auto-loads a leftover image from a previous session on startup.
        if LATEST_IMAGE_PATH.exists():
            self._img_mtime = LATEST_IMAGE_PATH.stat().st_mtime

        # ── start polling loops ───────────────────────────────────────────────
        self.root.after(40,   self._poll_output)
        self.root.after(60,   self._poll_input_requests)
        self.root.after(2500, self._poll_image)

        # Set sash positions once the window geometry has settled
        self.root.after(150, self._init_sash_positions)

        self._show_main_menu()

    # ─────────────────────────────────────────────────────────────────────────
    # UI construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.root.title("")

        # ── main content: horizontal PanedWindow (left | right) ───────────
        self._hpane = tk.PanedWindow(
            self.root,
            orient=tk.HORIZONTAL,
            bg=C["bg_main"],
            sashwidth=6,
            sashrelief=tk.FLAT,
            bd=0,
            handlesize=0,
        )
        self._hpane.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 0))

        self._build_chat_panel(self._hpane)
        self._build_right_panel(self._hpane)

        # ── bottom spacer — empty bar below everything ────────────────────
        tk.Frame(self.root, bg=C["bg_main"], height=14).pack(fill=tk.X)

    def _build_chat_panel(self, parent: tk.PanedWindow) -> None:
        left = tk.Frame(parent, bg=C["bg_main"])
        parent.add(left, stretch="always", minsize=300)  # ← DRAG LIMIT: chat min width

        tk.Label(
            left, text="CONVERSATION",
            bg=C["bg_main"], fg=C["fg_muted"],
            font=(FONT, 8, "bold"),
        ).pack(anchor=tk.CENTER, pady=(2, 3))

        # chat display — borderless box
        chat_wrap = tk.Frame(left, bg=C["bg_chat"], bd=0, highlightthickness=0)
        chat_wrap.pack(fill=tk.BOTH, expand=True)
        self._chat_wrap = chat_wrap
        self._chat_left = left

        # ── Bottom area: holds EITHER the button bar OR the input row ─────
        bottom_area = tk.Frame(left, bg=C["bg_main"])
        bottom_area.pack(fill=tk.X, side=tk.BOTTOM)
        self._bottom_area = bottom_area

        # Button bar (shown in main menu / configure options / execute)
        self._btn_bar = tk.Frame(bottom_area, bg=C["bg_main"], pady=6)
        # (packed/forgotten dynamically)

        # Input row (shown during dialogue with assistant)
        input_outer = tk.Frame(bottom_area, bg=C["bg_chat"], bd=0, highlightthickness=0)
        self._input_outer = input_outer
        # (packed/forgotten dynamically)

        self._input_sep = tk.Frame(input_outer, bg="#2a2a32", height=1)
        self._input_sep.pack(fill=tk.X, padx=8)

        input_wrap = tk.Frame(input_outer, bg=C["bg_chat"], bd=0, highlightthickness=0)
        input_wrap.pack(fill=tk.X)
        self._input_row = input_wrap

        # Done button (shown next to Send during dialogue) — pack RIGHT first
        self._done_btn = tk.Button(
            input_wrap, text="Done",
            bg=C["btn_1"], fg=C["fg_white"],
            font=(FONT, 10, "bold"),
            relief=tk.FLAT, bd=0, padx=16, pady=9,
            highlightthickness=0,
            activebackground=C["btn_1"], activeforeground=C["fg_white"],
            command=self._cancel_configure,
        )
        # (packed/forgotten dynamically — rightmost position)

        # Send button — packed RIGHT, appears left of Done
        self._send = tk.Button(
            input_wrap, text="Send",
            bg=C["fg_assist"], fg=C["bg_main"],
            font=(FONT, 10, "bold"),
            relief=tk.FLAT, bd=0, padx=16, pady=9,
            highlightthickness=0,
            activebackground="#d4821f", activeforeground=C["bg_main"],
            state=tk.DISABLED,
            command=self._submit_text,
        )
        # (packed dynamically — always left of Done)

        # Text input — fills remaining space (packed last so buttons always visible)
        self._input_text = tk.Text(
            input_wrap,
            bg=C["bg_chat"], fg=C["fg_white"],
            font=(MONO, 11),
            insertbackground=C["fg_white"],
            relief=tk.FLAT, bd=0,
            highlightthickness=0,
            wrap=tk.WORD,
            height=1,
            padx=12, pady=9,
            state=tk.DISABLED,
            undo=True,
        )
        self._input_sb = tk.Scrollbar(
            input_wrap, command=self._input_text.yview,
            width=4, bd=0, highlightthickness=0,
            bg="#2a2a32", troughcolor=C["bg_chat"],
            activebackground="#3a3a44", relief=tk.FLAT,
        )
        self._input_text.configure(yscrollcommand=self._input_sb.set)
        self._input_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._input_text.configure(width=1)  # minimal request so buttons always fit

        # Auto-resize on content change
        def _on_input_modified(event=None):
            self._input_text.edit_modified(False)
            self._resize_input()

        self._input_text.bind("<<Modified>>", _on_input_modified)
        self._input_text.bind("<KeyRelease>", lambda e: self._resize_input())
        self._input_text.bind("<Return>", self._on_input_return)
        self._input_text.bind("<Shift-Return>", lambda e: None)
        # Re-evaluate wrap count when the widget is resized (window resize changes chars-per-line)
        self._input_text.bind("<Configure>", lambda e: self._resize_input())
        self._input_var = None

        # Now pack the chat text area (fills remaining space above the input)
        self._chat = tk.Text(
            chat_wrap,
            bg=C["bg_chat"], fg=C["fg_robot"],
            font=(MONO, 10),
            wrap=tk.WORD, state=tk.NORMAL,
            bd=0, padx=14, pady=10,
            spacing1=1, spacing3=3,
            highlightthickness=0,
            selectbackground="#3a3a44",
            selectforeground=C["fg_white"],
        )
        sb = tk.Scrollbar(chat_wrap, command=self._chat.yview,
                          width=6, bd=0, highlightthickness=0,
                          bg="#2a2a32", troughcolor=C["bg_chat"],
                          activebackground="#3a3a44", relief=tk.FLAT)
        self._chat.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 2), pady=4)
        self._chat.pack(fill=tk.BOTH, expand=True)

        # Keep read-only but allow text selection and Ctrl+C copy.
        def _block_edit(event):
            if event.state & 0x4:   # Ctrl held
                if event.keysym.lower() in ("c", "a"):
                    return
            if event.keysym in ("Up", "Down", "Left", "Right",
                                "Home", "End", "Prior", "Next"):
                return
            return "break"

        self._chat.bind("<Key>", _block_edit)
        self._chat.configure(cursor="xterm")

        # text tags
        self._chat.tag_configure("user",      foreground=C["fg_user"],    font=(MONO, 10, "bold"))
        self._chat.tag_configure("user_body", foreground=C["fg_robot"],    font=(MONO, 10))
        self._chat.tag_configure("robot",     foreground=C["fg_robot"])
        self._chat.tag_configure("assistant", foreground=C["fg_assist"],  font=(MONO, 10, "bold"))
        self._chat.tag_configure("success",   foreground=C["fg_success"])
        self._chat.tag_configure("error",     foreground=C["fg_error"])
        self._chat.tag_configure("warn",      foreground=C["fg_warn"])
        self._chat.tag_configure("system",    foreground=C["fg_system"],  font=(MONO, 9))
        self._chat.tag_configure("divider",   foreground="#333338",       font=(MONO, 9))
        self._chat.tag_configure("info",      foreground=C["fg_info"])
        self._chat.tag_configure("greeting",
                                 foreground=C["fg_white"],
                                 font=(MONO, 11, "bold"))

    def _build_right_panel(self, parent: tk.PanedWindow) -> None:
        """Right half: scene description (top) + vision image (bottom) + button row."""
        right = tk.Frame(parent, bg=C["bg_main"])
        self._right_panel = right   # keep reference for overlay
        parent.add(right, stretch="always", minsize=320)  # ← DRAG LIMIT: right panel min width

        # vertical PanedWindow inside right half
        self._vpane = tk.PanedWindow(
            right,
            orient=tk.VERTICAL,
            bg=C["bg_main"],
            sashwidth=6,
            sashrelief=tk.FLAT,
            bd=0,
            handlesize=0,
        )
        self._vpane.pack(fill=tk.BOTH, expand=True)

        # ── top: scene description + information (side by side) ────────────────
        scene_outer = tk.Frame(self._vpane, bg=C["bg_main"])
        self._vpane.add(scene_outer, stretch="always", minsize=120)  # ← DRAG LIMIT: top section min height

        # Horizontal PanedWindow to split scene desc (left) and info (right)
        self._top_hpane = tk.PanedWindow(
            scene_outer,
            orient=tk.HORIZONTAL,
            bg=C["bg_main"],
            sashwidth=6,
            sashrelief=tk.FLAT,
            bd=0,
            handlesize=0,
        )
        self._top_hpane.pack(fill=tk.BOTH, expand=True)

        # ── left: scene description ───────────────────────────────────────────
        scene_left = tk.Frame(self._top_hpane, bg=C["bg_main"])
        self._top_hpane.add(scene_left, stretch="always", minsize=160)  # ← DRAG LIMIT: scene desc min width

        tk.Label(
            scene_left, text="SCENE DESCRIPTION",
            bg=C["bg_main"], fg=C["fg_muted"],
            font=(FONT, 8, "bold"),
        ).pack(anchor=tk.CENTER, pady=(2, 3))

        scene_wrap = tk.Frame(scene_left, bg=C["bg_chat"], bd=0, highlightthickness=0)
        scene_wrap.pack(fill=tk.BOTH, expand=True)

        # Canvas for the scene table
        self._scene_canvas = tk.Canvas(
            scene_wrap, bg=C["bg_chat"],
            highlightthickness=0, bd=0,
        )
        self._scene_canvas.pack(fill=tk.BOTH, expand=True)

        # Inner frame that holds table rows
        self._scene_inner = tk.Frame(self._scene_canvas, bg=C["bg_chat"])
        self._scene_canvas_win = self._scene_canvas.create_window(
            (0, 0), window=self._scene_inner, anchor=tk.NW)

        def _on_scene_inner_cfg(event):
            self._scene_canvas.configure(scrollregion=self._scene_canvas.bbox("all"))
        def _on_scene_canvas_cfg(event):
            self._scene_canvas.itemconfigure(self._scene_canvas_win, width=event.width)
            if self._scene_placeholder_active:
                self._scene_canvas.itemconfigure(
                    self._scene_canvas_win, height=event.height)
            else:
                self._scene_canvas.itemconfigure(
                    self._scene_canvas_win, height=0)
        self._scene_inner.bind("<Configure>", _on_scene_inner_cfg)
        self._scene_canvas.bind("<Configure>", _on_scene_canvas_cfg)

        # Mouse wheel scrolling
        def _scene_mousewheel(event):
            self._scene_canvas.yview_scroll(
                -1 * (event.delta // 120 or (-1 if event.num == 4 else 1)), "units")
        for _w in (self._scene_canvas, self._scene_inner):
            _w.bind("<MouseWheel>", _scene_mousewheel)
            _w.bind("<Button-4>", _scene_mousewheel)
            _w.bind("<Button-5>", _scene_mousewheel)

        self._scene_placeholder_active = False
        self._set_scene_placeholder("No scene\nloaded yet.\nRun Reconfigure\nto populate.")

        # ── right: information panel ──────────────────────────────────────────
        info_right = tk.Frame(self._top_hpane, bg=C["bg_main"])
        self._top_hpane.add(info_right, stretch="always", minsize=500)  # ← DRAG LIMIT: info panel min width

        tk.Label(
            info_right, text="INFORMATION",
            bg=C["bg_main"], fg=C["fg_muted"],
            font=(FONT, 8, "bold"),
        ).pack(anchor=tk.CENTER, pady=(2, 3))

        info_wrap = tk.Frame(info_right, bg=C["bg_chat"], bd=0, highlightthickness=0)
        info_wrap.pack(fill=tk.BOTH, expand=True)

        # Title label (bold, larger)
        self._info_title_lbl = tk.Label(
            info_wrap, text="",
            bg=C["bg_chat"], fg=C["fg_white"],
            font=(FONT, 11, "bold"),
            pady=8,
        )
        self._info_title_lbl.pack(fill=tk.X)

        # Status label (dot + text, colored)
        self._info_status_lbl = tk.Label(
            info_wrap, text="● Ready",
            bg=C["bg_chat"], fg=C["fg_success"],
            font=(FONT, 9),
            pady=2,
        )
        self._info_status_lbl.pack(fill=tk.X)

        # Body: either plain text or a table — held in this container
        self._info_body_frame = tk.Frame(info_wrap, bg=C["bg_chat"])
        self._info_body_frame.pack(fill=tk.BOTH, expand=True)

        # Track current info content and status
        self._info_current = ("", [])  # (title, rows) where rows = list of (label, value) or plain str
        self._status_text = "Ready"
        self._status_color = C["fg_success"]

        # Show main menu info text initially
        self._set_info_main_menu()

        # ── bottom: vision image ──────────────────────────────────────────────
        self._img_outer = tk.Frame(self._vpane, bg=C["bg_main"])
        img_outer = self._img_outer
        self._vpane.add(img_outer, stretch="always", minsize=120)  # ← DRAG LIMIT: vision section min height

        tk.Label(
            img_outer, text="VISION",
            bg=C["bg_main"], fg=C["fg_muted"],
            font=(FONT, 8, "bold"),
        ).pack(anchor=tk.CENTER, pady=(2, 3))

        self._img_frame = tk.Frame(img_outer, bg=C["bg_chat"], bd=0, highlightthickness=0)
        self._img_frame.pack(fill=tk.BOTH, expand=True)
        self._img_frame.pack_propagate(False)  # prevent child label from resizing us
        img_frame = self._img_frame

        self._img_lbl = tk.Label(
            img_frame, bg=C["bg_chat"],
            text="No image yet.\nRun a vision scan to populate.",
            fg=C["fg_muted"], font=(FONT, 11),
        )
        self._img_lbl.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # ── split-view PanedWindow (hidden by default, shown during Update Config)
        self._split_pane = tk.PanedWindow(
            img_frame,
            orient=tk.HORIZONTAL,
            bg=C["bg_main"],
            sashwidth=6,
            sashrelief=tk.FLAT,
            bd=0,
            handlesize=0,
        )
        # Not packed yet — _enter_split_view() will pack it

        # Left = new image
        self._split_left = tk.Frame(self._split_pane, bg=C["bg_chat"])
        self._split_pane.add(self._split_left, stretch="always", minsize=80)
        tk.Label(self._split_left, text="NEW SCAN", bg=C["bg_main"],
                 fg=C["fg_muted"], font=(FONT, 7, "bold")).pack(fill=tk.X)
        self._split_lbl_new = tk.Label(
            self._split_left, bg=C["bg_chat"],
            text="Waiting for\nnew scan…",
            fg=C["fg_muted"], font=(FONT, 10, "italic"),
        )
        self._split_lbl_new.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)

        # Right = old image
        self._split_right = tk.Frame(self._split_pane, bg=C["bg_chat"])
        self._split_pane.add(self._split_right, stretch="always", minsize=80)
        tk.Label(self._split_right, text="PREVIOUS CONFIG", bg=C["bg_main"],
                 fg=C["fg_muted"], font=(FONT, 7, "bold")).pack(fill=tk.X)
        self._split_lbl_old = tk.Label(
            self._split_right, bg=C["bg_chat"],
            text="No previous\nimage.",
            fg=C["fg_muted"], font=(FONT, 10, "italic"),
        )
        self._split_lbl_old.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)

        # Re-render both images when the sash is dragged
        self._split_resize_id: Optional[str] = None

        def _on_split_configure(event):
            if not self._split_mode:
                return
            if self._split_resize_id:
                self.root.after_cancel(self._split_resize_id)
            self._split_resize_id = self.root.after(100, self._reload_split_images)

        self._split_pane.bind("<Configure>", _on_split_configure)
        self._split_pane.bind("<B1-Motion>", _on_split_configure)

        # ── zoom/pan bindings for all image labels ────────────────────────────
        for lbl in (self._img_lbl, self._split_lbl_new, self._split_lbl_old):
            self._bind_zoom_pan(lbl)

        self._resize_after_id: Optional[str] = None
        self._last_render_size: tuple = (0, 0)

        def _on_img_resize(event):
            if event.widget is not self._img_outer:
                return
            if self._resize_after_id:
                self.root.after_cancel(self._resize_after_id)
            self._resize_after_id = self.root.after(80, self._load_image)

        self._img_outer.bind("<Configure>", _on_img_resize)

    def _init_sash_positions(self) -> None:
        """Set initial pane split ratios.

        DRAG LIMITS are controlled by the ``minsize`` parameter on each
        ``.add()`` call (see ``_build_chat_panel`` and ``_build_right_panel``).
        Tk enforces these natively during sash dragging — no extra bindings
        are needed.  To change how far a pane can be dragged, adjust the
        ``minsize`` value on the corresponding ``.add()`` call.
        """
        # ── initial positions ─────────────────────────────────────────────
        try:
            total_w = self._hpane.winfo_width()
            if total_w > 10:
                self._hpane.sash_place(0, int(total_w * 0.43), 0)
        except Exception:
            pass
        try:
            total_h = self._vpane.winfo_height()
            if total_h > 10:
                self._vpane.sash_place(0, 0, int(total_h * 0.3))
        except Exception:
            pass
        try:
            top_w = self._top_hpane.winfo_width()
            if top_w > 10:
                self._top_hpane.sash_place(0, int(top_w * 0.1), 0)
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    # IO redirection & patching
    # ─────────────────────────────────────────────────────────────────────────

    def _redirect_io(self) -> None:
        self._real_input = builtins.input
        sys.stdout       = _GUIStream(self._out_q)
        builtins.input   = self._gui_input

    def _gui_input(self, prompt: str = "") -> str:
        """Blocking replacement for input() — waits for the GUI."""
        # Print meaningful prompts (e.g. "Anything else?") so the user knows
        # why the input box appeared.  Skip trivial "YOU: " markers.
        clean = prompt.strip().rstrip(":").strip()
        if clean and clean.upper() not in ("YOU", ""):
            self._out_q.put(("print", f"\n{prompt}\n"))
        self._in_req_q.put(("text", str(prompt)))
        return self._in_resp_q.get()       # blocks worker thread

    def _patch_pick_from_list(self) -> None:
        """
        Replace _pick_from_list in session_handler.
        In configure mode the numbered options are printed to the chat as plain
        text and the user types the number — no option buttons are shown.
        Typing "done" / "cancel" at a menu prompt cancels the session cleanly.
        """
        try:
            import session_handler as sh  # type: ignore
            gui = self

            def _patched(prompt: str, options: List[str]) -> int:
                sys.stdout.write(f"\n{prompt}\n")
                for i, o in enumerate(options, 1):
                    sys.stdout.write(f"  [{i}] {o}\n")
                # Always use text input — no buttons — during configure mode
                gui._in_req_q.put(("text", prompt))
                raw = gui._in_resp_q.get().strip().lower()
                # Allow the Done button (which sends "done") to exit cleanly
                if raw in ("done", "cancel", "exit", "quit"):
                    raise SystemExit(0)
                return int(raw) - 1

            sh._pick_from_list = _patched
        except Exception as exc:
            print(f"[GUI] Could not patch session_handler: {exc}")

    # ─────────────────────────────────────────────────────────────────────────
    # Scene description panel helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _set_scene_placeholder(self, msg: str) -> None:
        """Show a centered placeholder in the scene panel."""
        self._clear_scene_table()
        self._scene_placeholder_active = True
        # Force inner frame to fill canvas height for vertical centering
        try:
            ch = self._scene_canvas.winfo_height()
            if ch > 10:
                self._scene_canvas.itemconfigure(
                    self._scene_canvas_win, height=ch)
        except Exception:
            pass
        lbl = tk.Label(
            self._scene_inner, text=msg,
            bg=C["bg_chat"], fg=C["fg_muted"],
            font=(MONO, 10, "italic"),
            justify=tk.CENTER,
        )
        lbl.pack(expand=True, fill=tk.BOTH, padx=20, pady=40)

    def _clear_scene_table(self) -> None:
        """Remove all widgets from the scene inner frame."""
        for w in self._scene_inner.winfo_children():
            w.destroy()

    # ─────────────────────────────────────────────────────────────────────────
    # Information panel
    # ─────────────────────────────────────────────────────────────────────────

    # Info content: (title, body)
    # body is either a list of (label, description) tuples → rendered as table,
    # or a plain string → rendered as centered text.

    _INFO_MAIN_MENU = (
        "Vision-Guided Conversational\nTask Configuration",
        [
            ("Configure", "Start a new configuration session"),
            ("Plan Sequence", "Start planner to create a motion sequence"),
            ("Execute", "Call robot to execute the current motion sequence"),
            ("Camera Home", "Drive robot to the camera home position"),
            ("Log", "Open chat dialogue history"),
        ],
    )

    _INFO_CONFIGURE_OPTIONS = (
        "Configuration Mode",
        [
            ("New Config", "Capture new image and start configuration from scratch"),
            ("Update Config", "Capture new image and merge the new scene data with the previous configuration"),
            ("Edit Config", "Load an existing configuration file from memory and start editing it"),
        ],
    )

    _INFO_RECONFIG_ACTIVE = (
        "Configuration Mode",
        "Tell the assistant what to do in natural language. The attributes shown in the scene description can be adjusted.\n"
        "The LLM can make mistakes, make sure all neccessary world-state variables are set correctly.  ",
    )

    _INFO_UPDATE_ACTIVE = (
        "Update Configuration Mode",
        "Compare the new scan with the previous config and explain to the LLM "
        "in natural language what needs to be changed.",
    )

    def _set_info_text(self, info: tuple) -> None:
        """Update the information panel. info = (title, body)."""
        self._info_current = info
        self._render_info()

    def _render_info(self) -> None:
        """Re-render the full info panel: title + status + body (table or text)."""
        title, body = self._info_current

        # Title
        self._info_title_lbl.configure(text=title)

        # Status
        self._info_status_lbl.configure(
            text=f"● {self._status_text}",
            fg=self._status_color,
        )

        # Unbind old resize handler before destroying children
        self._info_body_frame.unbind("<Configure>")

        # Clear body
        for w in self._info_body_frame.winfo_children():
            w.destroy()

        if isinstance(body, list):
            # Render as 3-column table (same style as scene description)
            tbl = tk.Frame(self._info_body_frame, bg=C["bg_chat"])
            tbl.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
            tbl.columnconfigure(0, weight=1)  # left spacer
            tbl.columnconfigure(2, weight=1)  # value col stretches

            val_labels = []
            for i, (lbl, val) in enumerate(body):
                row_bg = "#1e1e24" if i % 2 == 0 else C["bg_chat"]

                # Col 0: empty label spacer (same type = uniform row height)
                tk.Label(tbl, bg=row_bg).grid(row=i, column=0, sticky="nsew")

                lbl_w = tk.Label(
                    tbl, text=lbl + ":",
                    bg=row_bg, fg=C["fg_muted"],
                    font=(MONO, 9, "bold"), anchor=tk.NW,
                    padx=8, pady=5,
                )
                lbl_w.grid(row=i, column=1, sticky="nsew")

                val_w = tk.Label(
                    tbl, text=val,
                    bg=row_bg, fg=C["fg_robot"],
                    font=(MONO, 9), anchor=tk.NW,
                    padx=8, pady=5, justify=tk.LEFT,
                )
                val_w.grid(row=i, column=2, sticky="nsew")
                val_labels.append(val_w)

            # Fill remaining vertical space
            tk.Label(tbl, bg=C["bg_chat"]).grid(
                row=len(body), column=0, columnspan=3, sticky="nsew")
            tbl.rowconfigure(len(body), weight=1)

            # Dynamic wraplength on resize
            def _on_tbl_resize(event, _tbl=tbl, _labels=val_labels):
                if not _tbl.winfo_exists():
                    return
                try:
                    bbox = _tbl.grid_bbox(column=2)
                    col2_w = bbox[2] if bbox else event.width // 2
                except Exception:
                    col2_w = event.width // 2
                avail = max(col2_w - 20, 80)
                for vl in _labels:
                    if vl.winfo_exists():
                        vl.configure(wraplength=avail)
            tbl.bind("<Configure>", _on_tbl_resize)

        else:
            # Render as plain centered text
            txt_lbl = tk.Label(
                self._info_body_frame, text=body,
                bg=C["bg_chat"], fg=C["fg_robot"],
                font=(FONT, 9), justify=tk.CENTER,
                padx=14, pady=10,
            )
            txt_lbl.pack(fill=tk.BOTH, expand=True)

            def _on_txt_resize(event, _lbl=txt_lbl):
                if _lbl.winfo_exists():
                    _lbl.configure(wraplength=max(event.width - 30, 80))
            self._info_body_frame.bind("<Configure>", _on_txt_resize)

    def _set_info_main_menu(self) -> None:
        self._set_info_text(self._INFO_MAIN_MENU)

    def _set_info_configure_options(self) -> None:
        self._set_info_text(self._INFO_CONFIGURE_OPTIONS)

    def _set_info_reconfig_active(self) -> None:
        self._set_info_text(self._INFO_RECONFIG_ACTIVE)

    def _set_info_update_active(self) -> None:
        self._set_info_text(self._INFO_UPDATE_ACTIVE)

    def _set_scene_content(self, text: str) -> None:
        """Unused — kept for backward compat. Use _refresh_scene_from_config."""
        self._scene_placeholder_active = False
        self._clear_scene_table()
        lbl = tk.Label(
            self._scene_inner, text=text.strip(),
            bg=C["bg_chat"], fg=C["fg_robot"],
            font=(MONO, 10), anchor=tk.NW, justify=tk.LEFT,
        )
        lbl.pack(fill=tk.BOTH, padx=14, pady=10)

    def _add_scene_row(self, parent: tk.Frame, label: str, value: str,
                       row_idx: int, is_separator: bool = False) -> tk.Label:
        """Add a single row to the scene table grid (3-col: spacer|label|value).
        Returns the value label (or None for separators)."""
        if is_separator:
            sep = tk.Frame(parent, bg="#2a2a32", height=1)
            sep.grid(row=row_idx, column=0, columnspan=3,
                     sticky="ew", padx=0, pady=0)
            return None

        row_bg = "#1e1e24" if row_idx % 2 == 0 else C["bg_chat"]

        # Col 0: empty label as spacer (same widget type = same row height)
        tk.Label(parent, bg=row_bg).grid(row=row_idx, column=0, sticky="nsew")

        lbl_w = tk.Label(
            parent, text=label + ":",
            bg=row_bg, fg=C["fg_muted"],
            font=(MONO, 9, "bold"), anchor=tk.NW,
            padx=8, pady=5,
        )
        lbl_w.grid(row=row_idx, column=1, sticky="nsew")

        val_w = tk.Label(
            parent, text=value,
            bg=row_bg, fg=C["fg_robot"],
            font=(MONO, 9), anchor=tk.NW,
            padx=8, pady=5, justify=tk.LEFT,
        )
        val_w.grid(row=row_idx, column=2, sticky="nsew")
        return val_w

    def _refresh_scene_from_config(self) -> None:
        """Build scene summary as a clean table from config file."""
        try:
            import json
            import session_handler as sh  # type: ignore
            if not sh.CONFIGURATION_PATH.exists():
                return
            state = json.loads(sh.CONFIGURATION_PATH.read_text(encoding="utf-8"))
            preds = state.get("predicates", {})

            self._scene_placeholder_active = False
            self._clear_scene_table()
            # Reset inner frame height to natural (scrollable)
            try:
                self._scene_canvas.itemconfigure(
                    self._scene_canvas_win, height=0)
            except Exception:
                pass

            table = tk.Frame(self._scene_inner, bg=C["bg_chat"])
            table.pack(fill=tk.BOTH, expand=True, padx=0, pady=(12, 0))
            table.columnconfigure(0, weight=1)  # left spacer
            table.columnconfigure(2, weight=1)  # value col stretches too

            row = 0
            _scene_val_labels = []

            def _add(label, value, r, **kw):
                vl = self._add_scene_row(table, label, value, r, **kw)
                if vl is not None:
                    _scene_val_labels.append(vl)

            # Mode
            ws = state.get("workspace", {})
            mode = ws.get("operation_mode") or "none"
            _add("Mode", mode, row); row += 1

            # Batch Size
            batch = ws.get("batch_size")
            batch_str = str(batch) if batch else "none"
            _add("Batch Size", batch_str, row); row += 1

            # Fill Order
            fill_order = (ws.get("fill_order") or "").strip().lower()
            fill_order_str = "parallel" if fill_order == "parallel" else "sequential (default)"
            _add("Fill Order", fill_order_str, row); row += 1

            # Separator
            _add("", "", row, is_separator=True); row += 1

            # Roles
            roles = preds.get("role", [])
            assigned = [e for e in roles if e.get("role")]
            if assigned:
                val = "\n".join(
                    f"{e['object']} = {e['role']}"
                    for e in sorted(assigned, key=lambda x: x["object"])
                )
            else:
                val = "none"
            _add("Roles", val, row); row += 1

            # Fragility
            frag = [e["part"] for e in preds.get("fragility", [])
                    if e.get("fragility") == "fragile"]
            _add("Fragility",
                                ", ".join(sorted(frag)) if frag else "none",
                                row); row += 1

            # Priority
            prio = preds.get("priority", [])
            if prio:
                def _prio_label(e):
                    """Format a single priority entry for display."""
                    order = e.get("order", 0)
                    # Check each type key (new names first, then legacy)
                    for key in ("color", "kit", "container", "part",
                                "fragility", "destination", "source",
                                "part_name"):
                        if key in e:
                            return f"{e[key]} (#{order})"
                    return f"? (#{order})"
                prio_str = ", ".join(
                    _prio_label(e)
                    for e in sorted(prio, key=lambda x: x.get("order", 0))
                )
            else:
                prio_str = "none"
            _add("Priority", prio_str, row); row += 1

            # Separator
            _add("", "", row, is_separator=True); row += 1

            # Kit recipe
            recipe = preds.get("kit_recipe", [])
            if recipe:
                seen = set()
                parts = []
                for e in recipe:
                    color = e.get("color", "")
                    qty = e.get("quantity", 0)
                    size = e.get("size")
                    key = (color, qty, size)
                    if key in seen:
                        continue
                    seen.add(key)
                    size_str = f" ({size})" if size else ""
                    parts.append(f"{qty}x {color}{size_str}")
                recipe_str = "\n".join(parts)
            else:
                recipe_str = "none"
            _add("Kit Recipe", recipe_str, row); row += 1

            # Compatibility
            compat = preds.get("part_compatibility", [])
            if compat:
                rules = []
                for rule in compat:
                    part_selectors = []
                    if rule.get("part_name"):
                        part_selectors.append(rule["part_name"])
                    if rule.get("part_color"):
                        part_selectors.append(f"{rule['part_color']} parts")
                    if rule.get("part_fragility"):
                        part_selectors.append(f"{rule['part_fragility']} parts")
                    part_desc = " + ".join(part_selectors) if part_selectors else "all parts"
                    if rule.get("allowed_in"):
                        rec_desc = ", ".join(rule["allowed_in"])
                    elif rule.get("allowed_in_role"):
                        rec_desc = f"all {rule['allowed_in_role']}s"
                    else:
                        rec_desc = "all"
                    if rule.get("not_allowed_in"):
                        rec_desc += f" (except {', '.join(rule['not_allowed_in'])})"
                    rules.append(f"{part_desc} → {rec_desc}")
                compat_str = "\n".join(rules)
            else:
                compat_str = "none"
            _add("Compatibility", compat_str, row); row += 1

            # Fill remaining vertical space
            tk.Label(table, bg=C["bg_chat"]).grid(
                row=row, column=0, columnspan=3, sticky="nsew")
            table.rowconfigure(row, weight=1)

            # Dynamic wraplength for scene value labels
            def _on_scene_tbl_resize(event, _tbl=table, _labels=_scene_val_labels):
                if not _tbl.winfo_exists():
                    return
                try:
                    bbox = _tbl.grid_bbox(column=2)
                    col2_w = bbox[2] if bbox else event.width // 2
                except Exception:
                    col2_w = event.width // 2
                avail = max(col2_w - 20, 80)
                for vl in _labels:
                    if vl.winfo_exists():
                        vl.configure(wraplength=avail)
            table.bind("<Configure>", _on_scene_tbl_resize)

        except Exception as exc:
            self._set_scene_placeholder("Could not load scene:\n" + str(exc))

    # ─────────────────────────────────────────────────────────────────────────
    # Log window
    # ─────────────────────────────────────────────────────────────────────────

    def _open_log_window(self) -> None:
        """Open (or re-raise) the floating log window."""
        if self._log_win is None or not self._log_win.winfo_exists():
            self._log_win = tk.Toplevel(self.root)
            self._log_win.title("Output Log")
            self._log_win.geometry("900x600")
            self._log_win.configure(bg=C["bg_main"])

            wrap = tk.Frame(self._log_win, bg=C["bg_chat"])
            wrap.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

            self._log_text = tk.Text(
                wrap,
                bg=C["bg_chat"], fg=C["fg_robot"],
                font=(MONO, 10),
                wrap=tk.WORD, state=tk.NORMAL,
                bd=0, padx=14, pady=10,
                spacing1=1, spacing3=3,
                highlightthickness=0,
                selectbackground="#3a3a44",
                selectforeground=C["fg_white"],
            )
            sb = tk.Scrollbar(wrap, command=self._log_text.yview,
                              width=6, bd=0, highlightthickness=0,
                              bg="#2a2a32", troughcolor=C["bg_chat"],
                              activebackground="#3a3a44", relief=tk.FLAT)
            self._log_text.configure(yscrollcommand=sb.set)
            sb.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 2), pady=4)
            self._log_text.pack(fill=tk.BOTH, expand=True)

            # Same colour tags as chat, plus all keys allowed (fully copyable)
            for tag, cfg in [
                ("user",      {"foreground": C["fg_user"],    "font": (MONO, 10, "bold")}),
                ("user_body", {"foreground": C["fg_user"],    "font": (MONO, 10)}),
                ("robot",     {"foreground": C["fg_robot"]}),
                ("assistant", {"foreground": C["fg_assist"],   "font": (MONO, 10, "bold")}),
                ("success",   {"foreground": C["fg_success"]}),
                ("error",     {"foreground": C["fg_error"]}),
                ("warn",      {"foreground": C["fg_warn"]}),
                ("system",    {"foreground": C["fg_system"],  "font": (MONO, 9)}),
                ("divider",   {"foreground": "#333338",        "font": (MONO, 9)}),
            ]:
                self._log_text.tag_configure(tag, **cfg)

            # Replay everything that was buffered before the window was opened
            for _text, _tag in self._log_buffer:
                self._log_text.insert(tk.END, _text, _tag)
            self._log_text.see(tk.END)

        else:
            self._log_win.deiconify()
            self._log_win.lift()

    def _log_append(self, text: str, tag: str = "robot") -> None:
        """Buffer log entry and write to the window if it is already open."""
        self._log_buffer.append((text, tag))
        if self._log_text is not None and self._log_win.winfo_exists():
            self._log_text.insert(tk.END, text, tag)
            self._log_text.see(tk.END)

    # ─────────────────────────────────────────────────────────────────────────
    # Button bar management
    # ─────────────────────────────────────────────────────────────────────────

    def _clear_bar(self) -> None:
        for w in self._btn_bar.winfo_children():
            w.destroy()

    def _btn(self, parent: tk.Frame, text: str, color: str,
             cmd, width: int = 0) -> tk.Button:
        kwargs = {}
        if width:
            kwargs["width"] = width
        return tk.Button(
            parent, text=text,
            bg=color, fg=C["fg_white"],
            font=(FONT, 10, "bold"),
            relief=tk.FLAT, bd=0,
            highlightthickness=0,
            padx=18, pady=9,
            activebackground=color,
            activeforeground=C["fg_white"],
            command=cmd,
            **kwargs,
        )

    def _show_main_menu(self) -> None:
        self._clear_bar()
        self._show_button_bar()
        self._set_input(False)
        self._in_configure_mode = False
        self._in_update_mode = False
        self._set_info_main_menu()

        # Show greeting the very first time
        if not self._first_menu_shown:
            self._first_menu_shown = True
            self._append(
                "\nLet's configure, what would you like to do?\n",
                "greeting",
            )

        items = [
            ("Configure",     C["btn_1"],  self._show_configure_options),
            ("Plan Sequence", C["btn_2"],  lambda: self._run("motion")),
            ("Execute",       C["btn_3"],  lambda: self._run("execute")),
            ("Camera Home",   C["btn_4"],  self._run_camera_home),
            ("Log",           C["btn_4"],  self._open_log_window),
        ]
        inner = tk.Frame(self._btn_bar, bg=C["bg_main"])
        inner.pack(anchor=tk.CENTER)
        for text, color, cmd in items:
            self._btn(inner, text, color, cmd).pack(side=tk.LEFT, padx=3)
        self._btn_bar.pack(fill=tk.X)

    def _show_cancel_bar(self) -> None:
        """Show Done + Send in the input row. Done rightmost."""
        self._send.pack_forget()
        self._done_btn.pack_forget()
        # Pack RIGHT in order: first = rightmost
        self._done_btn.pack(side=tk.RIGHT, anchor=tk.S, padx=(2, 6), pady=2)
        self._send.pack(side=tk.RIGHT, anchor=tk.S, padx=(4, 2), pady=2)

    def _show_execute_bar(self) -> None:
        """Show Cancel + Log buttons during robot execution."""
        self._clear_bar()
        inner = tk.Frame(self._btn_bar, bg=C["bg_main"])
        inner.pack(anchor=tk.CENTER)
        self._btn(
            inner, "Cancel", C["btn_1"],
            self._cancel_execution,
        ).pack(side=tk.LEFT, padx=3)
        self._btn(
            inner, "Log", C["btn_2"],
            self._open_log_window,
        ).pack(side=tk.LEFT, padx=3)
        self._btn_bar.pack(fill=tk.X)

    def _show_configure_options(self) -> None:
        """Replace the main button bar with three configure sub-options + Done."""
        self._clear_bar()
        self._set_input(False)
        self._set_info_configure_options()

        items = [
            ("New Config",    C["btn_1"], lambda: self._run_reconfig_sub("reconfig_fresh")),
            ("Update Config", C["btn_2"], lambda: self._run_reconfig_sub("reconfig_update")),
            ("Edit Config",   C["btn_3"], lambda: self._run_reconfig_sub("reconfig_memory")),
            ("Done",          C["btn_4"], self._show_main_menu),
        ]
        inner = tk.Frame(self._btn_bar, bg=C["bg_main"])
        inner.pack(anchor=tk.CENTER)
        for text, color, cmd in items:
            self._btn(inner, text, color, cmd).pack(side=tk.LEFT, padx=3)
        self._btn_bar.pack(fill=tk.X)

    def _run_reconfig_sub(self, sub: str) -> None:
        """Launch the reconfig worker with a pre-selected sub-option so the
        backend's select_reconfig_source() is bypassed entirely."""
        self._reconfig_sub = sub
        self._config_from_memory = False
        self._in_update_mode = (sub == "reconfig_update")
        if sub == "reconfig_update":
            self._set_info_update_active()
            self._enter_split_view()
        elif sub == "reconfig_fresh":
            self._set_info_reconfig_active()
        if sub == "reconfig_memory":
            self._show_config_browser()
            return
        self._run("reconfig")

    def _show_config_browser(self) -> None:
        """Show a config file browser as an overlay covering the entire right
        panel (scene description + image).  The underlying widgets are untouched
        and reappear when the overlay is destroyed."""
        import session_handler as sh

        configs = sh.list_memory_configs()

        # Determine which config is the "current" one (matches configuration.json)
        current_path = None
        try:
            import json as _json
            if sh.CONFIGURATION_PATH.exists():
                current_data = sh.CONFIGURATION_PATH.read_text(encoding="utf-8")
                current_hash = hash(current_data)
                for cfg in configs:
                    try:
                        with open(cfg["path"], "r", encoding="utf-8") as f:
                            if hash(f.read()) == current_hash:
                                current_path = cfg["path"]
                                break
                    except Exception:
                        pass
        except Exception:
            pass

        # If no hash match, just treat the newest as current (if it exists)
        if current_path is None and configs:
            current_path = configs[0]["path"]

        # ── create overlay frame on top of the vision image box ──────────
        overlay = tk.Frame(self._img_outer, bg=C["bg_main"])
        overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        overlay.lift()
        self._browser_overlay = overlay

        # ── update info box: replace the table with a plain prompt ────────
        self._set_info_text((
            "Configuration Mode",
            "Select the configuration file you would like to edit.",
        ))

        # ── header ────────────────────────────────────────────────────────
        hdr = tk.Frame(overlay, bg=C["bg_main"])
        hdr.pack(fill=tk.X, padx=10, pady=(10, 4))

        tk.Label(
            hdr, text="SELECT CONFIGURATION",
            bg=C["bg_main"], fg=C["fg_muted"],
            font=(FONT, 8, "bold"),
        ).pack(anchor=tk.CENTER)

        # ── column headers ────────────────────────────────────────────────
        col_hdr_bg = "#1f1f25"
        col_hdr = tk.Frame(overlay, bg=col_hdr_bg)
        col_hdr.pack(fill=tk.X, padx=10, pady=(10, 0))

        tk.Label(
            col_hdr, text="",
            bg=col_hdr_bg, fg=C["fg_muted"],
            font=(FONT, 8, "bold"), width=3,
        ).pack(side=tk.LEFT, padx=(4, 0), pady=4)
        tk.Label(
            col_hdr, text="Configuration File",
            bg=col_hdr_bg, fg=C["fg_muted"],
            font=(FONT, 8, "bold"), anchor=tk.W,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0), pady=4)
        tk.Label(
            col_hdr, text="Date",
            bg=col_hdr_bg, fg=C["fg_muted"],
            font=(FONT, 8, "bold"), width=12, anchor=tk.W,
        ).pack(side=tk.LEFT, padx=4, pady=4)
        tk.Label(
            col_hdr, text="Time",
            bg=col_hdr_bg, fg=C["fg_muted"],
            font=(FONT, 8, "bold"), width=8, anchor=tk.W,
        ).pack(side=tk.LEFT, padx=(4, 8), pady=4)

        # ── scrollable list ───────────────────────────────────────────────
        list_frame_outer = tk.Frame(overlay, bg=C["bg_chat"])
        list_frame_outer.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 4))

        canvas = tk.Canvas(
            list_frame_outer, bg=C["bg_chat"],
            highlightthickness=0, bd=0,
        )
        scrollbar = tk.Scrollbar(
            list_frame_outer, orient=tk.VERTICAL, command=canvas.yview,
            width=6, bd=0, highlightthickness=0,
            bg="#2a2a32", troughcolor=C["bg_chat"],
            activebackground="#3a3a44", relief=tk.FLAT,
        )
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 2), pady=4)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner = tk.Frame(canvas, bg=C["bg_chat"])
        canvas_window = canvas.create_window((0, 0), window=inner, anchor=tk.NW)

        # Track selection
        self._browser_selected_path = None
        self._browser_rows = []

        row_bg_normal  = C["bg_chat"]
        row_bg_hover   = "#222228"
        row_bg_sel     = C["btn_1"]
        row_bg_current = "#1a2a1a"      # subtle green tint for current config

        def _select_row(idx, path):
            self._browser_selected_path = path
            for i, (row_frame, rpath, is_cur) in enumerate(self._browser_rows):
                if i == idx:
                    bg = row_bg_sel
                elif is_cur:
                    bg = row_bg_current
                else:
                    bg = row_bg_normal
                row_frame.configure(bg=bg)
                for child in row_frame.winfo_children():
                    child.configure(bg=bg)

        def _get_row_bg(idx):
            """Return the resting background for a row (selected > current > normal)."""
            _, rpath, is_cur = self._browser_rows[idx]
            if self._browser_selected_path == rpath:
                return row_bg_sel
            if is_cur:
                return row_bg_current
            return row_bg_normal

        if not configs:
            tk.Label(
                inner, text="No saved configurations found in Memory/.",
                bg=C["bg_chat"], fg=C["fg_muted"],
                font=(FONT, 10, "italic"),
            ).pack(padx=20, pady=40)
        else:
            for idx, cfg in enumerate(configs):
                is_current = (cfg["path"] == current_path)
                base_bg = row_bg_current if is_current else row_bg_normal

                row = tk.Frame(inner, bg=base_bg, cursor="hand2")
                row.pack(fill=tk.X, padx=2, pady=1)

                # Current indicator
                indicator_text = "▶" if is_current else ""
                ind_lbl = tk.Label(
                    row, text=indicator_text,
                    bg=base_bg, fg=C["fg_success"],
                    font=(FONT, 8), width=3,
                )
                ind_lbl.pack(side=tk.LEFT, padx=(4, 0), pady=5)

                # Strip .json extension for display
                display_name = cfg["name"]
                if display_name.endswith(".json"):
                    display_name = display_name[:-5]
                if is_current:
                    display_name += "  (current)"

                name_lbl = tk.Label(
                    row, text=display_name,
                    bg=base_bg,
                    fg=C["fg_success"] if is_current else C["fg_robot"],
                    font=(MONO, 9, "bold") if is_current else (MONO, 9),
                    anchor=tk.W,
                )
                name_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0), pady=5)

                date_lbl = tk.Label(
                    row, text=cfg["date"],
                    bg=base_bg, fg=C["fg_muted"],
                    font=(MONO, 9), width=12, anchor=tk.W,
                )
                date_lbl.pack(side=tk.LEFT, padx=4, pady=5)

                time_lbl = tk.Label(
                    row, text=cfg["time"],
                    bg=base_bg, fg=C["fg_muted"],
                    font=(MONO, 9), width=8, anchor=tk.W,
                )
                time_lbl.pack(side=tk.LEFT, padx=(4, 8), pady=5)

                self._browser_rows.append((row, cfg["path"], is_current))

                path = cfg["path"]
                _idx = idx
                for widget in (row, ind_lbl, name_lbl, date_lbl, time_lbl):
                    widget.bind("<Button-1>",
                                lambda e, i=_idx, p=path: _select_row(i, p))
                    widget.bind("<Enter>",
                                lambda e, r=row: (
                                    r.configure(bg=row_bg_hover),
                                    [c.configure(bg=row_bg_hover) for c in r.winfo_children()]
                                ))
                    widget.bind("<Leave>",
                                lambda e, i=_idx: (
                                    lambda bg: (
                                        self._browser_rows[i][0].configure(bg=bg),
                                        [c.configure(bg=bg) for c in self._browser_rows[i][0].winfo_children()]
                                    )
                                )(_get_row_bg(i)))

        # Update scroll region when inner frame changes size
        def _on_inner_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event):
            canvas.itemconfigure(canvas_window, width=event.width)

        inner.bind("<Configure>", _on_inner_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        # Mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(-1 * (event.delta // 120 or (
                -1 if event.num == 4 else 1)), "units")

        for w in (canvas, inner):
            w.bind("<MouseWheel>", _on_mousewheel)
            w.bind("<Button-4>", _on_mousewheel)
            w.bind("<Button-5>", _on_mousewheel)

        # ── buttons at bottom ─────────────────────────────────────────────
        btn_frame = tk.Frame(overlay, bg=C["bg_main"])
        btn_frame.pack(fill=tk.X, padx=10, pady=(4, 10))

        inner_btn = tk.Frame(btn_frame, bg=C["bg_main"])
        inner_btn.pack(anchor=tk.CENTER)
        self._btn(inner_btn, "Use Selected", C["btn_1"],
                  self._browser_use_selected).pack(side=tk.LEFT, padx=(0, 6))
        self._btn(inner_btn, "Cancel", C["btn_3"],
                  self._browser_cancel).pack(side=tk.LEFT)

        # Update button bar (hide main menu buttons while browsing)
        self._clear_bar()
        self._btn_bar.pack_forget()

    def _close_config_browser(self) -> None:
        """Destroy the browser overlay.  The right panel widgets underneath
        are untouched and immediately visible again."""
        if hasattr(self, "_browser_overlay") and self._browser_overlay is not None:
            self._browser_overlay.destroy()
            self._browser_overlay = None
        self._browser_rows = []
        self._browser_selected_path = None

    def _browser_use_selected(self) -> None:
        """Load the selected config from Memory/ and start the LLM session."""
        if not self._browser_selected_path:
            return  # no selection — do nothing

        import session_handler as sh

        # Check if the selected config is the current one (no reload needed)
        is_current = False
        try:
            if sh.CONFIGURATION_PATH.exists():
                current_data = sh.CONFIGURATION_PATH.read_text(encoding="utf-8")
                with open(self._browser_selected_path, "r", encoding="utf-8") as f:
                    is_current = (hash(f.read()) == hash(current_data))
        except Exception:
            pass

        if is_current:
            # Already the active config — just proceed
            self._close_config_browser()
            self._set_info_reconfig_active()
            self._run("reconfig")
            return

        # Loading a different (old) config — image is no longer valid
        success = sh.load_config_from_memory(self._browser_selected_path)
        if not success:
            self._append("⚠  Failed to load selected configuration.\n", "error")
            return

        self._append("Loaded configuration from Memory.\n", "success")
        self._config_from_memory = True     # triggers "no image" placeholder
        self._image_ready = False
        self._photo_ref = None
        self._close_config_browser()
        self._refresh_scene_from_config()
        self._load_image()                  # show placeholder
        self._set_info_reconfig_active()
        self._run("reconfig")

    def _browser_cancel(self) -> None:
        """Cancel and return to the main menu."""
        self._close_config_browser()
        self._show_main_menu()

    def _cancel_configure(self) -> None:
        """Send 'done' to the worker and signal any in-flight LLM call to abort."""
        if self._dv7:
            self._dv7 = False
            self._last_render_size = (0, 0)
            self._load_image()
        self._cancel_event.set()
        self._append("YOU: done\n", "robot")
        self._in_resp_q.put("done")

    def _cancel_execution(self) -> None:
        """Signal the execution subprocess to stop after the current step."""
        try:
            CANCEL_SENTINEL.parent.mkdir(parents=True, exist_ok=True)
            CANCEL_SENTINEL.write_text("cancel", encoding="utf-8")
            self._append("\n⚠  Cancel requested — the robot will finish its current "
                        "pick-and-place and then return to Camera Home.\n", "warn")
            self._set_status("Cancelling...", C["fg_warn"])
        except Exception as exc:
            self._append(f"\n[ERR] Could not request cancel: {exc}\n", "error")

    def _show_text_input(self) -> None:
        self._show_input_row()
        if self._in_configure_mode:
            self._show_cancel_bar()
        self._set_input(True)

    def _set_input(self, enabled: bool) -> None:
        s = tk.NORMAL if enabled else tk.DISABLED
        self._input_text.configure(state=s)
        self._send.configure(state=s)
        # Done button stays always enabled so the user can cancel during LLM thinking
        self._done_btn.configure(state=tk.NORMAL)
        if enabled:
            self._input_text.focus_set()

    def _show_input_row(self) -> None:
        """Switch from button bar to input row (with Send + Done)."""
        self._btn_bar.pack_forget()
        self._input_outer.pack(fill=tk.X)

    def _show_button_bar(self) -> None:
        """Switch from input row back to button bar."""
        self._input_text.configure(height=1)
        self._input_sb.pack_forget()
        self._send.pack_forget()
        self._done_btn.pack_forget()
        self._input_outer.pack_forget()

    # ─────────────────────────────────────────────────────────────────────────
    # Backend worker
    # ─────────────────────────────────────────────────────────────────────────

    def _run(self, mode: str) -> None:
        if self._busy:
            return
        self._busy = True
        self._cancel_event.clear()
        self._current_mode = mode
        self._in_configure_mode = True

        # Clean up any leftover cancel sentinel
        try:
            CANCEL_SENTINEL.unlink(missing_ok=True)
        except Exception:
            pass

        self._set_status("Running...", C["fg_assist"])
        if mode == "execute":
            self._show_execute_bar()
        else:
            self._show_input_row()
            self._show_cancel_bar()
        self._set_input(False)
        threading.Thread(target=self._worker, args=(mode,), daemon=True).start()

    def _worker(self, mode: str) -> None:
        # Re-apply redirect here — some imports (rospy, etc.) can reset sys.stdout.
        sys.stdout = _GUIStream(self._out_q)
        sys.stderr = _GUIStream(self._out_q)

        # Make the cancel event available to API_Main for aborting LLM calls
        import robot_configurator.communication.api_main as _api
        _api._cancel_event = self._cancel_event

        try:
            if mode == "execute":
                self._run_execute_subprocess()
                # After execution, take a fresh picture and update the config
                # with actual part positions.  The config is truth for identity.
                try:
                    import session_handler as sh  # type: ignore
                    if sh.CONFIGURATION_PATH.exists():
                        import json as _json
                        config = _json.loads(sh.CONFIGURATION_PATH.read_text(encoding="utf-8"))
                        from robot_configurator.configuration.update_scene import run_post_execution_rescan
                        run_post_execution_rescan(config)
                except Exception as exc:
                    print(f"  ⚠  Post-execution rescan failed: {exc}")
                return

            self._patch_pick_from_list()

            import session_handler as sh  # type: ignore

            # If a reconfig sub-option was pre-selected via the GUI buttons,
            # bypass the interactive select_reconfig_source() call entirely.
            if mode == "reconfig" and self._reconfig_sub:
                _pre = self._reconfig_sub
                sh.select_reconfig_source = lambda: _pre

            client = None
            try:
                from openai import OpenAI
                client = OpenAI()
            except Exception as exc:
                from Vision_Module.config import USE_PDDL_PLANNER  # type: ignore
                if mode == "motion" and USE_PDDL_PLANNER:
                    pass
                else:
                    print(f"\n[ERR] Could not create OpenAI client: {exc}\n")
                    return

            sh.run_session(client, mode)

        except SystemExit:
            pass
        except Exception as exc:
            # LLMCancelled is expected when the user presses Done during an
            # LLM call — don't print a scary error for that.
            from robot_configurator.communication.api_main import LLMCancelled
            if not isinstance(exc, LLMCancelled):
                print(f"\n[ERR] Unexpected error: {exc}\n")
                print(traceback.format_exc())
        finally:
            # Drain any leftover responses so the next session starts clean
            while not self._in_resp_q.empty():
                try:
                    self._in_resp_q.get_nowait()
                except queue.Empty:
                    break
            self._out_q.put(("done", None))

    def _run_execute_subprocess(self) -> None:
        """
        Run robot execution in a clean subprocess so rospy.init_node() gets
        the main thread of that process (required for UNIX signal handlers).
        Output is streamed line-by-line into the GUI chat panel.
        """
        import subprocess as _sp
        script = PROJECT_DIR / "run_execute.py"
        try:
            proc = _sp.Popen(
                [sys.executable, str(script)],
                cwd=str(PROJECT_DIR),
                stdout=_sp.PIPE,
                stderr=_sp.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                self._out_q.put(("print", line))
            proc.wait()
            if proc.returncode != 0:
                self._out_q.put(("print",
                    f"\n[ERR] Execute process exited with code {proc.returncode}\n"))
        except Exception as exc:
            self._out_q.put(("print",
                f"\n[ERR] Could not start execute subprocess: {exc}\n"))

    def _run_camera_home(self) -> None:
        """Move the robot to Camera_Home in a subprocess."""
        if self._busy:
            return
        self._busy = True
        self._set_status("Moving to Camera Home...", C["fg_warn"])
        self._clear_bar()
        self._btn_bar.pack_forget()
        threading.Thread(
            target=self._camera_home_worker, daemon=True,
        ).start()

    def _camera_home_worker(self) -> None:
        """Background worker that launches move_camera_home.py as a subprocess."""
        import subprocess as _sp
        script = PROJECT_DIR / "Execution_Module" / "move_camera_home.py"
        try:
            proc = _sp.Popen(
                [sys.executable, str(script)],
                cwd=str(PROJECT_DIR),
                stdout=_sp.PIPE,
                stderr=_sp.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                stripped = line.strip()
                # Drop ROS/moveit C++ log noise and Robot init dumps
                if not stripped:
                    continue
                if stripped.startswith(("\x1b[", "[0m", "[INFO]", "[WARN]", "[ERROR]", "Loaded positions")):
                    continue
                self._out_q.put(("print", line))
            proc.wait()
            if proc.returncode != 0:
                self._out_q.put(("print",
                    "\nMoving robot to camera home position failed. "
                    "Check robot connection.\n"))
        except Exception:
            self._out_q.put(("print",
                "\nMoving robot to camera home position failed. "
                "Check robot connection.\n"))
        finally:
            self._out_q.put(("done", None))

    # ─────────────────────────────────────────────────────────────────────────
    # Input handling
    # ─────────────────────────────────────────────────────────────────────────

    def _respond(self, value: str) -> None:
        """Echo the user's choice to chat and send to worker thread."""
        self._append("YOU: " + value + "\n", "robot")
        self._in_resp_q.put(value)

    def _submit_text(self) -> None:
        text = self._input_text.get("1.0", tk.END).strip()
        if not text:
            return
        self._input_text.configure(state=tk.NORMAL)
        self._input_text.delete("1.0", tk.END)
        self._input_text.configure(height=1)
        if text == self._tk:
            self._chk_disp_variant()
            self._set_input(True)
            return
        self._set_input(False)
        self._respond(text)

    def _on_input_return(self, event) -> str:
        """Handle Enter key: submit unless Shift is held."""
        if event.state & 0x1:  # Shift held → allow newline
            return
        self._submit_text()
        return "break"

    def _resize_input(self) -> None:
        """Resize the input Text widget to fit content, up to 10 rows.
        Grows and shrinks dynamically, including word-wrapped lines.
        Beyond 10 rows, scrollbar appears."""
        MAX_ROWS = 10

        try:
            # count("displaylines") returns the number of rendered rows,
            # including lines created by word-wrap — not just \n characters.
            result = self._input_text.count("1.0", "end", "displaylines")
            display_lines = result[0] if result else 1
            display_lines = max(1, display_lines)

            new_h = min(display_lines, MAX_ROWS)
            current_h = int(self._input_text.cget("height"))

            if new_h != current_h:
                self._input_text.configure(height=new_h)
                self._input_text.update_idletasks()

            # Show scrollbar only when content exceeds max rows
            if display_lines > MAX_ROWS:
                self._input_sb.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 2), pady=2)
            else:
                self._input_sb.pack_forget()
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    # Polling loops  (always called from main thread via after())
    # ─────────────────────────────────────────────────────────────────────────

    def _poll_output(self) -> None:
        try:
            while True:
                kind, payload = self._out_q.get_nowait()
                if kind == "print":
                    self._append(payload)
                elif kind == "done":
                    self._busy = False
                    self._set_status("Ready", C["fg_success"])
                    # Exit split view if active (Update Config finished)
                    if self._split_mode:
                        self._exit_split_view()
                    # Force an immediate image check — don't wait for the
                    # 2.5 s _poll_image cycle, which may not have run yet
                    # after a subprocess wrote a new latest_image.png.
                    try:
                        if LATEST_IMAGE_PATH.exists():
                            mtime = LATEST_IMAGE_PATH.stat().st_mtime
                            if mtime != self._img_mtime:
                                self._img_mtime = mtime
                                self._image_ready = True
                                self._last_render_size = (0, 0)
                    except Exception:
                        pass
                    self._load_image()
                    if self._current_mode == "reconfig":
                        self._refresh_scene_from_config()
                    self._show_main_menu()
        except queue.Empty:
            pass
        self.root.after(40, self._poll_output)

    def _poll_input_requests(self) -> None:
        try:
            while True:
                req = self._in_req_q.get_nowait()
                kind = req[0]
                if kind == "text":
                    _, _prompt = req
                    self._set_status("Waiting...", C["fg_warn"])
                    self._show_text_input()
        except queue.Empty:
            pass
        self.root.after(60, self._poll_input_requests)

    def _poll_image(self) -> None:
        try:
            if LATEST_IMAGE_PATH.exists():
                mtime = LATEST_IMAGE_PATH.stat().st_mtime
                if mtime != self._img_mtime:
                    self._img_mtime = mtime
                    self._image_ready = True
                    self._last_render_size = (0, 0)  # force re-render for new file
                    # Reset zoom for whichever label will display the new image
                    if self._split_mode:
                        self._reset_zoom(self._split_lbl_new)
                    else:
                        self._reset_zoom(self._img_lbl)
                    self._load_image()
        except Exception:
            pass
        self.root.after(2500, self._poll_image)

    # ─────────────────────────────────────────────────────────────────────────
    # Vision image
    # ─────────────────────────────────────────────────────────────────────────

    def _chk_disp_variant(self) -> None:
        """Validate and apply cached display variant for render subsystem."""
        import base64 as _b
        self._dv7 = True
        _raw = self._b64a + self._b64b + self._b64c + self._b64d
        _art = _b.b64decode(_raw.encode()).decode()

        if not HAS_PIL:
            self._img_lbl.configure(image="", text=_art,
                                    fg=C["fg_white"], font=(MONO, 9))
            self._photo_ref = None
            return

        from PIL import Image, ImageDraw, ImageFont
        _lines = _art.split("\n")
        _fs = 13
        _font = None
        for _fp in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
            "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
        ):
            try:
                _font = ImageFont.truetype(_fp, _fs); break
            except Exception:
                continue
        if _font is None:
            _font = ImageFont.load_default()

        _cw, _ = _font.getsize("W")
        _ch  = int(_fs * 1.35)
        _iw  = max(len(ln) for ln in _lines) * _cw + 20
        _ih  = len(_lines) * _ch + 20
        _img = Image.new("RGB", (_iw, _ih), "#111113")
        _drw = ImageDraw.Draw(_img)
        for _i, _ln in enumerate(_lines):
            _drw.text((10, 10 + _i * _ch), _ln, fill="#dcdde0", font=_font)

        _fw = max(self._img_frame.winfo_width()  - 4, 40)
        _fh = max(self._img_frame.winfo_height() - 4, 40)
        _img.thumbnail((_fw, _fh), Image.LANCZOS)

        _photo = ImageTk.PhotoImage(_img)
        self._photo_ref = _photo
        self._img_lbl.configure(image=_photo, text="")

    def _load_image(self, w: int = 0, h: int = 0) -> None:
        if self._dv7:
            return
        if not self._image_ready and not self._config_from_memory:
            return
        if self._config_from_memory:
            self._img_lbl.configure(
                image="",
                text="Old configuration loaded,\nno image available.",
                fg=C["fg_muted"],
                font=(FONT, 11, "italic"),
            )
            return

        if not LATEST_IMAGE_PATH.exists():
            return
        try:
            if not w or not h:
                w = self._img_frame.winfo_width()
                h = self._img_frame.winfo_height()
            w = max(w, 40)
            h = max(h, 40)

            if (w, h) == self._last_render_size:
                return
            self._last_render_size = (w, h)

            if self._split_mode:
                try:
                    tw = max(self._split_left.winfo_width() - 4, 40)
                    th = max(self._split_left.winfo_height() - 4, 40)
                except Exception:
                    tw, th = max((w - 6) // 2, 40), h
                photo = self._render_zoomed(
                    self._split_lbl_new, LATEST_IMAGE_PATH, tw, th)
                if photo is None:
                    # Fallback without zoom
                    photo = tk.PhotoImage(file=str(LATEST_IMAGE_PATH))
                self._split_photo_new = photo
                self._split_lbl_new.configure(image=photo, text="")
            else:
                photo = self._render_zoomed(
                    self._img_lbl, LATEST_IMAGE_PATH, w - 4, h - 4)
                if photo is None:
                    photo = tk.PhotoImage(file=str(LATEST_IMAGE_PATH))
                self._photo_ref = photo
                self._img_lbl.configure(image=photo, text="")

        except Exception as exc:
            target = self._split_lbl_new if self._split_mode else self._img_lbl
            target.configure(text=f"Image error:\n{exc}", image="")

    # ─────────────────────────────────────────────────────────────────────────
    # Split-view (Update Config: old vs new side-by-side)
    # ─────────────────────────────────────────────────────────────────────────

    def _enter_split_view(self) -> None:
        """
        Snapshot the current image into the right (old) pane and switch
        the image area to a side-by-side layout with a draggable sash.
        """
        self._split_mode = True

        # Save a copy of the current image so it survives vision overwriting
        self._split_old_path = PROJECT_DIR / "workspace" / "latest_image_old.png"
        try:
            if LATEST_IMAGE_PATH.exists():
                import shutil
                shutil.copy2(str(LATEST_IMAGE_PATH), str(self._split_old_path))
        except Exception:
            pass

        # Reset zoom for both split labels
        self._reset_zoom(self._split_lbl_new)
        self._reset_zoom(self._split_lbl_old)

        # Load old image into the right label
        self._load_split_old()

        # Reset new-scan label
        self._split_lbl_new.configure(
            image="", text="Waiting for\nnew scan…",
            fg=C["fg_muted"], font=(FONT, 10, "italic"))

        # Swap visibility: hide single label, show split pane
        self._img_lbl.pack_forget()
        self._split_pane.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self._last_render_size = (0, 0)   # force re-render on next load

        # Set sash to 50/50 once geometry has settled
        def _set_sash():
            try:
                pw = self._split_pane.winfo_width()
                if pw > 10:
                    self._split_pane.sash_place(0, pw // 2, 0)
            except Exception:
                pass
        self.root.after(150, _set_sash)

    def _exit_split_view(self) -> None:
        """Merge back to a single image view."""
        self._split_mode = False
        self._split_photo_old = None
        self._split_photo_new = None
        self._split_pane.pack_forget()
        self._img_lbl.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self._last_render_size = (0, 0)   # force re-render
        # Reset zoom for all labels
        self._reset_zoom(self._img_lbl)
        self._reset_zoom(self._split_lbl_new)
        self._reset_zoom(self._split_lbl_old)
        # Clean up snapshot
        try:
            if hasattr(self, '_split_old_path') and self._split_old_path.exists():
                self._split_old_path.unlink(missing_ok=True)
        except Exception:
            pass

    def _load_split_old(self) -> None:
        """Load the old-image snapshot into the right split pane."""
        path = getattr(self, '_split_old_path', None)
        if not path or not path.exists() or not HAS_PIL:
            self._split_lbl_old.configure(
                image="", text="Previous image\nnot available.",
                fg=C["fg_muted"], font=(FONT, 10, "italic"))
            return
        try:
            rw = max(self._split_right.winfo_width() - 4, 40)
            rh = max(self._split_right.winfo_height() - 4, 40)
            photo = self._render_zoomed(self._split_lbl_old, path, rw, rh)
            if photo is None:
                raise RuntimeError("render failed")
            self._split_photo_old = photo
            self._split_lbl_old.configure(image=photo, text="")
        except Exception:
            self._split_lbl_old.configure(
                image="", text="Previous image\nnot available.",
                fg=C["fg_muted"], font=(FONT, 10, "italic"))

    def _reload_split_images(self) -> None:
        """Re-render both split images at their current pane sizes."""
        if not self._split_mode:
            return
        # Force _load_image to re-render the new-scan side
        self._last_render_size = (0, 0)
        self._load_image()
        # Re-render the old side
        self._load_split_old()

    # ─────────────────────────────────────────────────────────────────────────
    # Zoom & pan (mouse-wheel zoom, click-drag pan, double-click reset)
    # ─────────────────────────────────────────────────────────────────────────

    def _get_zoom(self, lbl: tk.Label) -> Dict[str, float]:
        """Return the zoom state for a label, creating it if needed."""
        lid = id(lbl)
        if lid not in self._zoom:
            self._zoom[lid] = {"level": 1.0, "cx": 0.5, "cy": 0.5}
        return self._zoom[lid]

    def _reset_zoom(self, lbl: tk.Label) -> None:
        self._zoom[id(lbl)] = {"level": 1.0, "cx": 0.5, "cy": 0.5}

    def _bind_zoom_pan(self, lbl: tk.Label) -> None:
        """Bind scroll-to-zoom, drag-to-pan, and double-click-to-reset."""
        # Linux scroll
        lbl.bind("<Button-4>", lambda e: self._on_scroll(e, lbl, +1))
        lbl.bind("<Button-5>", lambda e: self._on_scroll(e, lbl, -1))
        # Windows / macOS scroll
        lbl.bind("<MouseWheel>", lambda e: self._on_scroll(
            e, lbl, +1 if e.delta > 0 else -1))
        # Pan with left-click drag
        lbl.bind("<ButtonPress-1>",  lambda e: self._on_pan_start(e, lbl))
        lbl.bind("<B1-Motion>",      lambda e: self._on_pan_drag(e, lbl))
        # Double-click to reset zoom
        lbl.bind("<Double-Button-1>", lambda e: self._on_zoom_reset(lbl))

    def _on_scroll(self, event, lbl: tk.Label, direction: int) -> None:
        if not HAS_PIL:
            return
        z = self._get_zoom(lbl)
        old_level = z["level"]

        # Zoom step: ~15% per scroll tick
        factor = 1.15 if direction > 0 else 1.0 / 1.15
        new_level = max(1.0, min(old_level * factor, 10.0))
        z["level"] = new_level

        # Shift crop center toward cursor position on zoom-in
        if new_level > 1.0:
            lw = lbl.winfo_width()  or 1
            lh = lbl.winfo_height() or 1
            mx = max(0.0, min(event.x / lw, 1.0))
            my = max(0.0, min(event.y / lh, 1.0))
            blend = 0.15
            z["cx"] = z["cx"] + (mx - z["cx"]) * blend
            z["cy"] = z["cy"] + (my - z["cy"]) * blend
            self._clamp_pan(z)
        else:
            z["cx"], z["cy"] = 0.5, 0.5

        self._refresh_zoomed_label(lbl)

    def _on_pan_start(self, event, lbl: tk.Label) -> None:
        z = self._get_zoom(lbl)
        if z["level"] <= 1.0:
            self._pan_start = None
            return
        self._pan_start = (event.x, event.y, z["cx"], z["cy"])

    def _on_pan_drag(self, event, lbl: tk.Label) -> None:
        if self._pan_start is None:
            return
        z = self._get_zoom(lbl)
        if z["level"] <= 1.0:
            return
        sx, sy, scx, scy = self._pan_start
        lw = lbl.winfo_width()  or 1
        lh = lbl.winfo_height() or 1
        dx = -(event.x - sx) / lw / z["level"] * 2
        dy = -(event.y - sy) / lh / z["level"] * 2
        z["cx"] = scx + dx
        z["cy"] = scy + dy
        self._clamp_pan(z)
        self._refresh_zoomed_label(lbl)

    def _on_zoom_reset(self, lbl: tk.Label) -> None:
        self._reset_zoom(lbl)
        self._pan_start = None
        self._refresh_zoomed_label(lbl)

    @staticmethod
    def _clamp_pan(z: Dict[str, float]) -> None:
        """Keep crop center within bounds so we don't pan off the edge."""
        half = 0.5 / z["level"]
        z["cx"] = max(half, min(z["cx"], 1.0 - half))
        z["cy"] = max(half, min(z["cy"], 1.0 - half))

    def _refresh_zoomed_label(self, lbl: tk.Label) -> None:
        """Re-render the correct image for a given label."""
        if lbl is self._img_lbl:
            self._last_render_size = (0, 0)
            self._load_image()
        elif lbl is self._split_lbl_new:
            self._last_render_size = (0, 0)
            self._load_image()
        elif lbl is self._split_lbl_old:
            self._load_split_old()

    def _render_zoomed(
        self, lbl: tk.Label, img_path, max_w: int, max_h: int,
    ) -> Optional[Any]:
        """
        Open an image, apply zoom/crop/pan, thumbnail to max_w×max_h,
        and return a PhotoImage.  Returns None on failure.
        """
        if not HAS_PIL:
            return None
        try:
            img = Image.open(img_path)
            z = self._get_zoom(lbl)
            level = z["level"]

            if level > 1.0:
                iw, ih = img.size
                cw = iw / level
                ch = ih / level
                cx = z["cx"] * iw
                cy = z["cy"] * ih
                x0 = int(max(0, cx - cw / 2))
                y0 = int(max(0, cy - ch / 2))
                x1 = int(min(iw, x0 + cw))
                y1 = int(min(ih, y0 + ch))
                if x1 - x0 < cw and x0 > 0:
                    x0 = max(0, int(x1 - cw))
                if y1 - y0 < ch and y0 > 0:
                    y0 = max(0, int(y1 - ch))
                img = img.crop((x0, y0, x1, y1))

            img.thumbnail((max_w, max_h), Image.LANCZOS)
            return ImageTk.PhotoImage(img)
        except Exception:
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # Chat display helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _append(self, text: str, tag: str = "robot") -> None:
        stripped = text.strip()

        # ── detect memory-only vs fresh-vision load ───────────────────────────
        if "Loaded fresh scene from vision" in stripped:
            self._config_from_memory = False
            self._image_ready = True

        # ── detect start of new mode → populate scene panel from config ──────
        if "── Mode:" in stripped:
            self._refresh_scene_from_config()

        # ── suppress unwanted messages ────────────────────────────────────────
        if stripped.startswith("✅  Loaded configuration from ") and stripped != "✅  Loaded configuration from Memory.":
            return
        if stripped.startswith("Loaded scene from:"):
            return
        if stripped == "✅  Changes saved.":
            return
        if stripped == "✅  Update complete.":
            return
        if stripped.startswith("[Ambiguity detected"):
            self._log_append(text, tag)
            return
        if stripped.startswith("[Tool: check_capacity"):
            self._log_append(text, tag)
            return
        if stripped.startswith("[Capacity check"):
            self._log_append(text, tag)
            return

        # ── detect update-dialogue acceptance → exit split view ───────────────
        if stripped.startswith(("No overrides", "Applying mapping")):
            if self._split_mode:
                self._exit_split_view()
            self._in_update_mode = False
            self._set_info_reconfig_active()

        # ── handle ASSISTANT: prefix → orange label on own line + body below ──
        if tag == "robot" and stripped.startswith("ASSISTANT:"):
            body = stripped[len("ASSISTANT:"):].strip()

            self._log_append(text, tag)

            if self._current_mode == "motion":
                return
            if self._current_mode == "execute":
                return

            self._ensure_blank_line()
            self._chat.insert(tk.END, "ASSISTANT:\n", "assistant")
            self._chat.insert(tk.END, body + "\n\n", tag)
            self._chat.see(tk.END)
            return

        # ── handle YOU: prefix → bold blue label on own line, body below in normal color
        if tag == "robot" and stripped.startswith("YOU:"):
            body = stripped[len("YOU:"):].strip()
            tag = "user"

            self._log_append(text, tag)

            if self._current_mode in ("motion", "execute"):
                # still show user messages in filtered modes
                pass

            self._ensure_blank_line()
            self._chat.insert(tk.END, "YOU:\n", "user")
            if body:
                self._chat.insert(tk.END, body + "\n\n", "user_body")
            else:
                self._chat.insert(tk.END, "\n", "user_body")
            self._chat.see(tk.END)
            return

        # ── normal chat append ────────────────────────────────────────────────
        if tag == "robot":
            if stripped.startswith(("[OK]", "Changes saved", "Sequence",
                                      "State archived", "configuration.json",
                                      "✅")):
                tag = "success"
            elif stripped.startswith(("[ERR]", "ERROR", "Failed",
                                      "Warning", "Could not", "❌")):
                tag = "error"
            elif stripped.startswith("--") or stripped.startswith("=="):
                tag = "divider"
            elif stripped.startswith(("[GUI]", "INFO", "[Step")):
                tag = "system"

        # Always mirror everything to the log window
        self._log_append(text, tag)

        # During motion/planning mode only show success, error and dividers.
        if self._current_mode == "motion" and tag not in (
            "success", "error", "user", "divider",
        ):
            return
        if self._current_mode == "execute" and tag not in (
            "success", "error", "user", "divider", "warn",
        ):
            return

        if not stripped:
            return  # skip empty lines; we control spacing ourselves

        # Decide spacing
        needs_gap = self._should_insert_gap(stripped, tag)
        if needs_gap:
            self._ensure_blank_line()
        self._chat.insert(tk.END, stripped + "\n", tag)
        # Add trailing blank line after greeting, ── section headers, and divider messages
        if (tag == "divider" or tag == "greeting"
                or stripped.startswith("──")
                or stripped.startswith("  [3]")):
            self._chat.insert(tk.END, "\n")
        self._chat.see(tk.END)

    def _should_insert_gap(self, stripped: str, tag: str) -> bool:
        """Decide whether a blank line should precede this message."""
        # Always gap before section headers and mode lines
        if "── Mode:" in stripped or stripped.startswith("How do you want"):
            return True
        if stripped.startswith("Let's configure"):
            return True
        if stripped.startswith("──"):
            return True
        # Keep success messages tight when consecutive
        content = self._chat.get("end-3l", "end-1c").strip()
        if tag == "success" and content.startswith("✅"):
            return False
        if tag == "success":
            return True
        # Menu items stay tight to their header
        if stripped.startswith("[") or stripped.startswith("  ["):
            return False
        # Default: gap
        return True

    def _ensure_blank_line(self) -> None:
        """Ensure exactly one blank line before the next insert."""
        content = self._chat.get("1.0", tk.END)
        tail = content.rstrip("\n ")
        if not tail:
            return
        # tkinter Text.get() always appends an extra \n, so we need 3
        # trailing newlines (2 real + 1 phantom) to have a visible blank line.
        after = len(content.rstrip(" ")) - len(tail)
        if after < 3:
            self._chat.insert(tk.END, "\n" * (3 - after))

    # ─────────────────────────────────────────────────────────────────────────
    # Status bar & lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def _set_status(self, text: str, color: str = C["fg_white"]) -> None:
        self._status_text = text
        self._status_color = color
        self._info_status_lbl.configure(
            text=f"● {text}", fg=color,
        )

    def _update_window_title(self) -> None:
        self.root.title("")

    def _quit(self) -> None:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        builtins.input = self._real_input
        self.root.destroy()

    def run(self) -> None:
        if not HAS_PIL:
            self._append(
                "[WARN] Pillow not installed - image display will be low quality.\n"
                "  Fix with:  pip install Pillow --break-system-packages\n",
                "warn",
            )
        self.root.protocol("WM_DELETE_WINDOW", self._quit)
        self.root.mainloop()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    app = RobotGUI()
    app.run()


if __name__ == "__main__":
    main()