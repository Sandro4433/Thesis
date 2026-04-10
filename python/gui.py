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
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# ── PIL (optional but recommended) ────────────────────────────────────────────
try:
    from PIL import Image, ImageTk   # pip install Pillow
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

LATEST_IMAGE_PATH = PROJECT_DIR / "File_Exchange" / "latest_image.png"
CANCEL_SENTINEL   = PROJECT_DIR / "File_Exchange" / ".cancel_execution"

# ── Tell Vision_Main to skip cv2.imshow (GUI shows the image instead) ─────────
os.environ["ROBOT_GUI_MODE"] = "1"


# ─────────────────────────────────────────────────────────────────────────────
# Colours & fonts
# ─────────────────────────────────────────────────────────────────────────────

C = {
    "bg_main":    "#2b2d31",
    "bg_title":   "#1e1f22",
    "bg_chat":    "#313338",
    "bg_input":   "#383a40",
    "bg_btn":     "#4f545c",
    "bg_accent":  "#5865f2",
    "bg_green":   "#248046",
    "bg_red":     "#da373c",
    "bg_orange":  "#c27c0e",

    "fg_white":   "#ffffff",
    "fg_muted":   "#96989d",
    "fg_robot":   "#dcddde",
    "fg_user":    "#00b0f4",
    "fg_system":  "#72767d",
    "fg_success": "#57f287",
    "fg_error":   "#ed4245",
    "fg_info":    "#5865f2",
    "fg_warn":    "#faa61a",
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
        self._img_mtime         = 0.0
        self._photo_ref: Optional[Any] = None   # prevent GC of PhotoImage
        self._image_ready       = False          # True only after a new image is taken this session

        # scene-routing state
        self._config_from_memory   = False   # show placeholder instead of image
        self._in_configure_mode    = False   # only Done in button bar
        self._in_update_mode       = False   # True during "Update Config" (enables Recapture)
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

        # ── window ────────────────────────────────────────────────────────────
        self.root = tk.Tk()
        self.root.title("Robot Configuration System")
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
        # ── title bar ─────────────────────────────────────────────────────────
        title = tk.Frame(self.root, bg=C["bg_title"], height=50)
        title.pack(fill=tk.X)
        title.pack_propagate(False)

        tk.Label(
            title, text="Robot Configuration System",
            bg=C["bg_title"], fg=C["fg_white"],
            font=(FONT, 14, "bold"), padx=18,
        ).pack(side=tk.LEFT, pady=10)

        self._status = tk.Label(
            title, text="Ready",
            bg=C["bg_title"], fg=C["fg_success"],
            font=(FONT, 10), padx=18,
        )
        self._status.pack(side=tk.RIGHT)

        # ── main content: horizontal PanedWindow (left | right) ───────────────
        self._hpane = tk.PanedWindow(
            self.root,
            orient=tk.HORIZONTAL,
            bg=C["bg_main"],
            sashwidth=6,
            sashrelief=tk.FLAT,
            bd=0,
            handlesize=0,
        )
        self._hpane.pack(fill=tk.BOTH, expand=True, padx=10, pady=(6, 0))

        self._build_chat_panel(self._hpane)
        self._build_right_panel(self._hpane)

        # ── button bar ────────────────────────────────────────────────────────
        bar_outer = tk.Frame(self.root, bg=C["bg_title"], pady=10)
        bar_outer.pack(fill=tk.X, side=tk.BOTTOM)
        self._btn_bar = tk.Frame(bar_outer, bg=C["bg_title"])
        self._btn_bar.pack()

    def _build_chat_panel(self, parent: tk.PanedWindow) -> None:
        left = tk.Frame(parent, bg=C["bg_main"])
        parent.add(left, stretch="always", minsize=320)

        tk.Label(
            left, text="CONVERSATION",
            bg=C["bg_main"], fg=C["fg_muted"],
            font=(FONT, 8, "bold"),
        ).pack(anchor=tk.W, pady=(2, 3))

        # chat display
        chat_wrap = tk.Frame(left, bg=C["bg_chat"])
        chat_wrap.pack(fill=tk.BOTH, expand=True)

        self._chat = tk.Text(
            chat_wrap,
            bg=C["bg_chat"], fg=C["fg_robot"],
            font=(MONO, 10),
            wrap=tk.WORD, state=tk.NORMAL,
            bd=0, padx=14, pady=10,
            spacing1=1, spacing3=3,
            selectbackground=C["bg_btn"],
            selectforeground=C["fg_white"],
        )
        sb = tk.Scrollbar(chat_wrap, command=self._chat.yview,
                          bg=C["bg_chat"], troughcolor=C["bg_chat"],
                          activebackground=C["bg_btn"])
        self._chat.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
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
        self._chat.tag_configure("user",    foreground=C["fg_user"],    font=(MONO, 10, "bold"))
        self._chat.tag_configure("robot",   foreground=C["fg_robot"])
        self._chat.tag_configure("success", foreground=C["fg_success"])
        self._chat.tag_configure("error",   foreground=C["fg_error"])
        self._chat.tag_configure("warn",    foreground=C["fg_warn"])
        self._chat.tag_configure("system",  foreground=C["fg_system"],  font=(MONO, 9))
        self._chat.tag_configure("divider", foreground="#4f545c",       font=(MONO, 9))
        self._chat.tag_configure("info",    foreground=C["fg_info"])
        self._chat.tag_configure("greeting",
                                 foreground=C["fg_white"],
                                 font=(MONO, 11, "bold"))

        # input row
        row = tk.Frame(left, bg=C["bg_main"], pady=6)
        row.pack(fill=tk.X)

        self._input_var = tk.StringVar()
        self._entry = tk.Entry(
            row,
            textvariable=self._input_var,
            bg=C["bg_input"], fg=C["fg_white"],
            font=(MONO, 11),
            insertbackground=C["fg_white"],
            relief=tk.FLAT, bd=0,
            state=tk.DISABLED,
        )
        self._entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=9, ipadx=12)
        self._entry.bind("<Return>", lambda _e: self._submit_text())

        self._send = tk.Button(
            row, text="Send",
            bg=C["bg_accent"], fg=C["fg_white"],
            font=(FONT, 10, "bold"),
            relief=tk.FLAT, bd=0, padx=16, pady=9,
            activebackground="#4752c4", activeforeground=C["fg_white"],
            state=tk.DISABLED,
            command=self._submit_text,
        )
        self._send.pack(side=tk.LEFT, padx=(6, 0))

    def _build_right_panel(self, parent: tk.PanedWindow) -> None:
        """Right half: scene description (top) + vision image (bottom)."""
        right = tk.Frame(parent, bg=C["bg_main"])
        parent.add(right, stretch="always", minsize=320)

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

        # ── top: scene description ────────────────────────────────────────────
        scene_outer = tk.Frame(self._vpane, bg=C["bg_main"])
        self._vpane.add(scene_outer, stretch="always", minsize=120)

        tk.Label(
            scene_outer, text="SCENE DESCRIPTION",
            bg=C["bg_main"], fg=C["fg_muted"],
            font=(FONT, 8, "bold"),
        ).pack(anchor=tk.W, pady=(2, 3))

        scene_wrap = tk.Frame(scene_outer, bg=C["bg_chat"])
        scene_wrap.pack(fill=tk.BOTH, expand=True)

        self._scene_text = tk.Text(
            scene_wrap,
            bg=C["bg_chat"], fg=C["fg_robot"],
            font=(MONO, 10),
            wrap=tk.WORD, state=tk.DISABLED,
            bd=0, padx=14, pady=10,
            spacing1=1, spacing3=3,
            selectbackground=C["bg_btn"],
            selectforeground=C["fg_white"],
        )
        scene_sb = tk.Scrollbar(scene_wrap, command=self._scene_text.yview,
                                bg=C["bg_chat"], troughcolor=C["bg_chat"],
                                activebackground=C["bg_btn"])
        self._scene_text.configure(yscrollcommand=scene_sb.set)
        scene_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._scene_text.pack(fill=tk.BOTH, expand=True)

        self._scene_text.tag_configure("scene_text", foreground=C["fg_robot"])
        self._scene_text.tag_configure("placeholder",
                                       foreground=C["fg_muted"],
                                       font=(MONO, 10, "italic"))

        self._set_scene_placeholder("No scene loaded yet.\nRun Reconfigure to populate.")

        # ── bottom: vision image ──────────────────────────────────────────────
        self._img_outer = tk.Frame(self._vpane, bg=C["bg_main"])
        img_outer = self._img_outer
        self._vpane.add(img_outer, stretch="always", minsize=120)

        tk.Label(
            img_outer, text="VISION",
            bg=C["bg_main"], fg=C["fg_muted"],
            font=(FONT, 8, "bold"),
        ).pack(anchor=tk.W, pady=(2, 3))

        self._img_frame = tk.Frame(img_outer, bg=C["bg_title"], bd=0)
        self._img_frame.pack(fill=tk.BOTH, expand=True)
        self._img_frame.pack_propagate(False)  # prevent child label from resizing us
        img_frame = self._img_frame

        self._img_lbl = tk.Label(
            img_frame, bg=C["bg_title"],
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
        self._split_left = tk.Frame(self._split_pane, bg=C["bg_title"])
        self._split_pane.add(self._split_left, stretch="always", minsize=80)
        tk.Label(self._split_left, text="NEW SCAN", bg=C["bg_main"],
                 fg=C["fg_muted"], font=(FONT, 7, "bold")).pack(fill=tk.X)
        self._split_lbl_new = tk.Label(
            self._split_left, bg=C["bg_title"],
            text="Waiting for\nnew scan…",
            fg=C["fg_muted"], font=(FONT, 10, "italic"),
        )
        self._split_lbl_new.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)

        # Right = old image
        self._split_right = tk.Frame(self._split_pane, bg=C["bg_title"])
        self._split_pane.add(self._split_right, stretch="always", minsize=80)
        tk.Label(self._split_right, text="PREVIOUS CONFIG", bg=C["bg_main"],
                 fg=C["fg_muted"], font=(FONT, 7, "bold")).pack(fill=tk.X)
        self._split_lbl_old = tk.Label(
            self._split_right, bg=C["bg_title"],
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

        # Re-scale the image live when the container frame is resized (e.g. by
        # dragging a pane divider).  We bind to the *frame* so we always get the
        # true available size, not the size of the image already on the label.
        # The event carries the new width/height directly, so we pass them into
        # _load_image to avoid a second winfo call.
        self._resize_after_id: Optional[str] = None
        self._last_render_size: tuple = (0, 0)

        def _on_img_resize(event):
            # Ignore <Configure> events that bubble up from child widgets
            # (e.g. the label resizing when a new photo is placed on it).
            # We only want events that originate on _img_outer itself —
            # those are the ones caused by sash drags or window resizes.
            if event.widget is not self._img_outer:
                return
            if self._resize_after_id:
                self.root.after_cancel(self._resize_after_id)
            self._resize_after_id = self.root.after(80, self._load_image)

        self._img_outer.bind("<Configure>", _on_img_resize)

        # timestamp strip (no refresh button)
        strip = tk.Frame(img_outer, bg=C["bg_main"], pady=4)
        strip.pack(fill=tk.X)

        self._img_ts = tk.Label(
            strip, text="", bg=C["bg_main"], fg=C["fg_muted"],
            font=(FONT, 8),
        )
        self._img_ts.pack(side=tk.LEFT)

    def _init_sash_positions(self) -> None:
        """Set the two PanedWindows to equal 50/50 splits."""
        try:
            total_w = self._hpane.winfo_width()
            if total_w > 10:
                self._hpane.sash_place(0, total_w // 2, 0)
        except Exception:
            pass
        try:
            total_h = self._vpane.winfo_height()
            if total_h > 10:
                self._vpane.sash_place(0, 0, total_h // 2)
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
        self._scene_text.configure(state=tk.NORMAL)
        self._scene_text.delete("1.0", tk.END)
        self._scene_text.insert(tk.END, msg, "placeholder")
        self._scene_text.configure(state=tk.DISABLED)

    def _set_scene_content(self, text: str) -> None:
        self._scene_text.configure(state=tk.NORMAL)
        self._scene_text.delete("1.0", tk.END)
        self._scene_text.insert(tk.END, text.strip(), "scene_text")
        self._scene_text.see("1.0")
        self._scene_text.configure(state=tk.DISABLED)

    def _refresh_scene_from_config(self) -> None:
        """Build scene summary from config file. Always shows ALL high-level
        attributes, displaying 'none' for those not yet configured."""
        try:
            import json
            import session_handler as sh  # type: ignore
            if not sh.CONFIGURATION_PATH.exists():
                return
            state = json.loads(sh.CONFIGURATION_PATH.read_text(encoding="utf-8"))
            preds = state.get("predicates", {})
            lines = []

            # Mode
            ws = state.get("workspace", {})
            mode = ws.get("operation_mode") or "not set"
            lines.append(f"Mode: {mode}")

            # Batch Size
            batch = ws.get("batch_size")
            batch_str = str(batch) if batch else "not set"
            lines.append(f"Batch Size: {batch_str}")

            # Roles
            roles = preds.get("role", [])
            assigned = [e for e in roles if e.get("role")]
            lines.append("")
            if assigned:
                lines.append("Roles:")
                for e in sorted(assigned, key=lambda x: x["object"]):
                    lines.append(f"  {e['object']} = {e['role']}")
            else:
                lines.append("Roles: none")

            # Part Fragility
            frag = [e["part"] for e in preds.get("fragility", [])
                    if e.get("fragility") == "fragile"]
            lines.append("")
            if frag:
                lines.append(f"Part Fragility: {', '.join(sorted(frag))}")
            else:
                lines.append("Part Fragility: none")

            # Priority
            prio = preds.get("priority", [])
            lines.append("")
            if prio:
                prio_str = ", ".join(
                    f"{e.get('color')} (#{e.get('order')})"
                    for e in sorted(prio, key=lambda x: x.get("order", 0))
                )
                lines.append(f"Priority: {prio_str}")
            else:
                lines.append("Priority: none")

            # Kit recipe
            recipe = preds.get("kit_recipe", [])
            lines.append("")
            if recipe:
                lines.append("Kit Recipe:")
                # Group by color+quantity+size (ignore kit field for display)
                seen = set()
                for e in recipe:
                    color = e.get("color", "")
                    qty = e.get("quantity", 0)
                    size = e.get("size")
                    key = (color, qty, size)
                    if key in seen:
                        continue
                    seen.add(key)
                    size_str = f" ({size})" if size else ""
                    lines.append(f"  {qty}x {color}{size_str}")
            else:
                lines.append("Kit Recipe: none")

            # Part compatibility
            compat = preds.get("part_compatibility", [])
            lines.append("")
            if compat:
                lines.append("Compatibility:")
                for rule in compat:
                    # Build part selector description
                    part_selectors = []
                    if rule.get("part_name"):
                        part_selectors.append(rule["part_name"])
                    if rule.get("part_color"):
                        part_selectors.append(f"{rule['part_color']} parts")
                    if rule.get("part_fragility"):
                        part_selectors.append(f"{rule['part_fragility']} parts")
                    
                    part_desc = " + ".join(part_selectors) if part_selectors else "all parts"
                    
                    # Build receptacle selector description
                    if rule.get("allowed_in"):
                        rec_desc = ", ".join(rule["allowed_in"])
                    elif rule.get("allowed_in_role"):
                        rec_desc = f"all {rule['allowed_in_role']}s"
                    else:
                        rec_desc = "all"
                    
                    # Add exclusions
                    if rule.get("not_allowed_in"):
                        rec_desc += f" (except {', '.join(rule['not_allowed_in'])})"
                    
                    lines.append(f"  {part_desc} → {rec_desc}")
            else:
                lines.append("Compatibility: none")

            self._set_scene_content("\n".join(lines))

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

            tk.Label(
                self._log_win, text="FULL OUTPUT LOG",
                bg=C["bg_main"], fg=C["fg_muted"],
                font=(FONT, 8, "bold"),
            ).pack(anchor=tk.W, padx=10, pady=(8, 2))

            wrap = tk.Frame(self._log_win, bg=C["bg_chat"])
            wrap.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

            self._log_text = tk.Text(
                wrap,
                bg=C["bg_chat"], fg=C["fg_robot"],
                font=(MONO, 10),
                wrap=tk.WORD, state=tk.NORMAL,
                bd=0, padx=14, pady=10,
                spacing1=1, spacing3=3,
                selectbackground=C["bg_btn"],
                selectforeground=C["fg_white"],
            )
            sb = tk.Scrollbar(wrap, command=self._log_text.yview,
                              bg=C["bg_chat"], troughcolor=C["bg_chat"],
                              activebackground=C["bg_btn"])
            self._log_text.configure(yscrollcommand=sb.set)
            sb.pack(side=tk.RIGHT, fill=tk.Y)
            self._log_text.pack(fill=tk.BOTH, expand=True)

            # Same colour tags as chat, plus all keys allowed (fully copyable)
            for tag, cfg in [
                ("user",    {"foreground": C["fg_user"],    "font": (MONO, 10, "bold")}),
                ("robot",   {"foreground": C["fg_robot"]}),
                ("success", {"foreground": C["fg_success"]}),
                ("error",   {"foreground": C["fg_error"]}),
                ("warn",    {"foreground": C["fg_warn"]}),
                ("system",  {"foreground": C["fg_system"],  "font": (MONO, 9)}),
                ("divider", {"foreground": "#4f545c",        "font": (MONO, 9)}),
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
            padx=18, pady=10,
            activeforeground=C["fg_white"],
            command=cmd,
            **kwargs,
        )

    def _show_main_menu(self) -> None:
        self._clear_bar()
        self._set_input(False)
        self._in_configure_mode = False
        self._in_update_mode = False

        # Show greeting the very first time
        if not self._first_menu_shown:
            self._first_menu_shown = True
            self._append(
                "Let's configure, what would you like to do?\n",
                "greeting",
            )

        items = [
            ("Configure",     C["bg_accent"],  self._show_configure_options),
            ("Plan Sequence", "#4752c4",        lambda: self._run("motion")),
            ("Execute",       C["bg_green"],    lambda: self._run("execute")),
            ("Log",           C["bg_orange"],   self._open_log_window),
            ("Exit",          C["bg_red"],      self._quit),
        ]
        for text, color, cmd in items:
            self._btn(self._btn_bar, text, color, cmd).pack(side=tk.LEFT, padx=5)

    def _show_cancel_bar(self) -> None:
        """Show only the Done button (used during configure/planning mode)."""
        self._clear_bar()
        self._btn(
            self._btn_bar, "Done", C["bg_accent"],
            self._cancel_configure,
        ).pack(side=tk.LEFT, padx=5)

    def _show_update_bar(self) -> None:
        """Show Recapture + Done buttons during Update Config mode."""
        self._clear_bar()
        self._btn(
            self._btn_bar, "Recapture Image", C["bg_orange"],
            self._request_recapture,
        ).pack(side=tk.LEFT, padx=5)
        self._btn(
            self._btn_bar, "Done", C["bg_accent"],
            self._cancel_configure,
        ).pack(side=tk.LEFT, padx=5)

    def _request_recapture(self) -> None:
        """Send the recapture sentinel through the input queue so the
        backend re-runs vision while keeping the original old config."""
        self._set_input(False)
        self._append("Recapturing image …\n", "info")
        self._set_status("Recapturing...", C["fg_warn"])
        # Reset the new-scan label to show "waiting" while vision runs
        self._split_lbl_new.configure(
            image="", text="Recapturing …",
            fg=C["fg_muted"], font=(FONT, 10, "italic"))
        self._split_photo_new = None
        self._last_render_size = (0, 0)
        # Send sentinel — the backend worker thread is blocked on input()
        self._in_resp_q.put("__RECAPTURE__")

    def _show_execute_bar(self) -> None:
        """Show Cancel + Log buttons during robot execution."""
        self._clear_bar()
        self._btn(
            self._btn_bar, "Cancel", C["bg_red"],
            self._cancel_execution,
        ).pack(side=tk.LEFT, padx=5)
        self._btn(
            self._btn_bar, "Log", C["bg_orange"],
            self._open_log_window,
        ).pack(side=tk.LEFT, padx=5)

    def _show_configure_options(self) -> None:
        """Replace the main button bar with three configure sub-options + Done."""
        self._clear_bar()
        self._set_input(False)

        self._append(
            "\nHow do you want to load the scene?\n"
            "  [1] New Configuration — Capture new image and configure from scratch\n"
            "  [2] Update Configuration — Update configuration with new vision data\n"
            "  [3] Edit Configuration — Edit current configuration (no new image)\n",
            "robot",
        )

        items = [
            ("New Config",    C["bg_accent"], lambda: self._run_reconfig_sub("reconfig_fresh")),
            ("Update Config", C["bg_accent"], lambda: self._run_reconfig_sub("reconfig_update")),
            ("Edit Config",   C["bg_accent"], lambda: self._run_reconfig_sub("reconfig_memory")),
            ("Done",          C["bg_accent"], self._show_main_menu),
        ]
        for text, color, cmd in items:
            self._btn(self._btn_bar, text, color, cmd).pack(side=tk.LEFT, padx=5)

    def _run_reconfig_sub(self, sub: str) -> None:
        """Launch the reconfig worker with a pre-selected sub-option so the
        backend's select_reconfig_source() is bypassed entirely."""
        self._reconfig_sub = sub
        self._config_from_memory = False
        self._in_update_mode = (sub == "reconfig_update")
        if sub == "reconfig_update":
            self._enter_split_view()
        self._run("reconfig")

    def _cancel_configure(self) -> None:
        """Send 'done' to the worker, as if the user typed it."""
        self._append("YOU: done\n", "user")
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
        if self._in_configure_mode:
            if self._in_update_mode:
                self._show_update_bar()
            else:
                self._show_cancel_bar()
        else:
            self._clear_bar()
        self._set_input(True)
        self._entry.focus_set()

    def _set_input(self, enabled: bool) -> None:
        s = tk.NORMAL if enabled else tk.DISABLED
        self._entry.configure(state=s)
        self._send.configure(state=s)

    # ─────────────────────────────────────────────────────────────────────────
    # Backend worker
    # ─────────────────────────────────────────────────────────────────────────

    def _run(self, mode: str) -> None:
        if self._busy:
            return
        self._busy = True
        self._current_mode = mode
        self._in_configure_mode = True

        # Clean up any leftover cancel sentinel
        try:
            CANCEL_SENTINEL.unlink(missing_ok=True)
        except Exception:
            pass

        self._set_status("Running...", C["fg_muted"])
        if mode == "execute":
            self._show_execute_bar()
        else:
            self._show_cancel_bar()
        self._set_input(False)
        threading.Thread(target=self._worker, args=(mode,), daemon=True).start()

    def _worker(self, mode: str) -> None:
        # Re-apply redirect here — some imports (rospy, etc.) can reset sys.stdout.
        sys.stdout = _GUIStream(self._out_q)
        sys.stderr = _GUIStream(self._out_q)
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
                        from Configuration_Module.Update_Scene import run_post_execution_rescan  # type: ignore
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
            print(f"\n[ERR] Unexpected error: {exc}\n")
            print(traceback.format_exc())
        finally:
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

    # ─────────────────────────────────────────────────────────────────────────
    # Input handling
    # ─────────────────────────────────────────────────────────────────────────

    def _respond(self, value: str) -> None:
        """Echo the user's choice to chat and send to worker thread."""
        self._append("YOU: " + value + "\n", "user")
        self._in_resp_q.put(value)

    def _submit_text(self) -> None:
        text = self._input_var.get().strip()
        if not text:
            return
        self._input_var.set("")
        self._set_input(False)
        self._respond(text)

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

    def _load_image(self, w: int = 0, h: int = 0) -> None:
        if not self._image_ready and not self._config_from_memory:
            return
        if self._config_from_memory:
            self._img_lbl.configure(
                image="",
                text="Old configuration loaded,\nno image available.",
                fg=C["fg_muted"],
                font=(FONT, 11, "italic"),
            )
            self._img_ts.configure(text="")
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

            ts = datetime.fromtimestamp(LATEST_IMAGE_PATH.stat().st_mtime)
            self._img_ts.configure(text=f"Updated {ts.strftime('%H:%M:%S')}")

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
        self._split_old_path = PROJECT_DIR / "File_Exchange" / "latest_image_old.png"
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

        # ── normal chat append ────────────────────────────────────────────────
        if tag == "robot":
            if stripped.startswith("YOU:"):
                tag = "user"
            elif stripped.startswith(("[OK]", "Changes saved", "Sequence",
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
        # During execute mode show nothing — all detail goes to the log only.
        if self._current_mode == "motion" and tag not in (
            "success", "error", "user", "divider",
        ):
            return
        if self._current_mode == "execute" and tag not in (
            "success", "error", "user", "divider", "warn",
        ):
            return

        self._chat.insert(tk.END, text, tag)
        self._chat.see(tk.END)

    # ─────────────────────────────────────────────────────────────────────────
    # Status bar & lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def _set_status(self, text: str, color: str = C["fg_white"]) -> None:
        self._status.configure(text=text, fg=color)

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