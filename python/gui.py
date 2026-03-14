"""
gui.py — Graphical front-end for the Robot Configuration System.

Run this instead of Main.py.

Requirements:
    pip install Pillow --break-system-packages

The GUI provides:
  - Chat panel  : all terminal output + LLM dialogue
  - Button bar  : menu navigation (replaces numbered prompts)
  - Vision panel: latest annotated camera image, auto-refreshed
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
from typing import Any, List, Optional

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
        self._busy        = False
        self._img_mtime   = 0.0
        self._photo_ref: Optional[Any] = None   # prevent GC of PhotoImage

        # ── window ────────────────────────────────────────────────────────────
        self.root = tk.Tk()
        self.root.title("Robot Configuration System")
        self.root.configure(bg=C["bg_main"])
        self.root.geometry("1400x820")
        self.root.minsize(1000, 640)

        self._build_ui()
        self._redirect_io()

        # ── start polling loops ───────────────────────────────────────────────
        self.root.after(40,   self._poll_output)
        self.root.after(60,   self._poll_input_requests)
        self.root.after(2500, self._poll_image)

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

        # ── main content ──────────────────────────────────────────────────────
        content = tk.Frame(self.root, bg=C["bg_main"])
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=(6, 0))

        self._build_chat_panel(content)
        self._build_vision_panel(content)

        # ── button bar ────────────────────────────────────────────────────────
        bar_outer = tk.Frame(self.root, bg=C["bg_title"], pady=10)
        bar_outer.pack(fill=tk.X, side=tk.BOTTOM)
        self._btn_bar = tk.Frame(bar_outer, bg=C["bg_title"])
        self._btn_bar.pack()

    def _build_chat_panel(self, parent: tk.Frame) -> None:
        left = tk.Frame(parent, bg=C["bg_main"])
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

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

        # Allow text selection and Ctrl+C copy while keeping the widget read-only.
        # We keep state=NORMAL so selection works, but swallow all keypresses
        # except Ctrl+C, Ctrl+A, and navigation keys.
        def _block_edit(event):
            # Allow: Ctrl+C, Ctrl+A, arrow keys, home/end, page up/down
            if event.state & 0x4:   # Ctrl held
                if event.keysym.lower() in ("c", "a"):
                    return          # let copy / select-all through
            if event.keysym in ("Up", "Down", "Left", "Right",
                                 "Home", "End", "Prior", "Next"):
                return
            return "break"          # swallow everything else

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
            state=tk.NORMAL,
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

    def _build_vision_panel(self, parent: tk.Frame) -> None:
        right = tk.Frame(parent, bg=C["bg_main"], width=430)
        right.pack(side=tk.RIGHT, fill=tk.BOTH)
        right.pack_propagate(False)

        tk.Label(
            right, text="VISION",
            bg=C["bg_main"], fg=C["fg_muted"],
            font=(FONT, 8, "bold"),
        ).pack(anchor=tk.W, pady=(2, 3))

        # image frame
        img_frame = tk.Frame(right, bg=C["bg_title"], bd=0)
        img_frame.pack(fill=tk.BOTH, expand=True)

        self._img_lbl = tk.Label(
            img_frame, bg=C["bg_title"],
            text="No image yet.\nRun a vision scan to populate.",
            fg=C["fg_muted"], font=(FONT, 11),
        )
        self._img_lbl.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # bottom strip: timestamp + refresh button
        strip = tk.Frame(right, bg=C["bg_main"], pady=4)
        strip.pack(fill=tk.X)

        self._img_ts = tk.Label(
            strip, text="", bg=C["bg_main"], fg=C["fg_muted"],
            font=(FONT, 8),
        )
        self._img_ts.pack(side=tk.LEFT)

        tk.Button(
            strip, text="Refresh",
            bg=C["bg_btn"], fg=C["fg_white"],
            font=(FONT, 9), relief=tk.FLAT, bd=0,
            padx=10, pady=5,
            activebackground="#5d6269",
            command=self._load_image,
        ).pack(side=tk.RIGHT)

    # ─────────────────────────────────────────────────────────────────────────
    # IO redirection & patching
    # ─────────────────────────────────────────────────────────────────────────

    def _redirect_io(self) -> None:
        self._real_input = builtins.input
        sys.stdout       = _GUIStream(self._out_q)
        builtins.input   = self._gui_input

    def _gui_input(self, prompt: str = "") -> str:
        """Blocking replacement for input() — waits for the GUI."""
        self._in_req_q.put(("text", str(prompt)))
        return self._in_resp_q.get()       # blocks worker thread

    def _patch_pick_from_list(self) -> None:
        """
        Replace _pick_from_list in API_Main so menu options become GUI buttons.
        Called once from the worker thread before running any session.
        """
        try:
            import Communication_Module.API_Main as api  # type: ignore
            gui = self

            def _patched(prompt: str, options: List[str]) -> int:
                sys.stdout.write(f"\n{prompt}\n")
                for i, o in enumerate(options, 1):
                    sys.stdout.write(f"  [{i}] {o}\n")
                gui._in_req_q.put(("menu", prompt, options))
                raw = gui._in_resp_q.get()
                return int(raw) - 1

            api._pick_from_list = _patched
        except Exception as exc:
            print(f"[GUI] Could not patch API_Main: {exc}")

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
        items = [
            ("Reconfigure",     C["bg_accent"],  lambda: self._run("reconfig")),
            ("Plan Sequence",   "#4752c4",        lambda: self._run("motion")),
            ("Execute",          C["bg_green"],   lambda: self._run("execute")),
            ("Exit",             C["bg_red"],     self._quit),
        ]
        for text, color, cmd in items:
            self._btn(self._btn_bar, text, color, cmd).pack(side=tk.LEFT, padx=5)

    def _show_option_buttons(self, options: List[str]) -> None:
        self._clear_bar()
        self._set_input(False)
        palette = ["#5865f2", "#4752c4", "#3b4294", "#2e3679", C["bg_btn"]]
        for i, opt in enumerate(options, 1):
            color = palette[min(i - 1, len(palette) - 1)]
            self._btn(
                self._btn_bar,
                f"[{i}]  {opt}", color,
                lambda n=str(i): self._respond(n),
            ).pack(side=tk.LEFT, padx=4)

    def _show_text_input(self) -> None:
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
        self._set_status("Running...", C["fg_muted"])
        self._clear_bar()
        threading.Thread(target=self._worker, args=(mode,), daemon=True).start()

    def _worker(self, mode: str) -> None:
        # Re-apply redirect here — some imports (rospy, etc.) can reset sys.stdout.
        sys.stdout = _GUIStream(self._out_q)
        sys.stderr = _GUIStream(self._out_q)
        try:
            # Execute must run in its own subprocess: rospy.init_node() registers
            # UNIX signal handlers which Python only permits in the main thread.
            if mode == "execute":
                self._run_execute_subprocess()
                return

            self._patch_pick_from_list()

            import Communication_Module.API_Main as api  # type: ignore

            client = None
            try:
                from openai import OpenAI
                client = OpenAI()
            except Exception as exc:
                from Vision_Module.config import USE_PDDL_PLANNER  # type: ignore
                if mode == "motion" and USE_PDDL_PLANNER:
                    pass   # PDDL path — no LLM needed
                else:
                    print(f"\n[ERR] Could not create OpenAI client: {exc}\n")
                    return

            api.run_session(client, mode)

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
                self._out_q.put(("print", f"\n[ERR] Execute process exited with code {proc.returncode}\n"))
        except Exception as exc:
            self._out_q.put(("print", f"\n[ERR] Could not start execute subprocess: {exc}\n"))

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
                    self._load_image()
                    self._show_main_menu()
        except queue.Empty:
            pass
        self.root.after(40, self._poll_output)

    def _poll_input_requests(self) -> None:
        try:
            while True:
                req = self._in_req_q.get_nowait()
                kind = req[0]
                if kind == "menu":
                    _, _prompt, options = req
                    self._set_status("Waiting...", C["fg_warn"])
                    self._show_option_buttons(options)
                elif kind == "text":
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
                    self._load_image()
        except Exception:
            pass
        self.root.after(2500, self._poll_image)

    # ─────────────────────────────────────────────────────────────────────────
    # Vision image
    # ─────────────────────────────────────────────────────────────────────────

    def _load_image(self) -> None:
        if not LATEST_IMAGE_PATH.exists():
            return
        try:
            w = max(self._img_lbl.winfo_width(),  400)
            h = max(self._img_lbl.winfo_height(), 300)

            if HAS_PIL:
                img = Image.open(LATEST_IMAGE_PATH)
                img.thumbnail((w - 4, h - 4), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
            else:
                photo = tk.PhotoImage(file=str(LATEST_IMAGE_PATH))
                # rough integer downscale without PIL
                scale = min(w / photo.width(), h / photo.height())
                if scale < 1:
                    step = max(1, int(1 / scale))
                    photo = photo.subsample(step, step)

            self._photo_ref = photo    # keep ref so GC doesn't collect it
            self._img_lbl.configure(image=photo, text="")

            ts = datetime.fromtimestamp(LATEST_IMAGE_PATH.stat().st_mtime)
            self._img_ts.configure(text=f"Updated {ts.strftime('%H:%M:%S')}")

        except Exception as exc:
            self._img_lbl.configure(text=f"Image error:\n{exc}", image="")

    # ─────────────────────────────────────────────────────────────────────────
    # Chat display helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _append(self, text: str, tag: str = "robot") -> None:
        if tag == "robot":
            # Auto-detect semantic tags from content
            stripped = text.strip()
            if stripped.startswith("YOU:"):
                tag = "user"
            elif stripped.startswith(("[OK]", "Changes saved", "Sequence", "State archived", "configuration.json")):
                tag = "success"
            elif stripped.startswith(("[ERR]", "ERROR", "Failed", "Warning", "Could not")):
                tag = "error"
            elif stripped.startswith("--") or stripped.startswith("=="):
                tag = "divider"
            elif stripped.startswith(("[GUI]", "INFO", "[Step")):
                tag = "system"

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