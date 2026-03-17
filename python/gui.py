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
  Configure: (text-input row enabled) + Cancel button only
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
        self._busy              = False
        self._img_mtime         = 0.0
        self._photo_ref: Optional[Any] = None   # prevent GC of PhotoImage
        self._image_ready       = False          # True only after a new image is taken this session

        # scene-routing state
        self._awaiting_scene_desc  = False   # next ASSISTANT: block → scene panel
        self._config_from_memory   = False   # show placeholder instead of image
        self._in_configure_mode    = False   # only Cancel in button bar
        self._first_menu_shown     = False   # greeting shown exactly once
        self._current_mode: str    = ""      # "reconfig" | "motion" | "execute"
        self._reconfig_sub: str    = ""      # pre-selected reconfig sub-option

        # accumulator for multi-line scene text arriving in fragments
        self._scene_buffer: str = ""
        self._scene_capture = False          # currently capturing scene text

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
        Replace _pick_from_list in API_Main.
        In configure mode the numbered options are printed to the chat as plain
        text and the user types the number — no option buttons are shown.
        Typing "done" / "cancel" at a menu prompt cancels the session cleanly.
        """
        try:
            import Communication_Module.API_Main as api  # type: ignore
            gui = self

            def _patched(prompt: str, options: List[str]) -> int:
                sys.stdout.write(f"\n{prompt}\n")
                for i, o in enumerate(options, 1):
                    sys.stdout.write(f"  [{i}] {o}\n")
                # Always use text input — no buttons — during configure mode
                gui._in_req_q.put(("text", prompt))
                raw = gui._in_resp_q.get().strip().lower()
                # Allow the Cancel button (which sends "done") to exit cleanly
                if raw in ("done", "cancel", "exit", "quit"):
                    raise SystemExit(0)
                return int(raw) - 1

            api._pick_from_list = _patched
        except Exception as exc:
            print(f"[GUI] Could not patch API_Main: {exc}")

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

    def _flush_scene_buffer(self) -> None:
        """Split _scene_buffer into scene description (→ panel) and trailing
        follow-up question (→ chat).  The LLM always ends its first response
        with a question asking what the user wants to do; we peel that off so
        it appears in the conversation panel where the user can see it.

        Once a successful split is made we stop capturing so that the extra
        trailing write("\n") that Python's print() emits does not cause a
        second flush of the same content."""
        full = self._scene_buffer.strip()
        if not full:
            return

        # Try to split on the last blank line
        parts = full.rsplit("\n\n", 1)
        if len(parts) == 2:
            scene_part, question_part = parts
            q = question_part.strip()
            # Treat it as a follow-up if it contains "?" or is short prose
            if q and ("?" in q or len(q) < 350):
                self._set_scene_content(scene_part.strip())
                # Show the question in the chat panel and log
                self._chat.insert(tk.END, "\n" + q + "\n", "robot")
                self._chat.see(tk.END)
                self._log_append("\n" + q + "\n", "robot")
                # Stop capturing — prevents double-print from print()'s extra \n
                self._scene_capture = False
                return

        # No clear split — put everything in the scene panel and stop
        self._set_scene_content(full)
        self._scene_capture = False

    def _refresh_scene_from_config(self) -> None:
        """After a reconfig session, ask the LLM for a fresh natural-language
        scene summary and display it in the scene panel."""
        try:
            import json
            import Communication_Module.API_Main as api  # type: ignore
            if not api.CONFIGURATION_PATH.exists():
                return
            state = json.loads(api.CONFIGURATION_PATH.read_text(encoding="utf-8"))
            scene = api.slim_scene(state)
            self._set_scene_placeholder("Updating scene description…")
            threading.Thread(
                target=self._llm_scene_summary,
                args=(scene,),
                daemon=True,
            ).start()
        except Exception as exc:
            self._set_scene_placeholder("Could not reload scene:\n" + str(exc))

    def _llm_scene_summary(self, scene: dict) -> None:
        """Background thread: call the LLM to summarise the updated scene."""
        try:
            import json
            from openai import OpenAI  # type: ignore
            client = OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4.1",
                max_tokens=400,
                temperature=0.1,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a concise robot workspace describer. "
                            "Given a scene JSON, write a short natural-language "
                            "paragraph that covers: the operation mode, every "
                            "receptacle (name, role, what parts it holds with "
                            "their colour/size), and any kit recipe or priority "
                            "rules.  Be complete but brief — no bullet lists, "
                            "no headers, plain prose only."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(scene, indent=2, ensure_ascii=False),
                    },
                ],
            )
            summary = (resp.choices[0].message.content or "").strip()
            self.root.after(0, lambda: self._set_scene_content(summary))
        except Exception as exc:
            self.root.after(
                0,
                lambda: self._set_scene_placeholder(
                    "Could not generate summary:\n" + str(exc)
                ),
            )

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
        """Show only the Cancel button (used during configure mode)."""
        self._clear_bar()
        self._btn(
            self._btn_bar, "Cancel", C["bg_red"],
            self._cancel_configure,
        ).pack(side=tk.LEFT, padx=5)

    def _show_configure_options(self) -> None:
        """Replace the main button bar with three configure sub-options + Cancel."""
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
            ("Cancel",        C["bg_red"],    self._show_main_menu),
        ]
        for text, color, cmd in items:
            self._btn(self._btn_bar, text, color, cmd).pack(side=tk.LEFT, padx=5)

    def _run_reconfig_sub(self, sub: str) -> None:
        """Launch the reconfig worker with a pre-selected sub-option so the
        backend's select_reconfig_source() is bypassed entirely."""
        self._reconfig_sub = sub
        self._config_from_memory = False
        self._run("reconfig")

    def _cancel_configure(self) -> None:
        """Send 'done' to the worker, as if the user typed it."""
        self._append("YOU: done\n", "user")
        self._in_resp_q.put("done")

    def _show_text_input(self) -> None:
        if self._in_configure_mode:
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
        self._awaiting_scene_desc = False
        self._scene_buffer = ""
        self._scene_capture = False

        self._set_status("Running...", C["fg_muted"])
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
                    import Communication_Module.API_Main as api  # type: ignore
                    if api.CONFIGURATION_PATH.exists():
                        import json as _json
                        config = _json.loads(api.CONFIGURATION_PATH.read_text(encoding="utf-8"))
                        from Configuration_Module.Update_Scene import run_post_execution_rescan  # type: ignore
                        run_post_execution_rescan(config)
                except Exception as exc:
                    print(f"  ⚠  Post-execution rescan failed: {exc}")
                return

            self._patch_pick_from_list()

            import Communication_Module.API_Main as api  # type: ignore

            # If a reconfig sub-option was pre-selected via the GUI buttons,
            # bypass the interactive select_reconfig_source() call entirely.
            if mode == "reconfig" and self._reconfig_sub:
                _pre = self._reconfig_sub
                api.select_reconfig_source = lambda: _pre

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
                    self._load_image()
        except Exception:
            pass
        self.root.after(2500, self._poll_image)

    # ─────────────────────────────────────────────────────────────────────────
    # Vision image
    # ─────────────────────────────────────────────────────────────────────────

    def _load_image(self, w: int = 0, h: int = 0) -> None:
        # Never show a leftover image from a previous session — wait until the
        # vision module produces a fresh one (or the user loads from memory).
        if not self._image_ready and not self._config_from_memory:
            return
        # Memory-only config: show placeholder text instead of image
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
            # Use caller-supplied dimensions (from resize event) when available;
            # otherwise query the frame — never the label, whose size is locked
            # to whatever image is already displayed on it.
            if not w or not h:
                w = self._img_frame.winfo_width()
                h = self._img_frame.winfo_height()
            w = max(w, 40)
            h = max(h, 40)

            # Skip if the rendered size hasn't changed — prevents the feedback
            # loop where placing a new image on the label causes the frame to
            # emit another <Configure> event, triggering an infinite cascade.
            if (w, h) == self._last_render_size:
                return
            self._last_render_size = (w, h)

            if HAS_PIL:
                img = Image.open(LATEST_IMAGE_PATH)
                img.thumbnail((w - 4, h - 4), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
            else:
                photo = tk.PhotoImage(file=str(LATEST_IMAGE_PATH))
                scale = min(w / photo.width(), h / photo.height())
                if scale < 1:
                    step = max(1, int(1 / scale))
                    photo = photo.subsample(step, step)

            self._photo_ref = photo
            self._img_lbl.configure(image=photo, text="")

            ts = datetime.fromtimestamp(LATEST_IMAGE_PATH.stat().st_mtime)
            self._img_ts.configure(text=f"Updated {ts.strftime('%H:%M:%S')}")

        except Exception as exc:
            self._img_lbl.configure(text=f"Image error:\n{exc}", image="")

    # ─────────────────────────────────────────────────────────────────────────
    # Chat display helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _append(self, text: str, tag: str = "robot") -> None:
        stripped = text.strip()

        # ── detect memory-only vs fresh-vision load ───────────────────────────
        if "Loaded fresh scene from vision" in stripped:
            self._config_from_memory = False
            self._image_ready = True

        # ── detect start of new mode (scene description is next) ─────────────
        if "── Mode:" in stripped:
            self._awaiting_scene_desc = True
            self._scene_buffer = ""
            self._scene_capture = False
            # Clear the scene panel until the new description arrives
            self._set_scene_placeholder("Loading scene description…")

        # ── route scene description to the scene panel ────────────────────────
        # The backend prints:  "\nASSISTANT:\n{scene_text}\n"
        # This arrives as a single write() call most of the time, but we
        # handle fragmented delivery robustly via the capture buffer.

        if self._awaiting_scene_desc:
            if not self._scene_capture and "ASSISTANT:" in text:
                # First ASSISTANT block after mode start.
                # Capture it, then split scene description from trailing question.
                self._scene_capture = True
                after = text.split("ASSISTANT:", 1)[1]
                self._scene_buffer = after
                self._awaiting_scene_desc = False
                self._flush_scene_buffer()
                return
        elif self._scene_capture:
            # Stop capturing when the conversation moves on.
            if "ASSISTANT:" in text or stripped.startswith("YOU:")                     or stripped.startswith("[Sequence") or stripped.startswith("[Changes"):
                self._scene_capture = False
                # fall through — append this text to chat below
            else:
                # Still within the first ASSISTANT block (fragmented delivery)
                self._scene_buffer += text
                self._flush_scene_buffer()
                return

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
            "success", "error", "user", "divider",
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