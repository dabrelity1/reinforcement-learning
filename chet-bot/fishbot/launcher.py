"""Launcher GUI for FishBot + Minigame.

Provides a small tkinter window to configure and launch both the fullscreen
minigame and the bot in line-mode with chosen physics parameters.

User Flow:
 1. Adjust options (fullscreen forced for minigame, rect accel/drag, seed, debug, dry-run)
 2. Click 'Launch'. Launcher spawns two subprocesses:
      - Minigame process
      - Bot process (after short delay) with line detection mode
 3. Option to terminate both via 'Stop' button.

Note: This is a convenience layer; all options still available via CLI.
"""
from __future__ import annotations

import subprocess
import sys
import time
import tkinter as tk
from tkinter import ttk, messagebox
import threading

PY_CMD = sys.executable or "python"

class Launcher:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("FishBot Launcher")
        root.geometry("420x420")
        root.resizable(False, False)
        self.proc_game = None
        self.proc_bot = None
        self._build()

    def _build(self):
        pad = {"padx": 6, "pady": 4}
        frm = ttk.Frame(self.root)
        frm.pack(fill="both", expand=True, padx=10, pady=10)

        self.seed_var = tk.StringVar(value="")
        self.accel_var = tk.StringVar(value="800")
        self.drag_var = tk.StringVar(value="0.97")
        self.rectw_var = tk.StringVar(value="80")
        self.linem_var = tk.StringVar(value="1.25")
        self.debug_var = tk.BooleanVar(value=True)
        self.dry_var = tk.BooleanVar(value=True)
        self.delay_var = tk.StringVar(value="0.8")
        self.full_var = tk.BooleanVar(value=True)
        self.auto_var = tk.BooleanVar(value=True)
        self.headless_var = tk.BooleanVar(value=True)

        ttk.Label(frm, text="Seed (blank=random)").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.seed_var, width=12).grid(row=0, column=1, **pad)

        ttk.Label(frm, text="Rect Accel").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.accel_var, width=12).grid(row=1, column=1, **pad)

        ttk.Label(frm, text="Rect Drag").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.drag_var, width=12).grid(row=2, column=1, **pad)

        ttk.Label(frm, text="Rect Width").grid(row=3, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.rectw_var, width=12).grid(row=3, column=1, **pad)

        ttk.Label(frm, text="Line Speed Mult").grid(row=4, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.linem_var, width=12).grid(row=4, column=1, **pad)

        ttk.Label(frm, text="Bot Start Delay (s)").grid(row=5, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.delay_var, width=12).grid(row=5, column=1, **pad)

        ttk.Checkbutton(frm, text="Fullscreen", variable=self.full_var).grid(row=6, column=0, sticky="w", **pad)
        ttk.Checkbutton(frm, text="Debug Overlay", variable=self.debug_var).grid(row=6, column=1, sticky="w", **pad)
        ttk.Checkbutton(frm, text="Dry Run (no input)", variable=self.dry_var).grid(row=7, column=0, sticky="w", **pad)
        ttk.Checkbutton(frm, text="Auto Minigame ROI", variable=self.auto_var).grid(row=7, column=1, sticky="w", **pad)
        ttk.Checkbutton(frm, text="Headless Debug", variable=self.headless_var).grid(row=8, column=0, sticky="w", **pad)

        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(frm, textvariable=self.status_var, foreground="#00aa88").grid(row=9, column=0, columnspan=2, sticky="w", **pad)

        btn_frame = ttk.Frame(frm)
        btn_frame.grid(row=10, column=0, columnspan=2, pady=12)
        ttk.Button(btn_frame, text="Launch", command=self.launch).grid(row=0, column=0, padx=8)
        ttk.Button(btn_frame, text="Stop", command=self.stop).grid(row=0, column=1, padx=8)
        ttk.Button(btn_frame, text="Quit", command=self.quit_all).grid(row=0, column=2, padx=8)

        note = ("Workflow: Launch spawns minigame then bot (line-mode). "
                "Adjust Rect Width & Line Speed for desired difficulty.")
        ttk.Label(frm, text=note, wraplength=380, foreground="#666").grid(row=11, column=0, columnspan=2, **pad)

    def launch(self):
        if self.proc_game or self.proc_bot:
            messagebox.showwarning("Running", "Processes already running.")
            return
        try:
            accel = float(self.accel_var.get())
            drag = float(self.drag_var.get())
            delay = float(self.delay_var.get())
            rect_w = int(self.rectw_var.get())
            line_mult = float(self.linem_var.get())
        except ValueError:
            messagebox.showerror("Invalid", "Acceleration, drag, and delay must be numeric.")
            return
        seed = self.seed_var.get().strip()
        seed_arg = ["--seed", seed] if seed else []
        fullscreen_arg = ["--fullscreen"] if self.full_var.get() else []
        game_cmd = [PY_CMD, "-m", "fishbot.minigame", *seed_arg, *fullscreen_arg,
                    "--rect-accel", str(accel), "--rect-drag", str(drag),
                    "--rect-width", str(rect_w), "--line-speed-mult", str(line_mult)]
        self.status_var.set("Starting minigame ...")
        self.proc_game = subprocess.Popen(game_cmd)
        def start_bot_later():
            time.sleep(delay)
            if self.proc_game and self.proc_game.poll() is None:
                bot_cmd = [PY_CMD, "-m", "fishbot.main", "--line-mode"]
                if self.auto_var.get():
                    bot_cmd.append("--auto-minigame")
                if self.debug_var.get() and not self.headless_var.get():
                    bot_cmd.append("--debug")
                if self.headless_var.get():
                    bot_cmd.append("--headless-debug")
                if self.dry_var.get():
                    bot_cmd.append("--dry-run")
                self.status_var.set("Starting bot ...")
                self.proc_bot = subprocess.Popen(bot_cmd)
                self.status_var.set("Running")
        threading.Thread(target=start_bot_later, daemon=True).start()

    def stop(self):
        for proc in (self.proc_bot, self.proc_game):
            if proc and proc.poll() is None:
                proc.terminate()
        self.proc_bot = None
        self.proc_game = None
        self.status_var.set("Stopped")

    def quit_all(self):
        self.stop()
        self.root.destroy()


def main():
    root = tk.Tk()
    Launcher(root)
    root.mainloop()


if __name__ == "__main__":
    main()
