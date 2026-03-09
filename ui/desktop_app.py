"""
desktop_app.py
--------------
Retro arcade-styled Tkinter GUI for Muud — Music Intelligence System.
Browse audio → Analyze → View genre/emotion → Get recommendations.
Live mic mode: real-time spectrogram + rolling genre/emotion predictions.

Visual theme: dark navy background, neon accents (cyan / magenta / green),
pixel-style fonts, 3D raised buttons with hover glow.
"""

import os
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
from collections import deque

import sounddevice as sd
import scipy.io.wavfile as wav_io
import numpy as np
import librosa

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from engine.fusion import emotion_similarity, genre_similarity


# ════════════════════════════════════════════════════════════════
#  COLOUR PALETTE
# ════════════════════════════════════════════════════════════════
BG_DARK      = "#0b0e1a"       # deep navy
BG_PANEL     = "#111528"       # slightly lighter panel
BG_INPUT     = "#181d35"       # input / text area background
BORDER       = "#1e2444"       # subtle panel borders

NEON_CYAN    = "#00e5ff"
NEON_MAGENTA = "#ff2eaa"
NEON_GREEN   = "#39ff14"
NEON_YELLOW  = "#ffe600"
TEXT_PRIMARY  = "#e0e6f0"      # off-white readable text
TEXT_DIM      = "#6b7394"      # muted secondary text

# ════════════════════════════════════════════════════════════════
#  FONT STACK
# ════════════════════════════════════════════════════════════════
FONT_TITLE    = ("Terminal", 24, "bold")
FONT_SUBTITLE = ("Terminal", 10)
FONT_BODY     = ("Consolas", 10)
FONT_BODY_B   = ("Consolas", 10, "bold")
FONT_BTN      = ("Consolas", 11, "bold")
FONT_STATUS   = ("Consolas", 9)
FONT_SMALL    = ("Consolas", 9)


# ════════════════════════════════════════════════════════════════
#  NEON BUTTON — custom canvas widget
# ════════════════════════════════════════════════════════════════
class NeonButton(tk.Canvas):
    """
    A flat canvas-drawn button with:
      • 3-D raised border illusion (shadow rectangle)
      • Neon glow outline on hover
      • Custom colour per button
    """

    def __init__(self, parent, text, command, color=NEON_CYAN,
                 width=170, height=40, **kw):
        super().__init__(parent, width=width, height=height,
                         bg=BG_DARK, highlightthickness=0, **kw)
        self.command = command
        self.color = color
        self.text = text
        self.w = width
        self.h = height
        self._enabled = True

        self._draw_normal()

        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)

    # ── Drawing states ──────────────────────────────────────────

    def _draw_normal(self):
        self.delete("all")
        # Shadow (3-D raised effect)
        self.create_rectangle(4, 4, self.w, self.h,
                              fill="#050810", outline="")
        # Body
        self.create_rectangle(0, 0, self.w - 4, self.h - 4,
                              fill=BG_PANEL, outline=self.color, width=1)
        self.create_text(
            (self.w - 4) // 2, (self.h - 4) // 2,
            text=self.text, fill=self.color if self._enabled else TEXT_DIM,
            font=FONT_BTN,
        )

    def _draw_hover(self):
        self.delete("all")
        self.create_rectangle(4, 4, self.w, self.h,
                              fill="#050810", outline="")
        self.create_rectangle(0, 0, self.w - 4, self.h - 4,
                              fill="#1a1f3a", outline=self.color, width=2)
        self.create_text(
            (self.w - 4) // 2, (self.h - 4) // 2,
            text=self.text, fill=self.color, font=FONT_BTN,
        )

    def _draw_pressed(self):
        self.delete("all")
        # No shadow — looks "pushed in"
        self.create_rectangle(2, 2, self.w - 2, self.h - 2,
                              fill="#0d1020", outline=self.color, width=2)
        self.create_text(
            self.w // 2, self.h // 2,
            text=self.text, fill=NEON_GREEN, font=FONT_BTN,
        )

    # ── Events ──────────────────────────────────────────────────

    def _on_enter(self, _):
        if self._enabled:
            self._draw_hover()

    def _on_leave(self, _):
        self._draw_normal()

    def _on_press(self, _):
        if self._enabled:
            self._draw_pressed()

    def _on_release(self, _):
        if self._enabled:
            self._draw_normal()
            self.command()

    # ── Enable / Disable ────────────────────────────────────────

    def set_enabled(self, flag):
        self._enabled = flag
        self._draw_normal()


# ════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ════════════════════════════════════════════════════════════════
class MuudApp:
    """Retro arcade-themed desktop application window."""

    def __init__(self, recommender):
        self.recommender = recommender
        self.result = None
        self._selected_path = None
        self._temp_recording = None   # path to current temp WAV (if any)

        # ── Live mic state ──────────────────────────────────────
        self._live_active = False
        self._live_stream = None            # sd.InputStream
        self._live_buffer = deque(maxlen=22050 * 30)  # up to 30 s ring buffer
        self._live_inference_job = None     # root.after id
        self._live_spec_job = None          # root.after id for spectrogram refresh
        self._live_sr = 22050

        # ── Root window ─────────────────────────────────────────
        self.root = tk.Tk()
        self.root.title("MUUD — Music Intelligence")
        self.root.geometry("1350x860")
        self.root.minsize(1200, 750)
        self.root.configure(bg=BG_DARK)

        self._build_ui()

    # ════════════════════════════════════════════════════════════
    #  BUILD UI
    # ════════════════════════════════════════════════════════════

    def _build_ui(self):
        root = self.root
        # ── ttk Treeview theme ──────────────────────────────
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure(
            "Neon.Treeview",
            background=BG_INPUT,
            foreground=TEXT_PRIMARY,
            fieldbackground=BG_INPUT,
            font=("Consolas", 9),
            rowheight=28,
            borderwidth=0,
        )
        style.map(
            "Neon.Treeview",
            background=[("selected", "#1a2a4a")],
            foreground=[("selected", NEON_CYAN)],
        )
        style.configure(
            "Neon.Treeview.Heading",
            background=BG_PANEL,
            foreground=NEON_CYAN,
            font=("Consolas", 9, "bold"),
            borderwidth=1,
            relief="flat",
        )
        style.map(
            "Neon.Treeview.Heading",
            background=[("active", "#1a1f3a")],
        )
        # ── Top scanline decoration ─────────────────────────────
        scanline = tk.Canvas(root, height=3, bg=BG_DARK, highlightthickness=0)
        scanline.pack(fill="x")
        scanline.create_line(0, 1, 2000, 1, fill=NEON_CYAN, width=1)

        # ── Title Block ─────────────────────────────────────────
        title_frame = tk.Frame(root, bg=BG_DARK)
        title_frame.pack(pady=(14, 0))

        self._title_label = tk.Label(
            title_frame, text="\u25c6  M U U D  \u25c6",
            font=FONT_TITLE, fg=NEON_CYAN, bg=BG_DARK,
        )
        self._title_label.pack()

        sub_frame = tk.Frame(title_frame, bg=BG_DARK)
        sub_frame.pack(pady=(2, 0))
        tk.Label(
            sub_frame,
            text="HYBRID  SOFT  COMPUTING  MUSIC  INTELLIGENCE",
            font=FONT_SUBTITLE, fg=TEXT_DIM, bg=BG_DARK,
        ).pack(side="left")
        tk.Label(
            sub_frame, text="  v2.0",
            font=("Consolas", 8), fg="#3a4060", bg=BG_DARK,
        ).pack(side="left", padx=(6, 0))

        # Dashed glow line under title
        glow = tk.Canvas(root, height=5, bg=BG_DARK, highlightthickness=0)
        glow.pack(fill="x", padx=60, pady=(6, 0))
        glow.create_line(0, 2, 2000, 2, fill=NEON_MAGENTA, width=1, dash=(6, 4))

        # Start title pulse animation
        self._title_colors = [NEON_CYAN, "#00ccdd", "#00b3bb", "#009999",
                              "#00b3bb", "#00ccdd"]
        self._title_color_idx = 0
        self._pulse_title()

        # ── File Selection Panel ────────────────────────────────
        file_panel = self._make_panel(root, " \u25b8 AUDIO INPUT ")
        file_panel.pack(fill="x", padx=30, pady=(14, 0))

        file_inner = tk.Frame(file_panel, bg=BG_PANEL)
        file_inner.pack(fill="x", padx=12, pady=10)

        self.file_var = tk.StringVar(value="No file selected \u2026")

        file_label = tk.Label(
            file_inner, textvariable=self.file_var,
            font=FONT_BODY, fg=NEON_YELLOW, bg=BG_INPUT,
            anchor="w", padx=10, pady=6, relief="flat",
        )
        file_label.pack(side="left", fill="x", expand=True, padx=(0, 10))

        self.browse_btn = NeonButton(
            file_inner, text="BROWSE", command=self._browse_file,
            color=NEON_MAGENTA, width=130, height=34,
        )
        self.browse_btn.pack(side="right")

        # ── Action Buttons ──────────────────────────────────────
        btn_frame = tk.Frame(root, bg=BG_DARK)
        btn_frame.pack(pady=14)

        self.analyze_btn = NeonButton(
            btn_frame, text="\u26a1 ANALYZE", command=self._run_analyze,
            color=NEON_CYAN, width=190, height=42,
        )
        self.analyze_btn.pack(side="left", padx=12)
        self.analyze_btn.set_enabled(False)

        self.recommend_btn = NeonButton(
            btn_frame, text="\u266b RECOMMEND", command=self._run_recommend,
            color=NEON_GREEN, width=190, height=42,
        )
        self.recommend_btn.pack(side="left", padx=12)
        self.recommend_btn.set_enabled(False)

        self.explain_btn = NeonButton(
            btn_frame, text="? EXPLAIN", command=self._toggle_explain,
            color=NEON_YELLOW, width=190, height=42,
        )
        self.explain_btn.pack(side="left", padx=12)
        self.explain_btn.set_enabled(False)

        self.record_btn = NeonButton(
            btn_frame, text="\u23fa REC 5s", command=self._run_record,
            color=NEON_MAGENTA, width=190, height=42,
        )
        self.record_btn.pack(side="left", padx=12)

        self.live_btn = NeonButton(
            btn_frame, text="\U0001f3a4 LIVE MIC", command=self._toggle_live_mic,
            color="#ff6600", width=190, height=42,
        )
        self.live_btn.pack(side="left", padx=12)

        self._blink_job = None          # after-id for recording blink

        # ── Status Bar ──────────────────────────────────────────
        self.status_var = tk.StringVar(value="[ READY ]")
        tk.Label(
            root, textvariable=self.status_var,
            font=FONT_STATUS, fg=NEON_GREEN, bg=BG_DARK, anchor="w",
        ).pack(padx=34, anchor="w")

        # ── Content Area (results + VA plot side by side) ───────
        content_frame = tk.Frame(root, bg=BG_DARK)
        content_frame.pack(fill="both", expand=True, padx=30, pady=(6, 18))

        # Left — Results / Recommendations
        results_panel = self._make_panel(content_frame, " \u25b8 RESULTS ")
        results_panel.pack(side="left", fill="both", expand=True, padx=(0, 6))

        results_inner = tk.Frame(results_panel, bg=BG_PANEL)
        results_inner.pack(fill="both", expand=True, padx=8, pady=8)

        # ── Sub-frame A: Analysis text ───────────────────────
        self._analysis_frame = tk.Frame(results_inner, bg=BG_PANEL)

        self.results_text = tk.Text(
            self._analysis_frame, wrap="word",
            font=FONT_BODY, fg=TEXT_PRIMARY, bg=BG_INPUT,
            insertbackground=NEON_CYAN,
            selectbackground="#1a3050",
            relief="flat", padx=12, pady=10,
            state="disabled", cursor="arrow",
        )

        a_scroll = tk.Scrollbar(
            self._analysis_frame, command=self.results_text.yview,
            bg=BG_PANEL, troughcolor=BG_INPUT,
            activebackground=NEON_CYAN, width=10,
        )
        self.results_text.configure(yscrollcommand=a_scroll.set)

        a_scroll.pack(side="right", fill="y")
        self.results_text.pack(side="left", fill="both", expand=True)

        self.results_text.tag_configure("heading",   foreground=NEON_CYAN,    font=FONT_BODY_B)
        self.results_text.tag_configure("highlight",  foreground=NEON_MAGENTA)
        self.results_text.tag_configure("value",      foreground=NEON_GREEN)
        self.results_text.tag_configure("dim",        foreground=TEXT_DIM)
        self.results_text.tag_configure("bar",        foreground=NEON_CYAN)

        self._analysis_frame.pack(fill="both", expand=True)

        # ── Sub-frame B: Recommendation table ───────────────
        self._recommend_frame = tk.Frame(results_inner, bg=BG_PANEL)
        self._build_recommend_table()

        # Right — Valence-Arousal plot / Live spectrogram (stacked)
        right_frame = tk.Frame(content_frame, bg=BG_DARK, width=380)
        right_frame.pack(side="right", fill="both", padx=(6, 0))
        right_frame.pack_propagate(False)

        va_panel = self._make_panel(right_frame, " \u25b8 V\u2013A SPACE ")
        va_panel.pack(fill="both", expand=True)

        va_inner = tk.Frame(va_panel, bg=BG_PANEL)
        va_inner.pack(fill="both", expand=True, padx=6, pady=6)
        self._build_va_plot(va_inner)

        # Live spectrogram panel (hidden by default — shown when LIVE MIC active)
        self._live_panel = self._make_panel(right_frame, " \u25b8 LIVE SPECTROGRAM ")
        self._live_info_var = tk.StringVar(value="")
        self._live_info_label = tk.Label(
            self._live_panel, textvariable=self._live_info_var,
            font=FONT_SMALL, fg=NEON_GREEN, bg=BG_PANEL, anchor="w", padx=8,
        )
        self._live_info_label.pack(fill="x", pady=(4, 0))
        live_inner = tk.Frame(self._live_panel, bg=BG_PANEL)
        live_inner.pack(fill="both", expand=True, padx=6, pady=6)
        self._build_live_spectrogram(live_inner)

        # ── Bottom scanline ─────────────────────────────────────
        bot = tk.Canvas(root, height=3, bg=BG_DARK, highlightthickness=0)
        bot.pack(fill="x", side="bottom")
        bot.create_line(0, 1, 2000, 1, fill=NEON_CYAN, width=1)

    # ── Panel factory ───────────────────────────────────────────

    @staticmethod
    def _make_panel(parent, title_text):
        """Create a dark framed panel with a neon-outlined border."""
        outer = tk.LabelFrame(
            parent, text=title_text,
            font=FONT_SMALL, fg=NEON_CYAN, bg=BG_PANEL,
            bd=1, relief="groove", labelanchor="nw",
            highlightbackground=BORDER,
            highlightcolor=NEON_CYAN,
            highlightthickness=1,
        )
        return outer

    # ════════════════════════════════════════════════════════════
    #  FILE BROWSE
    # ════════════════════════════════════════════════════════════

    def _cleanup_temp(self):
        """Delete any previous temp recording file and drop its cached analysis."""
        if self._temp_recording:
            self.recommender.invalidate_cache(self._temp_recording)
            if os.path.exists(self._temp_recording):
                try:
                    os.unlink(self._temp_recording)
                except OSError:
                    pass
            self._temp_recording = None

    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[
                ("Audio Files", "*.wav *.mp3 *.flac *.ogg"),
                ("All Files", "*.*"),
            ],
        )
        if path:
            self._cleanup_temp()
            self.file_var.set(f"\u266a  {os.path.basename(path)}")
            self._selected_path = path
            self.analyze_btn.set_enabled(True)
            self.recommend_btn.set_enabled(True)
            self.explain_btn.set_enabled(False)
            if self._explain_visible:
                self._explain_outer.pack_forget()
                self._explain_visible = False
            self.status_var.set("[ FILE LOADED \u2014 SELECT ACTION ]")

    # ════════════════════════════════════════════════════════════
    #  ANALYSIS
    # ════════════════════════════════════════════════════════════

    def _run_analyze(self):
        self._run_in_thread(self._do_analyze)

    def _do_analyze(self):
        self.status_var.set("[ ANALYZING \u2026 ]")
        self._disable_buttons()

        try:
            result = self.recommender.analyze(self._selected_path)
            self.result = result
            self._show_analysis(result)
            self.status_var.set("[ ANALYSIS COMPLETE ]")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_var.set("[ ERROR ]")
        finally:
            self._enable_buttons()

    def _show_analysis(self, result):
        self._show_analysis_frame()
        self._clear_results()
        t = self.results_text
        t.config(state="normal")

        t.insert("end", f"  FILE:  ", "dim")
        t.insert("end", f"{result['file']}\n\n", "value")

        # ── Genre ───────────────────────────────────────────────
        g = result["genre"]
        t.insert("end", "  \u250c\u2500\u2500\u2500 GENRE CLASSIFICATION \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510\n", "heading")
        t.insert("end", f"  \u2502  Top Genre:  ", "dim")
        t.insert("end", f"{g['top_genre'].upper()}", "highlight")
        t.insert("end", f"   ({g['confidence']:.1%} confidence)\n", "dim")
        t.insert("end", "  \u2502\n", "heading")

        # ── Top-3 Genre Probabilities ───────────────────────────
        top3 = sorted(g["fuzzy_memberships"].items(), key=lambda x: -x[1])[:3]
        t.insert("end", "  \u2502  Genre Probabilities:\n", "dim")
        for label, score in top3:
            t.insert("end", f"  \u2502    {label}", "highlight")
            t.insert("end", f" \u2014 {score * 100:.1f}%\n", "value")
        t.insert("end", "  \u2502\n", "heading")

        # ── Full Fuzzy Memberships ──────────────────────────────
        t.insert("end", "  \u2502  Fuzzy Memberships:\n", "dim")

        for label, score in sorted(g["fuzzy_memberships"].items(), key=lambda x: -x[1]):
            bar_len = int(score * 25)
            bar = "\u2588" * bar_len + "\u2591" * (25 - bar_len)
            t.insert("end", f"  \u2502  {label:<10s} ", "dim")
            t.insert("end", f"{bar}", "bar")
            t.insert("end", f"  {score:.3f}\n", "value")

        t.insert("end", "  \u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518\n\n", "heading")

        # ── Emotion ─────────────────────────────────────────────
        e = result["emotion"]
        t.insert("end", "  \u250c\u2500\u2500\u2500 EMOTION ANALYSIS \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510\n", "heading")
        t.insert("end", f"  \u2502  Valence:  ", "dim")
        t.insert("end", f"{e['valence']:.2f}", "value")
        t.insert("end", f"     Arousal:  ", "dim")
        t.insert("end", f"{e['arousal']:.2f}\n", "value")
        t.insert("end", f"  \u2502  Mood:     ", "dim")
        t.insert("end", f"{e['mood_label']}\n", "highlight")
        t.insert("end", "  \u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518\n", "heading")

        t.config(state="disabled")

        # Update V-A scatter plot
        self._update_va_plot(e['valence'], e['arousal'], g['confidence'])

    # ════════════════════════════════════════════════════════════
    #  RECOMMENDATION
    # ════════════════════════════════════════════════════════════

    def _run_recommend(self):
        self._run_in_thread(self._do_recommend)

    def _do_recommend(self):
        self.status_var.set("[ ANALYZING & RECOMMENDING \u2026 ]")
        self._disable_buttons()

        try:
            result = self.recommender.recommend(self._selected_path, top_n=5)
            self.result = result
            self._show_recommendations(result)
            self.status_var.set("[ RECOMMENDATIONS READY ]")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_var.set("[ ERROR ]")
        finally:
            self._enable_buttons()

    def _show_recommendations(self, result):
        self._show_recommend_frame()

        # Reset explain panel for fresh results
        if self._explain_visible:
            self._explain_outer.pack_forget()
            self._explain_visible = False
        self.explain_btn.set_enabled(True)

        q = result["query"]
        g = q["genre"]
        e = q["emotion"]

        # Update query summary ribbon
        self._query_label.config(
            text=(
                f"  \u266a {q['file']}   \u00b7   "
                f"Genre: {g['top_genre'].upper()} ({g['confidence']:.0%})   \u00b7   "
                f"Mood: {e['mood_label']}  (V={e['valence']:.1f}  A={e['arousal']:.1f})"
            )
        )

        # Clear previous rows
        for iid in self._rec_tree.get_children():
            self._rec_tree.delete(iid)

        # Insert recommendations
        for i, rec in enumerate(result["recommendations"], 1):
            tag = "top" if i == 1 else "normal"
            self._rec_tree.insert("", "end", values=(
                i,
                rec["title"],
                rec["artist"],
                rec["genre"].upper(),
                f"{rec['valence']:.2f}",
                f"{rec['arousal']:.2f}",
                f"{rec['score']:.4f}",
            ), tags=(tag,))

        # Reset sort state
        self._sort_reverse.clear()

        # Update V-A scatter plot
        self._update_va_plot(e['valence'], e['arousal'], g['confidence'])

    # ════════════════════════════════════════════════════════════
    #  RECOMMENDATION TABLE + UI FRAME HELPERS
    # ════════════════════════════════════════════════════════════

    def _build_recommend_table(self):
        """Build the sortable recommendation Treeview inside _recommend_frame."""
        rf = self._recommend_frame

        # ── Query summary ribbon ────────────────────────────────
        self._query_label = tk.Label(
            rf, text="", font=FONT_SMALL,
            fg=NEON_YELLOW, bg=BG_INPUT, anchor="w",
            padx=10, pady=6,
        )
        self._query_label.pack(fill="x", pady=(0, 6))

        # ── Treeview wrapper ────────────────────────────────────
        self._tree_frame = tk.Frame(rf, bg=BG_PANEL)
        self._tree_frame.pack(fill="both", expand=True)

        columns = ("rank", "song", "artist", "genre",
                   "valence", "arousal", "score")
        self._rec_tree = ttk.Treeview(
            self._tree_frame, columns=columns, show="headings",
            style="Neon.Treeview", selectmode="browse",
        )
        self._sort_reverse = {}  # tracks sort direction per column

        col_cfg = {
            "rank":    ("#",       50,  "center"),
            "song":    ("Song",    155, "w"),
            "artist":  ("Artist",  125, "w"),
            "genre":   ("Genre",   80,  "center"),
            "valence": ("Val",     60,  "center"),
            "arousal": ("Aro",     60,  "center"),
            "score":   ("Fusion",  80,  "center"),
        }
        for cid, (heading, width, anchor) in col_cfg.items():
            self._rec_tree.heading(
                cid, text=heading,
                command=lambda c=cid: self._sort_treeview(c),
            )
            self._rec_tree.column(cid, width=width, anchor=anchor, minwidth=35)

        # Row tags
        self._rec_tree.tag_configure(
            "top", background="#0f2a18", foreground=NEON_GREEN)
        self._rec_tree.tag_configure(
            "normal", foreground=TEXT_PRIMARY)

        # Scrollbar
        tree_scroll = tk.Scrollbar(
            self._tree_frame, command=self._rec_tree.yview,
            bg=BG_PANEL, troughcolor=BG_INPUT,
            activebackground=NEON_CYAN, width=10,
        )
        self._rec_tree.configure(yscrollcommand=tree_scroll.set)

        tree_scroll.pack(side="right", fill="y")
        self._rec_tree.pack(fill="both", expand=True)

        # ── Explain panel (collapsible, initially hidden) ───────
        self._explain_visible = False
        self._explain_outer = tk.Frame(rf, bg=BG_PANEL)
        self._build_explain_panel()

    def _sort_treeview(self, col):
        """Sort recommendation table by *col*, toggling direction."""
        tree = self._rec_tree
        items = [(tree.set(iid, col), iid) for iid in tree.get_children()]
        reverse = self._sort_reverse.get(col, False)
        try:
            items.sort(key=lambda x: float(x[0]), reverse=reverse)
        except ValueError:
            items.sort(key=lambda x: x[0].lower(), reverse=reverse)
        for idx, (_, iid) in enumerate(items):
            tree.move(iid, "", idx)
        self._sort_reverse[col] = not reverse

    # ════════════════════════════════════════════════════════════
    #  EXPLAINABILITY PANEL
    # ════════════════════════════════════════════════════════════

    def _build_explain_panel(self):
        """Construct the collapsible explain panel (widgets only, not packed)."""
        ef = self._explain_outer

        # Separator line
        sep = tk.Canvas(ef, height=2, bg=BG_PANEL, highlightthickness=0)
        sep.pack(fill="x", pady=(6, 0))
        sep.create_line(0, 1, 3000, 1, fill=NEON_YELLOW, width=1, dash=(4, 3))

        # Header label
        tk.Label(
            ef, text=" \u25b8 EXPLAINABILITY \u2014 TOP RECOMMENDATION",
            font=FONT_SMALL, fg=NEON_YELLOW, bg=BG_PANEL,
            anchor="w", pady=4,
        ).pack(fill="x")

        # Scrollable text area
        self._explain_text = tk.Text(
            ef, wrap="word", height=18,
            font=FONT_BODY, fg=TEXT_PRIMARY, bg=BG_INPUT,
            insertbackground=NEON_CYAN,
            selectbackground="#1a3050",
            relief="flat", padx=12, pady=8,
            state="disabled", cursor="arrow",
        )
        e_scroll = tk.Scrollbar(
            ef, command=self._explain_text.yview,
            bg=BG_PANEL, troughcolor=BG_INPUT,
            activebackground=NEON_CYAN, width=10,
        )
        self._explain_text.configure(yscrollcommand=e_scroll.set)
        e_scroll.pack(side="right", fill="y")
        self._explain_text.pack(fill="both", expand=True)

        # Colour tags (same retro palette)
        for tag, fg in [
            ("heading",   NEON_CYAN),
            ("highlight", NEON_MAGENTA),
            ("value",     NEON_GREEN),
            ("dim",       TEXT_DIM),
            ("formula",   NEON_YELLOW),
            ("bar",       NEON_CYAN),
        ]:
            kw = {"foreground": fg}
            if tag == "heading":
                kw["font"] = FONT_BODY_B
            self._explain_text.tag_configure(tag, **kw)

    def _toggle_explain(self):
        """Show / hide the explainability panel below the recommendation table."""
        if not self.result or "recommendations" not in self.result:
            return
        if self._explain_visible:
            self._explain_outer.pack_forget()
            self._explain_visible = False
        else:
            self._populate_explain()
            self._explain_outer.pack(side="bottom", fill="x", pady=(4, 0))
            self._explain_visible = True

    def _populate_explain(self):
        """Compute and render intermediate fusion values for the top hit."""
        result = self.result
        q = result["query"]
        top = result["recommendations"][0]

        g = q["genre"]
        e = q["emotion"]

        # Re-derive intermediate scores using the same fusion functions
        g_sim = genre_similarity(
            g["fuzzy_memberships"], top["genre"].lower()
        )
        e_sim = emotion_similarity(
            (e["valence"], e["arousal"]),
            (top["valence"], top["arousal"]),
        )
        alpha = 0.4
        beta = 0.6
        final = alpha * g_sim + beta * e_sim

        t = self._explain_text
        t.config(state="normal")
        t.delete("1.0", "end")

        # ── Fusion Formula ──────────────────────────────────────
        t.insert("end", "  FUSION FORMULA\n", "heading")
        t.insert("end", "  score = ", "dim")
        t.insert("end", "\u03b1", "highlight")
        t.insert("end", " \u00d7 genre_sim  +  ", "dim")
        t.insert("end", "\u03b2", "highlight")
        t.insert("end", " \u00d7 emotion_sim\n\n", "dim")
        t.insert("end", f"  \u03b1 (w_genre)   = ", "dim")
        t.insert("end", f"{alpha:.2f}\n", "formula")
        t.insert("end", f"  \u03b2 (w_emotion) = ", "dim")
        t.insert("end", f"{beta:.2f}\n\n", "formula")

        # ── Genre Membership Vector ─────────────────────────────
        t.insert("end", "  GENRE MEMBERSHIP VECTOR  (Softmax)\n", "heading")
        for label, score in sorted(
            g["fuzzy_memberships"].items(), key=lambda x: -x[1]
        ):
            bar_len = int(score * 20)
            bar = "\u2588" * bar_len + "\u2591" * (20 - bar_len)
            marker = " \u25c0" if label == top["genre"].lower() else ""
            t.insert("end", f"  {label:<10s} ", "dim")
            t.insert("end", f"{bar}", "bar")
            t.insert("end", f"  {score:.4f}", "value")
            if marker:
                t.insert("end", marker, "highlight")
            t.insert("end", "\n")

        # ── Top Match Breakdown ─────────────────────────────────
        t.insert("end", "\n")
        t.insert("end", f"  TOP MATCH BREAKDOWN\n", "heading")
        t.insert("end", f"  \"{top['title']}\"", "value")
        t.insert("end", f"  by {top['artist']}", "dim")
        t.insert("end", f"  \u00b7  {top['genre'].upper()}\n\n", "highlight")

        t.insert("end", "  Genre Similarity\n", "highlight")
        t.insert("end", f"    genre_sim(\"{top['genre']}\") = ", "dim")
        t.insert("end", f"{g_sim:.4f}\n\n", "value")

        t.insert("end", "  Emotion Similarity\n", "highlight")
        t.insert("end", f"    query  = (V={e['valence']:.2f}, A={e['arousal']:.2f})\n", "dim")
        t.insert("end", f"    match  = (V={top['valence']:.2f}, A={top['arousal']:.2f})\n", "dim")
        t.insert("end", f"    emo_sim = ", "dim")
        t.insert("end", f"{e_sim:.4f}\n\n", "value")

        # ── Final Computation ───────────────────────────────────
        t.insert("end", "  FUSION COMPUTATION\n", "heading")
        t.insert("end", f"    {alpha:.2f}", "formula")
        t.insert("end", f" \u00d7 {g_sim:.4f}", "value")
        t.insert("end", f"  +  ", "dim")
        t.insert("end", f"{beta:.2f}", "formula")
        t.insert("end", f" \u00d7 {e_sim:.4f}", "value")
        t.insert("end", f"\n    = ", "dim")
        t.insert("end", f"{final:.4f}\n", "value")

        t.config(state="disabled")

    def _show_analysis_frame(self):
        """Switch the left panel to show the analysis text output."""
        self._recommend_frame.pack_forget()
        self._analysis_frame.pack(fill="both", expand=True)
        self.explain_btn.set_enabled(False)
        self._explain_visible = False

    def _show_recommend_frame(self):
        """Switch the left panel to show the recommendation table."""
        self._analysis_frame.pack_forget()
        self._recommend_frame.pack(fill="both", expand=True)

    # ════════════════════════════════════════════════════════════
    #  HELPERS
    # ════════════════════════════════════════════════════════════

    def _pulse_title(self):
        """Subtle colour-cycling animation on the title label."""
        color = self._title_colors[self._title_color_idx]
        self._title_label.config(fg=color)
        self._title_color_idx = (self._title_color_idx + 1) % len(self._title_colors)
        self.root.after(600, self._pulse_title)

    # ════════════════════════════════════════════════════════════
    #  VALENCE-AROUSAL PLOT
    # ════════════════════════════════════════════════════════════

    def _build_va_plot(self, parent):
        """Create the retro-styled Valence-Arousal 2D scatter plot."""
        fig = Figure(figsize=(3.6, 3.4), dpi=96, facecolor=BG_DARK)
        ax = fig.add_subplot(111)

        ax.set_facecolor(BG_INPUT)
        ax.set_xlim(1, 9)
        ax.set_ylim(1, 9)
        ax.set_xlabel("Valence \u2192", color=NEON_CYAN, fontsize=9,
                      fontfamily="Consolas")
        ax.set_ylabel("Arousal \u2192", color=NEON_CYAN, fontsize=9,
                      fontfamily="Consolas")
        ax.set_xticks(range(1, 10))
        ax.set_yticks(range(1, 10))

        # Neon grid
        ax.grid(True, color=BORDER, linewidth=0.5, alpha=0.6)
        ax.axhline(y=5, color=NEON_CYAN, linewidth=0.8, alpha=0.35,
                   linestyle="--")
        ax.axvline(x=5, color=NEON_CYAN, linewidth=0.8, alpha=0.35,
                   linestyle="--")

        # Tick & spine styling
        ax.tick_params(colors=TEXT_DIM, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(BORDER)

        # Quadrant labels
        kw = dict(fontsize=8, fontfamily="Consolas", alpha=0.55,
                  ha="center", va="center", weight="bold")
        ax.text(7, 7, "Happy /\nEnergetic", color=NEON_GREEN,   **kw)  # Q1
        ax.text(7, 3, "Happy /\nCalm",      color=NEON_CYAN,    **kw)  # Q2
        ax.text(3, 3, "Sad /\nCalm",        color=NEON_MAGENTA, **kw)  # Q3
        ax.text(3, 7, "Angry /\nIntense",   color=NEON_YELLOW,  **kw)  # Q4

        fig.tight_layout(pad=1.5)

        self._va_fig = fig
        self._va_ax = ax
        self._va_artists = []  # mutable list for previous-dot cleanup

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        self._va_canvas = canvas
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _update_va_plot(self, valence, arousal, confidence=0.5):
        """Plot (or re-plot) the current song as a confidence-scaled dot.

        Marker size scales with *confidence* (genre prediction strength)
        so that high-confidence predictions appear visually stronger.
        """
        ax = self._va_ax
        conf = max(0.1, min(float(confidence), 1.0))  # clamp to [0.1, 1.0]

        # Remove previous dot artists
        for art in self._va_artists:
            art.remove()
        self._va_artists.clear()

        # Sizes scale with confidence: base × confidence
        core_size = 200 * conf

        # Glow layers (large → small, increasing alpha)
        for scale, alpha in [(4.0, 0.08), (2.5, 0.16), (1.6, 0.28)]:
            d = ax.scatter(valence, arousal, s=core_size * scale,
                           c=NEON_MAGENTA, alpha=alpha * conf,
                           zorder=5, edgecolors="none")
            self._va_artists.append(d)

        # Core dot
        d = ax.scatter(valence, arousal, s=core_size, c=NEON_MAGENTA,
                       alpha=0.5 + 0.45 * conf,
                       zorder=6, edgecolors=NEON_CYAN, linewidths=1.2)
        self._va_artists.append(d)

        # Coordinate label
        lbl = ax.annotate(
            f"({valence:.1f}, {arousal:.1f})  {conf:.0%}",
            (valence, arousal),
            textcoords="offset points", xytext=(12, 10),
            fontsize=8, fontfamily="Consolas",
            color=NEON_GREEN, alpha=0.9,
            arrowprops=dict(arrowstyle="->", color=TEXT_DIM, lw=0.7),
        )
        self._va_artists.append(lbl)

        self._va_canvas.draw_idle()

    def _clear_results(self):
        self.results_text.config(state="normal")
        self.results_text.delete("1.0", "end")
        self.results_text.config(state="disabled")
        # Also clear recommendation table
        for iid in self._rec_tree.get_children():
            self._rec_tree.delete(iid)

    def _set_results(self, text):
        """Plain-text fallback (no colour tags)."""
        self.results_text.config(state="normal")
        self.results_text.delete("1.0", "end")
        self.results_text.insert("1.0", text)
        self.results_text.config(state="disabled")

    def _disable_buttons(self):
        self.analyze_btn.set_enabled(False)
        self.recommend_btn.set_enabled(False)
        self.explain_btn.set_enabled(False)
        self.record_btn.set_enabled(False)
        self.live_btn.set_enabled(False)

    def _enable_buttons(self):
        self.analyze_btn.set_enabled(True)
        self.recommend_btn.set_enabled(True)
        self.record_btn.set_enabled(True)
        self.live_btn.set_enabled(True)

    def _run_in_thread(self, target):
        """Run a function in a background thread to keep UI responsive."""
        thread = threading.Thread(target=target, daemon=True)
        thread.start()

    # ════════════════════════════════════════════════════════════
    #  MICROPHONE RECORDING
    # ════════════════════════════════════════════════════════════

    def _run_record(self):
        self._run_in_thread(self._do_record)

    def _do_record(self):
        """Record 5 s from the default mic, save temp WAV, auto-analyze."""
        SR = 22050
        DURATION = 5
        self._disable_buttons()
        self._cleanup_temp()           # remove previous temp recording
        self._blink_on = True
        self._blink_recording()

        try:
            audio = sd.rec(
                int(DURATION * SR), samplerate=SR, channels=1, dtype="float32",
            )
            sd.wait()  # block until recording finishes

            # Write to a temporary WAV file
            tmp = tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False, prefix="muud_rec_",
            )
            tmp_path = tmp.name
            tmp.close()
            wav_io.write(tmp_path, SR, audio)

            # Keep track so it can be cleaned up later
            self._temp_recording = tmp_path

            # Update UI path display
            self.file_var.set("\u266a  [mic recording]")
            self._selected_path = tmp_path

            # Stop blink, show analysing status
            self._stop_blink()
            self.status_var.set("[ ANALYZING RECORDING \u2026 ]")

            # Reuse the normal analysis pipeline
            result = self.recommender.analyze(tmp_path)
            self.result = result
            self._show_analysis(result)
            self.status_var.set("[ RECORDING ANALYSIS COMPLETE ]")

        except Exception as e:
            self._stop_blink()
            messagebox.showerror("Recording Error", str(e))
            self.status_var.set("[ RECORDING ERROR ]")
        finally:
            self._enable_buttons()
            self.explain_btn.set_enabled(True)

    def _blink_recording(self):
        """Toggle a red blinking \u25cf RECORDING indicator on the status bar."""
        if not self._blink_on:
            return
        current = self.status_var.get()
        if "\u25cf" in current:
            self.status_var.set("[   RECORDING \u2026 ]")
        else:
            self.status_var.set("[ \u25cf RECORDING \u2026 ]")
        self._blink_job = self.root.after(500, self._blink_recording)

    def _stop_blink(self):
        """Cancel the blink loop."""
        self._blink_on = False
        if self._blink_job is not None:
            self.root.after_cancel(self._blink_job)
            self._blink_job = None

    # ════════════════════════════════════════════════════════════
    #  LIVE MICROPHONE — spectrogram + rolling inference
    # ════════════════════════════════════════════════════════════

    def _build_live_spectrogram(self, parent):
        """Create the live mel-spectrogram plot (hidden until activated)."""
        fig = Figure(figsize=(3.9, 2.8), dpi=96, facecolor=BG_DARK)
        ax = fig.add_subplot(111)

        ax.set_facecolor(BG_INPUT)
        ax.set_xlabel("Time (frames)", color=NEON_CYAN, fontsize=8,
                       fontfamily="Consolas")
        ax.set_ylabel("Mel bin", color=NEON_CYAN, fontsize=8,
                       fontfamily="Consolas")
        ax.tick_params(colors=TEXT_DIM, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(BORDER)

        # Initialise with blank image (128 mel bins × 200 time frames)
        self._live_spec_data = np.zeros((128, 200), dtype=np.float32)
        self._live_im = ax.imshow(
            self._live_spec_data, aspect="auto", origin="lower",
            cmap="magma", vmin=-3, vmax=3,
            interpolation="nearest",
        )
        fig.tight_layout(pad=1.0)

        self._live_fig = fig
        self._live_ax = ax
        self._live_canvas = FigureCanvasTkAgg(fig, master=parent)
        self._live_canvas.draw()
        self._live_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _toggle_live_mic(self):
        """Start / stop the live microphone stream."""
        if self._live_active:
            self._stop_live_mic()
        else:
            self._start_live_mic()

    def _start_live_mic(self):
        """Open mic stream, show spectrogram panel, begin updates."""
        self._live_active = True
        self._live_buffer.clear()
        self._live_spec_data = np.zeros((128, 200), dtype=np.float32)
        self.recommender.genre_clf.clear_live_history()

        # Update button text
        self.live_btn.text = "\u25a0 STOP MIC"
        self.live_btn.color = "#ff3333"
        self.live_btn._draw_normal()

        # Disable file-based buttons while live
        self.analyze_btn.set_enabled(False)
        self.recommend_btn.set_enabled(False)
        self.explain_btn.set_enabled(False)
        self.record_btn.set_enabled(False)

        # Show live panel
        self._live_panel.pack(fill="both", expand=True, pady=(6, 0))

        self.status_var.set("[ \u25cf LIVE MIC ACTIVE \u2014 listening \u2026 ]")
        self._live_info_var.set("  Buffering audio \u2026")

        try:
            self._live_stream = sd.InputStream(
                samplerate=self._live_sr,
                channels=1,
                dtype="float32",
                blocksize=1024,
                callback=self._mic_callback,
            )
            self._live_stream.start()
        except Exception as e:
            messagebox.showerror("Mic Error", str(e))
            self._stop_live_mic()
            return

        # Schedule periodic updates
        self._live_spec_job = self.root.after(150, self._update_live_spectrogram)
        self._live_inference_job = self.root.after(6000, self._schedule_live_inference)

    def _stop_live_mic(self):
        """Stop mic stream and hide live panel."""
        self._live_active = False

        if self._live_stream is not None:
            try:
                self._live_stream.stop()
                self._live_stream.close()
            except Exception:
                pass
            self._live_stream = None

        # Cancel scheduled jobs
        if self._live_spec_job is not None:
            self.root.after_cancel(self._live_spec_job)
            self._live_spec_job = None
        if self._live_inference_job is not None:
            self.root.after_cancel(self._live_inference_job)
            self._live_inference_job = None

        # Restore button
        self.live_btn.text = "\U0001f3a4 LIVE MIC"
        self.live_btn.color = "#ff6600"
        self.live_btn._draw_normal()

        # Hide live panel
        self._live_panel.pack_forget()

        # Re-enable buttons
        self._enable_buttons()
        if self._selected_path:
            self.analyze_btn.set_enabled(True)
            self.recommend_btn.set_enabled(True)
        else:
            self.analyze_btn.set_enabled(False)
            self.recommend_btn.set_enabled(False)
        self.explain_btn.set_enabled(bool(self.result and "recommendations" in self.result))

        self.status_var.set("[ LIVE MIC STOPPED ]")
        self._live_info_var.set("")

    def _mic_callback(self, indata, frames, time_info, status):
        """sounddevice callback — append samples to the ring buffer."""
        self._live_buffer.extend(indata[:, 0])

    def _update_live_spectrogram(self):
        """Periodically redraw the rolling mel spectrogram from the buffer."""
        if not self._live_active:
            return

        buf = np.array(self._live_buffer, dtype=np.float32)
        if len(buf) > 2048:
            # Compute mel spectrogram of the last ~3 s (or whatever is available)
            tail = buf[-self._live_sr * 3:]
            try:
                mel = librosa.feature.melspectrogram(
                    y=tail, sr=self._live_sr, n_mels=128,
                    n_fft=2048, hop_length=512,
                )
                mel_db = librosa.power_to_db(mel, ref=np.max)
                std = mel_db.std()
                if std > 0:
                    mel_db = (mel_db - mel_db.mean()) / std

                # Roll existing data left and paste new columns
                n_new = mel_db.shape[1]
                if n_new >= 200:
                    self._live_spec_data = mel_db[:, -200:]
                else:
                    self._live_spec_data = np.roll(self._live_spec_data, -n_new, axis=1)
                    self._live_spec_data[:, -n_new:] = mel_db

                self._live_im.set_data(self._live_spec_data)
                self._live_canvas.draw_idle()
            except Exception:
                pass  # skip frame on error

        self._live_spec_job = self.root.after(200, self._update_live_spectrogram)

    def _schedule_live_inference(self):
        """Schedule a background inference on accumulated live audio."""
        if not self._live_active:
            return
        buf = np.array(self._live_buffer, dtype=np.float32)
        if len(buf) >= self._live_sr * 3:  # need at least 3 s
            thread = threading.Thread(
                target=self._do_live_inference, args=(buf.copy(),), daemon=True,
            )
            thread.start()
        # Re-schedule next inference cycle
        self._live_inference_job = self.root.after(7000, self._schedule_live_inference)

    def _do_live_inference(self, signal):
        """Run genre + emotion on the buffered signal (background thread)."""
        try:
            result = self.recommender.analyze_signal(signal, self._live_sr, "live mic")
            g = result["genre"]
            e = result["emotion"]

            info = (
                f"  Genre: {g['top_genre'].upper()}  ({g['confidence']:.0%})   "
                f"\u00b7   Mood: {e['mood_label']}  "
                f"(V={e['valence']:.1f}  A={e['arousal']:.1f})"
            )
            # Update UI from main thread
            self.root.after(0, self._live_info_var.set, info)
            self.root.after(0, lambda v=e["valence"], a=e["arousal"], c=g["confidence"]: self._update_va_plot(v, a, c))
            self.root.after(0, self.status_var.set,
                            f"[ \u25cf LIVE MIC \u2014 {g['top_genre'].upper()} ]")
            # Populate the analysis panel so the user sees full details
            self.root.after(0, self._show_live_analysis, result)
        except Exception as exc:
            self.root.after(0, self._live_info_var.set, f"  Inference error: {exc}")

    def _show_live_analysis(self, result):
        """Update the analysis text panel with live mic inference results."""
        self.result = result
        self._show_analysis(result)

    # ── Launch ──────────────────────────────────────────────────

    def run(self):
        self.root.mainloop()
