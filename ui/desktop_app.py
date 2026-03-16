"""
desktop_app.py
--------------
CustomTkinter GUI for Muud — Music Intelligence System.
Maintains the ORIGINAL retro-arcade theme (dark navy, neon accents)
but uses modern CustomTkinter for rounded borders and positioning.
"""
import os, tempfile, tkinter as tk, threading, urllib.request, webbrowser
from tkinter import filedialog, messagebox
from collections import deque
from io import BytesIO

import customtkinter as ctk
import sounddevice as sd
import scipy.io.wavfile as wav_io
import numpy as np
import librosa
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

try:
    from PIL import Image, ImageTk, ImageDraw, ImageFilter, ImageEnhance
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

try:
    import pygame; pygame.mixer.init(); _HAS_PYGAME = True
except Exception:
    _HAS_PYGAME = False

from engine.fusion import emotion_similarity, genre_similarity

ctk.set_appearance_mode("dark")

# ════════════════════════════════════════════════════════════════
#  ORIGINAL COLOUR PALETTE
# ════════════════════════════════════════════════════════════════
BG_DARK      = "#0b0e1a"       # deep navy (base window)
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
#  ORIGINAL FONT STACK
# ════════════════════════════════════════════════════════════════
FONT_TITLE    = ("Terminal", 24, "bold")
FONT_SUBTITLE = ("Terminal", 10)
FONT_BODY     = ("Consolas", 10)
FONT_BODY_B   = ("Consolas", 10, "bold")
FONT_BTN      = ("Consolas", 11, "bold")
FONT_STATUS   = ("Consolas", 9)
FONT_SMALL    = ("Consolas", 9)


class MuudApp:
    def __init__(self, recommender):
        self.recommender = recommender
        self.result = None
        self._selected_path = None
        self._temp_recording = None
        
        self._preview_playing = False
        self._preview_tmp_path = None
        self._preview_art_refs = []
        
        self._live_active = False
        self._live_stream = None
        self._live_buffer = deque(maxlen=22050 * 30)
        self._live_inference_job = None
        self._live_spec_job = None
        self._live_sr = 22050
        
        self._explain_visible = False
        self._blink_job = None
        self._title_color_idx = 0
        self._title_colors = [NEON_CYAN, "#00ccdd", "#00b3bb", "#009999", "#00b3bb", "#00ccdd"]

        # Carousel state
        self._carousel_recs = []
        self._carousel_idx = 0
        self._carousel_auto_job = None
        self._carousel_art_cache = {}  # idx -> CTkImage
        self._carousel_animating = False

        self.root = ctk.CTk()
        self.root.title("MUUD — Music Intelligence")
        self.root.geometry("1350x860")
        self.root.minsize(1200, 750)
        self.root.configure(fg_color=BG_DARK)

        self._build_ui()

    # ════════════════════════════════════════════════════════════
    #  BUILD UI
    # ════════════════════════════════════════════════════════════
    def _build_ui(self):
        # Scanline top
        scanline = tk.Canvas(self.root, height=3, bg=BG_DARK, highlightthickness=0)
        scanline.pack(fill="x")
        scanline.create_line(0, 1, 3000, 1, fill=NEON_CYAN, width=1)
        
        # ── Title Block ──
        title_frame = tk.Frame(self.root, bg=BG_DARK)
        title_frame.pack(pady=(14, 0))

        self._title_label = tk.Label(
            title_frame, text="◆  M U U D  ◆",
            font=FONT_TITLE, fg=NEON_CYAN, bg=BG_DARK,
        )
        self._title_label.pack()

        sub_frame = tk.Frame(title_frame, bg=BG_DARK)
        sub_frame.pack(pady=(2, 0))
        tk.Label(
            sub_frame, text="HYBRID  SOFT  COMPUTING  MUSIC  INTELLIGENCE",
            font=FONT_SUBTITLE, fg=TEXT_DIM, bg=BG_DARK,
        ).pack(side="left")
        tk.Label(
            sub_frame, text="  v2.0",
            font=("Consolas", 8), fg="#3a4060", bg=BG_DARK,
        ).pack(side="left", padx=(6, 0))

        glow = tk.Canvas(self.root, height=5, bg=BG_DARK, highlightthickness=0)
        glow.pack(fill="x", padx=60, pady=(6, 0))
        glow.create_line(0, 2, 3000, 2, fill=NEON_MAGENTA, width=1, dash=(6, 4))
        self._pulse_title()

        # ── Actions & File ──
        control_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        control_frame.pack(fill="x", padx=30, pady=10)
        
        file_panel = ctk.CTkFrame(control_frame, fg_color=BG_PANEL, border_color=NEON_CYAN, border_width=1, corner_radius=8)
        file_panel.pack(fill="x", pady=5)
        
        file_inner = ctk.CTkFrame(file_panel, fg_color="transparent")
        file_inner.pack(fill="x", padx=12, pady=10)
        
        self.file_var = tk.StringVar(value="No file selected …")
        tk.Label(
            file_inner, textvariable=self.file_var, font=FONT_BODY,
            fg=NEON_YELLOW, bg=BG_INPUT, anchor="w", padx=10, pady=6
        ).pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        self.browse_btn = ctk.CTkButton(
            file_inner, text="BROWSE", command=self._browse_file,
            font=FONT_BTN, fg_color=BG_PANEL, text_color=NEON_MAGENTA,
            border_color=NEON_MAGENTA, border_width=1, hover_color="#1a1f3a", width=130, height=34
        )
        self.browse_btn.pack(side="right")
        
        # Action Buttons
        btn_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        btn_frame.pack(pady=5)
        
        btn_specs = [
            ("⚡ ANALYZE",   NEON_CYAN,   self._run_analyze,     "analyze_btn"),
            ("♫ RECOMMEND",  NEON_GREEN,  self._run_recommend,   "recommend_btn"),
            ("? EXPLAIN",    NEON_YELLOW, self._toggle_explain,  "explain_btn"),
            ("⏺ REC 5s",    NEON_MAGENTA,self._run_record,      "record_btn"),
            ("🎤 LIVE MIC",  "#ff6600",   self._toggle_live_mic, "live_btn"),
        ]
        
        for text, col, cmd, attr in btn_specs:
            b = ctk.CTkButton(
                btn_frame, text=text, font=FONT_BTN, fg_color=BG_PANEL,
                text_color=col, border_color=col, border_width=1,
                hover_color="#1a1f3a", height=40, width=170, command=cmd
            )
            b.pack(side="left", padx=10)
            setattr(self, attr, b)
            
        self.analyze_btn.configure(state="disabled")
        self.recommend_btn.configure(state="disabled")
        self.explain_btn.configure(state="disabled")
        
        # Status Bar
        self.status_var = tk.StringVar(value="[ READY ]")
        tk.Label(
            self.root, textvariable=self.status_var,
            font=FONT_STATUS, fg=NEON_GREEN, bg=BG_DARK, anchor="w"
        ).pack(padx=34, anchor="w", pady=(5,0))
        
        # ── Main Content Split ──
        content_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        content_frame.pack(fill="both", expand=True, padx=30, pady=(5, 15))
        
        # Left Panel (Results / Recommendations)
        left_panel = ctk.CTkFrame(content_frame, fg_color=BG_PANEL, border_color=NEON_CYAN, border_width=1, corner_radius=8)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 6))
        
        lbl_left = tk.Label(left_panel, text=" ▸ RESULTS ", font=FONT_SMALL, fg=NEON_CYAN, bg=BG_PANEL)
        lbl_left.place(x=10, y=-8) # simulated LabelFrame logic
        
        left_inner = ctk.CTkFrame(left_panel, fg_color="transparent")
        left_inner.pack(fill="both", expand=True, padx=8, pady=15)
        
        # Sub-frames
        self._analysis_frame = ctk.CTkFrame(left_inner, fg_color="transparent")
        self.results_text = tk.Text(
            self._analysis_frame, wrap="word", font=FONT_BODY, fg=TEXT_PRIMARY, bg=BG_INPUT,
            insertbackground=NEON_CYAN, selectbackground="#1a3050", relief="flat", padx=12, pady=10,
            state="disabled", cursor="arrow"
        )
        a_scroll = tk.Scrollbar(self._analysis_frame, command=self.results_text.yview, bg=BG_PANEL, troughcolor=BG_INPUT, activebackground=NEON_CYAN, width=10)
        self.results_text.configure(yscrollcommand=a_scroll.set)
        a_scroll.pack(side="right", fill="y")
        self.results_text.pack(side="left", fill="both", expand=True)
        
        self.results_text.tag_configure("heading",   foreground=NEON_CYAN,    font=FONT_BODY_B)
        self.results_text.tag_configure("highlight",  foreground=NEON_MAGENTA)
        self.results_text.tag_configure("value",      foreground=NEON_GREEN)
        self.results_text.tag_configure("dim",        foreground=TEXT_DIM)
        self.results_text.tag_configure("bar",        foreground=NEON_CYAN)
        
        self._analysis_frame.pack(fill="both", expand=True)
        
        self._recommend_frame = ctk.CTkFrame(left_inner, fg_color="transparent")
        self._build_recommend_area()
        
        # Right Panel (V-A Plot / Live Mic / AI Summary)
        right_panel = ctk.CTkFrame(content_frame, fg_color="transparent", width=420)
        right_panel.pack(side="right", fill="both", padx=(6, 0))
        right_panel.pack_propagate(False)
        
        va_panel = ctk.CTkFrame(right_panel, fg_color=BG_PANEL, border_color=NEON_CYAN, border_width=1, corner_radius=8)
        va_panel.pack(fill="x", pady=(0, 10))
        lbl_va = tk.Label(va_panel, text=" ▸ V-A SPACE ", font=FONT_SMALL, fg=NEON_CYAN, bg=BG_PANEL)
        lbl_va.place(x=10, y=-8)
        va_inner = ctk.CTkFrame(va_panel, fg_color="transparent")
        va_inner.pack(fill="both", expand=True, padx=6, pady=15)
        self._build_va_plot(va_inner)
        
        # Replace the live mic panel completely. Make a dynamic panel for LiveMic/Explain in the right.
        # Actually in the original, Explain was below recommendations, and Live Mic was below VA Plot.
        self._live_panel = ctk.CTkFrame(right_panel, fg_color=BG_PANEL, border_color=NEON_CYAN, border_width=1, corner_radius=8)
        lbl_live = tk.Label(self._live_panel, text=" ▸ LIVE SPECTROGRAM ", font=FONT_SMALL, fg=NEON_CYAN, bg=BG_PANEL)
        lbl_live.place(x=10, y=-8)
        self._live_info_var = tk.StringVar(value="")
        self._live_info_label = tk.Label(self._live_panel, textvariable=self._live_info_var, font=FONT_SMALL, fg=NEON_GREEN, bg=BG_PANEL, anchor="w", padx=8)
        self._live_info_label.pack(fill="x", pady=(15, 0))
        live_inner = ctk.CTkFrame(self._live_panel, fg_color="transparent")
        live_inner.pack(fill="both", expand=True, padx=6, pady=6)
        self._build_live_spectrogram(live_inner)

        # Bottom scanline
        bot = tk.Canvas(self.root, height=3, bg=BG_DARK, highlightthickness=0)
        bot.pack(fill="x", side="bottom")
        bot.create_line(0, 1, 3000, 1, fill=NEON_CYAN, width=1)

    def _build_recommend_area(self):
        rf = self._recommend_frame
        self._query_label = tk.Label(rf, text="", font=FONT_SMALL, fg=NEON_YELLOW, bg=BG_INPUT, anchor="w", padx=10, pady=6)
        self._query_label.pack(fill="x", pady=(0, 4))

        # ── Hero Carousel Container ──
        self._hero_frame = ctk.CTkFrame(rf, fg_color=BG_PANEL, border_color=NEON_CYAN, border_width=1, corner_radius=12)
        self._hero_frame.pack(fill="both", expand=True, pady=(0, 4))

        # Navigation row (arrows + hero card + arrows)
        nav_row = ctk.CTkFrame(self._hero_frame, fg_color="transparent")
        nav_row.pack(fill="both", expand=True, padx=8, pady=12)

        # Left arrow
        self._left_arrow = ctk.CTkButton(
            nav_row, text="◀", font=("Segoe UI", 24, "bold"), fg_color="transparent",
            text_color=NEON_CYAN, hover_color=BG_INPUT, width=44, height=44,
            corner_radius=22, command=self._carousel_prev
        )
        self._left_arrow.pack(side="left", padx=(4, 8), pady=20)

        # Center hero content
        self._hero_content = ctk.CTkFrame(nav_row, fg_color="transparent")
        self._hero_content.pack(side="left", fill="both", expand=True)

        # Top: circular album art (centered)
        self._hero_art_frame = ctk.CTkFrame(self._hero_content, fg_color="transparent")
        self._hero_art_frame.pack(pady=(8, 10))
        self._hero_art_label = ctk.CTkLabel(self._hero_art_frame, text="♪", font=("Segoe UI", 64), text_color=TEXT_DIM, fg_color="transparent")
        self._hero_art_label.pack()

        # Song info
        self._hero_title = ctk.CTkLabel(self._hero_content, text="", font=("Segoe UI", 20, "bold"), text_color=NEON_GREEN, wraplength=480)
        self._hero_title.pack(pady=(0, 2))
        self._hero_artist = ctk.CTkLabel(self._hero_content, text="", font=("Segoe UI", 14), text_color=TEXT_PRIMARY)
        self._hero_artist.pack(pady=(0, 4))
        self._hero_meta = ctk.CTkLabel(self._hero_content, text="", font=("Segoe UI", 12), text_color=TEXT_DIM)
        self._hero_meta.pack(pady=(0, 8))
        
        # Score + VA bar
        self._hero_stats = ctk.CTkLabel(self._hero_content, text="", font=("Segoe UI", 11), text_color=NEON_CYAN)
        self._hero_stats.pack(pady=(0, 10))

        # Action buttons row
        self._hero_btns = ctk.CTkFrame(self._hero_content, fg_color="transparent")
        self._hero_btns.pack(pady=(0, 6))

        self._hero_play_btn = ctk.CTkButton(
            self._hero_btns, text="▶ PLAY", font=("Segoe UI", 12, "bold"), fg_color=BG_INPUT,
            text_color=NEON_GREEN, border_color=NEON_GREEN, border_width=1,
            hover_color="#1a2f1a", height=38, width=150
        )
        self._hero_play_btn.pack(side="left", padx=8)
        self._hero_spotify_btn = ctk.CTkButton(
            self._hero_btns, text="♫ SPOTIFY", font=("Segoe UI", 12, "bold"), fg_color=BG_INPUT,
            text_color=NEON_CYAN, border_color=NEON_CYAN, border_width=1,
            hover_color="#1a2a3f", height=38, width=150
        )
        self._hero_spotify_btn.pack(side="left", padx=8)

        # Right arrow
        self._right_arrow = ctk.CTkButton(
            nav_row, text="▶", font=("Segoe UI", 24, "bold"), fg_color="transparent",
            text_color=NEON_CYAN, hover_color=BG_INPUT, width=44, height=44,
            corner_radius=22, command=self._carousel_next
        )
        self._right_arrow.pack(side="right", padx=(8, 4), pady=20)

        # Dot indicators row
        self._dots_frame = ctk.CTkFrame(self._hero_frame, fg_color="transparent", height=24)
        self._dots_frame.pack(pady=(0, 10))
        self._dot_labels = []

        # Explain panel (retro style)
        self._explain_outer = ctk.CTkFrame(rf, fg_color="transparent")
        self._build_explain_panel()

    def _build_explain_panel(self):
        ef = self._explain_outer
        sep = tk.Canvas(ef, height=2, bg=BG_PANEL, highlightthickness=0)
        sep.pack(fill="x", pady=(6, 0))
        sep.create_line(0, 1, 3000, 1, fill=NEON_YELLOW, width=1, dash=(4, 3))
        
        tk.Label(ef, text=" ▸ EXPLAINABILITY — TOP RECOMMENDATION", font=FONT_SMALL, fg=NEON_YELLOW, bg=BG_PANEL, anchor="w", pady=4).pack(fill="x")
        
        self._explain_text = tk.Text(ef, wrap="word", height=18, font=FONT_BODY, fg=TEXT_PRIMARY, bg=BG_INPUT, insertbackground=NEON_CYAN, selectbackground="#1a3050", relief="flat", padx=12, pady=8, state="disabled", cursor="arrow")
        e_scroll = tk.Scrollbar(ef, command=self._explain_text.yview, bg=BG_PANEL, troughcolor=BG_INPUT, activebackground=NEON_CYAN, width=10)
        self._explain_text.configure(yscrollcommand=e_scroll.set)
        e_scroll.pack(side="right", fill="y")
        self._explain_text.pack(fill="both", expand=True)
        
        for tag, fg in [("heading", NEON_CYAN), ("highlight", NEON_MAGENTA), ("value", NEON_GREEN), ("dim", TEXT_DIM), ("formula", NEON_YELLOW), ("bar", NEON_CYAN)]:
            kw = {"foreground": fg}
            if tag == "heading": kw["font"] = FONT_BODY_B
            self._explain_text.tag_configure(tag, **kw)


    def _pulse_title(self):
        color = self._title_colors[self._title_color_idx]
        self._title_label.config(fg=color)
        self._title_color_idx = (self._title_color_idx + 1) % len(self._title_colors)
        self.root.after(600, self._pulse_title)

    # ════════════════════════════════════════════════════════════
    #  FILE BROWSE
    # ════════════════════════════════════════════════════════════
    def _cleanup_temp(self):
        if self._temp_recording:
            self.recommender.invalidate_cache(self._temp_recording)
            if os.path.exists(self._temp_recording):
                try: os.unlink(self._temp_recording)
                except OSError: pass
            self._temp_recording = None

    def _browse_file(self):
        path = filedialog.askopenfilename(title="Select Audio File", filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.ogg"), ("All Files", "*.*")])
        if path:
            self._cleanup_temp()
            self.file_var.set(f"♪  {os.path.basename(path)}")
            self._selected_path = path
            self.analyze_btn.configure(state="normal")
            self.recommend_btn.configure(state="normal")
            self.explain_btn.configure(state="disabled")
            if self._explain_visible:
                self._explain_outer.pack_forget()
                self._explain_visible = False
            self.status_var.set("[ FILE LOADED — SELECT ACTION ]")

    # ════════════════════════════════════════════════════════════
    #  ANALYSIS
    # ════════════════════════════════════════════════════════════
    def _run_analyze(self):
        self._run_in_thread(self._do_analyze)

    def _do_analyze(self):
        self.status_var.set("[ ANALYZING … ]")
        self._disable_buttons()
        try:
            result = self.recommender.analyze(self._selected_path)
            self.result = result
            self.root.after(0, self._show_analysis, result)
            self.status_var.set("[ ANALYSIS COMPLETE ]")
        except Exception as e:
            self.root.after(0, messagebox.showerror, "Error", str(e))
            self.status_var.set("[ ERROR ]")
        finally:
            self.root.after(0, self._enable_buttons)

    def _show_analysis(self, result):
        self._show_analysis_frame()
        self._clear_results()
        t = self.results_text
        t.config(state="normal")

        t.insert("end", f"  FILE:  ", "dim")
        t.insert("end", f"{result['file']}\n\n", "value")

        g = result["genre"]
        t.insert("end", "  ┌─── GENRE CLASSIFICATION ──────────────────────────────────────┐\n", "heading")
        t.insert("end", f"  │  Top Genre:  ", "dim")
        t.insert("end", f"{g['top_genre'].upper()}", "highlight")
        t.insert("end", f"   ({g['confidence']:.1%} confidence)\n", "dim")
        t.insert("end", "  │\n", "heading")

        top3 = sorted(g["fuzzy_memberships"].items(), key=lambda x: -x[1])[:3]
        t.insert("end", "  │  Genre Probabilities:\n", "dim")
        for label, score in top3:
            t.insert("end", f"  │    {label}", "highlight")
            t.insert("end", f" — {score * 100:.1f}%\n", "value")
        t.insert("end", "  │\n", "heading")

        t.insert("end", "  │  Fuzzy Memberships:\n", "dim")
        for label, score in sorted(g["fuzzy_memberships"].items(), key=lambda x: -x[1]):
            bar_len = int(score * 25)
            bar = "█" * bar_len + "░" * (25 - bar_len)
            t.insert("end", f"  │  {label:<10s} ", "dim")
            t.insert("end", f"{bar}", "bar")
            t.insert("end", f"  {score:.3f}\n", "value")
        t.insert("end", "  └─────────────────────────────────────────────────────────────┘\n\n", "heading")

        e = result["emotion"]
        t.insert("end", "  ┌─── EMOTION ANALYSIS ────────────────────────────────────────┐\n", "heading")
        t.insert("end", f"  │  Valence:  ", "dim")
        t.insert("end", f"{e['valence']:.2f}", "value")
        t.insert("end", f"     Arousal:  ", "dim")
        t.insert("end", f"{e['arousal']:.2f}\n", "value")
        t.insert("end", f"  │  Mood:     ", "dim")
        t.insert("end", f"{e['mood_label']}\n", "highlight")
        t.insert("end", "  └─────────────────────────────────────────────────────────────┘\n", "heading")
        t.config(state="disabled")

        self._update_va_plot(e['valence'], e['arousal'], g['confidence'])

    # ════════════════════════════════════════════════════════════
    #  RECOMMENDATION
    # ════════════════════════════════════════════════════════════
    def _run_recommend(self):
        self._run_in_thread(self._do_recommend)

    def _do_recommend(self):
        self.status_var.set("[ ANALYZING & RECOMMENDING … ]")
        self._disable_buttons()
        try:
            result = self.recommender.recommend(self._selected_path, top_n=6)
            self.result = result
            self.root.after(0, self._show_recommendations, result)
            self.status_var.set("[ RECOMMENDATIONS READY ]")
        except Exception as e:
            self.root.after(0, messagebox.showerror, "Error", str(e))
            self.status_var.set("[ ERROR ]")
        finally:
            self.root.after(0, self._enable_buttons)

    def _show_recommendations(self, result):
        self._show_recommend_frame()
        if self._explain_visible:
            self._explain_outer.pack_forget()
            self._explain_visible = False
        self.explain_btn.configure(state="normal")

        q = result["query"]; g = q["genre"]; e = q["emotion"]
        self._query_label.config(
            text=f"  ♪ {q['file']}   ·   Genre: {g['top_genre'].upper()} ({g['confidence']:.0%})   ·   "
                 f"Mood: {e['mood_label']}  (V={e['valence']:.1f}  A={e['arousal']:.1f})"
        )
        self._update_va_plot(e['valence'], e['arousal'], g['confidence'])
        self._build_recommend_cards(result["recommendations"])

    # ════════════════════════════════════════════════════════════
    #  HERO CAROUSEL
    # ════════════════════════════════════════════════════════════
    def _build_recommend_cards(self, recs):
        """Populate the hero carousel with recommendation data."""
        self._stop_preview()
        self._preview_art_refs.clear()
        self._carousel_art_cache.clear()
        self._carousel_recs = list(recs)
        self._carousel_idx = 0

        # Rebuild dot indicators
        for d in self._dot_labels:
            d.destroy()
        self._dot_labels.clear()
        for i in range(len(recs)):
            dot = tk.Label(
                self._dots_frame, text="●", font=("Consolas", 12),
                fg=NEON_CYAN if i == 0 else TEXT_DIM, bg=BG_PANEL, cursor="hand2"
            )
            dot.pack(side="left", padx=4)
            dot.bind("<Button-1>", lambda e, idx=i: self._carousel_goto(idx))
            self._dot_labels.append(dot)

        # Pre-fetch all album art in background threads
        for i, rec in enumerate(recs):
            if rec.get("album_art") and _HAS_PIL:
                threading.Thread(target=self._load_carousel_art, args=(i, rec["album_art"]), daemon=True).start()

        # Show first item
        self._carousel_show(0)
        self._carousel_auto_start()

    def _generate_placeholder_art(self, idx, rec):
        """Create a colorful circular placeholder with the track's initial using CTkImage."""
        if not _HAS_PIL:
            return None
        size = 240
        # Pick a gradient color based on index
        hue_palette = [
            (0, 229, 255),   # cyan
            (255, 46, 170),  # magenta
            (57, 255, 20),   # green
            (255, 230, 0),   # yellow
            (138, 43, 226),  # violet
            (255, 100, 50),  # orange
        ]
        c1 = hue_palette[idx % len(hue_palette)]
        c2 = hue_palette[(idx + 2) % len(hue_palette)]

        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        # Draw gradient circle (concentric rings)
        for r in range(size // 2, 0, -1):
            t = r / (size // 2)
            color = tuple(int(c1[j] * t + c2[j] * (1 - t)) for j in range(3)) + (255,)
            x0, y0 = size // 2 - r, size // 2 - r
            x1, y1 = size // 2 + r, size // 2 + r
            draw.ellipse((x0, y0, x1, y1), fill=color)

        # Draw the first letter of the track name
        track_name = rec.get("title") or rec.get("song") or "?"
        initial = track_name[0].upper() if track_name else "♪"
        try:
            from PIL import ImageFont
            font = ImageFont.truetype("segoeui.ttf", 96)
        except Exception:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), initial, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = (size - tw) // 2
        ty = (size - th) // 2 - 8
        draw.text((tx + 3, ty + 3), initial, fill=(0, 0, 0, 180), font=font)
        draw.text((tx, ty), initial, fill=(255, 255, 255, 240), font=font)

        photo = ctk.CTkImage(light_image=img, dark_image=img, size=(size, size))
        self._carousel_art_cache[idx] = photo
        return photo

    def _load_carousel_art(self, idx, url):
        """Download and circularly mask album art for a carousel item using CTkImage."""
        try:
            data = urllib.request.urlopen(url, timeout=5).read()
            img = Image.open(BytesIO(data))

            # Now create circular 240px album art
            img = img.resize((240, 240), Image.LANCZOS)
            mask = Image.new("L", (240, 240), 0)
            draw = ImageDraw.Draw(mask)
            draw.ellipse((2, 2, 237, 237), fill=255) # slightly inset to prevent edge artifacts
            out = Image.new("RGBA", (240, 240), (0, 0, 0, 0))
            img = img.convert("RGBA")
            out.paste(img, (0, 0), mask)
            
            photo = ctk.CTkImage(light_image=out, dark_image=out, size=(240, 240))
            self._carousel_art_cache[idx] = photo

            # If this is the currently displayed item, update it
            if idx == self._carousel_idx:
                def update_art():
                    self._hero_art_label.configure(image=photo, text="")
                self.root.after(0, update_art)
        except Exception:
            pass

    def _carousel_show(self, idx):
        """Display recommendation at index `idx` in the hero section."""
        if not self._carousel_recs:
            return
        idx = idx % len(self._carousel_recs)
        self._carousel_idx = idx
        rec = self._carousel_recs[idx]

        # Update art
        if idx in self._carousel_art_cache:
            photo = self._carousel_art_cache[idx]
            self._hero_art_label.configure(image=photo, text="")
        else:
            # Generate placeholder on demand if not loading real art
            if not rec.get("album_art"):
                photo = self._generate_placeholder_art(idx, rec)
                if photo:
                    self._hero_art_label.configure(image=photo, text="")
                else:
                    self._hero_art_label.configure(image="", text="♪")
            else:
                self._hero_art_label.configure(image="", text="♪")

        # Get title — CSV uses "song", Spotify uses "title"
        track_title = rec.get("title") or rec.get("song") or "Unknown"

        # Update text
        rank_label = "★ TOP PICK" if idx == 0 else f"{idx + 1}"
        title_color = NEON_GREEN if idx == 0 else NEON_CYAN
        self._hero_title.configure(text=f"{rank_label}  ·  {track_title}", text_color=title_color)
        self._hero_artist.configure(text=rec.get("artist", "Unknown"))
        self._hero_meta.configure(text=f"{rec['genre'].upper()}  ·  Score: {rec['score']:.4f}")

        va = f"V={rec['valence']:.1f}   A={rec['arousal']:.1f}"
        if rec.get("emotion_analyzed"):
            va += "   ✓ emotion-analyzed"
        self._hero_stats.configure(text=va, text_color=NEON_CYAN if rec.get("emotion_analyzed") else TEXT_DIM)

        # Update buttons
        p_url = rec.get("preview_url")
        if p_url and _HAS_PYGAME:
            self._hero_play_btn.configure(
                state="normal", text="▶ PLAY", text_color=NEON_GREEN, border_color=NEON_GREEN,
                command=lambda: self._toggle_preview(p_url, self._hero_play_btn)
            )
        else:
            self._hero_play_btn.configure(state="disabled", text="▶ PLAY")

        s_url = rec.get("spotify_url")
        if s_url:
            self._hero_spotify_btn.configure(state="normal", command=lambda: webbrowser.open(s_url))
        else:
            self._hero_spotify_btn.configure(state="disabled")

        # Update dots
        for i, dot in enumerate(self._dot_labels):
            dot.config(fg=NEON_CYAN if i == idx else TEXT_DIM)

    def _carousel_next(self):
        if self._carousel_recs:
            self._carousel_auto_stop()
            new_idx = (self._carousel_idx + 1) % len(self._carousel_recs)
            self._carousel_animate_to(new_idx)
            self._carousel_auto_start()

    def _carousel_prev(self):
        if self._carousel_recs:
            self._carousel_auto_stop()
            new_idx = (self._carousel_idx - 1) % len(self._carousel_recs)
            self._carousel_animate_to(new_idx)
            self._carousel_auto_start()

    def _carousel_goto(self, idx):
        if self._carousel_recs:
            self._carousel_auto_stop()
            self._carousel_animate_to(idx)
            self._carousel_auto_start()

    def _carousel_animate_to(self, target_idx):
        """Fade-out → switch → fade-in animation."""
        if self._carousel_animating:
            return
        self._carousel_animating = True
        # Quick fade-out via text color dimming
        self._hero_content.configure(fg_color=BG_PANEL)
        self.root.after(80, lambda: self._carousel_show(target_idx))
        self.root.after(160, self._carousel_animation_done)

    def _carousel_animation_done(self):
        self._carousel_animating = False

    def _carousel_auto_start(self):
        self._carousel_auto_stop()
        self._carousel_auto_job = self.root.after(4000, self._carousel_auto_advance)

    def _carousel_auto_stop(self):
        if self._carousel_auto_job is not None:
            self.root.after_cancel(self._carousel_auto_job)
            self._carousel_auto_job = None

    def _carousel_auto_advance(self):
        if self._carousel_recs and not self._carousel_animating:
            new_idx = (self._carousel_idx + 1) % len(self._carousel_recs)
            self._carousel_show(new_idx)
        self._carousel_auto_job = self.root.after(4000, self._carousel_auto_advance)

    def _load_album_art(self, url, label, size):
        """Legacy album art loader (non-carousel usage)."""
        try:
            data = urllib.request.urlopen(url, timeout=5).read()
            img = Image.open(BytesIO(data)).resize((size, size), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self._preview_art_refs.append(photo)
            self.root.after(0, lambda: label.config(image=photo))
        except Exception:
            pass

    # ════════════════════════════════════════════════════════════
    #  PREVIEW AUDIO
    # ════════════════════════════════════════════════════════════
    def _toggle_preview(self, url, btn):
        if self._preview_playing:
            self._stop_preview()
            btn.configure(text="▶ PLAY", text_color=NEON_GREEN, border_color=NEON_GREEN)
        else:
            self._stop_preview()
            btn.configure(text="■ STOP", text_color=NEON_MAGENTA, border_color=NEON_MAGENTA)
            threading.Thread(target=self._play_preview, args=(url,), daemon=True).start()

    def _play_preview(self, url):
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False, prefix="muud_")
            tmp_path = tmp.name; tmp.close()
            urllib.request.urlretrieve(url, tmp_path)
            self._preview_tmp_path = tmp_path
            self._preview_playing = True
            pygame.mixer.music.load(tmp_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy() and self._preview_playing:
                pygame.time.wait(100)
            self._preview_playing = False
        except Exception:
            self._preview_playing = False
        finally:
            self._cleanup_preview_tmp()

    def _stop_preview(self):
        self._preview_playing = False
        if _HAS_PYGAME:
            try: pygame.mixer.music.stop()
            except Exception: pass
        self._cleanup_preview_tmp()

    def _cleanup_preview_tmp(self):
        if self._preview_tmp_path:
            try: pygame.mixer.music.unload()
            except Exception: pass
            try: os.unlink(self._preview_tmp_path)
            except OSError: pass
            self._preview_tmp_path = None

    # ════════════════════════════════════════════════════════════
    #  EXPLAINABILITY
    # ════════════════════════════════════════════════════════════
    def _toggle_explain(self):
        if not self.result or "recommendations" not in self.result: return
        if self._explain_visible:
            self._explain_outer.pack_forget()
            self._explain_visible = False
        else:
            self._populate_explain()
            self._explain_outer.pack(side="bottom", fill="x", pady=(4, 0))
            self._explain_visible = True

    def _populate_explain(self):
        result = self.result; q = result["query"]; top = result["recommendations"][0]
        g = q["genre"]; e = q["emotion"]
        g_sim = genre_similarity(g["fuzzy_memberships"], top["genre"].lower())
        e_sim = emotion_similarity((e["valence"], e["arousal"]), (top["valence"], top["arousal"]))
        alpha, beta = 0.4, 0.6
        final = alpha * g_sim + beta * e_sim

        t = self._explain_text; t.config(state="normal"); t.delete("1.0", "end")
        t.insert("end", "  FUSION FORMULA\n", "heading")
        t.insert("end", "  score = ", "dim"); t.insert("end", "α", "highlight")
        t.insert("end", " × genre_sim  +  ", "dim"); t.insert("end", "β", "highlight")
        t.insert("end", " × emotion_sim\n\n", "dim")
        t.insert("end", f"  α (w_genre)   = ", "dim"); t.insert("end", f"{alpha:.2f}\n", "formula")
        t.insert("end", f"  β (w_emotion) = ", "dim"); t.insert("end", f"{beta:.2f}\n\n", "formula")

        t.insert("end", "  GENRE MEMBERSHIP VECTOR  (Softmax)\n", "heading")
        for label, score in sorted(g["fuzzy_memberships"].items(), key=lambda x: -x[1]):
            bar_len = int(score * 20); bar = "█" * bar_len + "░" * (20 - bar_len)
            marker = " ◀" if label == top["genre"].lower() else ""
            t.insert("end", f"  {label:<10s} ", "dim"); t.insert("end", f"{bar}", "bar"); t.insert("end", f"  {score:.4f}", "value")
            if marker: t.insert("end", marker, "highlight")
            t.insert("end", "\n")

        t.insert("end", "\n"); t.insert("end", f"  TOP MATCH BREAKDOWN\n", "heading")
        t.insert("end", f"  \"{top['title']}\"", "value"); t.insert("end", f"  by {top['artist']}", "dim")
        t.insert("end", f"  ·  {top['genre'].upper()}\n\n", "highlight")
        t.insert("end", "  Genre Similarity\n", "highlight"); t.insert("end", f"    genre_sim(\"{top['genre']}\") = ", "dim"); t.insert("end", f"{g_sim:.4f}\n\n", "value")
        t.insert("end", "  Emotion Similarity\n", "highlight"); t.insert("end", f"    query  = (V={e['valence']:.2f}, A={e['arousal']:.2f})\n", "dim")
        t.insert("end", f"    match  = (V={top['valence']:.2f}, A={top['arousal']:.2f})\n", "dim")
        t.insert("end", f"    emo_sim = ", "dim"); t.insert("end", f"{e_sim:.4f}\n\n", "value")
        t.insert("end", "  FUSION COMPUTATION\n", "heading")
        t.insert("end", f"    {alpha:.2f}", "formula"); t.insert("end", f" × {g_sim:.4f}", "value"); t.insert("end", f"  +  ", "dim")
        t.insert("end", f"{beta:.2f}", "formula"); t.insert("end", f" × {e_sim:.4f}", "value"); t.insert("end", f"\n    = ", "dim"); t.insert("end", f"{final:.4f}\n", "value")
        t.config(state="disabled")

    # ════════════════════════════════════════════════════════════
    #  FRAME SWITCHING
    # ════════════════════════════════════════════════════════════
    def _show_analysis_frame(self):
        self._recommend_frame.pack_forget()
        self._analysis_frame.pack(fill="both", expand=True)
        self.explain_btn.configure(state="disabled")
        self._explain_visible = False

    def _show_recommend_frame(self):
        self._analysis_frame.pack_forget()
        self._recommend_frame.pack(fill="both", expand=True)

    # ════════════════════════════════════════════════════════════
    #  V-A PLOT (Original Styling, Large Size)
    # ════════════════════════════════════════════════════════════
    def _build_va_plot(self, parent):
        fig = Figure(figsize=(4.2, 4.0), dpi=96, facecolor=BG_PANEL)
        ax = fig.add_subplot(111)
        ax.set_facecolor(BG_INPUT)
        ax.set_xlim(1, 9)
        ax.set_ylim(1, 9)
        ax.set_xlabel("Valence →", color=NEON_CYAN, fontsize=9, fontfamily="Consolas")
        ax.set_ylabel("Arousal →", color=NEON_CYAN, fontsize=9, fontfamily="Consolas")
        ax.set_xticks(range(1, 10))
        ax.set_yticks(range(1, 10))

        ax.grid(True, color=BORDER, linewidth=0.5, alpha=0.6)
        ax.axhline(y=5, color=NEON_CYAN, linewidth=0.8, alpha=0.35, linestyle="--")
        ax.axvline(x=5, color=NEON_CYAN, linewidth=0.8, alpha=0.35, linestyle="--")

        ax.tick_params(colors=TEXT_DIM, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(BORDER)

        kw = dict(fontsize=8, fontfamily="Consolas", alpha=0.55, ha="center", va="center", weight="bold")
        ax.text(7, 7, "Happy /\nEnergetic", color=NEON_GREEN,   **kw)
        ax.text(7, 3, "Happy /\nCalm",      color=NEON_CYAN,    **kw)
        ax.text(3, 3, "Sad /\nCalm",        color=NEON_MAGENTA, **kw)
        ax.text(3, 7, "Angry /\nIntense",   color=NEON_YELLOW,  **kw)
        fig.tight_layout(pad=1.5)

        self._va_fig = fig
        self._va_ax = ax
        self._va_artists = []

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        self._va_canvas = canvas
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _update_va_plot(self, valence, arousal, confidence=0.5):
        ax = self._va_ax
        conf = max(0.1, min(float(confidence), 1.0))

        for art in self._va_artists:
            art.remove()
        self._va_artists.clear()

        core_size = 200 * conf
        for scale, alpha in [(4.0, 0.08), (2.5, 0.16), (1.6, 0.28)]:
            d = ax.scatter(valence, arousal, s=core_size * scale, c=NEON_MAGENTA, alpha=alpha * conf, zorder=5, edgecolors="none")
            self._va_artists.append(d)

        d = ax.scatter(valence, arousal, s=core_size, c=NEON_MAGENTA, alpha=0.5 + 0.45 * conf, zorder=6, edgecolors=NEON_CYAN, linewidths=1.2)
        self._va_artists.append(d)

        lbl = ax.annotate(f"({valence:.1f}, {arousal:.1f})  {conf:.0%}", (valence, arousal), textcoords="offset points", xytext=(12, 10), fontsize=8, fontfamily="Consolas", color=NEON_GREEN, alpha=0.9, arrowprops=dict(arrowstyle="->", color=TEXT_DIM, lw=0.7))
        self._va_artists.append(lbl)
        self._va_canvas.draw_idle()

    # ════════════════════════════════════════════════════════════
    #  HELPERS
    # ════════════════════════════════════════════════════════════
    def _clear_results(self):
        self.results_text.config(state="normal")
        self.results_text.delete("1.0", "end")
        self.results_text.config(state="disabled")

    def _disable_buttons(self):
        for attr in ("analyze_btn", "recommend_btn", "explain_btn", "record_btn", "live_btn"): getattr(self, attr).configure(state="disabled")

    def _enable_buttons(self):
        for attr in ("analyze_btn", "recommend_btn", "record_btn", "live_btn"): getattr(self, attr).configure(state="normal")

    def _run_in_thread(self, target):
        threading.Thread(target=target, daemon=True).start()

    # ════════════════════════════════════════════════════════════
    #  MICROPHONE RECORDING
    # ════════════════════════════════════════════════════════════
    def _run_record(self):
        self._run_in_thread(self._do_record)

    def _do_record(self):
        SR = 22050; DURATION = 5
        self._disable_buttons(); self._cleanup_temp(); self._blink_on = True; self._blink_recording()
        try:
            audio = sd.rec(int(DURATION * SR), samplerate=SR, channels=1, dtype="float32")
            sd.wait()
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix="muud_rec_")
            tmp_path = tmp.name; tmp.close()
            wav_io.write(tmp_path, SR, audio)
            self._temp_recording = tmp_path
            self.file_var.set("♪  [mic recording]")
            self._selected_path = tmp_path
            self._stop_blink(); self.status_var.set("[ ANALYZING RECORDING … ]")
            result = self.recommender.analyze(tmp_path)
            self.result = result
            self._show_analysis(result)
            self.status_var.set("[ RECORDING ANALYSIS COMPLETE ]")
        except Exception as e:
            self._stop_blink(); messagebox.showerror("Recording Error", str(e)); self.status_var.set("[ RECORDING ERROR ]")
        finally:
            self._enable_buttons(); self.explain_btn.configure(state="normal")

    def _blink_recording(self):
        if not self._blink_on: return
        current = self.status_var.get()
        if "●" in current: self.status_var.set("[   RECORDING … ]")
        else: self.status_var.set("[ ● RECORDING … ]")
        self._blink_job = self.root.after(500, self._blink_recording)

    def _stop_blink(self):
        self._blink_on = False
        if self._blink_job is not None:
            self.root.after_cancel(self._blink_job); self._blink_job = None

    # ════════════════════════════════════════════════════════════
    #  LIVE MICROPHONE 
    # ════════════════════════════════════════════════════════════
    def _build_live_spectrogram(self, parent):
        fig = Figure(figsize=(4.2, 3.2), dpi=96, facecolor=BG_PANEL)
        ax = fig.add_subplot(111)
        ax.set_facecolor(BG_INPUT)
        ax.set_xlabel("Time (frames)", color=NEON_CYAN, fontsize=8, fontfamily="Consolas")
        ax.set_ylabel("Mel bin", color=NEON_CYAN, fontsize=8, fontfamily="Consolas")
        ax.tick_params(colors=TEXT_DIM, labelsize=7)
        for spine in ax.spines.values(): spine.set_color(BORDER)
        self._live_spec_data = np.zeros((128, 200), dtype=np.float32)
        self._live_im = ax.imshow(self._live_spec_data, aspect="auto", origin="lower", cmap="magma", vmin=-3, vmax=3, interpolation="nearest")
        fig.tight_layout(pad=1.0)
        self._live_fig = fig; self._live_ax = ax
        self._live_canvas = FigureCanvasTkAgg(fig, master=parent)
        self._live_canvas.draw()
        self._live_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _toggle_live_mic(self):
        if self._live_active: self._stop_live_mic()
        else: self._start_live_mic()

    def _start_live_mic(self):
        self._live_active = True; self._live_buffer.clear()
        self._live_spec_data = np.zeros((128, 200), dtype=np.float32)
        self.recommender.genre_clf.clear_live_history()
        self.live_btn.configure(text="■ STOP MIC", text_color="#ff3333", border_color="#ff3333")
        self._disable_buttons(); self.live_btn.configure(state="normal")
        self._live_panel.pack(fill="both", expand=True, pady=(6, 0))
        self.status_var.set("[ ● LIVE MIC ACTIVE — listening … ]")
        self._live_info_var.set("  Buffering audio …")
        try:
            self._live_stream = sd.InputStream(samplerate=self._live_sr, channels=1, dtype="float32", blocksize=1024, callback=self._mic_callback)
            self._live_stream.start()
        except Exception as e:
            messagebox.showerror("Mic Error", str(e)); self._stop_live_mic(); return
        self._live_spec_job = self.root.after(150, self._update_live_spectrogram)
        self._live_inference_job = self.root.after(6000, self._schedule_live_inference)

    def _stop_live_mic(self):
        self._live_active = False
        if self._live_stream is not None:
            try: self._live_stream.stop(); self._live_stream.close()
            except Exception: pass
            self._live_stream = None
        if self._live_spec_job is not None:
            self.root.after_cancel(self._live_spec_job); self._live_spec_job = None
        if self._live_inference_job is not None:
            self.root.after_cancel(self._live_inference_job); self._live_inference_job = None
        self.live_btn.configure(text="🎤 LIVE MIC", text_color="#ff6600", border_color="#ff6600")
        self._live_panel.pack_forget()
        self._enable_buttons()
        if self._selected_path: self.analyze_btn.configure(state="normal"); self.recommend_btn.configure(state="normal")
        else: self.analyze_btn.configure(state="disabled"); self.recommend_btn.configure(state="disabled")
        self.explain_btn.configure(state="normal" if self.result and "recommendations" in self.result else "disabled")
        self.status_var.set("[ LIVE MIC STOPPED ]")
        self._live_info_var.set("")

    def _mic_callback(self, indata, frames, time_info, status):
        self._live_buffer.extend(indata[:, 0])

    def _update_live_spectrogram(self):
        if not self._live_active: return
        buf = np.array(self._live_buffer, dtype=np.float32)
        if len(buf) > 2048:
            tail = buf[-self._live_sr * 3:]
            try:
                mel = librosa.feature.melspectrogram(y=tail, sr=self._live_sr, n_mels=128, n_fft=2048, hop_length=512)
                mel_db = librosa.power_to_db(mel, ref=np.max); std = mel_db.std()
                if std > 0: mel_db = (mel_db - mel_db.mean()) / std
                n_new = mel_db.shape[1]
                if n_new >= 200: self._live_spec_data = mel_db[:, -200:]
                else: self._live_spec_data = np.roll(self._live_spec_data, -n_new, axis=1); self._live_spec_data[:, -n_new:] = mel_db
                self._live_im.set_data(self._live_spec_data)
                self._live_canvas.draw_idle()
            except Exception: pass
        self._live_spec_job = self.root.after(200, self._update_live_spectrogram)

    def _schedule_live_inference(self):
        if not self._live_active: return
        buf = np.array(self._live_buffer, dtype=np.float32)
        if len(buf) >= self._live_sr * 3:
            threading.Thread(target=self._do_live_inference, args=(buf.copy(),), daemon=True).start()
        self._live_inference_job = self.root.after(7000, self._schedule_live_inference)

    def _do_live_inference(self, signal):
        try:
            result = self.recommender.analyze_signal(signal, self._live_sr, "live mic")
            g = result["genre"]; e = result["emotion"]
            info = f"  Genre: {g['top_genre'].upper()}  ({g['confidence']:.0%})   ·   Mood: {e['mood_label']}  (V={e['valence']:.1f}  A={e['arousal']:.1f})"
            self.root.after(0, self._live_info_var.set, info)
            self.root.after(0, lambda v=e["valence"], a=e["arousal"], c=g["confidence"]: self._update_va_plot(v, a, c))
            self.root.after(0, self.status_var.set, f"[ ● LIVE MIC — {g['top_genre'].upper()} ]")
            self.root.after(0, self._show_live_analysis, result)
        except Exception as exc:
            self.root.after(0, self._live_info_var.set, f"  Inference error: {exc}")

    def _show_live_analysis(self, result):
        self.result = result; self._show_analysis(result)

    def run(self):
        self.root.mainloop()
