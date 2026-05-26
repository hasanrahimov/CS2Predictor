#CS2-styled overlay window for the round predictor.


import tkinter as tk
from tkinter import font as tkfont

#CS2 palette
BG_DARK = "#15191d"  #HUD panel background
BG_PANEL = "#1f262d"  #secondary panel
BG_BAR = "#0a0d10"  #bar track
FG_MAIN = "#e4e4e4"  #primary text
FG_MUTED = "#8c98a5"  #secondary text
FG_ACCENT = "#f2b233"  #CS2 yellow accent (match type, version tag)

CT_COLOR = "#6cb8ec"  #CT blue (matches CS2 scoreboard)
CT_DARK = "#3a7cb8"
T_COLOR = "#eba200"  #T yellow/orange (CS2 scoreboard)
T_DARK = "#b37a00"

BORDER = "#2a343d"

class PredictorGUI:
    def __init__(self, model_version: str = "v6"):
        self.version = model_version

        self.root = tk.Tk()
        self.root.title("CS2 Round Predictor")
        self.root.overrideredirect(True)  # borderless
        self.root.attributes("-topmost", True)  # always on top
        self.root.attributes("-alpha", 0.94)  # slight transparency
        self.root.configure(bg=BG_DARK)

        #Position
        self.W, self.H = 420, 260
        sw = self.root.winfo_screenwidth()
        self.root.geometry(f"{self.W}x{self.H}+{sw - self.W - 20}+20")

        #Fonts
        self.f_title = tkfont.Font(family="Bahnschrift", size=11, weight="bold")
        self.f_big = tkfont.Font(family="Bahnschrift", size=22, weight="bold")
        self.f_mid = tkfont.Font(family="Bahnschrift", size=11, weight="bold")
        self.f_small = tkfont.Font(family="Segoe UI", size=9)

        self._build()
        self._bind_drag()
        self.root.bind("<Escape>", lambda e: self.close())

    #Layout
    def _build(self):
        #Outer border
        outer = tk.Frame(self.root, bg=BORDER)
        outer.pack(fill="both", expand=True, padx=0, pady=0)
        inner = tk.Frame(outer, bg=BG_DARK)
        inner.pack(fill="both", expand=True, padx=1, pady=1)

        #Title strip
        self.title_bar = tk.Frame(inner, bg=BG_PANEL, height=26)
        self.title_bar.pack(fill="x", side="top")
        self.title_bar.pack_propagate(False)

        tk.Label(
            self.title_bar,
            text="CS2 ROUND PREDICTOR",
            bg=BG_PANEL,
            fg=FG_MAIN,
            font=self.f_title,
        ).pack(side="left", padx=10)
        tk.Label(
            self.title_bar,
            text=f"model {self.version}",
            bg=BG_PANEL,
            fg=FG_ACCENT,
            font=self.f_small,
        ).pack(side="left")

        close_btn = tk.Label(
            self.title_bar,
            text="×",
            bg=BG_PANEL,
            fg=FG_MUTED,
            font=tkfont.Font(family="Segoe UI", size=14, weight="bold"),
            cursor="hand2",
            padx=10,
        )
        close_btn.pack(side="right")
        close_btn.bind("<Button-1>", lambda e: self.close())
        close_btn.bind("<Enter>", lambda e: close_btn.config(fg="#ff6666"))
        close_btn.bind("<Leave>", lambda e: close_btn.config(fg=FG_MUTED))
        self.timer_lbl = tk.Label(
            self.title_bar,
            text="⏱ --:--",
            bg=BG_PANEL,
            fg=FG_MUTED,
            font=self.f_small,
            padx=6,
        )
        self.timer_lbl.pack(side="right")

        #Probability row
        prob_row = tk.Frame(inner, bg=BG_DARK)
        prob_row.pack(fill="x", pady=(12, 4), padx=16)

        self.ct_label = tk.Label(
            prob_row, text="CT", bg=BG_DARK, fg=CT_COLOR, font=self.f_mid
        )
        self.ct_label.pack(side="left")
        self.ct_pct = tk.Label(
            prob_row, text="--.-%", bg=BG_DARK, fg=CT_COLOR, font=self.f_big
        )
        self.ct_pct.pack(side="left", padx=(6, 0))

        self.t_pct = tk.Label(
            prob_row, text="--.-%", bg=BG_DARK, fg=T_COLOR, font=self.f_big
        )
        self.t_pct.pack(side="right")
        self.t_label = tk.Label(
            prob_row, text="T", bg=BG_DARK, fg=T_COLOR, font=self.f_mid
        )
        self.t_label.pack(side="right", padx=(0, 6))

        #Split bar
        self.bar_canvas = tk.Canvas(
            inner, height=14, bg=BG_BAR, highlightthickness=0, bd=0
        )
        self.bar_canvas.pack(fill="x", padx=16, pady=(2, 14))
        self._bar_ct_rect = None
        self._bar_t_rect = None

        #Key factors header
        hdr = tk.Frame(inner, bg=BG_DARK)
        hdr.pack(fill="x", padx=16)
        tk.Frame(hdr, bg=FG_ACCENT, width=3, height=14).pack(side="left")
        tk.Label(
            hdr, text="LATEST FACTORS", bg=BG_DARK, fg=FG_MUTED, font=self.f_title
        ).pack(side="left", padx=8)

        #Top-3 rows
        self.factor_rows = []
        for _ in range(3):
            row = tk.Frame(inner, bg=BG_DARK)
            row.pack(fill="x", padx=16, pady=3)

            side_tag = tk.Label(row, text="  ", bg=BG_PANEL, width=3, font=self.f_small)
            side_tag.pack(side="left")
            name = tk.Label(
                row, text="--", bg=BG_DARK, fg=FG_MAIN, font=self.f_small, anchor="w"
            )
            name.pack(side="left", fill="x", expand=True, padx=8)
            val = tk.Label(row, text="", bg=BG_DARK, fg=FG_MUTED, font=self.f_small)
            val.pack(side="right")

            self.factor_rows.append((side_tag, name, val))

        #Status bar
        self.status = tk.Label(
            inner,
            text="Waiting for CS2...",
            bg=BG_DARK,
            fg=FG_MUTED,
            font=self.f_small,
            anchor="w",
        )
        self.status.pack(fill="x", side="bottom", padx=16, pady=(0, 8))

    #Window dragging
    def _bind_drag(self):
        def start(e):
            self._dx, self._dy = e.x, e.y

        def drag(e):
            x = self.root.winfo_x() + e.x - self._dx
            y = self.root.winfo_y() + e.y - self._dy
            self.root.geometry(f"+{x}+{y}")

        for w in (self.title_bar,):
            w.bind("<Button-1>", start)
            w.bind("<B1-Motion>", drag)

    #Public API
    def update(self, result: dict):
        ct, t = result["ct_pct"], result["t_pct"]
        self.ct_pct.config(text=f"{ct:.1f}%")
        self.t_pct.config(text=f"{t:.1f}%")

        #Redraw the split bar
        self.bar_canvas.delete("all")
        self.bar_canvas.update_idletasks()
        w = max(self.bar_canvas.winfo_width(), 1)
        h = int(self.bar_canvas["height"])
        ct_w = int(round(w * ct / 100.0))

        #Subtle gradient effect
        self.bar_canvas.create_rectangle(0, 0, ct_w, h, fill=CT_COLOR, outline="")
        self.bar_canvas.create_rectangle(0, h - 3, ct_w, h, fill=CT_DARK, outline="")
        self.bar_canvas.create_rectangle(ct_w, 0, w, h, fill=T_COLOR, outline="")
        self.bar_canvas.create_rectangle(ct_w, h - 3, w, h, fill=T_DARK, outline="")
        #Divider
        self.bar_canvas.create_line(ct_w, 0, ct_w, h, fill=BG_DARK, width=2)

        #Top-3 factors
        for i, (tag, name, val) in enumerate(self.factor_rows):
            if i < len(result["top3"]):
                f = result["top3"][i]
                is_t = f["direction"] == "T"
                tag.config(
                    bg=T_COLOR if is_t else CT_COLOR,
                    text="T" if is_t else "CT",
                    fg="#15191d",
                )
                name.config(text=f["feature"].replace("_", " "))
                text = f.get("value_str")
                if text is None:
                    try:
                        text = f"{float(f['value']):.2f}"
                    except (TypeError, ValueError):
                        text = str(f["value"])
                val.config(text=text)
            else:
                tag.config(bg=BG_PANEL, text="  ")
                name.config(text="--")
                val.config(text="")

        # Timer display
        bomb = result.get("bomb_planted", False)
        if bomb:
            bcd = max(0.0, float(result.get("bomb_countdown", 0)))
            self.timer_lbl.config(text=f"💣 {bcd:.1f}s", fg="#ff6666")
        else:
            secs = max(0, int(round(float(result.get("round_time_remaining", 0)))))
            self.timer_lbl.config(
                text=f"⏱ {secs // 60}:{secs % 60:02d}", fg=FG_MUTED
            )
            
        self.status.config(text="Live -- updating every snapshot", fg=FG_MUTED)
        self.root.update_idletasks()
        self.root.update()

    def set_status(self, text: str, color: str = FG_MUTED):
        self.status.config(text=text, fg=color)
        self.root.update_idletasks()
        self.root.update()

    def pump(self):
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            pass

    def close(self):
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    @property
    def alive(self) -> bool:
        try:
            return bool(self.root.winfo_exists())
        except tk.TclError:
            return False
