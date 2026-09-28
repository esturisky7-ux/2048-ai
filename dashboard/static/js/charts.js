/* Canvas line charts with a hover read-out.
 *
 * Moving the pointer over a chart snaps to the nearest sample and reports the
 * real value and the training game number it came from; arrow keys do the
 * same for keyboard users. Colours come from the theme's CSS custom
 * properties (series name them as "var(--chart-2)"), so the same chart
 * redraws correctly in either theme. No charting library, no dependency.
 */

/* A theme token, resolved to a colour string the canvas understands. */
function themeColor(value) {
  const m = /^var\((--[\w-]+)\)$/.exec(String(value || "").trim());
  if (!m) return value;
  return getComputedStyle(document.documentElement).getPropertyValue(m[1]).trim();
}

/* Series colours by meaning, from the chart ramp (--chart-1 … --chart-5). */
const SERIES_COLORS = {
  mean: "var(--chart-2)", median: "var(--chart-4)",
  eval: "var(--chart-4)", evalBand: "var(--chart-1)",
  tile: "var(--chart-3)",
  r512: "var(--chart-1)", r1024: "var(--chart-2)", r2048: "var(--chart-3)",
  r4096: "var(--chart-4)", r8192: "var(--chart-5)",
  games: "var(--chart-2)", speed: "var(--chart-4)", cumulative: "var(--chart-3)",
};

/* n colours spread evenly over the ramp: three series get 1, 3 and 5. */
function spreadColors(n) {
  if (n <= 1) return ["var(--chart-3)"];
  return Array.from({ length: n }, (_, i) =>
    `var(--chart-${n > 5 ? (i % 5) + 1 : Math.round(1 + (i * 4) / (n - 1))})`);
}

function compactNumber(v) {
  const a = Math.abs(v);
  if (a >= 1e6) return (v / 1e6).toFixed(a >= 1e7 ? 0 : 1).replace(/\.0$/, "") + "M";
  if (a >= 1e4) return Math.round(v / 1e3) + "k";
  if (a >= 1e3) return (v / 1e3).toFixed(1).replace(/\.0$/, "") + "k";
  if (a >= 10 || v === 0) return String(Math.round(v));
  return v.toFixed(1);
}

function axisLabel(v, unit) {
  if (unit === "hours") {
    if (v < 1) return `${Math.round(v * 60)}m`;
    return (v < 10 ? v.toFixed(1).replace(/\.0$/, "") : Math.round(v)) + "h";
  }
  return compactNumber(v);
}

function valueLabel(v, opts, full = false) {
  if (opts.tileScale) return v >= 1 ? F.n(Math.pow(2, Math.round(v))) : "0";
  if (opts.percent) return v.toFixed(v < 10 ? 1 : 0) + "%";
  if (full) return Math.abs(v) >= 100 ? F.n(Math.round(v)) : F.n(v, v % 1 ? 2 : 0);
  return compactNumber(v);
}

/* A round step at or above v, so an axis reads 0 / 6k / 12k / 18k … */
function niceStep(v) {
  if (!(v > 0)) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  for (const m of [1, 2, 2.5, 3, 4, 5, 6, 8, 10]) {
    if (m * p >= v * 0.999) return m * p;
  }
  return 10 * p;
}

/* A chart instance keeps its geometry so hover can map pixels back to data. */
class Chart {
  constructor(canvas, series, opts = {}) {
    this.canvas = canvas;
    this.series = series;
    this.opts = opts;
    this.geom = null;
    this.hover = null;
    this._bind();
    this.draw();
  }

  update(series, opts) {
    this.series = series;
    if (opts) this.opts = { ...this.opts, ...opts };
    this.hover = null;
    this.tip.classList.remove("show");
    this.draw();
  }

  static redrawAll() {
    $$(".chart").forEach((node) => node._chart && node._chart.draw());
  }

  _bind() {
    const box = this.canvas.parentElement;
    this.tip = el("div", { class: "chart-tip", "aria-hidden": "true" });
    box.append(this.tip);

    const move = (clientX, clientY) => {
      const rect = this.canvas.getBoundingClientRect();
      this._onHover(clientX - rect.left, clientY - rect.top);
    };
    this.canvas.addEventListener("pointermove", (e) => move(e.clientX, e.clientY));
    this.canvas.addEventListener("pointerleave", () => this._clearHover());
    this.canvas.addEventListener("blur", () => this._clearHover());
    // Keyboard users get the same read-out by tabbing to the canvas.
    this.canvas.tabIndex = 0;
    this.canvas.addEventListener("keydown", (e) => {
      if (!this.geom) return;
      const n = this.geom.count;
      if (!n) return;
      let i = this.hover === null ? n - 1 : this.hover;
      if (e.key === "ArrowRight") i = Math.min(n - 1, i + 1);
      else if (e.key === "ArrowLeft") i = Math.max(0, i - 1);
      else return;
      e.preventDefault();
      this.hover = i;
      this.draw();
      this._showTip(this.geom.X(this.geom.xs[i]), 24);
    });
    // Redraw whenever the canvas changes size: window resizes, the sidebar
    // collapsing, a card reflowing.
    if (window.ResizeObserver) {
      let frame = 0;
      new ResizeObserver(() => {
        cancelAnimationFrame(frame);
        frame = requestAnimationFrame(() => this.draw());
      }).observe(this.canvas);
    }
  }

  _clearHover() {
    if (this.hover === null) return;
    this.hover = null;
    this.tip.classList.remove("show");
    this.draw();
  }

  _onHover(px, py) {
    if (!this.geom || !this.geom.count) return;
    const { xs, X } = this.geom;
    let best = 0, bestD = Infinity;
    for (let i = 0; i < xs.length; i++) {
      const d = Math.abs(X(xs[i]) - px);
      if (d < bestD) { bestD = d; best = i; }
    }
    if (bestD > 40) { this._clearHover(); return; }
    this.hover = best;
    this.draw();
    this._showTip(X(xs[best]), py);
  }

  _showTip(px, py) {
    const i = this.hover;
    if (i === null || !this.geom) return;
    const { xs } = this.geom;
    const xv = xs[i];
    const rows = [el("div", { class: "tip-head" },
      `${this.opts.xLabel || "games"} ${this.opts.xUnit === "hours"
        ? axisLabel(xv, "hours") : F.n(xv)}`)];
    for (const s of this.geom.lines) {
      if (!s.y || s.y[i] === undefined || s.y[i] === null) continue;
      rows.push(el("div", { class: "row" },
        el("span", { class: "swatch", style: `background:${s.color}` }),
        el("span", { class: "k" }, s.label || ""),
        el("b", {}, s.format ? s.format(s.y[i]) : valueLabel(s.y[i], this.opts, true))));
      if (s.band && s.band[0][i] !== undefined) {
        rows.push(el("div", { class: "row" },
          el("span", { class: "swatch", style: `background:${s.bandColor || s.color};opacity:.5` }),
          el("span", { class: "k" }, "95% CI"),
          el("b", {}, `${valueLabel(s.band[0][i], this.opts)} – ${valueLabel(s.band[1][i], this.opts)}`)));
      }
    }
    this.tip.replaceChildren(...rows);
    this.tip.classList.add("show");
    const w = this.tip.offsetWidth, h = this.tip.offsetHeight;
    const boxW = this.canvas.clientWidth, boxH = this.canvas.clientHeight;
    let left = px + 14;
    if (left + w > boxW) left = Math.max(0, px - w - 14);
    this.tip.style.left = `${left}px`;
    this.tip.style.top = `${Math.max(0, Math.min(py - h / 2, boxH - h))}px`;
  }

  draw() {
    const canvas = this.canvas;
    const opts = this.opts;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const muted = themeColor("var(--muted-foreground)");
    const grid = themeColor("var(--border)");
    const surface = themeColor("var(--card)");
    const mono = `11px ${themeColor("var(--font-mono)") || "monospace"}`;
    const sans = `12px ${themeColor("var(--font-sans)") || "sans-serif"}`;

    const lines = this.series
      .filter((s) => s.y && s.y.length)
      .map((s) => ({ ...s, color: themeColor(s.color),
                     bandColor: themeColor(s.bandColor || s.color) }));
    if (!lines.length) {
      ctx.fillStyle = muted;
      ctx.font = sans;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(opts.emptyText || "No data yet", w / 2, h / 2);
      this.geom = null;
      return;
    }

    let xmin = Infinity, xmax = -Infinity, ymin = 0, ymax = -Infinity;
    for (const s of lines) {
      for (let i = 0; i < s.y.length; i++) {
        const x = s.x[i];
        if (x < xmin) xmin = x;
        if (x > xmax) xmax = x;
        const hi = s.band ? Math.max(s.y[i], s.band[1][i] ?? -Infinity) : s.y[i];
        if (hi > ymax) ymax = hi;
        if (opts.allowNegative && s.y[i] < ymin) ymin = s.y[i];
      }
    }
    if (xmax === xmin) { xmin -= 0.5; xmax += 0.5; }

    // Five y ticks on round numbers; tile charts step in whole powers of two.
    const ticks = [];
    if (opts.tileScale) {
      ymax = Math.max(ymin + 1, Math.ceil(ymax));
      const step = Math.max(1, Math.ceil((ymax - ymin) / 4));
      ymax = ymin + step * Math.ceil((ymax - ymin) / step);
      for (let v = ymin; v <= ymax + 1e-9; v += step) ticks.push(v);
    } else {
      const top = opts.percent ? Math.min(100, Math.max(ymax * 1.05, 5)) : ymax * 1.05;
      const step = niceStep(Math.max(top - ymin, 1e-9) / 4);
      ymax = ymin + step * 4;
      for (let i = 0; i <= 4; i++) ticks.push(ymin + step * i);
    }

    ctx.font = mono;
    const labelW = Math.max(...ticks.map((v) => ctx.measureText(valueLabel(v, opts)).width));
    const pad = { l: Math.ceil(labelW) + 12, r: 8, t: 10, b: 26 };
    const plotW = Math.max(10, w - pad.l - pad.r), plotH = Math.max(10, h - pad.t - pad.b);
    const X = (v) => pad.l + ((v - xmin) / (xmax - xmin)) * plotW;
    const Y = (v) => pad.t + plotH - ((v - ymin) / (ymax - ymin)) * plotH;

    // Grid and axes.
    ctx.lineWidth = 1;
    ctx.strokeStyle = grid;
    ctx.fillStyle = muted;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const v of ticks) {
      const y = Math.round(Y(v)) + 0.5;
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
      ctx.fillText(valueLabel(v, opts), pad.l - 8, y);
    }
    // X ticks on round numbers too, as many as fit.
    ctx.textBaseline = "alphabetic";
    const xstep = niceStep((xmax - xmin) / Math.max(3, Math.min(6, Math.floor(plotW / 110))));
    for (let v = Math.ceil(xmin / xstep - 1e-9) * xstep; v <= xmax + 1e-9; v += xstep) {
      const label = axisLabel(v, opts.xUnit);
      const half = ctx.measureText(label).width / 2;
      ctx.textAlign = "center";
      ctx.fillText(label, Math.max(pad.l + half, Math.min(w - pad.r - half, X(v))), h - 8);
    }

    // Series.
    for (const s of lines) {
      if (s.band) {
        ctx.fillStyle = s.bandColor;
        ctx.globalAlpha = 0.14;
        ctx.beginPath();
        for (let i = 0; i < s.x.length; i++) ctx[i ? "lineTo" : "moveTo"](X(s.x[i]), Y(s.band[0][i]));
        for (let i = s.x.length - 1; i >= 0; i--) ctx.lineTo(X(s.x[i]), Y(s.band[1][i]));
        ctx.closePath(); ctx.fill();
        ctx.globalAlpha = 1;
        ctx.strokeStyle = s.bandColor;
        ctx.lineWidth = 1.25;
        for (const edge of s.band) {
          ctx.beginPath();
          for (let i = 0; i < s.x.length; i++) ctx[i ? "lineTo" : "moveTo"](X(s.x[i]), Y(edge[i]));
          ctx.stroke();
        }
      }
      ctx.strokeStyle = s.color;
      ctx.lineWidth = s.width || 2;
      ctx.lineJoin = "round"; ctx.lineCap = "round";
      ctx.setLineDash(s.dash || []);
      ctx.beginPath();
      for (let i = 0; i < s.y.length; i++) {
        ctx[i ? "lineTo" : "moveTo"](X(s.x[i]), Y(s.y[i]));
      }
      ctx.stroke();
      ctx.setLineDash([]);
      if (s.points || s.y.length === 1) {
        ctx.fillStyle = s.color;
        for (let i = 0; i < s.y.length; i++) {
          ctx.beginPath(); ctx.arc(X(s.x[i]), Y(s.y[i]), 3, 0, 7); ctx.fill();
        }
      }
    }

    // Hover marker.
    const longest = lines.reduce((a, b) => (b.x.length > a.x.length ? b : a), lines[0]);
    this.geom = { X, Y, xs: longest.x, count: longest.x.length, lines };
    if (this.hover !== null && this.hover < longest.x.length) {
      const hx = Math.round(X(longest.x[this.hover])) + 0.5;
      ctx.strokeStyle = muted;
      ctx.globalAlpha = 0.6;
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(hx, pad.t); ctx.lineTo(hx, pad.t + plotH); ctx.stroke();
      ctx.globalAlpha = 1;
      for (const s of lines) {
        const v = s.y[this.hover];
        if (v === undefined || v === null) continue;
        ctx.fillStyle = s.color;
        ctx.beginPath(); ctx.arc(hx, Y(v), 4, 0, 7); ctx.fill();
        ctx.strokeStyle = surface; ctx.lineWidth = 2; ctx.stroke();
      }
    }
  }
}

/* A chart with its legend underneath; the node carries the Chart as _chart. */
function chartView(series, opts = {}) {
  const canvas = el("canvas", { role: "img", "aria-label": opts.label || "chart" });
  const legend = el("div", { class: "chart-legend" });
  const node = el("div", { class: "chart", style: `--chart-h:${opts.height || 260}px` },
    canvas, legend);
  node._legend = legend;
  paintLegend(node, series, opts);
  // The canvas needs to be in the document, with a size, before it can draw.
  requestAnimationFrame(() => { node._chart = new Chart(canvas, series, opts); });
  return node;
}

function paintLegend(node, series, opts = {}) {
  const items = [];
  for (const s of series) {
    if (!s.label || opts.legend === false) continue;
    items.push(el("span", {}, el("span", { class: "swatch", style: `background:${s.color}` }), s.label));
    if (s.band && s.bandLabel) {
      items.push(el("span", {},
        el("span", { class: "swatch", style: `background:${s.bandColor || s.color}` }), s.bandLabel));
    }
  }
  node._legend.replaceChildren(...items);
  node._legend.hidden = !items.length;
}

function updateChart(node, series, opts) {
  if (!node) return;
  paintLegend(node, series, { ...(node._chart?.opts || {}), ...(opts || {}) });
  if (opts?.label) node.querySelector("canvas").setAttribute("aria-label", opts.label);
  if (node._chart) node._chart.update(series, opts);
  else requestAnimationFrame(() => node._chart && node._chart.update(series, opts));
}

// Tick labels are drawn in Geist Mono; redraw once the font has loaded.
if (document.fonts && document.fonts.ready) {
  document.fonts.ready.then(() => Chart.redrawAll()).catch(() => { });
}
