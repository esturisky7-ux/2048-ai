/* Canvas charts with hover read-out.
 *
 * Evolved from the original dashboard renderer, with the interaction the
 * control center needs: moving the pointer over a chart snaps to the nearest
 * sample and reports the real value and the training game number it came from.
 * No charting library, no dependency, ~300 lines.
 */

const css = (v) =>
  getComputedStyle(document.documentElement).getPropertyValue(v).trim();

const SERIES_COLORS = {
  mean: "#edc22e", median: "#5ac8fa", eval: "#3ddc84", tile: "#edc22e",
  r512: "#9aa3b2", r1024: "#5ac8fa", r2048: "#edc22e", r4096: "#ff9f43",
  r8192: "#a78bfa", speed: "#5ac8fa", games: "#3ddc84",
};

function axisLabel(v, unit) {
  if (unit === "games") {
    if (v >= 1e6) return (v / 1e6).toFixed(1) + "M";
    if (v >= 1000) return (v / 1000).toFixed(v >= 10000 ? 0 : 1) + "k";
    return String(Math.round(v));
  }
  if (unit === "hours") return v.toFixed(v < 10 ? 1 : 0) + "h";
  return F.n(Math.round(v));
}

function valueLabel(v, opts) {
  if (opts.tileScale) return v >= 1 ? F.n(Math.pow(2, Math.round(v))) : "0";
  if (opts.percent) return v.toFixed(v < 10 ? 1 : 0) + "%";
  if (v >= 10000) return (v / 1000).toFixed(0) + "k";
  return F.n(Math.round(v));
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
    this.draw();
  }

  _bind() {
    if (this.canvas._chartBound) return;
    this.canvas._chartBound = true;
    const box = this.canvas.closest(".chart-box") || this.canvas.parentElement;
    let tip = box.querySelector(".chart-tip");
    if (!tip) {
      tip = el("div", { class: "chart-tip" });
      box.style.position = "relative";
      box.append(tip);
    }
    this.tip = tip;

    const move = (clientX, clientY) => {
      const rect = this.canvas.getBoundingClientRect();
      this._onHover(clientX - rect.left, clientY - rect.top, rect);
    };
    this.canvas.addEventListener("pointermove", (e) => move(e.clientX, e.clientY));
    this.canvas.addEventListener("pointerleave", () => {
      this.hover = null;
      this.tip.classList.remove("show");
      this.draw();
    });
    // Keyboard users get the same read-out by tabbing to the canvas.
    this.canvas.tabIndex = 0;
    this.canvas.addEventListener("keydown", (e) => {
      if (!this.geom) return;
      const n = this.geom.count;
      if (!n) return;
      let i = this.hover === null ? 0 : this.hover;
      if (e.key === "ArrowRight") i = Math.min(n - 1, i + 1);
      else if (e.key === "ArrowLeft") i = Math.max(0, i - 1);
      else return;
      e.preventDefault();
      this.hover = i;
      this.draw();
      this._showTip(this.geom.X(this.geom.xs[i]), 20);
    });
  }

  _onHover(px, py) {
    if (!this.geom || !this.geom.count) return;
    const { xs, X } = this.geom;
    let best = 0, bestD = Infinity;
    for (let i = 0; i < xs.length; i++) {
      const d = Math.abs(X(xs[i]) - px);
      if (d < bestD) { bestD = d; best = i; }
    }
    if (bestD > 40) {
      this.hover = null;
      this.tip.classList.remove("show");
      this.draw();
      return;
    }
    this.hover = best;
    this.draw();
    this._showTip(X(xs[best]), py);
  }

  _showTip(px, py) {
    const i = this.hover;
    if (i === null || !this.geom) return;
    const { xs } = this.geom;
    const rows = [el("div", { class: "row" },
      el("span", { class: "k" }, this.opts.xLabel || "games"),
      el("b", {}, F.n(xs[i])))];
    for (const s of this.series) {
      if (!s.y || s.y[i] === undefined || s.y[i] === null) continue;
      rows.push(el("div", { class: "row" },
        el("span", { class: "sw", style: `background:${s.color}` }),
        el("span", { class: "k" }, s.label || ""),
        el("b", {}, s.format ? s.format(s.y[i]) : valueLabel(s.y[i], this.opts))));
    }
    this.tip.innerHTML = "";
    this.tip.append(...rows);
    this.tip.classList.add("show");
    const w = this.tip.offsetWidth;
    const boxW = this.canvas.clientWidth;
    let left = px + 12;
    if (left + w > boxW) left = Math.max(4, px - w - 12);
    this.tip.style.left = `${left}px`;
    this.tip.style.top = `${Math.max(2, Math.min(py - 10, this.canvas.clientHeight - 40))}px`;
  }

  draw() {
    const canvas = this.canvas;
    const opts = this.opts;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    canvas.width = w * dpr; canvas.height = h * dpr;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const pad = { l: 46, r: 12, t: 12, b: 22 };
    const plotW = w - pad.l - pad.r, plotH = h - pad.t - pad.b;
    const lines = this.series.filter((s) => s.y && s.y.length);
    if (!lines.length) {
      ctx.fillStyle = css("--fg-faint");
      ctx.font = "12px system-ui";
      ctx.textAlign = "center";
      ctx.fillText(opts.emptyText || "no data yet", w / 2, h / 2);
      this.geom = null;
      return;
    }

    let xmin = Infinity, xmax = -Infinity, ymin = 0, ymax = -Infinity;
    for (const s of lines) {
      for (let i = 0; i < s.y.length; i++) {
        const x = s.x[i];
        if (x < xmin) xmin = x;
        if (x > xmax) xmax = x;
        const v = s.y[i];
        if (v > ymax) ymax = v;
        if (opts.allowNegative && v < ymin) ymin = v;
      }
    }
    if (xmax === xmin) xmax = xmin + 1;
    if (ymax <= ymin) ymax = ymin + 1;
    if (opts.percent) ymax = Math.min(100, Math.max(ymax * 1.15, 5));
    else ymax *= 1.1;

    const X = (v) => pad.l + ((v - xmin) / (xmax - xmin)) * plotW;
    const Y = (v) => pad.t + plotH - ((v - ymin) / (ymax - ymin)) * plotH;

    ctx.strokeStyle = css("--grid"); ctx.lineWidth = 1;
    ctx.fillStyle = css("--fg-faint");
    ctx.font = "10px ui-monospace, monospace"; ctx.textAlign = "right";
    for (let i = 0; i <= 4; i++) {
      const v = ymin + ((ymax - ymin) * i) / 4;
      const y = Math.round(Y(v)) + 0.5;
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
      ctx.fillText(valueLabel(v, opts), pad.l - 6, y + 3);
    }
    ctx.textAlign = "center";
    for (let i = 0; i <= 2; i++) {
      const v = xmin + ((xmax - xmin) * i) / 2;
      ctx.fillText(axisLabel(v, opts.xUnit), X(v), h - 7);
    }

    for (const s of lines) {
      if (s.band) {
        ctx.fillStyle = s.color + "22";
        ctx.beginPath();
        for (let i = 0; i < s.x.length; i++) ctx[i ? "lineTo" : "moveTo"](X(s.x[i]), Y(s.band[0][i]));
        for (let i = s.x.length - 1; i >= 0; i--) ctx.lineTo(X(s.x[i]), Y(s.band[1][i]));
        ctx.closePath(); ctx.fill();
      }
      ctx.strokeStyle = s.color;
      ctx.lineWidth = s.width || 1.8;
      ctx.lineJoin = "round"; ctx.lineCap = "round";
      ctx.setLineDash(s.dash || []);
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < s.y.length; i++) {
        const px = X(s.x[i]), py = Y(s.y[i]);
        if (!started) { ctx.moveTo(px, py); started = true; } else ctx.lineTo(px, py);
      }
      ctx.stroke();
      if (s.points) {
        ctx.fillStyle = s.color;
        for (let i = 0; i < s.y.length; i++) {
          ctx.beginPath(); ctx.arc(X(s.x[i]), Y(s.y[i]), 2.6, 0, 7); ctx.fill();
        }
      }
    }
    ctx.setLineDash([]);

    // hover marker
    const longest = lines.reduce((a, b) => (b.x.length > a.x.length ? b : a), lines[0]);
    this.geom = { X, Y, xs: longest.x, count: longest.x.length, pad, plotW, plotH };
    if (this.hover !== null && this.hover < longest.x.length) {
      const hx = X(longest.x[this.hover]);
      ctx.strokeStyle = css("--fg-faint");
      ctx.globalAlpha = 0.5;
      ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(hx, pad.t); ctx.lineTo(hx, pad.t + plotH); ctx.stroke();
      ctx.setLineDash([]); ctx.globalAlpha = 1;
      for (const s of lines) {
        const v = s.y[this.hover];
        if (v === undefined || v === null) continue;
        ctx.fillStyle = s.color;
        ctx.beginPath(); ctx.arc(hx, Y(v), 3.8, 0, 7); ctx.fill();
        ctx.strokeStyle = css("--panel-2"); ctx.lineWidth = 1.6; ctx.stroke();
      }
    }

    if (opts.legend) {
      ctx.font = "10px system-ui"; ctx.textAlign = "left";
      let lx = pad.l + 4;
      for (const s of lines) {
        if (!s.label) continue;
        ctx.fillStyle = s.color;
        ctx.fillRect(lx, pad.t - 5, 8, 2.5);
        ctx.fillStyle = css("--fg-dim");
        ctx.fillText(s.label, lx + 12, pad.t - 1);
        lx += 14 + ctx.measureText(s.label).width + 12;
      }
    }
  }
}

/* Create a titled chart card; returns {node, chart}. */
function chartCard(title, series, opts = {}) {
  const canvas = el("canvas", { "aria-label": title, role: "img" });
  const node = el("div", { class: "chart-box" },
    el("h4", {}, title), canvas);
  // Canvas needs to be in the DOM with a size before the first draw.
  requestAnimationFrame(() => {
    node._chart = new Chart(canvas, series, opts);
  });
  return node;
}

function updateChartCard(node, series, opts) {
  if (node && node._chart) node._chart.update(series, opts);
}

/* Redraw every live chart when the window resizes or the theme flips. */
let _resizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(_resizeTimer);
  _resizeTimer = setTimeout(() => {
    $$(".chart-box").forEach((b) => b._chart && b._chart.draw());
  }, 140);
});
