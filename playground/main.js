const $ = (id) => document.getElementById(id);

// ── DOM refs ─────────────────────────────────────────────────
const dropzone = $("dropzone");
const fileInput = $("fileInput");
const exampleSelect = $("exampleSelect");
const editorEl = $("editor");
const formatBtn = $("formatBtn");
const runBtn = $("runBtn");
const statusEl = $("status");
const errorBox = $("errorBox");
const emptyState = $("emptyState");
const wasmBanner = $("wasmBanner");
const wasmBannerText = $("wasmBannerText");

let currentOp = "brazil";
let isRunning = false;
let wasmReady = false;
let massScanActive = false;

const exampleGuides = {
  "simple_workspace.json": {
    title: "Simple Counting Experiment",
    desc: "2-bin channel with a signal (normfactor \u00b5) and background with uncorrelated shape uncertainties. Try Brazil Band to compute the 95% CL upper limit on \u00b5, or MLE Fit to find the best-fit signal strength.",
    suggestedOp: "brazil",
  },
  "shape_workspace.json": {
    title: "Shape Analysis",
    desc: "10-bin channel with shape-varying signal. The signal peaks in bins 4\u20136 while background is flat. Profile Scan shows the likelihood parabola around \u00b5\u0302.",
    suggestedOp: "scan",
  },
  "multichannel_workspace.json": {
    title: "Multi-Channel Combination",
    desc: "Two channels (5 bins each) with correlated normalization systematics across channels. Demonstrates how NextStat handles multi-channel HistFactory workspaces in the browser.",
    suggestedOp: "fit",
  },
  "discovery_workspace.json": {
    title: "Discovery (Excess)",
    desc: "Workspace with a large observed excess over background. Run Hypothesis Test with \u00b5\u209c\u2091\u209b\u209c=0 to compute CL\u209b for the background-only hypothesis \u2014 a small CL\u209b indicates a discovery-level excess.",
    suggestedOp: "hypotest",
  },
  "glm_linear.json": {
    title: "Linear Regression",
    desc: "20 observations with 2 features. Fits y = \u03b2\u2080 + \u03b2\u2081x\u2081 + \u03b2\u2082x\u2082 via MLE (equivalent to OLS). Switch to the GLM Regression tab.",
    suggestedOp: "glm",
  },
  "glm_logistic.json": {
    title: "Logistic Regression",
    desc: "Binary classification with 30 observations and 2 features. Fits P(y=1) = sigmoid(\u03b2\u2080 + \u03b2\u2081x\u2081 + \u03b2\u2082x\u2082) via MLE.",
    suggestedOp: "glm",
  },
  "glm_poisson.json": {
    title: "Poisson Regression",
    desc: "Count data (25 observations, 1 feature). Fits E[y] = exp(\u03b2\u2080 + \u03b2\u2081x) via MLE. Models count rates with a log link.",
    suggestedOp: "glm",
  },
};

// ── Helpers ──────────────────────────────────────────────────

function setStatus(msg) {
  statusEl.textContent = msg || "";
}

function showError(msg) {
  errorBox.hidden = !msg;
  errorBox.textContent = msg || "";
}

function fmt(x, digits = 6) {
  if (x === null || x === undefined) return "—";
  if (typeof x !== "number") return String(x);
  if (!Number.isFinite(x)) return String(x);
  return x.toPrecision(digits);
}

function numVal(id) {
  const n = Number($(id)?.value);
  return Number.isFinite(n) ? n : null;
}

function intVal(id) {
  const n = numVal(id);
  return n !== null ? Math.trunc(n) : null;
}

// ── SVG chart helpers (ported from production BrazilBandChart / ProfileScanChart) ──

const SVG_W = 720, SVG_H = 400;
const PAD = { top: 32, right: 24, bottom: 48, left: 56 };
const PW = SVG_W - PAD.left - PAD.right;
const PH = SVG_H - PAD.top - PAD.bottom;

function scaleLinear([d0, d1], [r0, r1]) {
  const m = (r1 - r0) / (d1 - d0 || 1);
  return (v) => r0 + m * (v - d0);
}

function linTicks(min, max, count) {
  return Array.from({ length: count }, (_, i) => min + (i / (count - 1)) * (max - min));
}

function svgEl(tag, attrs = {}) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

function buildBrazilSVG(r) {
  const { scan, observedCls, expectedCls, alpha, observedLimit, expectedLimits, poiName } = r;
  const xS = scaleLinear([scan[0], scan[scan.length - 1]], [0, PW]);
  const yS = scaleLinear([0, 1], [PH, 0]);

  const svg = svgEl("svg", { viewBox: `0 0 ${SVG_W} ${SVG_H}` });
  svg.style.fontFamily = "ui-monospace, SFMono-Regular, Menlo, monospace";
  const g = svgEl("g", { transform: `translate(${PAD.left},${PAD.top})` });

  const yTicks = [0, 0.2, 0.4, 0.6, 0.8, 1.0];
  const xTicks = linTicks(scan[0], scan[scan.length - 1], 6);
  yTicks.forEach((t) => g.appendChild(svgEl("line", { x1: 0, y1: yS(t), x2: PW, y2: yS(t), stroke: "rgba(255,255,255,0.06)" })));
  xTicks.forEach((t) => g.appendChild(svgEl("line", { x1: xS(t), y1: 0, x2: xS(t), y2: PH, stroke: "rgba(255,255,255,0.06)" })));

  const band2Up = scan.map((mu, i) => `${xS(mu)},${yS(expectedCls[i][0])}`).join(" ");
  const band2Dn = [...scan].reverse().map((mu, i) => `${xS(mu)},${yS(expectedCls[scan.length - 1 - i][4])}`).join(" ");
  g.appendChild(svgEl("polygon", { points: `${band2Up} ${band2Dn}`, fill: "rgba(212,175,55,0.18)" }));

  const band1Up = scan.map((mu, i) => `${xS(mu)},${yS(expectedCls[i][1])}`).join(" ");
  const band1Dn = [...scan].reverse().map((mu, i) => `${xS(mu)},${yS(expectedCls[scan.length - 1 - i][3])}`).join(" ");
  g.appendChild(svgEl("polygon", { points: `${band1Up} ${band1Dn}`, fill: "rgba(76,175,80,0.25)" }));

  const expMed = scan.map((mu, i) => `${xS(mu)},${yS(expectedCls[i][2])}`).join(" ");
  g.appendChild(svgEl("polyline", { points: expMed, fill: "none", stroke: "rgba(255,255,255,0.6)", "stroke-width": "1.5", "stroke-dasharray": "6 4" }));

  const obsLine = scan.map((mu, i) => `${xS(mu)},${yS(observedCls[i])}`).join(" ");
  g.appendChild(svgEl("polyline", { points: obsLine, fill: "none", stroke: "#D4AF37", "stroke-width": "2" }));

  g.appendChild(svgEl("line", { x1: 0, y1: yS(alpha), x2: PW, y2: yS(alpha), stroke: "rgba(239,68,68,0.7)", "stroke-width": "1.5", "stroke-dasharray": "4 3" }));
  const alphaLabel = svgEl("text", { x: PW - 4, y: yS(alpha) - 6, fill: "rgba(239,68,68,0.7)", "font-size": "10", "text-anchor": "end" });
  alphaLabel.textContent = `CLs = ${alpha}`;
  g.appendChild(alphaLabel);

  g.appendChild(svgEl("line", { x1: xS(observedLimit), y1: 0, x2: xS(observedLimit), y2: PH, stroke: "rgba(212,175,55,0.5)", "stroke-width": "1" }));
  g.appendChild(svgEl("line", { x1: xS(expectedLimits[2]), y1: 0, x2: xS(expectedLimits[2]), y2: PH, stroke: "rgba(255,255,255,0.3)", "stroke-width": "1", "stroke-dasharray": "4 3" }));

  yTicks.forEach((t) => { const tx = svgEl("text", { x: -8, y: yS(t) + 4, fill: "rgba(255,255,255,0.5)", "font-size": "10", "text-anchor": "end" }); tx.textContent = t.toFixed(1); g.appendChild(tx); });
  xTicks.forEach((t) => { const tx = svgEl("text", { x: xS(t), y: PH + 20, fill: "rgba(255,255,255,0.5)", "font-size": "10", "text-anchor": "middle" }); tx.textContent = t.toFixed(1); g.appendChild(tx); });

  const xTitle = svgEl("text", { x: PW / 2, y: PH + 40, fill: "rgba(255,255,255,0.6)", "font-size": "11", "text-anchor": "middle" });
  xTitle.textContent = poiName;
  g.appendChild(xTitle);
  const yTitle = svgEl("text", { x: -40, y: PH / 2, fill: "rgba(255,255,255,0.6)", "font-size": "11", "text-anchor": "middle", transform: `rotate(-90, -40, ${PH / 2})` });
  yTitle.textContent = "CLs";
  g.appendChild(yTitle);

  svg.appendChild(g);

  const legend = svgEl("g", { transform: `translate(${PAD.left + 12}, 12)` });
  legend.appendChild(svgEl("rect", { x: 0, y: 0, width: 10, height: 10, fill: "rgba(212,175,55,0.18)", stroke: "rgba(212,175,55,0.4)", "stroke-width": "0.5" }));
  let lt = svgEl("text", { x: 14, y: 9, fill: "rgba(255,255,255,0.5)", "font-size": "9" }); lt.textContent = "±2σ"; legend.appendChild(lt);
  legend.appendChild(svgEl("rect", { x: 48, y: 0, width: 10, height: 10, fill: "rgba(76,175,80,0.25)", stroke: "rgba(76,175,80,0.5)", "stroke-width": "0.5" }));
  lt = svgEl("text", { x: 62, y: 9, fill: "rgba(255,255,255,0.5)", "font-size": "9" }); lt.textContent = "±1σ"; legend.appendChild(lt);
  legend.appendChild(svgEl("line", { x1: 100, y1: 5, x2: 118, y2: 5, stroke: "rgba(255,255,255,0.6)", "stroke-width": "1.5", "stroke-dasharray": "4 3" }));
  lt = svgEl("text", { x: 122, y: 9, fill: "rgba(255,255,255,0.5)", "font-size": "9" }); lt.textContent = "Expected"; legend.appendChild(lt);
  legend.appendChild(svgEl("line", { x1: 178, y1: 5, x2: 196, y2: 5, stroke: "#D4AF37", "stroke-width": "2" }));
  lt = svgEl("text", { x: 200, y: 9, fill: "rgba(255,255,255,0.5)", "font-size": "9" }); lt.textContent = "Observed"; legend.appendChild(lt);
  svg.appendChild(legend);

  return svg;
}

function buildProfileSVG(r) {
  const { points, muHat, poiName } = r;
  const twoDnll = points.map((p) => ({ mu: p.mu, y: 2 * p.deltaNll }));
  const yMax = Math.max(4, ...twoDnll.map((p) => p.y).filter((v) => v < 100));

  const xS = scaleLinear([points[0].mu, points[points.length - 1].mu], [0, PW]);
  const yS = scaleLinear([0, yMax], [PH, 0]);

  const svg = svgEl("svg", { viewBox: `0 0 ${SVG_W} ${SVG_H}` });
  svg.style.fontFamily = "ui-monospace, SFMono-Regular, Menlo, monospace";
  const g = svgEl("g", { transform: `translate(${PAD.left},${PAD.top})` });

  const yTicks = [];
  for (let t = 0; t <= yMax; t += 1) yTicks.push(t);
  const xTicks = linTicks(points[0].mu, points[points.length - 1].mu, 6);

  yTicks.forEach((t) => g.appendChild(svgEl("line", { x1: 0, y1: yS(t), x2: PW, y2: yS(t), stroke: "rgba(255,255,255,0.06)" })));
  xTicks.forEach((t) => g.appendChild(svgEl("line", { x1: xS(t), y1: 0, x2: xS(t), y2: PH, stroke: "rgba(255,255,255,0.06)" })));

  const thresholds = [{ level: 1, label: "1σ (68%)", color: "rgba(76,175,80,0.5)" }, { level: 3.84, label: "95% CL", color: "rgba(239,68,68,0.5)" }];
  thresholds.forEach((th) => {
    if (th.level > yMax) return;
    g.appendChild(svgEl("line", { x1: 0, y1: yS(th.level), x2: PW, y2: yS(th.level), stroke: th.color, "stroke-width": "1", "stroke-dasharray": "6 4" }));
    const tl = svgEl("text", { x: PW - 4, y: yS(th.level) - 4, fill: th.color, "font-size": "9", "text-anchor": "end" });
    tl.textContent = th.label;
    g.appendChild(tl);
  });

  const curvePts = twoDnll.map((p) => `${xS(p.mu)},${yS(Math.min(p.y, yMax))}`).join(" ");
  g.appendChild(svgEl("polyline", { points: curvePts, fill: "none", stroke: "#D4AF37", "stroke-width": "2" }));

  const muX = xS(muHat);
  g.appendChild(svgEl("line", { x1: muX, y1: 0, x2: muX, y2: PH, stroke: "rgba(212,175,55,0.4)", "stroke-width": "1", "stroke-dasharray": "3 3" }));
  const muLabel = svgEl("text", { x: muX, y: -6, fill: "rgba(212,175,55,0.7)", "font-size": "9", "text-anchor": "middle" });
  muLabel.textContent = `μ̂ = ${muHat.toPrecision(4)}`;
  g.appendChild(muLabel);

  yTicks.forEach((t) => { const tx = svgEl("text", { x: -8, y: yS(t) + 4, fill: "rgba(255,255,255,0.5)", "font-size": "10", "text-anchor": "end" }); tx.textContent = String(t); g.appendChild(tx); });
  xTicks.forEach((t) => { const tx = svgEl("text", { x: xS(t), y: PH + 20, fill: "rgba(255,255,255,0.5)", "font-size": "10", "text-anchor": "middle" }); tx.textContent = t.toFixed(1); g.appendChild(tx); });

  const xTitle = svgEl("text", { x: PW / 2, y: PH + 40, fill: "rgba(255,255,255,0.6)", "font-size": "11", "text-anchor": "middle" });
  xTitle.textContent = poiName;
  g.appendChild(xTitle);
  const yTitle = svgEl("text", { x: -40, y: PH / 2, fill: "rgba(255,255,255,0.6)", "font-size": "11", "text-anchor": "middle", transform: `rotate(-90, -40, ${PH / 2})` });
  yTitle.textContent = "−2 Δln L";
  g.appendChild(yTitle);

  svg.appendChild(g);
  return svg;
}

// ── Mass Scan (Type B) ──────────────────────────────────────

function generateMassScanWorkspaces(workspaceJson, nMassPoints) {
  const ws = JSON.parse(workspaceJson);
  const nBins = ws.channels[0]?.samples[0]?.data?.length || 0;
  if (nBins < 2) return [];

  const poiName = ws.measurements?.[0]?.config?.poi || "mu";
  const origSignals = {};
  for (const ch of ws.channels) {
    for (const s of ch.samples) {
      const hasNormfactor = s.modifiers?.some(
        (m) => m.type === "normfactor" && m.name === poiName
      );
      if (hasNormfactor) {
        origSignals[`${ch.name}::${s.name}`] = { ch: ch.name, sample: s.name, data: [...s.data] };
      }
    }
  }
  if (Object.keys(origSignals).length === 0) return [];

  const massPoints = [];
  const sigma = Math.max(1.0, nBins / 5);
  for (let m = 0; m < nMassPoints; m++) {
    const center = (m / (nMassPoints - 1)) * (nBins - 1);
    const variant = JSON.parse(workspaceJson);

    for (const ch of variant.channels) {
      for (const s of ch.samples) {
        const key = `${ch.name}::${s.name}`;
        const orig = origSignals[key];
        if (!orig) continue;
        const totalYield = orig.data.reduce((a, b) => a + b, 0);
        const weights = Array.from({ length: nBins }, (_, j) =>
          Math.exp(-0.5 * ((j - center) / sigma) ** 2)
        );
        const wSum = weights.reduce((a, b) => a + b, 0);
        s.data = weights.map((w) => Math.max(0.01, (w / wSum) * totalYield));
      }
    }
    massPoints.push({
      massIndex: m,
      massLabel: (center + 1).toFixed(1),
      workspaceJson: JSON.stringify(variant),
    });
  }
  return massPoints;
}

function workerCall(msg) {
  return new Promise((resolve, reject) => {
    const handler = (e) => {
      const d = e.data;
      if (!d) return;
      if (d.type === "result" && d.operation === msg.operation) {
        worker.removeEventListener("message", handler);
        resolve(d.result);
      } else if (d.type === "error" && d.operation === msg.operation) {
        worker.removeEventListener("message", handler);
        reject(new Error(d.error));
      }
    };
    worker.addEventListener("message", handler);
    worker.postMessage(msg);
  });
}

async function runMassScan() {
  massScanActive = true;
  const json = editorEl.value;
  const nPts = intVal("massScanPoints") || 7;
  const massPoints = generateMassScanWorkspaces(json, nPts);
  if (massPoints.length === 0) {
    showError("Cannot generate mass scan: need signal samples with normfactor modifier and ≥2 bins.");
    setRunning(false);
    massScanActive = false;
    return;
  }

  const opts = {};
  const a = numVal("alpha"); if (a !== null) opts.alpha = a;
  const s0 = numVal("scanStart"); if (s0 !== null) opts.scanStart = s0;
  const s1 = numVal("scanStop"); if (s1 !== null) opts.scanStop = s1;
  const np = intVal("scanPoints"); if (np !== null) opts.scanPoints = np;

  const results = [];
  for (let i = 0; i < massPoints.length; i++) {
    setStatus(`Mass scan ${i + 1}/${massPoints.length}…`);
    try {
      const r = await workerCall({
        type: "run",
        operation: "brazil",
        workspaceJson: massPoints[i].workspaceJson,
        options: opts,
      });
      results.push({
        massIndex: massPoints[i].massIndex,
        massLabel: massPoints[i].massLabel,
        observedLimit: r.observedLimit,
        expectedLimits: r.expectedLimits,
      });
    } catch (err) {
      results.push({
        massIndex: massPoints[i].massIndex,
        massLabel: massPoints[i].massLabel,
        observedLimit: NaN,
        expectedLimits: [NaN, NaN, NaN, NaN, NaN],
      });
    }
  }

  massScanActive = false;
  setRunning(false);
  setStatus(`Mass scan done (${results.length} points)`);
  emptyState.hidden = true;

  const chartEl = $("massScanChart");
  chartEl.hidden = false;
  chartEl.innerHTML = "";
  chartEl.appendChild(buildTypeBSVG(results));
}

function buildTypeBSVG(results) {
  const valid = results.filter((r) => Number.isFinite(r.observedLimit));
  if (valid.length === 0) {
    const el = document.createElement("div");
    el.textContent = "No valid mass scan results.";
    el.style.color = "var(--muted)";
    return el;
  }

  const masses = valid.map((r) => r.massIndex);
  const obs = valid.map((r) => r.observedLimit);
  const expM2 = valid.map((r) => r.expectedLimits[0]);
  const expM1 = valid.map((r) => r.expectedLimits[1]);
  const expMed = valid.map((r) => r.expectedLimits[2]);
  const expP1 = valid.map((r) => r.expectedLimits[3]);
  const expP2 = valid.map((r) => r.expectedLimits[4]);

  const allLimits = [...obs, ...expM2, ...expP2].filter(Number.isFinite);
  const yMin = 0;
  const yMax = Math.max(1.2, ...allLimits) * 1.15;

  const xS = scaleLinear([masses[0], masses[masses.length - 1]], [0, PW]);
  const yS = scaleLinear([yMin, yMax], [PH, 0]);

  const svg = svgEl("svg", { viewBox: `0 0 ${SVG_W} ${SVG_H}` });
  svg.style.fontFamily = "ui-monospace, SFMono-Regular, Menlo, monospace";
  const g = svgEl("g", { transform: `translate(${PAD.left},${PAD.top})` });

  const yTicks = linTicks(yMin, yMax, 6);
  const xTicks = masses;

  yTicks.forEach((t) => g.appendChild(svgEl("line", { x1: 0, y1: yS(t), x2: PW, y2: yS(t), stroke: "rgba(255,255,255,0.06)" })));
  xTicks.forEach((t) => g.appendChild(svgEl("line", { x1: xS(t), y1: 0, x2: xS(t), y2: PH, stroke: "rgba(255,255,255,0.06)" })));

  const band2Fwd = valid.map((r, i) => `${xS(masses[i])},${yS(expM2[i])}`).join(" ");
  const band2Rev = [...valid].reverse().map((r, i) => `${xS(masses[valid.length - 1 - i])},${yS(expP2[valid.length - 1 - i])}`).join(" ");
  g.appendChild(svgEl("polygon", { points: `${band2Fwd} ${band2Rev}`, fill: "rgba(212,175,55,0.18)" }));

  const band1Fwd = valid.map((r, i) => `${xS(masses[i])},${yS(expM1[i])}`).join(" ");
  const band1Rev = [...valid].reverse().map((r, i) => `${xS(masses[valid.length - 1 - i])},${yS(expP1[valid.length - 1 - i])}`).join(" ");
  g.appendChild(svgEl("polygon", { points: `${band1Fwd} ${band1Rev}`, fill: "rgba(76,175,80,0.25)" }));

  const expLine = valid.map((r, i) => `${xS(masses[i])},${yS(expMed[i])}`).join(" ");
  g.appendChild(svgEl("polyline", { points: expLine, fill: "none", stroke: "rgba(255,255,255,0.6)", "stroke-width": "1.5", "stroke-dasharray": "6 4" }));

  const obsLine = valid.map((r, i) => `${xS(masses[i])},${yS(obs[i])}`).join(" ");
  g.appendChild(svgEl("polyline", { points: obsLine, fill: "none", stroke: "#D4AF37", "stroke-width": "2" }));
  valid.forEach((r, i) => {
    g.appendChild(svgEl("circle", { cx: xS(masses[i]), cy: yS(obs[i]), r: "3", fill: "#D4AF37" }));
  });

  g.appendChild(svgEl("line", { x1: 0, y1: yS(1), x2: PW, y2: yS(1), stroke: "rgba(239,68,68,0.6)", "stroke-width": "1", "stroke-dasharray": "4 3" }));
  const excLabel = svgEl("text", { x: PW - 4, y: yS(1) - 5, fill: "rgba(239,68,68,0.6)", "font-size": "9", "text-anchor": "end" });
  excLabel.textContent = "μ = 1 (excluded below)";
  g.appendChild(excLabel);

  yTicks.forEach((t) => { const tx = svgEl("text", { x: -8, y: yS(t) + 4, fill: "rgba(255,255,255,0.5)", "font-size": "10", "text-anchor": "end" }); tx.textContent = t.toFixed(1); g.appendChild(tx); });
  xTicks.forEach((t) => { const tx = svgEl("text", { x: xS(t), y: PH + 20, fill: "rgba(255,255,255,0.5)", "font-size": "10", "text-anchor": "middle" }); tx.textContent = valid[t]?.massLabel || String(t); g.appendChild(tx); });

  const xTitle = svgEl("text", { x: PW / 2, y: PH + 40, fill: "rgba(255,255,255,0.6)", "font-size": "11", "text-anchor": "middle" });
  xTitle.textContent = "Signal peak position (bin)";
  g.appendChild(xTitle);
  const yTitle = svgEl("text", { x: -44, y: PH / 2, fill: "rgba(255,255,255,0.6)", "font-size": "11", "text-anchor": "middle", transform: `rotate(-90, -44, ${PH / 2})` });
  yTitle.textContent = "95% CL upper limit on μ";
  g.appendChild(yTitle);

  svg.appendChild(g);

  const legend = svgEl("g", { transform: `translate(${PAD.left + 12}, 12)` });
  legend.appendChild(svgEl("rect", { x: 0, y: 0, width: 10, height: 10, fill: "rgba(212,175,55,0.18)", stroke: "rgba(212,175,55,0.4)", "stroke-width": "0.5" }));
  let lt = svgEl("text", { x: 14, y: 9, fill: "rgba(255,255,255,0.5)", "font-size": "9" }); lt.textContent = "±2σ"; legend.appendChild(lt);
  legend.appendChild(svgEl("rect", { x: 48, y: 0, width: 10, height: 10, fill: "rgba(76,175,80,0.25)", stroke: "rgba(76,175,80,0.5)", "stroke-width": "0.5" }));
  lt = svgEl("text", { x: 62, y: 9, fill: "rgba(255,255,255,0.5)", "font-size": "9" }); lt.textContent = "±1σ"; legend.appendChild(lt);
  legend.appendChild(svgEl("line", { x1: 100, y1: 5, x2: 118, y2: 5, stroke: "rgba(255,255,255,0.6)", "stroke-width": "1.5", "stroke-dasharray": "4 3" }));
  lt = svgEl("text", { x: 122, y: 9, fill: "rgba(255,255,255,0.5)", "font-size": "9" }); lt.textContent = "Expected"; legend.appendChild(lt);
  legend.appendChild(svgEl("line", { x1: 178, y1: 5, x2: 196, y2: 5, stroke: "#D4AF37", "stroke-width": "2" }));
  legend.appendChild(svgEl("circle", { cx: 187, cy: 5, r: "3", fill: "#D4AF37" }));
  lt = svgEl("text", { x: 200, y: 9, fill: "rgba(255,255,255,0.5)", "font-size": "9" }); lt.textContent = "Observed"; legend.appendChild(lt);
  svg.appendChild(legend);

  return svg;
}

// ── Tab switching ────────────────────────────────────────────

function switchOp(op) {
  currentOp = op;
  document.querySelectorAll(".tab").forEach((t) => {
    const isActive = t.dataset.op === op;
    t.classList.toggle("active", isActive);
    t.setAttribute("aria-selected", String(isActive));
  });
  document.querySelectorAll(".op-settings").forEach((s) => {
    s.hidden = s.dataset.op !== op;
  });
  document.querySelectorAll(".op-results").forEach((r) => {
    r.hidden = r.dataset.op !== op;
  });
}

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => switchOp(tab.dataset.op));
});

// ── Running state ────────────────────────────────────────────

function setRunning(running) {
  isRunning = running;
  runBtn.disabled = running || !editorEl.value.trim();
  runBtn.classList.toggle("loading", running);
  runBtn.textContent = running ? "Running" : "Run";
  fileInput.disabled = running;
  exampleSelect.disabled = running;
}

function hasWorkspace() {
  return editorEl.value.trim().length > 0;
}

// ── Build worker message per operation ───────────────────────

function buildMessage() {
  const json = editorEl.value;
  const base = { type: "run", operation: currentOp, workspaceJson: json };

  switch (currentOp) {
    case "brazil": {
      const opts = {};
      const a = numVal("alpha");
      if (a !== null) opts.alpha = a;
      const s0 = numVal("scanStart");
      if (s0 !== null) opts.scanStart = s0;
      const s1 = numVal("scanStop");
      if (s1 !== null) opts.scanStop = s1;
      const np = intVal("scanPoints");
      if (np !== null) opts.scanPoints = np;
      return { ...base, options: opts };
    }
    case "fit":
      return base;
    case "scan": {
      const opts = {};
      const s0 = numVal("pScanStart");
      if (s0 !== null) opts.scanStart = s0;
      const s1 = numVal("pScanStop");
      if (s1 !== null) opts.scanStop = s1;
      const np = intVal("pScanPoints");
      if (np !== null) opts.scanPoints = np;
      return { ...base, options: opts };
    }
    case "hypotest":
      return { ...base, muTest: numVal("muTest") ?? 1.0 };
    case "glm": {
      const json = editorEl.value;
      const glmModel = $("glmModel")?.value || "linear";
      const intercept = $("glmIntercept")?.value !== "false";
      let parsed;
      try {
        parsed = JSON.parse(json);
      } catch {
        return { type: "run", operation: "glm", inputJson: json };
      }
      parsed.model = glmModel;
      parsed.includeIntercept = intercept;
      return { type: "run", operation: "glm", inputJson: JSON.stringify(parsed) };
    }
    default:
      return base;
  }
}

// ── Result renderers ─────────────────────────────────────────

function renderBrazil(r) {
  $("measurement").textContent = r.measurementName || "—";
  $("poi").textContent = r.poiName;
  $("nParams").textContent = String(r.nParams);
  $("muHat").textContent = fmt(r.muHat, 8);
  $("nll").textContent = fmt(r.freeDataNll, 10);
  $("obsLimit").textContent = fmt(r.observedLimit, 8);
  $("expMedLimit").textContent = fmt(r.expectedLimits[2], 8);
  $("brazilRuntime").textContent = `${Math.round(r.elapsedMs)} ms`;

  const chartEl = $("brazilChart");
  chartEl.innerHTML = "";
  chartEl.appendChild(buildBrazilSVG(r));
}

function renderFit(r) {
  $("fitMeasurement").textContent = r.measurementName || "—";
  $("fitNll").textContent = fmt(r.nll, 10);
  $("fitConverged").innerHTML = r.converged
    ? '<span class="badge badge-success">Converged</span>'
    : '<span class="badge badge-danger">Failed</span>';
  $("fitIter").textContent = String(r.nIter);
  $("fitRuntime").textContent = `${Math.round(r.elapsedMs)} ms`;

  const wrap = $("fitParamsWrap");
  if (!r.parameterNames || r.parameterNames.length === 0) {
    wrap.innerHTML = "";
    return;
  }
  let html = '<table class="params-table"><thead><tr><th>#</th><th>Parameter</th><th>Value</th><th>Uncertainty</th></tr></thead><tbody>';
  r.parameterNames.forEach((name, i) => {
    html += `<tr><td>${i}</td><td>${name}</td><td>${fmt(r.parameters[i], 8)}</td><td>${fmt(r.uncertainties[i], 4)}</td></tr>`;
  });
  html += "</tbody></table>";
  wrap.innerHTML = html;
}

function renderScan(r) {
  $("scanPoi").textContent = r.poiName;
  $("scanMuHat").textContent = fmt(r.muHat, 8);
  $("scanNllHat").textContent = fmt(r.nllHat, 10);
  $("scanNParams").textContent = String(r.nParams);
  $("scanRuntime").textContent = `${Math.round(r.elapsedMs)} ms`;

  const chartEl = $("scanChart");
  chartEl.innerHTML = "";
  chartEl.appendChild(buildProfileSVG(r));
}

function renderHypotest(r) {
  $("htMeasurement").textContent = r.measurementName || "—";
  $("htPoi").textContent = r.poiName;
  $("htMuTest").textContent = fmt(r.muTest, 6);
  $("htCls").textContent = fmt(r.cls, 6);
  $("htClsb").textContent = fmt(r.clsb, 6);
  $("htClb").textContent = fmt(r.clb, 6);
  $("htQmu").textContent = fmt(r.qMu, 6);
  $("htQmuA").textContent = fmt(r.qMuA, 6);
  $("htMuHat").textContent = fmt(r.muHat, 8);
  $("htRuntime").textContent = `${Math.round(r.elapsedMs)} ms`;
}

function renderGlm(r) {
  $("glmModelOut").innerHTML = `<span class="badge badge-info">${r.model}</span>`;
  $("glmConverged").innerHTML = r.converged
    ? '<span class="badge badge-success">Converged</span>'
    : '<span class="badge badge-danger">Failed</span>';
  $("glmNll").textContent = fmt(r.nll, 10);
  $("glmNobs").textContent = String(r.nObs);
  $("glmIter").textContent = String(r.nIter);
  $("glmRuntime").textContent = `${Math.round(r.elapsedMs)} ms`;

  const wrap = $("glmParamsWrap");
  if (!r.parameterNames || r.parameterNames.length === 0) {
    wrap.innerHTML = "";
    return;
  }
  let html = '<table class="params-table"><thead><tr><th>#</th><th>Parameter</th><th>Value</th></tr></thead><tbody>';
  r.parameterNames.forEach((name, i) => {
    html += `<tr><td>${i}</td><td>${name}</td><td>${fmt(r.parameters[i], 8)}</td></tr>`;
  });
  html += "</tbody></table>";
  wrap.innerHTML = html;
}

const renderers = { brazil: renderBrazil, fit: renderFit, scan: renderScan, hypotest: renderHypotest, glm: renderGlm };

// ── Worker ───────────────────────────────────────────────────

const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });

worker.onmessage = (e) => {
  const msg = e.data;
  if (!msg) return;

  if (massScanActive) return;

  if (msg.type === "result") {
    setRunning(false);
    const r = msg.result;

    if (msg.operation === "ingest_parquet" || msg.operation === "ingest_json") {
      loadJson(r.workspaceJson, "Imported from Parquet");
      return;
    }

    const ms = r.elapsedMs != null ? ` in ${Math.round(r.elapsedMs)} ms` : "";
    setStatus(`Done${ms}`);
    emptyState.hidden = true;
    switchOp(msg.operation);
    const render = renderers[msg.operation];
    if (render) render(r);
    return;
  }

  if (msg.type === "error") {
    setRunning(false);
    setStatus("");
    showError(msg.error || "Unknown error");
    return;
  }
};

worker.onerror = (e) => {
  setRunning(false);
  setStatus("");
  showError(
    `Worker failed to start. Build the WASM bundle first (see playground/README.md).\n\n${e.message || e.toString()}`
  );
};

// ── WASM init probe ──────────────────────────────────────────

function probeWasmReady() {
  worker.postMessage({ type: "ping" });
}

const origOnMessage = worker.onmessage;
worker.onmessage = (e) => {
  const msg = e.data;
  if (msg && msg.type === "pong") {
    wasmReady = true;
    wasmBanner.classList.add("ready");
    wasmBannerText.textContent = "WebAssembly ready";
    setTimeout(() => wasmBanner.classList.add("hidden"), 2000);
    runBtn.disabled = isRunning || !editorEl.value.trim();
    return;
  }
  origOnMessage.call(worker, e);
};

// ── Run ──────────────────────────────────────────────────────

function run() {
  if (!hasWorkspace()) return;
  showError("");
  setStatus("Running…");
  setRunning(true);

  const massScanOn = $("massScanToggle")?.checked && currentOp === "brazil";
  if (massScanOn) {
    runMassScan();
    return;
  }

  worker.postMessage(buildMessage());
}

runBtn.addEventListener("click", run);

$("massScanToggle")?.addEventListener("change", () => {
  const on = $("massScanToggle").checked;
  document.querySelectorAll(".mass-scan-field").forEach((el) => (el.hidden = !on));
  const msChart = $("massScanChart");
  if (msChart && !on) msChart.hidden = true;
});

// ── Editor ───────────────────────────────────────────────────

editorEl.addEventListener("input", () => {
  runBtn.disabled = isRunning || !hasWorkspace();
});

formatBtn.addEventListener("click", () => {
  try {
    const parsed = JSON.parse(editorEl.value);
    editorEl.value = JSON.stringify(parsed, null, 2);
  } catch {
    showError("Invalid JSON — cannot format.");
  }
});

// ── File loading ─────────────────────────────────────────────

function loadJson(text, label) {
  editorEl.value = text;
  runBtn.disabled = !text.trim();
  setStatus(label ? `Loaded: ${label}` : "Loaded");
  showError("");
  hideGuide();
}

function showGuide(filename) {
  const guide = exampleGuides[filename];
  if (!guide) { hideGuide(); return; }
  let el = $("guideBox");
  if (!el) {
    el = document.createElement("div");
    el.id = "guideBox";
    el.className = "guide-box";
    dropzone.parentNode.insertBefore(el, dropzone.nextSibling);
  }
  el.innerHTML = `<strong>${guide.title}</strong><br><span class="guide-desc">${guide.desc}</span>`;
  el.hidden = false;
  if (guide.suggestedOp) switchOp(guide.suggestedOp);
}

function hideGuide() {
  const el = $("guideBox");
  if (el) el.hidden = true;
}

async function handleFile(file) {
  if (file.name.endsWith(".parquet")) {
    setStatus("Ingesting Parquet…");
    const buf = await file.arrayBuffer();
    worker.postMessage({
      type: "run",
      operation: "ingest_parquet",
      bytes: new Uint8Array(buf),
    });
    return;
  }
  const text = await file.text();
  loadJson(text, file.name);
}

fileInput.addEventListener("change", async (e) => {
  const file = e.target.files?.[0];
  if (file) await handleFile(file);
});

exampleSelect.addEventListener("change", async () => {
  const val = exampleSelect.value;
  if (!val) return;
  showError("");
  setStatus("Loading example…");
  try {
    const res = await fetch(`./examples/${val}`);
    if (!res.ok) throw new Error(`Failed to fetch: ${res.status} ${res.statusText}`);
    const text = await res.text();
    loadJson(text, val);
    showGuide(val);
  } catch (err) {
    setStatus("");
    showError(String(err));
  }
  exampleSelect.value = "";
});

// ── Dropzone ─────────────────────────────────────────────────

function setupDropzone() {
  const onDragOver = (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  };
  const onDragLeave = () => dropzone.classList.remove("dragover");
  const onDrop = async (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    const file = e.dataTransfer?.files?.[0];
    if (file) await handleFile(file);
  };

  dropzone.addEventListener("dragover", onDragOver);
  dropzone.addEventListener("dragenter", onDragOver);
  dropzone.addEventListener("dragleave", onDragLeave);
  dropzone.addEventListener("drop", onDrop);

  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") fileInput.click();
  });
}

// ── Keyboard shortcut: Cmd/Ctrl + Enter ─────────────────────

document.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
    e.preventDefault();
    if (!isRunning && hasWorkspace()) run();
  }
});

// ── Init ─────────────────────────────────────────────────────

setupDropzone();
setRunning(false);
setStatus("Load a workspace to begin.");

// Auto-load simple example on first visit.
(async () => {
  try {
    const res = await fetch("./examples/simple_workspace.json");
    if (res.ok) {
      const text = await res.text();
      loadJson(text, "simple_workspace.json (auto-loaded)");
    }
  } catch { /* ignore */ }
})();
