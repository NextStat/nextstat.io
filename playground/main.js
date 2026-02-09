const $ = (id) => document.getElementById(id);

const dropzone = $("dropzone");
const fileInput = $("fileInput");
const loadExampleBtn = $("loadExample");
const runBtn = $("runBtn");
const statusEl = $("status");
const errorBox = $("errorBox");

const alphaInput = $("alpha");
const scanStartInput = $("scanStart");
const scanStopInput = $("scanStop");
const scanPointsInput = $("scanPoints");

const measurementEl = $("measurement");
const poiEl = $("poi");
const nParamsEl = $("nParams");
const muHatEl = $("muHat");
const nllEl = $("nll");
const obsLimitEl = $("obsLimit");
const expMedLimitEl = $("expMedLimit");
const runtimeEl = $("runtime");

let currentWorkspaceJson = null;

function setStatus(msg) {
  statusEl.textContent = msg || "";
}

function showError(msg) {
  errorBox.hidden = !msg;
  errorBox.textContent = msg || "";
}

function resetResults() {
  measurementEl.textContent = "—";
  poiEl.textContent = "—";
  nParamsEl.textContent = "—";
  muHatEl.textContent = "—";
  nllEl.textContent = "—";
  obsLimitEl.textContent = "—";
  expMedLimitEl.textContent = "—";
  runtimeEl.textContent = "—";
  Plotly.purge("plot");
}

function getOptions() {
  const numOrNull = (v) => {
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  };
  const intOrNull = (v) => {
    const n = Number(v);
    if (!Number.isFinite(n)) return null;
    return Math.trunc(n);
  };

  const opts = {};
  const alpha = numOrNull(alphaInput.value);
  const scanStart = numOrNull(scanStartInput.value);
  const scanStop = numOrNull(scanStopInput.value);
  const scanPoints = intOrNull(scanPointsInput.value);

  if (alpha !== null) opts.alpha = alpha;
  if (scanStart !== null) opts.scanStart = scanStart;
  if (scanStop !== null) opts.scanStop = scanStop;
  if (scanPoints !== null) opts.scanPoints = scanPoints;
  return opts;
}

function fmt(x, digits = 6) {
  if (x === null || x === undefined) return "—";
  if (typeof x !== "number") return String(x);
  if (!Number.isFinite(x)) return String(x);
  return x.toPrecision(digits);
}

function buildPlot(result) {
  const x = result.scan;
  const alpha = result.alpha;
  const expected = result.expectedCls;

  const expP2 = expected.map((v) => v[0]);
  const expP1 = expected.map((v) => v[1]);
  const expMed = expected.map((v) => v[2]);
  const expM1 = expected.map((v) => v[3]);
  const expM2 = expected.map((v) => v[4]);

  const traces = [
    {
      x,
      y: expP2,
      mode: "lines",
      line: { color: "rgba(0,0,0,0)" },
      hoverinfo: "skip",
      showlegend: false,
      name: "+2σ",
    },
    {
      x,
      y: expM2,
      mode: "lines",
      line: { color: "rgba(0,0,0,0)" },
      fill: "tonexty",
      fillcolor: "rgba(255, 193, 7, 0.30)",
      hoverinfo: "skip",
      name: "±2σ",
    },
    {
      x,
      y: expP1,
      mode: "lines",
      line: { color: "rgba(0,0,0,0)" },
      hoverinfo: "skip",
      showlegend: false,
      name: "+1σ",
    },
    {
      x,
      y: expM1,
      mode: "lines",
      line: { color: "rgba(0,0,0,0)" },
      fill: "tonexty",
      fillcolor: "rgba(76, 175, 80, 0.30)",
      hoverinfo: "skip",
      name: "±1σ",
    },
    {
      x,
      y: expMed,
      mode: "lines",
      line: { color: "rgba(255,255,255,0.85)", dash: "dash", width: 2 },
      name: "Expected (median)",
    },
    {
      x,
      y: result.observedCls,
      mode: "lines",
      line: { color: "rgba(255,255,255,0.95)", width: 2.5 },
      name: "Observed",
    },
    {
      x,
      y: x.map(() => alpha),
      mode: "lines",
      line: { color: "rgba(239,68,68,0.9)", dash: "dot", width: 2 },
      name: `CLs = ${alpha}`,
    },
  ];

  const shapes = [
    {
      type: "line",
      xref: "x",
      yref: "paper",
      x0: result.observedLimit,
      x1: result.observedLimit,
      y0: 0,
      y1: 1,
      line: { color: "rgba(255,255,255,0.55)", width: 1.5 },
    },
    {
      type: "line",
      xref: "x",
      yref: "paper",
      x0: result.expectedLimits[2],
      x1: result.expectedLimits[2],
      y0: 0,
      y1: 1,
      line: { color: "rgba(255,255,255,0.55)", width: 1.5, dash: "dash" },
    },
  ];

  const layout = {
    margin: { l: 54, r: 18, t: 18, b: 48 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "rgba(255,255,255,0.90)" },
    legend: { orientation: "h", x: 0, y: 1.08 },
    xaxis: {
      title: { text: result.poiName },
      gridcolor: "rgba(255,255,255,0.10)",
      zerolinecolor: "rgba(255,255,255,0.12)",
    },
    yaxis: {
      title: { text: "CLs" },
      range: [0, 1],
      gridcolor: "rgba(255,255,255,0.10)",
      zerolinecolor: "rgba(255,255,255,0.12)",
    },
    shapes,
  };

  Plotly.newPlot("plot", traces, layout, { displayModeBar: false, responsive: true });
}

function updateResults(result) {
  measurementEl.textContent = result.measurementName || "—";
  poiEl.textContent = result.poiName;
  nParamsEl.textContent = String(result.nParams);
  muHatEl.textContent = fmt(result.muHat, 8);
  nllEl.textContent = fmt(result.freeDataNll, 10);
  obsLimitEl.textContent = fmt(result.observedLimit, 8);
  expMedLimitEl.textContent = fmt(result.expectedLimits[2], 8);
  runtimeEl.textContent = `${Math.round(result.elapsedMs)} ms`;
  buildPlot(result);
}

function setRunning(running) {
  runBtn.disabled = running || !currentWorkspaceJson;
  alphaInput.disabled = running;
  scanStartInput.disabled = running;
  scanStopInput.disabled = running;
  scanPointsInput.disabled = running;
  fileInput.disabled = running;
  loadExampleBtn.disabled = running;
}

const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });

worker.onmessage = (e) => {
  const msg = e.data;
  if (!msg) return;

  if (msg.type === "result") {
    setRunning(false);
    setStatus(`Done in ${Math.round(msg.result.elapsedMs)} ms`);
    updateResults(msg.result);
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
    `Worker failed to start. If this says “failed to fetch dynamically imported module”, build the WASM bundle first.\n\n${e.message || e.toString()}`
  );
};

async function run() {
  if (!currentWorkspaceJson) return;
  showError("");
  setStatus("Running…");
  setRunning(true);
  resetResults();

  worker.postMessage({
    type: "run",
    workspaceJson: currentWorkspaceJson,
    options: getOptions(),
  });
}

runBtn.addEventListener("click", run);

function setWorkspaceJson(text, sourceLabel) {
  currentWorkspaceJson = text;
  runBtn.disabled = !text;
  setStatus(sourceLabel ? `Loaded: ${sourceLabel}` : "Loaded");
}

async function readFile(file) {
  const text = await file.text();
  setWorkspaceJson(text, file.name);
}

fileInput.addEventListener("change", async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;
  await readFile(file);
});

loadExampleBtn.addEventListener("click", async () => {
  showError("");
  setStatus("Loading example…");
  try {
    const res = await fetch("./examples/simple_workspace.json");
    if (!res.ok) throw new Error(`Failed to fetch example: ${res.status} ${res.statusText}`);
    const text = await res.text();
    setWorkspaceJson(text, "simple_workspace.json");
  } catch (err) {
    setStatus("");
    showError(String(err));
  }
});

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
    if (!file) return;
    await readFile(file);
  };

  dropzone.addEventListener("dragover", onDragOver);
  dropzone.addEventListener("dragenter", onDragOver);
  dropzone.addEventListener("dragleave", onDragLeave);
  dropzone.addEventListener("drop", onDrop);

  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      fileInput.click();
    }
  });
}

setupDropzone();
setRunning(false);
resetResults();
setStatus("Load a workspace to begin.");
