import init, {
  run_asymptotic_upper_limits,
  run_fit,
  run_profile_scan,
  run_hypotest,
  run_glm,
  workspace_from_histogram_rows_json,
  workspace_from_parquet_bytes,
} from "./pkg/ns_wasm.js";

let ready = false;
let readyPromise = null;

function ensureReady() {
  if (ready) return Promise.resolve();
  if (!readyPromise) {
    readyPromise = init().then(() => {
      ready = true;
    });
  }
  return readyPromise;
}

const handlers = {
  brazil(msg) {
    return run_asymptotic_upper_limits(msg.workspaceJson, msg.options ?? null);
  },
  fit(msg) {
    return run_fit(msg.workspaceJson);
  },
  scan(msg) {
    return run_profile_scan(msg.workspaceJson, msg.options ?? null);
  },
  hypotest(msg) {
    return run_hypotest(msg.workspaceJson, msg.muTest ?? 1.0);
  },
  ingest_json(msg) {
    return { workspaceJson: workspace_from_histogram_rows_json(msg.rowsJson, msg.options ?? null) };
  },
  ingest_parquet(msg) {
    return { workspaceJson: workspace_from_parquet_bytes(msg.bytes, msg.options ?? null) };
  },
  glm(msg) {
    return run_glm(msg.inputJson);
  },
};

self.onmessage = async (e) => {
  const msg = e.data;
  if (!msg) return;

  if (msg.type === "ping") {
    await ensureReady();
    self.postMessage({ type: "pong" });
    return;
  }

  if (msg.type !== "run") return;

  try {
    await ensureReady();
    const handler = handlers[msg.operation];
    if (!handler) throw new Error(`Unknown operation: ${msg.operation}`);
    const result = handler(msg);
    self.postMessage({ type: "result", operation: msg.operation, result });
  } catch (err) {
    const error = err?.message ? String(err.message) : String(err);
    self.postMessage({ type: "error", operation: msg.operation, error });
  }
};

// Start WASM init eagerly so the banner updates as soon as possible.
ensureReady().then(() => self.postMessage({ type: "pong" }));

