import init, { run_asymptotic_upper_limits } from "./pkg/ns_wasm.js";

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

self.onmessage = async (e) => {
  const msg = e.data;
  if (!msg || msg.type !== "run") return;

  try {
    await ensureReady();
    const result = run_asymptotic_upper_limits(msg.workspaceJson, msg.options ?? null);
    self.postMessage({ type: "result", result });
  } catch (err) {
    const error = err?.message ? String(err.message) : String(err);
    self.postMessage({ type: "error", error });
  }
};

