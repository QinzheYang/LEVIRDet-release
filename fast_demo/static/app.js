"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const desktop = window.levirDesktop;
  const SUPPORTED = /\.(jpg|jpeg|png|bmp|tif|tiff|webp|ppm|pgm)$/i;
  const ACTIVE = new Set(["created", "pending", "uploading", "queued", "starting", "running"]);
  const COMPLETE = new Set(["completed", "complete", "done", "succeeded"]);
  const TOKEN_KEY = "levir-fast-demo-token";
  let token = "";
  try {
    const incoming = new URLSearchParams(location.hash.slice(1)).get("token");
    token = incoming || sessionStorage.getItem(TOKEN_KEY) || "";
    if (incoming) {
      sessionStorage.setItem(TOKEN_KEY, incoming);
      history.replaceState(null, "", location.pathname + location.search);
    }
  } catch (_) { /* Private browsing can disable storage. The current token still works. */ }

  const state = {
    connected: false, selected: [], localUrls: new Map(), nativePreviews: new Map(), mediaUrls: new Map(),
    job: null, index: 0, busy: false, phase: "", retryUpload: false,
    uploaded: new Set(), pollTimer: null, renderId: 0, outputRoot: "", reconnecting: false,
  };

  function showNotice(message, reconnect = false) {
    $("notice-text").textContent = message;
    $("notice").hidden = !message;
    $("retry-connect").hidden = !reconnect;
  }

  function connectState(connected) {
    state.connected = connected;
    $("connection").className = `connection ${connected ? "connected" : "failed"}`;
    $("connection-text").textContent = connected ? "Local session" : "Disconnected";
  }

  async function api(path, options = {}) {
    const target = new URL(path, location.origin);
    if (target.origin !== location.origin || !target.pathname.startsWith("/api/")) throw new Error("The demo returned an invalid request location.");
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), options.upload ? 10 * 60 * 1000 : 30000);
    const headers = new Headers(options.headers || {});
    headers.set("X-Levir-Token", token);
    const { upload, ...request } = options;
    try {
      const response = await fetch(path, { ...request, headers, signal: controller.signal, cache: "no-store", credentials: "same-origin" });
      if (!response.ok) {
        let message = `Request failed (${response.status}).`;
        try { const data = await response.json(); message = data.error || data.message || data.detail || message; }
        catch (_) { /* Keep the safe HTTP error when the response is not JSON. */ }
        if (response.status === 401 || response.status === 403) message = "This session is not authorized. Reopen the demo using its launcher to receive a new session link.";
        throw new Error(typeof message === "string" ? message : JSON.stringify(message));
      }
      if (response.status === 204) return null;
      return response.headers.get("content-type")?.includes("application/json") ? response.json() : response;
    } catch (error) {
      if (error.name === "AbortError") throw new Error("The local demo took too long to respond. Check its launcher window, then try again.");
      throw error;
    } finally { clearTimeout(timeout); }
  }

  function post(path, body) {
    return api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  }

  function files() { return state.job?.files?.length ? state.job.files : state.selected; }
  function isActive() { return state.busy || !!state.job && ACTIVE.has(state.job.status); }
  function isComplete(job) { return job && COMPLETE.has(job.status); }
  function setStatus(message, kind = "", percent = null, detail = "") {
    $("status-text").textContent = message;
    $("run-dot").className = `run-dot ${kind}`;
    $("progress").hidden = percent === null;
    if (percent !== null) $("progress").value = Math.max(0, Math.min(100, percent));
    $("progress-label").textContent = detail;
  }

  function controls() {
    const active = isActive();
    const canReplaceInput = state.retryUpload && !state.busy;
    for (const id of ["choose-images", "choose-folder"]) $(id).disabled = active && !canReplaceInput;
    for (const id of ["model", "threshold"]) $(id).disabled = active;
    $("confirm").disabled = !state.connected || !state.selected.length || active && !state.retryUpload;
    $("confirm").replaceChildren(document.createTextNode(state.retryUpload ? "Retry upload" : active ? "Running…" : "Confirm & run"));
    const arrow = document.createElement("span"); arrow.setAttribute("aria-hidden", "true"); arrow.textContent = active ? "·" : "→"; $("confirm").append(arrow);
    const count = files().length;
    $("previous").disabled = count < 2 || state.index === 0;
    $("next").disabled = count < 2 || state.index >= count - 1;
    $("page-count").textContent = `${count ? state.index + 1 : 0} / ${count}`;
    $("input-count").textContent = count ? `${count.toLocaleString()} image${count === 1 ? "" : "s"}` : "No images";
    const completed = Number(state.job?.completed || 0);
    $("output-count").textContent = completed ? `${completed.toLocaleString()} ready` : "No predictions";
    const file = files()[state.index];
    $("filename").textContent = file?.relative_path || file?.name || "No image selected";
    $("filename").title = file?.relative_path || file?.name || "";
    $("save-json").disabled = !file?.prediction_url;
    const output = state.job?.output_dir || state.outputRoot;
    $("output-path").textContent = output || "Waiting for the local demo";
    $("copy-path").disabled = !output;
    $("open-output").hidden = !desktop?.openOutput;
    $("open-output").disabled = !output;
    $("footer-model").textContent = `${state.job?.model || $("model").value}-class model`;
  }

  function placeholder(side, title, detail) {
    $(`${side}-image`).hidden = true;
    $(`${side}-badge`).hidden = true;
    $(`${side}-empty`).hidden = false;
    $(`${side}-empty-title`).textContent = title;
    $(`${side}-empty-detail`).textContent = detail;
  }

  function showImage(side, url, file, renderId, onError) {
    if (renderId !== state.renderId) return;
    const img = $(`${side}-image`);
    const onLoad = () => {
      if (renderId !== state.renderId) return;
      img.hidden = false;
      $(`${side}-empty`).hidden = true;
      $(`${side}-badge`).hidden = false;
    };
    img.onload = onLoad;
    img.onerror = () => { if (renderId === state.renderId) onError(); };
    img.alt = `${side === "input" ? "Original image" : "Detection visualization"}: ${file.name}`;
    if (img.src === url && img.complete && img.naturalWidth) onLoad();
    else img.src = url;
  }

  async function media(url) {
    // Media is authenticated via a header. Never expose the session token in image URLs.
    if (!url || new URL(url, location.origin).origin !== location.origin) throw new Error("The demo returned an invalid image location.");
    if (!state.mediaUrls.has(url)) {
      const pending = api(url).then(async (response) => {
        if (!(response instanceof Response)) throw new Error("The demo returned an invalid image response.");
        return URL.createObjectURL(await response.blob());
      }).catch((error) => { state.mediaUrls.delete(url); throw error; });
      state.mediaUrls.set(url, pending);
      if (state.mediaUrls.size > 8) {
        const oldest = state.mediaUrls.keys().next().value;
        state.mediaUrls.get(oldest).then((value) => URL.revokeObjectURL(value)).catch(() => {});
        state.mediaUrls.delete(oldest);
      }
    }
    return state.mediaUrls.get(url);
  }

  async function render() {
    controls();
    const renderId = ++state.renderId;
    const file = files()[state.index];
    if (!file) {
      placeholder("input", "please choose input", "An image, a selection, or an entire folder");
      placeholder("output", "Visualization will appear here", "Confirm your input to start inference");
      return;
    }
    const local = state.selected[state.index];
    const localFile = local?.file;
    const previewUrl = file.preview_url || file.input_url;
    const remoteInput = async () => {
      if (!previewUrl || state.phase === "uploading" && !state.uploaded.has(String(file.id))) {
        placeholder("input", "Preview available after upload", "This image format needs the local image reader.");
        return;
      }
      try {
        const url = await media(previewUrl);
        showImage("input", url, file, renderId, () => placeholder("input", "Preview unavailable", "The selected image can still be processed by the model."));
      } catch (_) { if (renderId === state.renderId) placeholder("input", "Preview not ready", "The preview will appear when this image has been uploaded."); }
    };
    placeholder("input", "Loading image…", file.name);
    if (file.visualization_url) placeholder("output", "Loading visualization…", file.name);
    if (localFile) {
      if (!state.localUrls.has(localFile)) {
        state.localUrls.set(localFile, URL.createObjectURL(localFile));
        if (state.localUrls.size > 6) {
          const oldest = state.localUrls.keys().next().value;
          URL.revokeObjectURL(state.localUrls.get(oldest)); state.localUrls.delete(oldest);
        }
      }
      showImage("input", state.localUrls.get(localFile), file, renderId, remoteInput);
    } else if (local?.path && desktop?.preview) {
      try {
        if (!state.nativePreviews.has(local.path)) {
          state.nativePreviews.set(local.path, desktop.preview(local.path));
          if (state.nativePreviews.size > 6) state.nativePreviews.delete(state.nativePreviews.keys().next().value);
        }
        const preview = await state.nativePreviews.get(local.path);
        if (preview) showImage("input", preview, file, renderId, remoteInput);
        else await remoteInput();
      } catch (_) { await remoteInput(); }
    } else await remoteInput();

    if (file.visualization_url) {
      try {
        const url = await media(file.visualization_url);
        showImage("output", url, file, renderId, () => placeholder("output", "Preview unavailable", "The visualization is still saved in the output directory."));
      } catch (_) { if (renderId === state.renderId) placeholder("output", "Preview could not be loaded", "Use the arrows to try again, or check the output directory."); }
    } else {
      if (renderId !== state.renderId) return;
      const failed = state.job?.status === "failed" || file.status === "failed";
      placeholder("output", failed ? "No prediction for this image" : isActive() ? "Processing your images" : "Visualization will appear here", failed ? "See the message below for details." : isActive() ? "Results appear here as they become available." : "Confirm your input to start inference");
    }
  }

  async function revokeMedia() {
    for (const url of state.localUrls.values()) URL.revokeObjectURL(url);
    state.localUrls.clear();
    state.nativePreviews.clear();
    for (const promise of state.mediaUrls.values()) promise.then((url) => URL.revokeObjectURL(url)).catch(() => {});
    state.mediaUrls.clear();
  }

  function selectFiles(selection) {
    if (isActive() && !(state.retryUpload && !state.busy)) return;
    const chosen = Array.from(selection).filter((file) => SUPPORTED.test(file.name));
    chosen.sort((a, b) => (a.relative_path || a.webkitRelativePath || a.name).localeCompare(b.relative_path || b.webkitRelativePath || b.name, "en", { numeric: true }));
    if (!chosen.length) { showNotice("No supported images were found. Choose JPG, PNG, BMP, TIFF, WebP, PPM, or PGM images."); return; }
    revokeMedia();
    state.job = null; state.index = 0; state.uploaded.clear(); state.retryUpload = false; state.phase = "";
    state.selected = chosen.map((file, index) => ({ file: file instanceof File ? file : null, path: file.path, index, name: file.name, relative_path: file.relative_path || file.webkitRelativePath || file.name, size: file.size }));
    const ignored = selection.length - chosen.length;
    showNotice(ignored ? `${ignored.toLocaleString()} unsupported file${ignored === 1 ? " was" : "s were"} skipped.` : "");
    setStatus(`${chosen.length.toLocaleString()} image${chosen.length === 1 ? "" : "s"} selected. Ready when you are.`);
    render();
  }

  function uploadFile(jobId, entry, file, completedBytes, totalBytes, index) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("PUT", `/api/jobs/${encodeURIComponent(jobId)}/files/${encodeURIComponent(entry.id)}`);
      xhr.setRequestHeader("X-Levir-Token", token);
      xhr.setRequestHeader("Content-Type", "application/octet-stream");
      xhr.timeout = 10 * 60 * 1000;
      xhr.upload.onprogress = (event) => {
        const percent = totalBytes ? (completedBytes + event.loaded) / totalBytes * 100 : (index + 1) / state.selected.length * 100;
        setStatus(`Preparing image ${index + 1} of ${state.selected.length}: ${file.name}`, "active", percent, `Uploading · ${Math.round(percent)}%`);
      };
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) resolve();
        else {
          let message = `Upload failed (${xhr.status}).`;
          try { const body = JSON.parse(xhr.responseText); message = body.error || body.message || message; } catch (_) { /* Keep HTTP error. */ }
          reject(new Error(typeof message === "string" ? message : JSON.stringify(message)));
        }
      };
      xhr.onerror = () => reject(new Error("The local demo connection was interrupted during upload. Keep this window open and retry."));
      xhr.ontimeout = () => reject(new Error("This upload timed out. Check the launcher window, then retry."));
      xhr.send(file);
    });
  }

  async function run() {
    if (state.busy || !state.selected.length || !state.connected) return;
    const retry = state.retryUpload;
    state.busy = true; state.retryUpload = false; state.phase = "uploading";
    showNotice(""); controls();
    try {
      if (!retry) {
        clearTimeout(state.pollTimer);
        state.uploaded.clear();
        state.job = null;
        setStatus("Creating your inference run…", "active", 0);
        state.job = await post("/api/jobs", {
          model: $("model").value, threshold: Number($("threshold").value),
          files: state.selected.map(({ name, relative_path, size }) => ({ name, relative_path, size })),
        });
        await render();
      }
      if (!state.job?.id || state.job.files?.length !== state.selected.length) throw new Error("The local demo returned an incomplete upload job. Please restart the launcher.");
      if (retry) {
        const current = await api(`/api/jobs/${encodeURIComponent(state.job.id)}`);
        if (["queued", "starting", "running"].includes(current.status) || isComplete(current)) {
          state.job = current; state.busy = false; state.phase = "running";
          await poll(); return;
        }
      }
      const totalBytes = state.selected.reduce((sum, entry) => sum + entry.size, 0);
      let completedBytes = 0;
      for (let index = 0; index < state.selected.length; index++) {
        const entry = state.job.files[index];
        const local = state.selected[index];
        if (!state.uploaded.has(String(entry.id))) {
          if (local.path && desktop?.uploadFile) {
            setStatus(`Preparing image ${index + 1} of ${state.selected.length}: ${local.name}`, "active", totalBytes ? completedBytes / totalBytes * 100 : 0, `Uploading · ${index + 1} / ${state.selected.length}`);
            const url = new URL(`/api/jobs/${encodeURIComponent(state.job.id)}/files/${encodeURIComponent(entry.id)}`, location.origin).href;
            await desktop.uploadFile(local.path, url, token);
          } else await uploadFile(state.job.id, entry, local.file, completedBytes, totalBytes, index);
          state.uploaded.add(String(entry.id));
          if (index === state.index) render();
        }
        completedBytes += local.size;
      }
      state.phase = "starting";
      setStatus("Loading the model. The first run may take a moment…", "active", 0);
      const started = await post(`/api/jobs/${encodeURIComponent(state.job.id)}/start`, {});
      if (started?.id) state.job = started;
      state.busy = false; state.phase = "running";
      controls();
      await poll();
    } catch (error) {
      state.busy = false;
      state.retryUpload = !!state.job?.id && state.phase !== "running";
      setStatus(state.retryUpload ? "Preparation paused. Your selected images are kept for retry." : "Unable to start inference.", "error");
      showNotice(error.message);
      controls();
    }
  }

  function jobStatus() {
    const job = state.job;
    const total = Number(job.total || job.files?.length || 0);
    const completed = Number(job.completed || 0);
    if (isComplete(job)) {
      state.busy = false; state.phase = "";
      setStatus(`Finished. ${completed.toLocaleString()} image${completed === 1 ? "" : "s"} processed and saved.`, "done", 100, `${completed} / ${total}`);
    } else if (job.status === "failed" || job.status === "cancelled") {
      state.busy = false; state.phase = "";
      setStatus(`Inference ${job.status}. ${completed} of ${total} results saved.`, "error", total ? completed / total * 100 : 0, `${completed} / ${total}`);
      showNotice(job.error || "Check the launcher window for details. You can select images and try another run.");
    } else {
      setStatus(job.message || (completed ? `Processing images. ${completed} of ${total} ready to view.` : job.stage || "Loading the model and preparing inference…"), "active", total ? completed / total * 100 : 0, `${completed} / ${total}`);
    }
  }

  async function poll() {
    if (!state.job?.id) return;
    clearTimeout(state.pollTimer);
    try {
      state.job = await api(`/api/jobs/${encodeURIComponent(state.job.id)}`);
      connectState(true);
      if (state.reconnecting) { showNotice(""); state.reconnecting = false; }
      jobStatus();
      await render();
      if (ACTIVE.has(state.job.status)) state.pollTimer = setTimeout(poll, 1400);
    } catch (error) {
      connectState(false);
      state.reconnecting = true;
      showNotice(`${error.message} Inference may still be running. Reconnecting automatically…`);
      controls();
      state.pollTimer = setTimeout(poll, 4000);
    }
  }

  async function initialize() {
    try {
      const info = await api("/api/status");
      connectState(true); showNotice("");
      state.outputRoot = info.output_root || "";
      if (info.models?.length) {
        const previous = $("model").value;
        $("model").replaceChildren(...info.models.map((entry) => {
          const option = document.createElement("option"); option.value = String(entry.id); option.textContent = entry.label; return option;
        }));
        $("model").value = info.models.some((entry) => String(entry.id) === previous) ? previous : info.default_model || "30";
      }
      if (info.active_job) {
        state.job = info.active_job;
        if (state.job.model) $("model").value = String(state.job.model);
        if (state.job.status === "created" || state.job.status === "uploading") {
          setStatus("An unfinished upload exists. Keep the original demo window open to finish it.", "error");
          showNotice("If the original window was closed, restart the demo launcher to begin a new upload.");
          render();
        } else await poll();
      } else {
        setStatus("Ready. Choose your images to begin.");
        render();
      }
    } catch (error) {
      connectState(false);
      setStatus("Unable to connect to the local demo.", "error");
      showNotice(error.message || "Keep the launcher window open, then try again.", true);
      controls();
    }
  }

  async function chooseInput(folder) {
    const picker = folder ? desktop?.chooseFolder : desktop?.chooseFiles;
    if (!picker) { $(folder ? "folder-picker" : "image-picker").click(); return; }
    try { const selected = await picker(); if (selected?.length) selectFiles(selected); }
    catch (error) { showNotice(error.message || "The file selection could not be opened."); }
  }
  $("choose-images").addEventListener("click", () => chooseInput(false));
  $("choose-folder").addEventListener("click", () => chooseInput(true));
  for (const id of ["image-picker", "folder-picker"]) $(id).addEventListener("change", (event) => { if (event.target.files.length) selectFiles(event.target.files); event.target.value = ""; });
  $("confirm").addEventListener("click", run);
  $("retry-connect").addEventListener("click", initialize);
  $("threshold").addEventListener("input", () => { $("threshold-value").textContent = Number($("threshold").value).toFixed(2); });
  $("model").addEventListener("change", controls);
  function move(delta) { const next = state.index + delta; if (next >= 0 && next < files().length) { state.index = next; render(); } }
  $("previous").addEventListener("click", () => move(-1));
  $("next").addEventListener("click", () => move(1));
  document.addEventListener("keydown", (event) => {
    if (event.altKey || event.ctrlKey || event.metaKey || ["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(event.target.tagName) || event.target.isContentEditable) return;
    if (event.key === "ArrowLeft" || event.key === "ArrowRight") { event.preventDefault(); move(event.key === "ArrowLeft" ? -1 : 1); }
  });
  $("copy-path").addEventListener("click", async () => {
    try {
      const text = state.job?.output_dir || state.outputRoot;
      if (desktop?.copyText) await desktop.copyText(text);
      else await navigator.clipboard.writeText(text);
      $("copy-path").textContent = "Copied";
      setTimeout(() => { $("copy-path").textContent = "Copy path"; }, 1700);
    }
    catch (_) { showNotice("Select the output path to copy it manually."); }
  });
  $("open-output").addEventListener("click", async () => {
    try { await desktop.openOutput(state.job?.output_dir || state.outputRoot); }
    catch (error) { showNotice(error.message || "The output folder could not be opened."); }
  });
  $("save-json").addEventListener("click", async () => {
    const file = files()[state.index];
    if (!file?.prediction_url) return;
    try {
      const data = await api(file.prediction_url);
      if (desktop?.saveJSON) {
        const text = data instanceof Response ? await data.text() : JSON.stringify(data, null, 2);
        await desktop.saveJSON(file.name.replace(/\.[^.]+$/, "") + ".json", text);
        return;
      }
      const blob = data instanceof Response ? await data.blob() : new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob); const link = document.createElement("a");
      link.href = url; link.download = file.name.replace(/\.[^.]+$/, "") + ".json";
      document.body.append(link); link.click(); link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (error) { showNotice(error.message); }
  });
  initialize();
})();
