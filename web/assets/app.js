"use strict";

const FILE_TYPES = {
  image: { endpoint: "/api/v1/image/clean", extensions: ["jpg", "jpeg", "png", "bmp"] },
  video: { endpoint: "/api/v1/video/clean", extensions: ["mp4"] },
};

const ERROR_MESSAGES = {
  FILE_NOT_FOUND: "文件不存在或结果已被移除。",
  INVALID_FILE: "文件格式或内容无效，请重新选择。",
  MODEL_NOT_FOUND: "服务尚未加载 ROI 模型，请检查模型文件。",
  ROI_NOT_FOUND: "没有检测到有效超声区域，请检查该样本。",
  PROCESS_FAILED: "处理失败，请查看服务日志后重试。",
};

const elements = {
  input: document.querySelector("#file-input"),
  selectionName: document.querySelector("#selection-name"),
  selectionMeta: document.querySelector("#selection-meta"),
  clear: document.querySelector("#clear-button"),
  cleanCurrent: document.querySelector("#clean-current-button"),
  clean: document.querySelector("#clean-button"),
  stop: document.querySelector("#stop-button"),
  error: document.querySelector("#error-box"),
  emptyView: document.querySelector("#empty-view"),
  singleView: document.querySelector("#single-view"),
  singleImage: document.querySelector("#single-image"),
  singleVideo: document.querySelector("#single-video"),
  compareView: document.querySelector("#compare-view"),
  compareSourceImage: document.querySelector("#compare-source-image"),
  compareSourceVideo: document.querySelector("#compare-source-video"),
  compareResultImage: document.querySelector("#compare-result-image"),
  compareResultVideo: document.querySelector("#compare-result-video"),
  resultPlaceholder: document.querySelector("#result-placeholder"),
  resultMessage: document.querySelector("#result-message"),
  currentName: document.querySelector("#current-name"),
  currentPath: document.querySelector("#current-path"),
  currentStatus: document.querySelector("#current-status"),
  progress: document.querySelector("#metric-progress"),
  confidence: document.querySelector("#metric-confidence"),
  bbox: document.querySelector("#metric-bbox"),
  time: document.querySelector("#metric-time"),
  download: document.querySelector("#current-download"),
  fileCounter: document.querySelector("#file-counter"),
  fileSearch: document.querySelector("#file-search"),
  listEmpty: document.querySelector("#list-empty"),
  fileList: document.querySelector("#file-list"),
  summary: document.querySelector("#batch-summary"),
  healthDot: document.querySelector("#health-dot"),
  healthText: document.querySelector("#health-text"),
};

let items = [];
let selectedIndex = 0;
let sourceUrls = [];
let requestController = null;
let stopRequested = false;
let batchStartedAt = 0;

function extensionOf(filename) {
  return filename.includes(".") ? filename.split(".").pop().toLowerCase() : "";
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
}

function statusClasses(status) {
  if (status === "COMPLETE") return "border-signal/40 text-signal";
  if (status === "PROCESSING") return "border-amber/40 text-amber";
  if (status === "FAILED" || status === "STOPPED") return "border-[#e56a5d]/40 text-[#e56a5d]";
  return "border-white/10 text-mist/40";
}

function hideMedia(media) {
  if (media.tagName === "VIDEO") {
    media.pause();
    media.removeAttribute("src");
    media.load();
  } else {
    media.removeAttribute("src");
  }
  media.classList.add("hidden");
}

function hideAllMedia() {
  [elements.singleImage, elements.singleVideo, elements.compareSourceImage, elements.compareSourceVideo, elements.compareResultImage, elements.compareResultVideo].forEach(hideMedia);
}

function showMedia(imageElement, videoElement, url, kind) {
  hideMedia(imageElement);
  hideMedia(videoElement);
  const media = kind === "video" ? videoElement : imageElement;
  media.src = url;
  media.classList.remove("hidden");
}

function showError(message) {
  elements.error.textContent = message;
  elements.error.classList.remove("hidden");
}

function clearError() {
  elements.error.textContent = "";
  elements.error.classList.add("hidden");
}

function disableDownload() {
  elements.download.href = "#";
  elements.download.removeAttribute("download");
  elements.download.classList.add("pointer-events-none", "text-mist/25");
  elements.download.classList.remove("text-signal", "hover:underline");
}

function activateDownload(item) {
  elements.download.href = item.resultUrl;
  elements.download.download = item.payload.output;
  elements.download.classList.remove("pointer-events-none", "text-mist/25");
  elements.download.classList.add("text-signal", "hover:underline");
}

function revokeSourceUrls() {
  sourceUrls.forEach((url) => URL.revokeObjectURL(url));
  sourceUrls = [];
}

function updateListItem(item) {
  const active = item.index === selectedIndex;
  item.button.setAttribute("aria-selected", String(active));
  item.button.classList.toggle("border-l-signal", active);
  item.button.classList.toggle("border-l-transparent", !active);
  item.button.classList.toggle("bg-signal/10", active);
  item.statusNode.textContent = item.status;
  item.statusNode.className = `item-status font-mono text-[8px] tracking-[0.08em] ${statusClasses(item.status).split(" ").at(-1)}`;
}

function showItem(index) {
  const item = items[index];
  if (!item) return;
  if (!item.sourceUrl) {
    item.sourceUrl = URL.createObjectURL(item.file);
    sourceUrls.push(item.sourceUrl);
  }
  selectedIndex = index;
  items.forEach(updateListItem);
  elements.currentName.textContent = item.file.name;
  elements.currentPath.textContent = item.file.webkitRelativePath || item.file.name;
  elements.currentStatus.textContent = item.status;
  elements.currentStatus.className = `shrink-0 border px-2 py-1 font-mono text-[9px] tracking-[0.12em] ${statusClasses(item.status)}`;
  elements.emptyView.classList.add("hidden");
  hideAllMedia();
  disableDownload();

  if (!item.started) {
    elements.compareView.classList.add("hidden");
    elements.compareView.classList.remove("grid");
    elements.singleView.classList.remove("hidden");
    elements.singleView.classList.add("flex");
    showMedia(elements.singleImage, elements.singleVideo, item.sourceUrl, item.kind);
  } else {
    elements.singleView.classList.add("hidden");
    elements.singleView.classList.remove("flex");
    elements.compareView.classList.remove("hidden");
    elements.compareView.classList.add("grid");
    showMedia(elements.compareSourceImage, elements.compareSourceVideo, item.sourceUrl, item.kind);
    if (item.resultUrl) {
      showMedia(elements.compareResultImage, elements.compareResultVideo, item.resultUrl, item.kind);
      elements.resultPlaceholder.classList.add("hidden");
      activateDownload(item);
    } else {
      elements.resultPlaceholder.classList.remove("hidden");
      elements.resultMessage.textContent = item.error || (item.status === "PROCESSING" ? "正在定位顶部患者信息边界…" : "未生成脱敏结果");
    }
  }

  if (item.payload && item.kind === "image") {
    elements.confidence.textContent = `${(item.payload.confidence * 100).toFixed(1)}%`;
    elements.bbox.textContent = `[${item.payload.bbox.join(", ")}]`;
  } else if (item.payload) {
    elements.confidence.textContent = `${item.payload.frames} FRAMES`;
    elements.bbox.textContent = "MP4 / MP4V";
  } else {
    elements.confidence.textContent = "—";
    elements.bbox.textContent = "—";
  }
  elements.time.textContent = item.elapsed ? `${item.elapsed.toFixed(2)} s` : "—";
}

function navigateSelection(offset) {
  if (!items.length) return;
  const nextIndex = Math.max(0, Math.min(items.length - 1, selectedIndex + offset));
  if (nextIndex === selectedIndex) return;
  showItem(nextIndex);
  items[nextIndex].button.scrollIntoView({ block: "nearest" });
}

function createListItem(file, index, kind) {
  const button = document.createElement("button");
  button.type = "button";
  button.setAttribute("role", "option");
  button.className = "file-list-item grid w-full grid-cols-[2.25rem_minmax(0,1fr)] gap-2 border-l-2 border-l-transparent px-3 py-2.5 text-left transition hover:bg-white/[0.04]";
  button.innerHTML = `<span class="grid size-8 place-items-center border border-white/10 bg-black/20 font-mono text-[8px] text-mist/45">${kind === "video" ? "MP4" : String(index + 1).padStart(2, "0")}</span><span class="min-w-0"><span class="file-name block truncate text-[11px] text-paper/75"></span><span class="mt-1 flex items-center justify-between gap-2"><span class="font-mono text-[8px] text-mist/35">${formatBytes(file.size)}</span><span class="item-status font-mono text-[8px] tracking-[0.08em] text-mist/40">QUEUED</span></span></span>`;
  button.querySelector(".file-name").textContent = file.name;
  const item = { index, file, kind, sourceUrl: null, resultUrl: null, payload: null, status: "QUEUED", started: false, error: null, elapsed: 0, button, statusNode: button.querySelector(".item-status") };
  button.addEventListener("click", () => showItem(index));
  return item;
}

function clearSelection() {
  if (requestController) return;
  revokeSourceUrls();
  items = [];
  selectedIndex = 0;
  stopRequested = false;
  elements.input.value = "";
  elements.fileList.replaceChildren();
  elements.fileList.classList.add("hidden");
  elements.listEmpty.classList.remove("hidden");
  elements.fileSearch.value = "";
  elements.selectionName.textContent = "尚未导入数据";
  elements.selectionMeta.textContent = "支持 JPG / JPEG / PNG / BMP / MP4";
  elements.fileCounter.textContent = "0 FILES";
  elements.summary.textContent = "READY · 等待导入";
  elements.clean.disabled = true;
  elements.clean.textContent = "全部脱敏";
  elements.cleanCurrent.disabled = true;
  elements.cleanCurrent.textContent = "脱敏当前文件";
  elements.clear.disabled = true;
  elements.emptyView.classList.remove("hidden");
  elements.singleView.classList.add("hidden");
  elements.singleView.classList.remove("flex");
  elements.compareView.classList.add("hidden");
  elements.compareView.classList.remove("grid");
  hideAllMedia();
  elements.currentName.textContent = "未选择文件";
  elements.currentPath.textContent = "—";
  elements.currentStatus.textContent = "WAITING";
  elements.currentStatus.className = "shrink-0 border border-white/10 px-2 py-1 font-mono text-[9px] tracking-[0.12em] text-mist/40";
  elements.progress.textContent = "0 / 0";
  elements.confidence.textContent = "—";
  elements.bbox.textContent = "—";
  elements.time.textContent = "—";
  disableDownload();
  clearError();
}

function acceptFiles(fileList) {
  const incoming = [...fileList];
  if (!incoming.length) return;
  clearSelection();
  const supported = incoming
    .map((file) => {
      const extension = extensionOf(file.name);
      const kind = FILE_TYPES.image.extensions.includes(extension) ? "image" : FILE_TYPES.video.extensions.includes(extension) ? "video" : null;
      return { file, kind };
    })
    .filter((entry) => entry.kind);
  if (!supported.length) {
    showError("文件夹中没有找到支持的图片或 MP4 视频。");
    return;
  }
  const selected = supported.sort((a, b) => (a.file.webkitRelativePath || a.file.name).localeCompare(b.file.webkitRelativePath || b.file.name, "zh-CN", { numeric: true }));
  elements.fileList.replaceChildren();
  items = selected.map((entry, index) => createListItem(entry.file, index, entry.kind));
  const listFragment = document.createDocumentFragment();
  items.forEach((item) => listFragment.append(item.button));
  elements.fileList.append(listFragment);
  elements.listEmpty.classList.add("hidden");
  elements.fileList.classList.remove("hidden");
  const totalBytes = selected.reduce((total, entry) => total + entry.file.size, 0);
  const folderName = selected[0].file.webkitRelativePath?.split("/")[0];
  elements.selectionName.textContent = folderName || "已导入数据";
  elements.selectionMeta.textContent = `${selected.length} ${selected.length > 1 ? "FILES" : "FILE"} · ${formatBytes(totalBytes)}${incoming.length > selected.length ? ` · 忽略 ${incoming.length - selected.length}` : ""}`;
  elements.fileCounter.textContent = `${selected.length} ${selected.length > 1 ? "FILES" : "FILE"}`;
  elements.summary.textContent = `READY · ${selected.length} 个文件等待处理`;
  elements.clean.disabled = false;
  elements.cleanCurrent.disabled = false;
  elements.clear.disabled = false;
  elements.progress.textContent = `0 / ${selected.length}`;
  showItem(0);
}

async function requestClean(file, signal) {
  const formData = new FormData();
  formData.append("file", file.file, file.file.name);
  const response = await fetch(FILE_TYPES[file.kind].endpoint, { method: "POST", body: formData, signal });
  const payload = await response.json();
  if (!response.ok || !payload.success) throw new Error(ERROR_MESSAGES[payload.error] || payload.error || "处理失败，请重试。");
  return payload;
}

async function processItem(item) {
  const startedAt = performance.now();
  item.started = true;
  item.status = "PROCESSING";
  item.error = null;
  item.resultUrl = null;
  item.payload = null;
  item.elapsed = 0;
  updateListItem(item);
  if (selectedIndex === item.index) showItem(item.index);
  try {
    const payload = await requestClean(item, requestController.signal);
    item.payload = payload;
    item.resultUrl = `/api/v1/output/${encodeURIComponent(payload.output)}`;
    item.status = "COMPLETE";
  } catch (error) {
    item.status = error.name === "AbortError" ? "STOPPED" : "FAILED";
    item.error = error.name === "AbortError" ? "处理已停止" : error.message || "无法连接清洗服务";
  }
  item.elapsed = (performance.now() - startedAt) / 1000;
  updateListItem(item);
  if (selectedIndex === item.index) showItem(item.index);
}

function updateOverallProgress() {
  const completed = items.filter((item) => item.status === "COMPLETE").length;
  const failed = items.filter((item) => item.status === "FAILED").length;
  const processed = completed + failed;
  elements.progress.textContent = `${processed} / ${items.length}`;
  elements.summary.textContent = `PROCESSING · ${completed} 完成 · ${failed} 失败 · ${items.length - processed} 等待`;
  elements.clean.textContent = `处理中 ${processed} / ${items.length}`;
}

function setProcessingControls(processing) {
  elements.clean.disabled = processing;
  elements.cleanCurrent.disabled = processing;
  elements.clear.disabled = processing;
  elements.stop.classList.toggle("hidden", !processing);
}

async function processCurrent() {
  if (!items.length || requestController) return;
  clearError();
  stopRequested = false;
  requestController = new AbortController();
  const item = items[selectedIndex];
  setProcessingControls(true);
  elements.cleanCurrent.textContent = "正在脱敏…";
  await processItem(item);
  updateOverallProgress();
  const completed = items.filter((entry) => entry.status === "COMPLETE").length;
  const failed = items.filter((entry) => entry.status === "FAILED").length;
  elements.summary.textContent = item.status === "COMPLETE" ? `COMPLETE · 当前文件脱敏完成 · 共 ${completed} 完成` : `${item.status} · ${item.error || "当前文件未完成"} · 共 ${failed} 失败`;
  requestController = null;
  setProcessingControls(false);
  elements.cleanCurrent.textContent = "重新脱敏当前文件";
  elements.clean.textContent = "全部脱敏";
}

async function processSelection() {
  if (!items.length || requestController) return;
  clearError();
  stopRequested = false;
  requestController = new AbortController();
  batchStartedAt = performance.now();
  setProcessingControls(true);
  for (const item of items) {
    if (stopRequested) break;
    await processItem(item);
    updateOverallProgress();
    if (item.status === "STOPPED") break;
  }
  const completed = items.filter((item) => item.status === "COMPLETE").length;
  const failed = items.filter((item) => item.status === "FAILED").length;
  const elapsed = (performance.now() - batchStartedAt) / 1000;
  elements.summary.textContent = stopRequested ? `STOPPED · ${completed} 完成` : `COMPLETE · ${completed} 完成 · ${failed} 失败 · ${elapsed.toFixed(1)} 秒`;
  requestController = null;
  setProcessingControls(false);
  elements.clean.textContent = "重新脱敏";
  elements.cleanCurrent.textContent = "脱敏当前文件";
}

async function checkHealth() {
  try {
    const response = await fetch("/health", { headers: { Accept: "application/json" } });
    if (!response.ok) throw new Error("unavailable");
    elements.healthDot.classList.remove("bg-amber");
    elements.healthDot.classList.add("bg-signal");
    elements.healthText.textContent = "服务正常";
  } catch {
    elements.healthDot.classList.remove("bg-amber");
    elements.healthDot.classList.add("bg-[#e56a5d]");
    elements.healthText.textContent = "服务不可用";
  }
}

elements.input.addEventListener("change", () => acceptFiles(elements.input.files));
elements.clear.addEventListener("click", clearSelection);
elements.cleanCurrent.addEventListener("click", processCurrent);
elements.clean.addEventListener("click", processSelection);
elements.stop.addEventListener("click", () => {
  stopRequested = true;
  requestController?.abort();
});
elements.fileSearch.addEventListener("input", () => {
  const query = elements.fileSearch.value.trim().toLocaleLowerCase();
  items.forEach((item) => {
    const path = item.file.webkitRelativePath || item.file.name;
    item.button.classList.toggle("hidden", Boolean(query) && !path.toLocaleLowerCase().includes(query));
  });
});
document.addEventListener("keydown", (event) => {
  if (event.defaultPrevented || event.metaKey || event.ctrlKey || event.altKey) return;
  const target = event.target;
  if (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement || target?.isContentEditable) return;
  const key = event.key.toLowerCase();
  if (key !== "a" && key !== "d") return;
  event.preventDefault();
  navigateSelection(key === "a" ? -1 : 1);
});

clearSelection();
checkHealth();
