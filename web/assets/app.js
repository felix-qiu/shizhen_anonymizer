"use strict";

const MODES = {
  image: {
    endpoint: "/api/v1/image/clean",
    accept: ".jpg,.jpeg,.png,.bmp,image/jpeg,image/png,image/bmp",
    extensions: ["jpg", "jpeg", "png", "bmp"],
    title: "拖入超声图片",
    hint: "或点击选择 JPG、PNG、BMP 文件",
    action: "开始清洗图片",
  },
  video: {
    endpoint: "/api/v1/video/clean",
    accept: ".mp4,video/mp4",
    extensions: ["mp4"],
    title: "拖入超声视频",
    hint: "或点击选择 MP4 文件",
    action: "开始清洗视频",
  },
};

const ERROR_MESSAGES = {
  FILE_NOT_FOUND: "文件不存在或结果已被移除。",
  INVALID_FILE: "文件格式或内容无效，请重新选择。",
  MODEL_NOT_FOUND: "服务尚未加载 ROI 模型，请检查模型文件。",
  ROI_NOT_FOUND: "没有检测到有效超声区域，请更换样本或调整模型。",
  PROCESS_FAILED: "处理失败，请查看服务日志后重试。",
};

const elements = {
  tabs: [...document.querySelectorAll(".mode-tab")],
  input: document.querySelector("#file-input"),
  dropZone: document.querySelector("#drop-zone"),
  dropTitle: document.querySelector("#drop-title"),
  dropHint: document.querySelector("#drop-hint"),
  fileCard: document.querySelector("#file-card"),
  fileName: document.querySelector("#file-name"),
  fileMeta: document.querySelector("#file-meta"),
  removeFile: document.querySelector("#remove-file"),
  submit: document.querySelector("#submit-button"),
  cancel: document.querySelector("#cancel-button"),
  error: document.querySelector("#error-box"),
  stage: document.querySelector("#preview-stage"),
  state: document.querySelector("#preview-state"),
  sourcePlaceholder: document.querySelector("#source-placeholder"),
  sourceImage: document.querySelector("#source-image"),
  sourceVideo: document.querySelector("#source-video"),
  resultPlaceholder: document.querySelector("#result-placeholder"),
  resultMessage: document.querySelector("#result-message"),
  resultImage: document.querySelector("#result-image"),
  resultVideo: document.querySelector("#result-video"),
  download: document.querySelector("#download-button"),
  metricEndpoint: document.querySelector("#metric-endpoint"),
  metricPrimaryLabel: document.querySelector("#metric-primary-label"),
  metricPrimary: document.querySelector("#metric-primary"),
  metricSecondaryLabel: document.querySelector("#metric-secondary-label"),
  metricSecondary: document.querySelector("#metric-secondary"),
  metricTime: document.querySelector("#metric-time"),
  healthDot: document.querySelector("#health-dot"),
  healthText: document.querySelector("#health-text"),
};

let mode = "image";
let selectedFile = null;
let sourceUrl = null;
let requestController = null;

function extensionOf(filename) {
  return filename.includes(".") ? filename.split(".").pop().toLowerCase() : "";
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}

function hideMedia(element) {
  if (element.tagName === "VIDEO") {
    element.pause();
    element.removeAttribute("src");
    element.load();
  } else {
    element.removeAttribute("src");
  }
  element.classList.add("hidden");
}

function clearError() {
  elements.error.textContent = "";
  elements.error.classList.add("hidden");
}

function showError(message) {
  elements.error.textContent = message;
  elements.error.classList.remove("hidden");
}

function clearResult() {
  hideMedia(elements.resultImage);
  hideMedia(elements.resultVideo);
  elements.resultPlaceholder.classList.remove("hidden");
  elements.resultMessage.textContent = "清洗完成后，标准化 ROI 将显示在这里";
  elements.download.href = "#";
  elements.download.removeAttribute("download");
  elements.download.classList.add("pointer-events-none", "border-white/10", "text-mist/30");
  elements.download.classList.remove("border-signal/50", "text-signal", "hover:bg-signal/10");
  elements.metricPrimary.textContent = "—";
  elements.metricSecondary.textContent = "—";
  elements.metricTime.textContent = "—";
  elements.state.textContent = selectedFile ? "INPUT READY" : "WAITING";
  elements.state.classList.remove("border-signal/30", "text-signal", "border-amber/30", "text-amber");
}

function clearSelection() {
  selectedFile = null;
  elements.input.value = "";
  elements.fileCard.classList.add("hidden");
  elements.fileCard.classList.remove("flex");
  elements.submit.disabled = true;
  elements.submit.textContent = "选择文件后开始";
  if (sourceUrl) URL.revokeObjectURL(sourceUrl);
  sourceUrl = null;
  hideMedia(elements.sourceImage);
  hideMedia(elements.sourceVideo);
  elements.sourcePlaceholder.classList.remove("hidden");
  clearResult();
  clearError();
}

function selectMode(nextMode) {
  if (!MODES[nextMode] || nextMode === mode) return;
  mode = nextMode;
  clearSelection();
  const config = MODES[mode];
  elements.input.accept = config.accept;
  elements.dropTitle.textContent = config.title;
  elements.dropHint.textContent = config.hint;
  elements.metricEndpoint.textContent = config.endpoint;
  elements.metricPrimaryLabel.textContent = mode === "image" ? "CONFIDENCE" : "PROCESSED";
  elements.metricSecondaryLabel.textContent = mode === "image" ? "ROI BBOX" : "OUTPUT TYPE";

  elements.tabs.forEach((tab) => {
    const active = tab.dataset.mode === mode;
    tab.setAttribute("aria-selected", String(active));
    tab.classList.toggle("bg-paper", active);
    tab.classList.toggle("text-ink", active);
    tab.classList.toggle("font-semibold", active);
    tab.classList.toggle("text-mist/65", !active);
    tab.classList.toggle("font-medium", !active);
  });
}

function renderSource(file) {
  if (sourceUrl) URL.revokeObjectURL(sourceUrl);
  sourceUrl = URL.createObjectURL(file);
  elements.sourcePlaceholder.classList.add("hidden");
  hideMedia(elements.sourceImage);
  hideMedia(elements.sourceVideo);
  const media = mode === "image" ? elements.sourceImage : elements.sourceVideo;
  media.src = sourceUrl;
  media.classList.remove("hidden");
}

function acceptFile(file) {
  if (!file) return;
  const detectedMode = extensionOf(file.name) === "mp4" ? "video" : "image";
  if (detectedMode !== mode && MODES[detectedMode].extensions.includes(extensionOf(file.name))) {
    selectMode(detectedMode);
  }
  if (!MODES[mode].extensions.includes(extensionOf(file.name))) {
    showError(`当前仅支持 ${MODES[mode].extensions.join("、").toUpperCase()} 文件。`);
    return;
  }

  clearError();
  clearResult();
  selectedFile = file;
  elements.fileName.textContent = file.name;
  elements.fileMeta.textContent = `${formatBytes(file.size)} · ${mode === "image" ? "IMAGE" : "VIDEO"}`;
  elements.fileCard.classList.remove("hidden");
  elements.fileCard.classList.add("flex");
  elements.submit.disabled = false;
  elements.submit.textContent = MODES[mode].action;
  elements.state.textContent = "INPUT READY";
  renderSource(file);
}

function setProcessing(processing) {
  elements.stage.classList.toggle("is-processing", processing);
  elements.submit.disabled = processing;
  elements.cancel.classList.toggle("hidden", !processing);
  elements.state.textContent = processing ? "PROCESSING" : elements.state.textContent;
  elements.state.classList.toggle("border-amber/30", processing);
  elements.state.classList.toggle("text-amber", processing);
  if (processing) {
    elements.submit.textContent = mode === "image" ? "正在检测 ROI…" : "正在逐帧处理…";
    elements.resultMessage.textContent = mode === "image" ? "模型正在定位有效超声区域" : "长视频处理需要一些时间，请保持页面开启";
  }
}

function showResult(payload, elapsedSeconds) {
  const outputUrl = `/api/v1/output/${encodeURIComponent(payload.output)}`;
  elements.resultPlaceholder.classList.add("hidden");
  const media = mode === "image" ? elements.resultImage : elements.resultVideo;
  media.src = outputUrl;
  media.classList.remove("hidden");
  elements.download.href = outputUrl;
  elements.download.download = payload.output;
  elements.download.classList.remove("pointer-events-none", "border-white/10", "text-mist/30");
  elements.download.classList.add("border-signal/50", "text-signal", "hover:bg-signal/10");
  elements.state.textContent = "CLEAN COMPLETE";
  elements.state.classList.remove("border-amber/30", "text-amber");
  elements.state.classList.add("border-signal/30", "text-signal");
  elements.metricTime.textContent = `${elapsedSeconds.toFixed(2)} s`;

  if (mode === "image") {
    elements.metricPrimary.textContent = `${(payload.confidence * 100).toFixed(1)}%`;
    elements.metricSecondary.textContent = `[${payload.bbox.join(", ")}]`;
  } else {
    elements.metricPrimary.textContent = `${payload.frames} FRAMES`;
    elements.metricSecondary.textContent = "MP4 / MP4V";
  }
}

async function processFile() {
  if (!selectedFile || requestController) return;
  clearError();
  clearResult();
  setProcessing(true);
  requestController = new AbortController();
  const startedAt = performance.now();
  const formData = new FormData();
  formData.append("file", selectedFile, selectedFile.name);

  try {
    const response = await fetch(MODES[mode].endpoint, {
      method: "POST",
      body: formData,
      signal: requestController.signal,
    });
    const payload = await response.json();
    if (!response.ok || !payload.success) {
      throw new Error(ERROR_MESSAGES[payload.error] || payload.error || "处理失败，请重试。");
    }
    showResult(payload, (performance.now() - startedAt) / 1000);
  } catch (error) {
    if (error.name === "AbortError") {
      showError("已停止等待。服务端可能仍在完成当前视频处理。 ");
    } else {
      showError(error.message || "无法连接清洗服务，请确认服务已经启动。");
    }
    elements.state.textContent = "FAILED";
    elements.state.classList.remove("border-amber/30", "text-amber");
  } finally {
    requestController = null;
    setProcessing(false);
    elements.submit.disabled = !selectedFile;
    elements.submit.textContent = selectedFile ? MODES[mode].action : "选择文件后开始";
  }
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

elements.tabs.forEach((tab) => tab.addEventListener("click", () => selectMode(tab.dataset.mode)));
elements.input.addEventListener("change", () => acceptFile(elements.input.files[0]));
elements.removeFile.addEventListener("click", clearSelection);
elements.submit.addEventListener("click", processFile);
elements.cancel.addEventListener("click", () => requestController?.abort());

["dragenter", "dragover"].forEach((eventName) => {
  elements.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropZone.classList.add("drop-active");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  elements.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropZone.classList.remove("drop-active");
  });
});

elements.dropZone.addEventListener("drop", (event) => acceptFile(event.dataTransfer.files[0]));

checkHealth();
