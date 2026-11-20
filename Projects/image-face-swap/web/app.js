"use strict";

const $ = (id) => document.getElementById(id);
const state = { status: null, analysis: null, selectedFace: null, targetFilename: "", modelId: "", busy: false, libraryBusy: false, identityId: null, identity: null, statusRequest: 0, identityRequest: 0 };

async function api(path, options = {}) {
  const response = await fetch(path, { ...options, headers: { ...(options.body ? { "Content-Type": "application/json" } : {}), ...options.headers }, cache: "no-store" });
  const text = await response.text();
  let data;
  try { data = text ? JSON.parse(text) : {}; } catch { data = null; }
  if (!response.ok) {
    const detail = data?.error ?? data?.detail ?? data?.message;
    throw new Error(typeof detail === "string" ? detail : detail ? JSON.stringify(detail) : `本地服务返回错误（${response.status}）`);
  }
  if (data === null) throw new Error("本地服务返回了无法识别的数据，请检查服务日志。");
  return data;
}

function showError(error) {
  $("global-error-text").textContent = error instanceof Error ? error.message : String(error);
  $("global-error").hidden = false;
}

function clearError() { $("global-error").hidden = true; }
function deviceText(value) {
  if (typeof value === "string") return value;
  if (!value) return "设备信息未提供";
  return value.name || value.label || value.type || JSON.stringify(value);
}
function setPage(page) {
  const library = page === "library";
  $("workspace-page").hidden = library;
  $("library-page").hidden = !library;
  document.querySelectorAll("[data-page]").forEach((button) => { button.classList.toggle("active", button.dataset.page === page); button.setAttribute("aria-current", button.dataset.page === page ? "page" : "false"); });
  $("breadcrumb").replaceChildren(document.createTextNode(`工作空间 / ${library ? "人物素材库" : "图片换脸"}`));
  if (location.hash !== `#${page}`) history.replaceState(null, "", `#${page}`);
}

async function refreshStatus() {
  const request = ++state.statusRequest;
  $("refresh-status").disabled = true;
  $("refresh-models").disabled = true;
  try {
    const status = await api("/api/status");
    if (request !== state.statusRequest) return;
    state.status = status;
    const issues = Array.isArray(status.issues) ? status.issues : [];
    $("runtime-issues").textContent = issues.join("；");
    $("runtime-issues").hidden = !issues.length;
    $("connection-light").className = "status-light connected";
    $("runtime-state").textContent = "本地服务已连接";
    $("runtime-device").textContent = deviceText(status.device);
    $("runtime-device").title = deviceText(status.device);
    const models = Array.isArray(status.models) ? status.models : [];
    if (!models.some((model) => model.id === state.modelId)) state.modelId = models[0]?.id ?? "";
    $("model-select").replaceChildren();
    if (!models.length) $("model-select").add(new Option("暂无可用人物模型", ""));
    for (const model of models) $("model-select").add(new Option(model.name || model.id, model.id));
    $("model-select").value = state.modelId;
    $("model-select").disabled = !models.length || state.busy;
    $("no-models").hidden = models.length > 0;
    renderModelDescription();
    renderIdentities();
    updateSwapButton();
  } catch (error) {
    if (request !== state.statusRequest) return;
    $("connection-light").className = "status-light failed";
    $("runtime-state").textContent = "本地服务连接失败";
    $("runtime-device").textContent = "检查服务是否运行";
    showError(error);
  } finally {
    if (request === state.statusRequest) { $("refresh-status").disabled = false; $("refresh-models").disabled = false; }
  }
}

function renderModelDescription() {
  const model = state.status?.models?.find((entry) => entry.id === state.modelId);
  $("model-description").textContent = model ? [model.face_type ? `脸型 ${model.face_type}` : "", model.resolution ? `${model.resolution} × ${model.resolution}` : "", "已训练人物模型"].filter(Boolean).join(" · ") : "人物身份来自已训练模型。";
}

function updateSwapButton() {
  const ready = !state.busy && Boolean(state.analysis?.image_id) && state.selectedFace !== null && Boolean(state.modelId) && state.status?.assets_ready !== false;
  $("swap-button").disabled = !ready;
  let help = "准备就绪，可以开始处理。";
  if (state.busy) help = "正在处理，请稍候。";
  else if (state.status?.assets_ready === false) help = "检测或推理资源尚未就绪，请检查本地服务。";
  else if (!state.modelId) help = "请先导入一个已训练的人物模型。";
  else if (!state.analysis) help = "请导入一张目标图片。";
  else if (state.selectedFace === null) help = "当前图片没有可选择的人脸，请更换图片。";
  $("process-help").textContent = help;
}

function setBusy(busy, message = "正在处理…") {
  state.busy = busy;
  document.body.classList.toggle("busy", busy);
  $("work-progress").hidden = !busy;
  $("work-progress-text").textContent = message;
  $("reset-workspace").disabled = busy;
  $("target-file").disabled = busy;
  $("model-select").disabled = busy || !state.status?.models?.length;
  for (const id of ["feather", "erode", "color-match"]) $(id).disabled = busy;
  document.querySelectorAll(".face-card").forEach((button) => { button.disabled = busy; });
  $("swap-button").querySelector("span").textContent = busy ? "正在处理…" : "开始图片换脸";
  updateSwapButton();
}

function clearResult() {
  $("result-image").hidden = true;
  $("result-image").removeAttribute("src");
  $("result-empty").hidden = false;
  $("result-meta").textContent = "等待处理";
  $("download-result").removeAttribute("href");
  $("download-result").classList.add("disabled");
  $("download-result").setAttribute("aria-disabled", "true");
  $("result-warnings").hidden = true;
}

function resetWorkspace() {
  if (state.busy) return;
  state.analysis = null;
  state.selectedFace = null;
  state.targetFilename = "";
  $("target-file").value = "";
  $("target-image").hidden = true;
  $("target-image").removeAttribute("src");
  $("target-empty").hidden = false;
  $("replace-image").hidden = true;
  $("image-dimensions").textContent = "未导入";
  $("image-filename").textContent = "图片会在本地服务中处理";
  $("face-count").textContent = "尚未检测";
  $("face-list").replaceChildren();
  $("faces-empty").hidden = false;
  $("faces-empty").textContent = "导入图片后，选择需要替换的人脸。";
  $("face-quality").hidden = true;
  clearResult();
  clearError();
  updateSwapButton();
}

function imageBase64(file) {
  if (file?.size > 30 * 1024 * 1024) return Promise.reject(new Error("单张图片不能超过 30 MiB。"));
  if (!file || !(/^image\/(jpeg|png|webp)$/i.test(file.type) || /\.(jpe?g|png|webp)$/i.test(file.name))) return Promise.reject(new Error("请选择 JPG、PNG 或 WEBP 图片。"));
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",")[1]);
    reader.onerror = () => reject(new Error(`无法读取图片：${file.name}`));
    reader.readAsDataURL(file);
  });
}

async function analyzeImage(file) {
  if (!file || state.busy) return;
  resetWorkspace();
  clearError();
  setBusy(true, "正在读取图片、检测人脸并分析质量…");
  try {
    const image = await imageBase64(file);
    $("target-image").src = `data:${file.type || "image/jpeg"};base64,${image}`;
    $("target-image").hidden = false;
    $("target-empty").hidden = true;
    $("image-filename").textContent = file.name;
    const analysis = await api("/api/analyze", { method: "POST", body: JSON.stringify({ image }) });
    if (!analysis.image_id) throw new Error("人脸分析结果缺少图片标识，请检查本地服务。");
    state.analysis = analysis;
    state.targetFilename = file.name;
    const faces = Array.isArray(analysis.faces) ? analysis.faces : [];
    state.selectedFace = faces.length ? faces[0].index : null;
    if (analysis.url) $("target-image").src = analysis.url;
    $("replace-image").hidden = false;
    $("image-dimensions").textContent = `${analysis.width} × ${analysis.height}`;
    renderFaces();
  } catch (error) { showError(error); } finally { setBusy(false); }
}

function qualityChips(quality = {}) {
  const fragment = document.createDocumentFragment();
  for (const [key, label] of [["sharpness", "清晰度"], ["brightness", "亮度"], ["yaw", "朝向"]]) {
    if (quality[key] === undefined || quality[key] === null) continue;
    const chip = document.createElement("span");
    chip.className = "quality-chip";
    const number = Number(quality[key]);
    chip.textContent = `${label} ${Number.isFinite(number) ? number.toFixed(key === "yaw" ? 2 : 1) : String(quality[key])}`;
    fragment.append(chip);
  }
  for (const warning of Array.isArray(quality.warnings) ? quality.warnings : []) {
    const chip = document.createElement("span");
    chip.className = "quality-chip warning";
    chip.textContent = String(warning);
    fragment.append(chip);
  }
  return fragment;
}

function renderFaces() {
  const faces = state.analysis?.faces ?? [];
  $("face-count").textContent = `检测到 ${faces.length} 张人脸`;
  $("faces-empty").hidden = faces.length > 0;
  $("faces-empty").textContent = "没有检测到可用人脸。请尝试面部更清晰、角度更正的图片。";
  $("face-list").replaceChildren();
  faces.forEach((face, position) => {
    const button = document.createElement("button");
    button.className = `face-card${face.index === state.selectedFace ? " selected" : ""}`;
    button.type = "button";
    button.setAttribute("aria-pressed", face.index === state.selectedFace ? "true" : "false");
    if (face.crop_url) { const img = document.createElement("img"); img.src = face.crop_url; img.alt = `目标人脸 ${position + 1}`; button.append(img); }
    const caption = document.createElement("span");
    const title = document.createElement("strong"); title.textContent = `人脸 ${position + 1}`;
    const sub = document.createElement("small"); sub.textContent = face.index === state.selectedFace ? "当前选择" : "点击选择";
    caption.append(title, sub); button.append(caption);
    button.addEventListener("click", () => { if (state.busy || state.selectedFace === face.index) return; state.selectedFace = face.index; clearResult(); renderFaces(); updateSwapButton(); });
    button.disabled = state.busy;
    $("face-list").append(button);
  });
  const selected = faces.find((face) => face.index === state.selectedFace);
  $("face-quality").replaceChildren(qualityChips(selected?.quality));
  $("face-quality").hidden = !selected || !$("face-quality").childNodes.length;
}

async function swapImage() {
  if (state.busy || !state.analysis || state.selectedFace === null || !state.modelId) return;
  clearError(); clearResult();
  setBusy(true, "正在载入人物模型、生成换脸结果并融合边缘…首次加载可能需要较长时间。");
  try {
    const result = await api("/api/swap", { method: "POST", body: JSON.stringify({ image_id: state.analysis.image_id, face_index: state.selectedFace, model_id: state.modelId, feather: Number($("feather").value), erode: Number($("erode").value), color_match: $("color-match").checked }) });
    if (!result.result_url) throw new Error("处理返回成功，但没有生成结果图片地址。");
    $("result-image").src = result.result_url;
    $("result-image").hidden = false; $("result-empty").hidden = true;
    const elapsed = Number(result.elapsed_ms);
    $("result-meta").textContent = `${Number.isFinite(elapsed) ? (elapsed / 1000).toFixed(1) + " 秒 · " : ""}${deviceText(result.device ?? state.status?.device)}`;
    $("download-result").href = result.result_url;
    $("download-result").download = `${state.targetFilename.replace(/\.[^.]+$/, "") || "facework"}-swap.png`;
    $("download-result").classList.remove("disabled"); $("download-result").setAttribute("aria-disabled", "false");
    if (Array.isArray(result.warnings) && result.warnings.length) {
      const list = document.createElement("ul");
      result.warnings.forEach((warning) => { const li = document.createElement("li"); li.textContent = String(warning); list.append(li); });
      $("result-warnings").replaceChildren(list); $("result-warnings").hidden = false;
    }
  } catch (error) { showError(error); } finally { setBusy(false); }
}

function renderIdentities() {
  const identities = state.status?.identities ?? [];
  $("identity-count").textContent = String(identities.length);
  $("people-empty").hidden = identities.length > 0;
  $("identity-list").replaceChildren();
  identities.forEach((identity) => {
    const button = document.createElement("button");
    button.className = `identity-button${identity.id === state.identityId ? " selected" : ""}`;
    button.type = "button"; button.disabled = state.libraryBusy;
    const avatar = document.createElement("span"); avatar.className = "identity-avatar"; avatar.textContent = Array.from(identity.name || "人")[0];
    const caption = document.createElement("span");
    const name = document.createElement("strong"); name.textContent = identity.name;
    const count = document.createElement("small"); count.textContent = `${identity.count ?? 0} 张素材`;
    caption.append(name, count); button.append(avatar, caption);
    button.addEventListener("click", () => { if (!state.libraryBusy) loadIdentity(identity.id); });
    $("identity-list").append(button);
  });
  if (!state.identityId && identities.length) loadIdentity(identities[0].id);
}

async function loadIdentity(id) {
  const request = ++state.identityRequest;
  state.identityId = id;
  $("selected-identity-name").textContent = "正在读取素材…";
  $("library-items").replaceChildren();
  $("library-empty").hidden = true;
  $("add-library-images").disabled = true;
  $("export-identity").classList.add("disabled");
  $("export-identity").setAttribute("aria-disabled", "true");
  renderIdentities();
  try {
    const identity = await api(`/api/identities/${encodeURIComponent(id)}`);
    if (request !== state.identityRequest) return;
    state.identity = identity;
    renderLibraryItems();
  } catch (error) {
    if (request !== state.identityRequest) return;
    state.identity = null;
    $("selected-identity-name").textContent = "素材读取失败";
    $("library-summary").textContent = "点击左侧人物可重新读取。";
    showError(error);
  }
}

function renderLibraryItems() {
  const identity = state.identity;
  if (!identity) return;
  const items = Array.isArray(identity.items) ? identity.items : [];
  $("selected-identity-name").textContent = identity.name;
  const warningsCount = items.filter((item) => item.quality?.warnings?.length).length;
  $("library-summary").textContent = `${items.length} 张素材${warningsCount ? ` · ${warningsCount} 张有质量提示` : ""}`;
  if (identity.coverage) $("library-summary").textContent += ` · 朝向分布：左 ${identity.coverage.left} / 正 ${identity.coverage.front} / 右 ${identity.coverage.right}（估算）`;
  $("library-empty").hidden = items.length > 0;
  $("library-items").replaceChildren();
  $("add-library-images").disabled = state.libraryBusy;
  $("export-identity").href = `/api/identities/${encodeURIComponent(identity.id)}/export`;
  $("export-identity").download = `${identity.name}-faces.zip`;
  $("export-identity").classList.toggle("disabled", !items.length || state.libraryBusy);
  $("export-identity").setAttribute("aria-disabled", String(!items.length || state.libraryBusy));
  for (const item of items) {
    const card = document.createElement("article"); card.className = "library-item";
    const imageWrap = document.createElement("div"); imageWrap.className = "library-item-image";
    const image = document.createElement("img"); image.src = item.url; image.alt = item.filename || "人物素材"; image.loading = "lazy";
    const remove = document.createElement("button"); remove.className = "delete-item"; remove.type = "button"; remove.textContent = "×"; remove.title = "删除这张素材"; remove.setAttribute("aria-label", `删除素材 ${item.filename || item.id}`); remove.disabled = state.libraryBusy;
    remove.addEventListener("click", () => deleteItem(identity.id, item));
    imageWrap.append(image, remove);
    const info = document.createElement("div"); info.className = "library-item-info";
    const name = document.createElement("div"); name.className = "library-item-name"; name.textContent = item.filename || item.id; name.title = name.textContent;
    const quality = document.createElement("div"); quality.className = "quality-report"; quality.append(qualityChips(item.quality));
    info.append(name, quality); card.append(imageWrap, info); $("library-items").append(card);
  }
}

function setLibraryBusy(busy, message = "正在处理素材…") {
  state.libraryBusy = busy;
  $("library-progress").hidden = !busy;
  $("library-progress-text").textContent = message;
  $("create-identity-button").disabled = busy;
  $("identity-name").disabled = busy;
  $("library-files").disabled = busy;
  $("add-library-images").disabled = busy || !state.identity;
  document.querySelectorAll(".identity-button, .delete-item").forEach((button) => { button.disabled = busy; });
  if (state.identity) { $("export-identity").classList.toggle("disabled", busy || !state.identity.items?.length); $("export-identity").setAttribute("aria-disabled", String(busy || !state.identity.items?.length)); }
}

async function createIdentity(event) {
  event.preventDefault();
  const name = $("identity-name").value.trim();
  if (!name || state.libraryBusy) return;
  clearError(); setLibraryBusy(true, "正在创建人物素材库…");
  try {
    const identity = await api("/api/identities", { method: "POST", body: JSON.stringify({ name }) });
    if (!identity.id) throw new Error("创建人物后没有收到有效标识。");
    $("identity-name").value = "";
    state.identityId = identity.id;
    await refreshStatus();
    await loadIdentity(identity.id);
  } catch (error) { showError(error); } finally { setLibraryBusy(false); }
}

async function uploadLibraryFiles(files) {
  if (!files.length || !state.identityId || state.libraryBusy) return;
  const identityId = state.identityId;
  const errors = [];
  clearError(); setLibraryBusy(true);
  try {
    for (let index = 0; index < files.length; index += 1) {
      const file = files[index];
      $("library-progress-text").textContent = `正在整理 ${index + 1} / ${files.length}：${file.name}`;
      try {
        const image = await imageBase64(file);
        await api(`/api/identities/${encodeURIComponent(identityId)}/items`, { method: "POST", body: JSON.stringify({ image, filename: file.name }) });
      } catch (error) { errors.push(`${file.name}：${error.message}`); }
    }
    await loadIdentity(identityId);
    await refreshStatus();
    if (errors.length) showError(`以下 ${errors.length} 张图片未能加入素材库：${errors.join("；")}`);
  } finally { $("library-files").value = ""; setLibraryBusy(false); }
}

async function deleteItem(identityId, item) {
  if (state.libraryBusy || !window.confirm(`确定删除这张素材吗？\n${item.filename || item.id}\n原始上传照片不会被修改。`)) return;
  clearError(); setLibraryBusy(true, "正在删除素材…");
  try {
    await api(`/api/identities/${encodeURIComponent(identityId)}/items/${encodeURIComponent(item.id)}`, { method: "DELETE" });
    await loadIdentity(identityId);
    await refreshStatus();
  } catch (error) { showError(error); } finally { setLibraryBusy(false); }
}

async function importModel(event) {
  event.preventDefault();
  const button = $("import-model-button");
  button.disabled = true;
  button.textContent = "检查模型…";
  clearError();
  try {
    await api("/api/models", { method: "POST", body: JSON.stringify({ directory: $("model-directory").value.trim(), name: $("model-name").value.trim() }) });
    await refreshStatus();
    $("import-feedback").textContent = "模型已加入列表。请选择模型并生成预览，检查实际效果。";
  } catch (error) { $("import-feedback").textContent = ""; showError(error); }
  finally { button.disabled = false; button.textContent = "导入模型"; }
}

$("import-model-form").addEventListener("submit", importModel);
document.querySelectorAll("[data-page]").forEach((button) => button.addEventListener("click", () => setPage(button.dataset.page)));
window.addEventListener("hashchange", () => setPage(location.hash === "#library" ? "library" : "workspace"));
$("dismiss-error").addEventListener("click", clearError);
$("refresh-status").addEventListener("click", refreshStatus);
$("refresh-models").addEventListener("click", refreshStatus);
$("reset-workspace").addEventListener("click", resetWorkspace);
$("target-file").addEventListener("change", (event) => analyzeImage(event.target.files[0]));
$("target-dropzone").addEventListener("keydown", (event) => { if ((event.key === "Enter" || event.key === " ") && !state.busy) { event.preventDefault(); $("target-file").click(); } });
for (const name of ["dragenter", "dragover"]) $("target-dropzone").addEventListener(name, (event) => { event.preventDefault(); if (!state.busy) $("target-dropzone").classList.add("dragging"); });
for (const name of ["dragleave", "drop"]) $("target-dropzone").addEventListener(name, (event) => { event.preventDefault(); $("target-dropzone").classList.remove("dragging"); });
$("target-dropzone").addEventListener("drop", (event) => { if (!state.busy) analyzeImage(event.dataTransfer.files[0]); });
$("model-select").addEventListener("change", () => { state.modelId = $("model-select").value; renderModelDescription(); clearResult(); updateSwapButton(); });
for (const id of ["feather", "erode"]) $(id).addEventListener("input", () => { $(`${id}-value`).value = $(id).value; if (!$("result-image").hidden) $("process-help").textContent = "参数已改变，重新生成以应用调整。"; });
$("color-match").addEventListener("change", () => { if (!$("result-image").hidden) $("process-help").textContent = "参数已改变，重新生成以应用调整。"; });
$("swap-button").addEventListener("click", swapImage);
$("create-identity-form").addEventListener("submit", createIdentity);
$("add-library-images").addEventListener("click", () => $("library-files").click());
$("library-files").addEventListener("change", (event) => uploadLibraryFiles(Array.from(event.target.files)));
for (const id of ["download-result", "export-identity"]) $(id).addEventListener("click", (event) => { if ($(id).getAttribute("aria-disabled") === "true") event.preventDefault(); });
$("result-image").addEventListener("error", () => { if ($("result-image").hasAttribute("src")) { showError("无法加载结果图片。请检查本地服务后重新生成。"); clearResult(); } });
setPage(location.hash === "#library" ? "library" : "workspace");
refreshStatus();
