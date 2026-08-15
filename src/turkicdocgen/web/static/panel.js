const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
const uiLanguage = () => document.documentElement.lang === "en" ? "en" : "ru";
const message = (ru, en) => uiLanguage() === "ru" ? ru : en;

document.addEventListener("DOMContentLoaded", () => {
  initNavigation();
  initTheme();
  initLanguage();
  initAutoSubmit();
  initFilters();
  initModal();
  initRunActions();
  initGeneration();
  initTabs();
  initZoneOverlay();
  initGalleryZones();
  initImageZoom();
  initReview();
  initImageFallbacks();
  initKeyboardNavigation();
});

function initNavigation() {
  const button = $("[data-mobile-nav]");
  const nav = $(".primary-nav");
  button?.addEventListener("click", () => {
    const open = nav?.classList.toggle("open") || false;
    button.setAttribute("aria-expanded", String(open));
  });
}

function initTheme() {
  $("#theme-toggle")?.addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem("tdg-theme", next);
    document.cookie = `tdg-theme=${next}; path=/; max-age=31536000`;
  });
}

function initLanguage() {
  $$("[data-language]").forEach((button) => {
    button.addEventListener("click", () => {
      document.cookie = `lang=${button.dataset.language}; path=/; max-age=31536000`;
      const url = new URL(window.location.href);
      url.searchParams.set("lang", button.dataset.language);
      window.location.assign(url);
    });
  });
}

function initAutoSubmit() {
  $$("[data-submit-on-change]").forEach((control) => {
    control.addEventListener("change", () => control.form?.requestSubmit());
  });
}

function initFilters() {
  const button = $("[data-filter-toggle]");
  const filters = $("#gallery-filters");
  button?.addEventListener("click", () => {
    const open = filters?.classList.toggle("open") || false;
    button.setAttribute("aria-expanded", String(open));
  });
}

let deleteRunId = null;
let deleteOutputBase = "outputs";
let modalReturnFocus = null;

function initModal() {
  const overlay = $("#delete-modal");
  const dialog = $(".modal", overlay || document);
  $$("[data-close-modal]").forEach((button) => button.addEventListener("click", closeDeleteModal));
  overlay?.addEventListener("mousedown", (event) => {
    if (event.target === overlay) closeDeleteModal();
  });
  $("#delete-confirm-btn")?.addEventListener("click", confirmDeleteRun);
  document.addEventListener("keydown", (event) => {
    if (overlay?.classList.contains("hidden")) return;
    if (event.key === "Escape") closeDeleteModal();
    if (event.key === "Tab") trapFocus(event, dialog);
  });
}

function openDeleteModal(runId, outputBase, trigger) {
  deleteRunId = runId;
  deleteOutputBase = outputBase || "outputs";
  modalReturnFocus = trigger || document.activeElement;
  $("#delete-target-name").textContent = runId;
  $("#delete-modal")?.classList.remove("hidden");
  $(".modal")?.focus();
}

function closeDeleteModal() {
  $("#delete-modal")?.classList.add("hidden");
  deleteRunId = null;
  modalReturnFocus?.focus();
}

function trapFocus(event, root) {
  if (!root) return;
  const focusable = $$("button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), a[href]", root);
  if (!focusable.length) return;
  const first = focusable[0];
  const last = focusable[focusable.length - 1];
  if (event.shiftKey && document.activeElement === first) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault();
    first.focus();
  }
}

async function confirmDeleteRun() {
  if (!deleteRunId) return;
  const type = $("input[name='delete_type']:checked")?.value || "soft";
  const response = await fetch(`/api/runs/${encodeURIComponent(deleteRunId)}/delete`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({delete_type: type, output_base: deleteOutputBase}),
  });
  const data = await response.json();
  showToast(data.message || data.detail, response.ok ? "success" : "error");
  if (response.ok) window.setTimeout(() => window.location.assign("/gallery"), 600);
}

function initRunActions() {
  $$("[data-delete-run]").forEach((button) => {
    button.addEventListener("click", () => openDeleteModal(button.dataset.deleteRun, button.dataset.outputBase, button));
  });
  $$("[data-export-run]").forEach((button) => {
    button.addEventListener("click", () => exportRun(button.dataset.exportRun));
  });
  $$("[data-cancel-job]").forEach((button) => {
    button.addEventListener("click", async () => {
      const response = await fetch(`/api/jobs/${button.dataset.cancelJob}/cancel`, {method: "POST"});
      showToast(response.ok ? message("Задача отменена", "Job cancelled") : message("Не удалось отменить задачу", "Could not cancel job"), response.ok ? "success" : "error");
      if (response.ok) window.location.reload();
    });
  });
}

async function exportRun(runId) {
  const response = await fetch(`/api/runs/${encodeURIComponent(runId)}/export`, {method: "POST"});
  const data = await response.json();
  showToast(data.message || data.detail, response.ok ? "success" : "error");
}

function generationPayload() {
  const language = $("#gen-language")?.value;
  const layout = $("#gen-layout")?.value;
  const effect = $("#gen-effect")?.value;
  return {
    profile: $("#gen-profile")?.value,
    out_dir: $("#gen-out")?.value,
    count: Number($("#gen-count")?.value),
    seed: Number($("#gen-seed")?.value),
    languages: language ? [language] : [],
    layouts: layout ? [layout] : [],
    effects: effect ? [effect] : [],
  };
}

function updateCommandPreview() {
  const preview = $("#command-preview");
  if (!preview) return;
  const payload = generationPayload();
  const parts = ["turkicdocgen", "pipeline", "--profile", payload.profile, "--out", payload.out_dir, "--count", payload.count, "--seed", payload.seed, "--force"];
  if (payload.languages.length) parts.push("--language", payload.languages[0]);
  if (payload.layouts.length) parts.push("--layout", payload.layouts[0]);
  if (payload.effects.length) parts.push("--effect", payload.effects[0]);
  preview.textContent = parts.join(" ");
}

function initGeneration() {
  const form = $("#generation-form");
  if (!form) return;
  $$("input, select", form).forEach((control) => control.addEventListener("input", updateCommandPreview));
  updateCommandPreview();
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const errors = $("#generation-errors");
    if (!form.reportValidity()) return;
    const submit = $("#gen-submit");
    submit.disabled = true;
    errors?.classList.add("hidden");
    try {
      const response = await fetch("/api/jobs", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(generationPayload()),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || message("Не удалось запустить генерацию", "Could not start generation"));
      startJobStream(data.job_id, data.total);
    } catch (error) {
      errors.textContent = `${error.message}. ${message("Проверьте профиль и папку вывода.", "Check the profile and output directory.")}`;
      errors.classList.remove("hidden");
    } finally {
      submit.disabled = false;
    }
  });
}

function startJobStream(jobId, total) {
  const log = $("#log-panel");
  const cancel = $("#cancel-job");
  const section = $("#progress-section");
  const progress = $("#job-progress");
  const label = $("#progress-label");
  log.textContent = "";
  section.classList.remove("hidden");
  cancel.classList.remove("hidden");
  setJobStatus("running");
  cancel.onclick = async () => {
    await fetch(`/api/jobs/${jobId}/cancel`, {method: "POST"});
    setJobStatus("cancelled");
    cancel.classList.add("hidden");
  };
  const source = new EventSource(`/api/jobs/${jobId}/events`);
  source.onmessage = (event) => {
    if (event.data === "__DONE__") {
      source.close();
      finishJob(jobId);
      return;
    }
    log.textContent += `${event.data}\n`;
    log.scrollTop = log.scrollHeight;
    const match = /^progress:\s*(\d+)\/(\d+)/.exec(event.data);
    if (match) {
      const done = Number(match[1]);
      const max = Number(match[2]) || total;
      const percent = max ? Math.round(done / max * 100) : 0;
      progress.value = percent;
      progress.textContent = `${percent}%`;
      label.textContent = `${done} / ${max}`;
      $("#job-announcer").textContent = message(`Готово ${done} из ${max}`, `${done} of ${max} complete`);
    }
  };
  source.onerror = () => {
    source.close();
    finishJob(jobId);
  };
}

async function finishJob(jobId) {
  const response = await fetch(`/api/jobs/${jobId}`);
  const job = await response.json();
  setJobStatus(job.status);
  $("#cancel-job")?.classList.add("hidden");
  $("#job-announcer").textContent = job.status === "done" ? message("Генерация завершена", "Generation complete") : (job.error || job.status);
}

function setJobStatus(status) {
  const badge = $("#job-status-badge");
  if (!badge) return;
  badge.textContent = status;
  badge.className = `status status-${status}`;
}

function initTabs() {
  $$("[data-tab]").forEach((tab) => {
    tab.addEventListener("click", () => {
      $$("[data-tab]").forEach((item) => item.setAttribute("aria-selected", String(item === tab)));
      $$("[data-tab-panel]").forEach((panel) => panel.classList.toggle("hidden", panel.dataset.tabPanel !== tab.dataset.tab));
    });
  });
}

const zoneColors = {
  title: "#1d5fd1",
  body: "#167647",
  metadata: "#596579",
  recipient_block: "#b45309",
  sender_block: "#b45309",
  signature_zone: "#9d2672",
  stamp_zone: "#b4232c",
  table: "#5b7c12",
  cell: "#0d9488",
  line: "#4f46e5",
  decoration: "#0369a1",
  default: "#596579",
  selected: "#2563eb",
  error: "#b4232c",
  warning: "#8b5a00"
};

function colorWithOpacity(color, opacityPercent) {
  const alpha = Math.round(Math.max(0, Math.min(100, opacityPercent)) * 2.55);
  return `${color}${alpha.toString(16).padStart(2, "0")}`;
}

function isDecorationZone(zone) {
  return zone.zone_type === "decoration"
    || zone.role === "layout_separator"
    || zone.role === "section_rule";
}

let selectedZoneIndex = null;
let hoveredZoneIndex = null;

function initZoneOverlay() {
  const canvas = $("#zone-overlay-canvas");
  const image = $("#sample-main-img");
  if (!canvas || !image) return;

  const zoneDataEl = document.getElementById("zone-data");
  let zones = [];
  if (zoneDataEl) {
    try { zones = JSON.parse(zoneDataEl.textContent || "[]"); } catch { zones = []; }
  }

  // Settings
  let layerZones = true;
  let layerLines = true;
  let layerCells = true;
  let layerReadingOrder = false;
  let layerQA = true;
  let layerDecorations = true;
  let opacity = 25;
  let labelMode = "number"; // number, role, id, none
  let filterRole = "all";
  let showOnlyProblems = false;

  // Drawer Toggle
  const drawer = document.getElementById("inspector-controls-drawer");
  const drawerToggle = document.getElementById("controls-drawer-toggle");
  if (drawer && drawerToggle) {
    drawerToggle.addEventListener("click", () => {
      const hidden = drawer.classList.toggle("hidden");
      drawerToggle.setAttribute("aria-expanded", String(!hidden));
    });
  }

  // Populate dynamic role filter
  const filterSelect = document.getElementById("role-filter-select");
  if (filterSelect) {
    const uniqueRoles = new Set();
    zones.forEach(z => {
      if (z.role) uniqueRoles.add(z.role);
      if (z.zone_type) uniqueRoles.add(z.zone_type);
    });
    uniqueRoles.forEach(role => {
      const option = document.createElement("option");
      option.value = role;
      option.textContent = role;
      filterSelect.appendChild(option);
    });
  }

  // Bind settings to UI controls
  const bindCheckbox = (id, setter) => {
    const el = document.getElementById(id);
    if (el) {
      el.addEventListener("change", () => {
        setter(el.checked);
        localStorage.setItem(`tdg-${id}`, String(el.checked));
        render();
      });
      const saved = localStorage.getItem(`tdg-${id}`);
      if (saved !== null) {
        el.checked = saved === "true";
        setter(el.checked);
      }
    }
  };

  bindCheckbox("layer-zones", v => layerZones = v);
  bindCheckbox("layer-lines", v => layerLines = v);
  bindCheckbox("layer-cells", v => layerCells = v);
  bindCheckbox("layer-reading-order", v => layerReadingOrder = v);
  bindCheckbox("layer-qa-issues", v => layerQA = v);
  bindCheckbox("layer-decorations", v => layerDecorations = v);
  bindCheckbox("show-only-problems", v => showOnlyProblems = v);

  const opacitySlider = document.getElementById("layer-opacity");
  if (opacitySlider) {
    opacitySlider.addEventListener("input", () => {
      opacity = parseInt(opacitySlider.value, 10);
      localStorage.setItem("tdg-layer-opacity", opacitySlider.value);
      render();
    });
    const saved = localStorage.getItem("tdg-layer-opacity");
    if (saved !== null) {
      opacitySlider.value = saved;
      opacity = parseInt(saved, 10);
    }
  }

  const labelSelect = document.getElementById("label-mode-select");
  if (labelSelect) {
    labelSelect.addEventListener("change", () => {
      labelMode = labelSelect.value;
      localStorage.setItem("tdg-label-mode", labelMode);
      render();
    });
    const saved = localStorage.getItem("tdg-label-mode");
    if (saved !== null) {
      labelSelect.value = saved;
      labelMode = saved;
    }
  }

  if (filterSelect) {
    filterSelect.addEventListener("change", () => {
      filterRole = filterSelect.value;
      render();
    });
  }

  // Clear Selection Button
  document.getElementById("clear-selection-btn")?.addEventListener("click", () => {
    selectZone(null);
  });

  // Zoom / View Helpers
  const zoomInput = document.getElementById("image-zoom");
  const container = $(".sample-img-container");

  const fitPage = () => {
    if (!container || !image) return;
    const containerHeight = container.clientHeight;
    const naturalHeight = image.naturalHeight || 2339;
    const ratio = Math.round((containerHeight - 24) / naturalHeight * 100);
    const zoomVal = Math.max(50, Math.min(300, ratio));
    image.style.width = `${zoomVal}%`;
    if (zoomInput) zoomInput.value = zoomVal;
    render();
  };

  const fitWidth = () => {
    if (!image) return;
    image.style.width = "100%";
    if (zoomInput) zoomInput.value = 100;
    render();
  };

  const resetView = () => {
    if (!image || !container) return;
    image.style.width = "100%";
    if (zoomInput) zoomInput.value = 100;
    container.scrollLeft = 0;
    container.scrollTop = 0;
    selectZone(null);

    // Clear local storage overlay settings
    localStorage.removeItem("tdg-layer-zones");
    localStorage.removeItem("tdg-layer-lines");
    localStorage.removeItem("tdg-layer-cells");
    localStorage.removeItem("tdg-layer-reading-order");
    localStorage.removeItem("tdg-layer-qa-issues");
    localStorage.removeItem("tdg-layer-decorations");
    localStorage.removeItem("tdg-show-only-problems");
    localStorage.removeItem("tdg-layer-opacity");
    localStorage.removeItem("tdg-label-mode");

    // Restore default UI values and state variables
    layerZones = true;
    const elZones = document.getElementById("layer-zones");
    if (elZones) elZones.checked = true;

    layerLines = true;
    const elLines = document.getElementById("layer-lines");
    if (elLines) elLines.checked = true;

    layerCells = true;
    const elCells = document.getElementById("layer-cells");
    if (elCells) elCells.checked = true;

    layerReadingOrder = false;
    const elRO = document.getElementById("layer-reading-order");
    if (elRO) elRO.checked = false;

    layerQA = true;
    const elQA = document.getElementById("layer-qa-issues");
    if (elQA) elQA.checked = true;

    layerDecorations = true;
    const elDecorations = document.getElementById("layer-decorations");
    if (elDecorations) elDecorations.checked = true;

    showOnlyProblems = false;
    const elProblems = document.getElementById("show-only-problems");
    if (elProblems) elProblems.checked = false;

    opacity = 25;
    const elOpacity = document.getElementById("layer-opacity");
    if (elOpacity) elOpacity.value = 25;

    labelMode = "number";
    const elLabelMode = document.getElementById("label-mode-select");
    if (elLabelMode) elLabelMode.value = "number";

    filterRole = "all";
    const elFilterRole = document.getElementById("role-filter-select");
    if (elFilterRole) elFilterRole.value = "all";

    render();
  };

  document.getElementById("btn-fit-page")?.addEventListener("click", fitPage);
  document.getElementById("btn-fit-width")?.addEventListener("click", fitWidth);
  document.getElementById("btn-reset-view")?.addEventListener("click", resetView);

  // Render overlay canvas
  const render = () => {
    const rect = image.getBoundingClientRect();
    const parent = canvas.parentElement.getBoundingClientRect();
    canvas.style.position = "absolute";
    canvas.style.left = `${rect.left - parent.left + canvas.parentElement.scrollLeft}px`;
    canvas.style.top = `${rect.top - parent.top + canvas.parentElement.scrollTop}px`;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;

    const scaleX = rect.width / (image.naturalWidth || 1654);
    const scaleY = rect.height / (image.naturalHeight || 2339);

    const context = canvas.getContext("2d");
    context.scale(dpr, dpr);
    context.clearRect(0, 0, rect.width, rect.height);

    if (zones.length === 0) {
      // Draw visible empty/error state on canvas
      context.fillStyle = "rgba(220, 38, 38, 0.15)";
      context.fillRect(0, 0, rect.width, rect.height);

      context.strokeStyle = "rgba(220, 38, 38, 0.8)";
      context.lineWidth = 3;
      context.strokeRect(10, 10, rect.width - 20, rect.height - 20);

      context.fillStyle = "rgb(220, 38, 38)";
      context.font = "bold 18px sans-serif";
      context.textAlign = "center";
      context.textBaseline = "middle";
      context.fillText("No Zones / Error State", rect.width / 2, rect.height / 2);
      return;
    }

    // Draw Zones
    zones.forEach((zone, index) => {
      if (zone.drawable === false) return;
      const isSelected = selectedZoneIndex === index;
      const isHovered = hoveredZoneIndex === index;
      const points = zone.polygon?.length ? zone.polygon : bboxPolygon(zone.bbox || zone.bounding_box);

      if (!points?.length) return;

      // Filtering checks
      if (filterRole !== "all" && zone.role !== filterRole && zone.zone_type !== filterRole) return;
      if (showOnlyProblems && (!zone.qa_issues || zone.qa_issues.length === 0)) return;
      if (!layerDecorations && isDecorationZone(zone)) return;

      // Colors
      let color = zoneColors[zone.role] || zoneColors[zone.zone_type] || zoneColors.default;
      if (isSelected || isHovered) {
        color = zoneColors.selected;
      } else if (zone.qa_issues && zone.qa_issues.length > 0) {
        const hasError = zone.qa_issues.some(i => i.severity === "error");
        color = hasError ? zoneColors.error : zoneColors.warning;
      }

      // Draw Zone Outline & Fill
      if (layerZones) {
        context.beginPath();
        points.forEach(([x, y], pointIndex) => {
          const px = x * scaleX;
          const py = y * scaleY;
          if (pointIndex === 0) context.moveTo(px, py); else context.lineTo(px, py);
        });
        context.closePath();
        context.fillStyle = colorWithOpacity(color, isSelected ? Math.min(95, opacity * 2.5) : opacity);
        context.strokeStyle = color;
        context.lineWidth = isSelected ? 3.0 : (isHovered ? 2.0 : 1.5);
        context.fill();
        context.stroke();
      }

      // Draw Lines Layer
      if (layerLines && zone.lines) {
        zone.lines.forEach(line => {
          const lBbox = line.bbox;
          if (lBbox && lBbox.length === 4) {
            context.strokeStyle = zoneColors.line;
            context.lineWidth = 1;
            context.strokeRect(lBbox[0] * scaleX, lBbox[1] * scaleY, (lBbox[2] - lBbox[0]) * scaleX, (lBbox[3] - lBbox[1]) * scaleY);
          }
        });
      }

      // Draw Cells Layer
      if (layerCells && zone.cells) {
        zone.cells.forEach(cell => {
          const cBbox = cell.bbox;
          if (cBbox && cBbox.length === 4) {
            context.strokeStyle = zoneColors.cell;
            context.lineWidth = 1.2;
            context.strokeRect(cBbox[0] * scaleX, cBbox[1] * scaleY, (cBbox[2] - cBbox[0]) * scaleX, (cBbox[3] - cBbox[1]) * scaleY);
          }
        });
      }

      // Draw QA issues visual indicators
      if (layerQA && zone.qa_issues && zone.qa_issues.length > 0) {
        const hasError = zone.qa_issues.some(i => i.severity === "error");
        const qaColor = hasError ? zoneColors.error : zoneColors.warning;
        
        context.beginPath();
        points.forEach(([x, y], ptIdx) => {
          const px = x * scaleX;
          const py = y * scaleY;
          if (ptIdx === 0) context.moveTo(px, py); else context.lineTo(px, py);
        });
        context.closePath();
        context.setLineDash([3, 3]);
        context.strokeStyle = qaColor;
        context.lineWidth = 2.0;
        context.stroke();
        context.setLineDash([]);

        // indicator circle at top-left
        const [x0, y0] = points[0];
        context.beginPath();
        context.arc(x0 * scaleX, y0 * scaleY, 6, 0, 2 * Math.PI);
        context.fillStyle = qaColor;
        context.fill();
      }

      // Draw Labels
      if (labelMode !== "none" && points.length > 0) {
        let labelText = "";
        if (labelMode === "number") labelText = String(index + 1);
        else if (labelMode === "role") labelText = zone.role;
        else if (labelMode === "id") labelText = zone.zone_id;

        const [x, y] = points[0];
        context.fillStyle = isSelected || isHovered ? "white" : color;
        context.font = "bold 10px ui-monospace, monospace";
        context.textBaseline = "top";
        context.textAlign = "left";

        const textWidth = context.measureText(labelText).width;
        context.fillStyle = color;
        context.fillRect(x * scaleX, y * scaleY, textWidth + 6, 14);

        context.fillStyle = "white";
        context.fillText(labelText, x * scaleX + 3, y * scaleY + 2);
      }
    });

    // Draw Reading Order Path Layer
    if (layerReadingOrder) {
      const centers = zones
        .map(z => {
          if (z.drawable === false) return null;
          const bbox = z.bbox || z.bounding_box;
          if (!bbox || bbox.length !== 4) return null;
          return {
            x: ((bbox[0] + bbox[2]) / 2) * scaleX,
            y: ((bbox[1] + bbox[3]) / 2) * scaleY,
            order: z.reading_order
          };
        })
        .filter(c => c !== null)
        .sort((a, b) => a.order - b.order);

      // Draw path line
      if (centers.length > 1) {
        context.beginPath();
        centers.forEach((c, idx) => {
          if (idx === 0) context.moveTo(c.x, c.y); else context.lineTo(c.x, c.y);
        });
        context.strokeStyle = zoneColors.line;
        context.lineWidth = 2;
        context.setLineDash([4, 4]);
        context.stroke();
        context.setLineDash([]);
      }

      // Draw numbers in circles
      centers.forEach(c => {
        context.beginPath();
        context.arc(c.x, c.y, 10, 0, 2 * Math.PI);
        context.fillStyle = zoneColors.line;
        context.fill();
        context.fillStyle = "white";
        context.font = "bold 10px ui-monospace, monospace";
        context.textAlign = "center";
        context.textBaseline = "middle";
        context.fillText(String(c.order), c.x, c.y);
      });
    }
  };

  // Selection Manager
  const selectZone = (index) => {
    selectedZoneIndex = index;

    // Highlight list row
    document.querySelectorAll(".zone-item").forEach((el, idx) => {
      if (idx === index) {
        el.classList.add("selected-item");
        el.style.border = "2px solid var(--accent)";
        el.style.background = "var(--accent-soft)";
      } else {
        el.classList.remove("selected-item");
        el.style.border = "1px solid var(--border)";
        el.style.background = "var(--surface)";
      }
    });

    if (index !== null) {
      const zone = zones[index];
      updatePropertiesPanel(zone);
    } else {
      updatePropertiesPanel(null);
    }
    render();
  };

  // Point in Polygon PNPOLY Algorithm
  const pointInPolygon = (x, y, vs) => {
    let inside = false;
    for (let i = 0, j = vs.length - 1; i < vs.length; j = i++) {
      const xi = vs[i][0], yi = vs[i][1];
      const xj = vs[j][0], yj = vs[j][1];
      const intersect = ((yi > y) !== (yj > y)) &&
        (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
      if (intersect) inside = !inside;
    }
    return inside;
  };

  const isPointInZone = (x, y, zone) => {
    if (zone.drawable === false) return false;
    const vs = zone.polygon?.length ? zone.polygon : bboxPolygon(zone.bbox || zone.bounding_box);
    if (!vs || !vs.length) return false;
    return pointInPolygon(x, y, vs);
  };

  // Drag-to-pan & Selection on Canvas
  let isDragging = false;
  let startX = 0, startY = 0;
  let scrollLeft = 0, scrollTop = 0;
  let lastClickTime = 0;
  let lastClickedCoords = { x: 0, y: 0 };
  let overlapIndex = 0;

  canvas.addEventListener("mousedown", (e) => {
    isDragging = true;
    startX = e.clientX;
    startY = e.clientY;
    scrollLeft = container.scrollLeft;
    scrollTop = container.scrollTop;
    canvas.style.cursor = "grabbing";
  });

  window.addEventListener("mousemove", (e) => {
    if (!isDragging) return;
    const dx = e.clientX - startX;
    const dy = e.clientY - startY;
    container.scrollLeft = scrollLeft - dx;
    container.scrollTop = scrollTop - dy;
  });

  window.addEventListener("mouseup", (e) => {
    if (!isDragging) return;
    isDragging = false;
    canvas.style.cursor = "crosshair";

    const dx = Math.abs(e.clientX - startX);
    const dy = Math.abs(e.clientY - startY);
    if (dx < 5 && dy < 5) {
      // It's a click, handle selection
      const rect = image.getBoundingClientRect();
      const clickX = e.clientX - rect.left;
      const clickY = e.clientY - rect.top;
      const scaleX = (image.naturalWidth || 1654) / rect.width;
      const scaleY = (image.naturalHeight || 2339) / rect.height;
      const px = clickX * scaleX;
      const py = clickY * scaleY;

      // Find matched zones
      const matched = [];
      zones.forEach((zone, idx) => {
        if (zone.drawable === false) return;
        if (filterRole !== "all" && zone.role !== filterRole && zone.zone_type !== filterRole) return;
        if (showOnlyProblems && (!zone.qa_issues || zone.qa_issues.length === 0)) return;
        if (!layerDecorations && isDecorationZone(zone)) return;
        if (isPointInZone(px, py, zone)) {
          matched.push(idx);
        }
      });

      if (matched.length > 0) {
        const now = Date.now();
        const isSameLocation = Math.abs(px - lastClickedCoords.x) < 25 && Math.abs(py - lastClickedCoords.y) < 25;
        if (isSameLocation && now - lastClickTime < 2000) {
          overlapIndex = (overlapIndex + 1) % matched.length;
        } else {
          overlapIndex = 0;
        }
        lastClickTime = now;
        lastClickedCoords = { x: px, y: py };
        selectZone(matched[overlapIndex]);
      } else {
        selectZone(null);
      }
    }
  });

  // Bind list item clicks & hover
  document.querySelectorAll(".zone-item").forEach(item => {
    item.addEventListener("click", () => {
      const idx = parseInt(item.getAttribute("data-zone-index"), 10);
      selectZone(idx);

      // Scroll to show selected zone in view
      if (idx !== null && zones[idx]) {
        const zone = zones[idx];
        const bbox = zone.bbox || zone.bounding_box;
        if (bbox && bbox.length === 4) {
          const rect = image.getBoundingClientRect();
          const scaleX = rect.width / (image.naturalWidth || 1654);
          const scaleY = rect.height / (image.naturalHeight || 2339);

          const cx = ((bbox[0] + bbox[2]) / 2) * scaleX;
          const cy = ((bbox[1] + bbox[3]) / 2) * scaleY;

          container.scrollLeft = cx - container.clientWidth / 2;
          container.scrollTop = cy - container.clientHeight / 2;
        }
      }
    });

    item.addEventListener("mouseenter", () => {
      hoveredZoneIndex = parseInt(item.getAttribute("data-zone-index"), 10);
      render();
    });

    item.addEventListener("mouseleave", () => {
      hoveredZoneIndex = null;
      render();
    });
  });

  // Keyboard navigation & Esc key handler
  document.addEventListener("keydown", (event) => {
    if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName)) return;
    
    if (zones.length > 0) {
      if (event.key === "Escape") {
        selectZone(null);
      }
      if (event.key === "[") {
        let nextIdx = selectedZoneIndex === null ? zones.length - 1 : selectedZoneIndex - 1;
        if (nextIdx < 0) nextIdx = zones.length - 1;
        selectZone(nextIdx);
        // Scroll item into view
        document.querySelector(`.zone-item[data-zone-index="${nextIdx}"]`)?.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }
      if (event.key === "]") {
        let nextIdx = selectedZoneIndex === null ? 0 : selectedZoneIndex + 1;
        if (nextIdx >= zones.length) nextIdx = 0;
        selectZone(nextIdx);
        // Scroll item into view
        document.querySelector(`.zone-item[data-zone-index="${nextIdx}"]`)?.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }
    }
  });

  // Direct navigation from QA issue click (e.g. data-zone-id)
  document.addEventListener("click", (e) => {
    const issueLink = e.target.closest("[data-zone-id]");
    if (issueLink && !issueLink.closest(".zone-list") && !issueLink.closest(".image-inspector")) {
      const zoneId = issueLink.getAttribute("data-zone-id");
      const idx = zones.findIndex(z => z.zone_id === zoneId);
      if (idx !== -1) {
        // Toggle tab to zones
        document.querySelector('[data-tab="zones"]')?.click();
        selectZone(idx);
        document.querySelector(`.zone-item[data-zone-id="${zoneId}"]`)?.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }
    }
  });

  // Setup image & window listeners
  image.addEventListener("load", render);
  if (image.complete) render();
  window.addEventListener("resize", render);

  $("#zone-overlay-toggle")?.addEventListener("click", (event) => {
    const hidden = canvas.classList.toggle("hidden");
    event.currentTarget.lastChild.textContent = hidden ? message("Показать зоны", "Show zones") : message("Скрыть зоны", "Hide zones");
  });

  canvas.renderOverlay = render;
}

function escapeHtml(str) {
  if (str === null || str === undefined) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function updatePropertiesPanel(zone) {
  const panel = document.getElementById("selected-zone-properties");
  const content = panel?.querySelector(".properties-content");
  if (!panel || !content) return;

  if (!zone) {
    panel.classList.add("hidden");
    return;
  }
  panel.classList.remove("hidden");

  let qaHtml = "";
  if (zone.qa_issues && zone.qa_issues.length > 0) {
    qaHtml = `
      <div style="margin-top: 8px; border-top: 1px solid var(--border); padding-top: 8px;">
        <strong style="color: var(--danger); font-size: 0.75rem;">QA Issues (${zone.qa_issues.length}):</strong>
        <ul style="margin: 4px 0 0 0; padding-left: 14px; color: var(--danger); font-size: 0.7rem;">
          ${zone.qa_issues.map(issue => `
            <li style="margin-bottom: 3px;">
              <strong>${escapeHtml(issue.code)}</strong> (${escapeHtml(issue.severity)}): ${escapeHtml(issue.message)}
            </li>
          `).join("")}
        </ul>
      </div>
    `;
  }

  let cellsHtml = "";
  if (zone.cells && zone.cells.length > 0) {
    cellsHtml = `
      <div style="margin-top: 8px; border-top: 1px solid var(--border); padding-top: 8px;">
        <strong style="font-size: 0.75rem;">Table Cells (${zone.cells.length}):</strong>
        <div style="font-size: 0.7rem; color: var(--muted); max-height: 100px; overflow-y: auto; margin-top: 3px; background: var(--surface-subtle); padding: 4px; border-radius: 4px;">
          ${zone.cells.map(c => `R${escapeHtml(c.row)}C${escapeHtml(c.col)}: "${escapeHtml(c.text)}"`).join("<br>")}
        </div>
      </div>
    `;
  }

  let linesHtml = "";
  if (zone.lines && zone.lines.length > 0) {
    linesHtml = `
      <div style="margin-top: 8px; border-top: 1px solid var(--border); padding-top: 8px;">
        <strong style="font-size: 0.75rem;">Lines (${zone.lines.length}):</strong>
        <div style="font-size: 0.7rem; color: var(--muted); max-height: 100px; overflow-y: auto; margin-top: 3px; background: var(--surface-subtle); padding: 4px; border-radius: 4px;">
          ${zone.lines.map(l => `[${escapeHtml(l.reading_order)}]: "${escapeHtml(l.text)}"`).join("<br>")}
        </div>
      </div>
    `;
  }

  let styleHtml = "";
  if (zone.style) {
    styleHtml = `
      <div><strong>Font:</strong> ${escapeHtml(zone.style.font_family)} (${escapeHtml(zone.style.font_size_px)}px)</div>
      <div><strong>Align:</strong> ${escapeHtml(zone.style.align)} | <strong>Spacing:</strong> ${escapeHtml(zone.style.line_spacing)}</div>
      <div><strong>Style:</strong> ${zone.style.bold ? 'Bold' : 'Normal'} ${zone.style.italic ? 'Italic' : ''}</div>
    `;
  }

  content.innerHTML = `
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px; border-bottom: 1px solid var(--border); padding-bottom: 6px;">
      <div><strong>ID:</strong> <span class="mono">${escapeHtml(zone.zone_id)}</span></div>
      <div><strong>Role:</strong> <span class="status status-idle" style="padding: 1px 4px; font-size: 0.7rem;">${escapeHtml(zone.role)}</span></div>
      <div><strong>Type:</strong> ${escapeHtml(zone.zone_type)}</div>
      <div><strong>Lang:</strong> ${escapeHtml(zone.language || '—')}</div>
      <div><strong>Order:</strong> ${escapeHtml(zone.reading_order)}</div>
    </div>
    
    <div style="margin-top: 6px;">
      <strong>Text:</strong>
      <textarea readonly rows="3" style="margin-top: 3px; font-size: 0.7rem; font-family: ui-monospace, monospace; width: 100%; resize: vertical; background: var(--surface); color: var(--text); padding: 4px; border: 1px solid var(--border-strong);">${escapeHtml(zone.text || '')}</textarea>
    </div>
    
    <div style="margin-top: 6px; font-size: 0.7rem;">
      <strong>BBox:</strong> <span class="mono">[${escapeHtml(zone.bbox?.join(", ") || '')}]</span>
    </div>
    
    <div style="margin-top: 6px; border-top: 1px solid var(--border); padding-top: 6px; display: grid; grid-template-columns: 1fr; gap: 3px; font-size: 0.7rem;">
      ${styleHtml}
    </div>
    
    ${linesHtml}
    ${cellsHtml}
    ${qaHtml}
  `;
}

function initGalleryZones() {
  const toggle = document.getElementById("gallery-zones-toggle");
  if (!toggle) return;

  const saved = localStorage.getItem("tdg-gallery-zones");
  toggle.checked = saved === "true";
  const visibleCanvases = () => Array.from(document.querySelectorAll(".gallery-zones-canvas"))
    .filter(canvas => {
      const rect = canvas.getBoundingClientRect();
      return rect.bottom >= 0 && rect.top <= window.innerHeight && rect.right >= 0 && rect.left <= window.innerWidth;
    });

  const updateCanvases = () => {
    const canvases = document.querySelectorAll(".gallery-zones-canvas");
    canvases.forEach(canvas => {
      canvas.classList.toggle("hidden", !toggle.checked);
    });
    if (toggle.checked) visibleCanvases().forEach(canvas => renderGalleryCanvas(canvas));
  };

  toggle.addEventListener("change", () => {
    localStorage.setItem("tdg-gallery-zones", String(toggle.checked));
    updateCanvases();
  });

  document.querySelectorAll(".document-preview img").forEach(img => {
    img.addEventListener("load", () => {
      const canvas = img.parentElement.querySelector(".gallery-zones-canvas");
      if (canvas && toggle.checked) {
        renderGalleryCanvas(canvas);
      }
    });
    if (img.complete) {
      const canvas = img.parentElement.querySelector(".gallery-zones-canvas");
      if (canvas && toggle.checked) {
        renderGalleryCanvas(canvas);
      }
    }
  });

  setTimeout(updateCanvases, 300);
  window.addEventListener("scroll", updateCanvases, {passive: true});
  window.addEventListener("resize", updateCanvases);
}

function renderGalleryCanvas(canvas) {
  const img = canvas.parentElement.querySelector("img");
  if (!img || !img.complete || img.naturalWidth === 0) return;

  if (!canvas.dataset.zonesLoaded) {
    loadGalleryZones(canvas).then(() => renderGalleryCanvas(canvas)).catch(() => {});
    return;
  }

  let zones = canvas._zones || [];
  if (!Array.isArray(zones)) {
    return;
  }

  const rect = img.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  canvas.style.width = `${rect.width}px`;
  canvas.style.height = `${rect.height}px`;

  const coordinateWidth = Number.parseFloat(img.getAttribute("width")) || img.naturalWidth;
  const coordinateHeight = Number.parseFloat(img.getAttribute("height")) || img.naturalHeight;
  const scaleX = rect.width / coordinateWidth;
  const scaleY = rect.height / coordinateHeight;

  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  zones.forEach(zone => {
    if (zone.drawable === false) return;
    const points = zone.polygon?.length ? zone.polygon : bboxPolygon(zone.bbox || zone.bounding_box);
    if (!points || !points.length) return;

    const color = zoneColors[zone.role] || zoneColors[zone.zone_type] || zoneColors.default;

    ctx.beginPath();
    points.forEach(([x, y], idx) => {
      const px = x * scaleX;
      const py = y * scaleY;
      if (idx === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.closePath();
    ctx.fillStyle = `${color}15`;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.fill();
    ctx.stroke();
  });
}

async function loadGalleryZones(canvas) {
  if (canvas.dataset.zonesLoading === "true" || canvas.dataset.zonesLoaded === "true") return;
  const runId = canvas.dataset.runId;
  const sampleId = canvas.dataset.sampleId;
  const outputBase = canvas.dataset.outputBase || "outputs";
  if (!runId || !sampleId) return;
  canvas.dataset.zonesLoading = "true";
  await scheduleGalleryZoneLoad();
  try {
    const url = `/api/runs/${encodeURIComponent(runId)}/samples/${encodeURIComponent(sampleId)}/zones?output_base=${encodeURIComponent(outputBase)}`;
    const response = await fetch(url);
    if (!response.ok) return;
    const payload = await response.json();
    canvas._zones = Array.isArray(payload.zones) ? payload.zones : [];
    canvas.dataset.zonesLoaded = "true";
  } finally {
    canvas.dataset.zonesLoading = "false";
    galleryZoneInFlight = Math.max(0, galleryZoneInFlight - 1);
    const next = galleryZoneQueue.shift();
    if (next) next();
  }
}

let galleryZoneInFlight = 0;
const galleryZoneQueue = [];
const maxGalleryZoneRequests = 4;

function scheduleGalleryZoneLoad() {
  if (galleryZoneInFlight < maxGalleryZoneRequests) {
    galleryZoneInFlight += 1;
    return Promise.resolve();
  }
  return new Promise(resolve => {
    galleryZoneQueue.push(() => {
      galleryZoneInFlight += 1;
      resolve();
    });
  });
}

function bboxPolygon(bbox) {
  if (!bbox || bbox.length !== 4) return [];
  return [[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]];
}

function initImageZoom() {
  const control = $("#image-zoom");
  const image = $("#sample-main-img");
  control?.addEventListener("input", () => {
    image.style.width = `${control.value}%`;
    window.requestAnimationFrame(() => $("#zone-overlay-canvas")?.renderOverlay?.());
  });
}

function initReview() {
  $$("[data-visual-qa-btn]").forEach((button) => {
    button.addEventListener("click", async () => {
      button.disabled = true;
      const response = await fetch(`/api/samples/${encodeURIComponent(button.dataset.sampleId)}/visual-status`, {
        method: "PATCH",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          visual_qa_status: button.dataset.visualQaBtn,
          reviewer_note: $("#visual-qa-note")?.value || "",
          run_id: button.dataset.runId,
          output_base: button.dataset.outputBase || "outputs",
        }),
      });
      const data = await response.json();
      button.disabled = false;
      if (response.ok) {
        const badge = $("#visual-qa-status-badge");
        badge.textContent = data.visual_qa_status;
        badge.className = `status status-${data.visual_qa_status}`;
      }
      showToast(response.ok ? message("Решение сохранено", "Review saved") : data.detail, response.ok ? "success" : "error");
    });
  });
}

function initImageFallbacks() {
  $$("img").forEach((image) => image.addEventListener("error", () => {
    image.src = "/static/no-image.svg";
  }, {once: true}));
}

function initKeyboardNavigation() {
  document.addEventListener("keydown", (event) => {
    if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName)) return;
    if (event.key === "ArrowLeft") $("#nav-prev-btn")?.click();
    if (event.key === "ArrowRight") $("#nav-next-btn")?.click();
  });
}

function showToast(text, type = "info") {
  if (!text) return;
  const toast = document.createElement("div");
  toast.className = `toast-msg toast-${type}`;
  toast.textContent = text;
  $("#toast-container")?.appendChild(toast);
  window.setTimeout(() => toast.remove(), 4000);
}

