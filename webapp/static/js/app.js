/**
 * PolypNet – Frontend Application Logic
 * Handles: drag-and-drop upload, AJAX pipeline call,
 *          progress animation, and results rendering.
 */

(function () {
  "use strict";

  /* ─── PAGE DETECTION ──────────────────────────────────── */
  const isResults = document.getElementById("results-container") !== null;
  const isIndex   = document.getElementById("drop-zone") !== null;

  if (isIndex)   initUploadPage();
  if (isResults) initResultsPage();


  /* ═══════════════════════════════════════════════════════
     UPLOAD PAGE
     ═══════════════════════════════════════════════════════ */
  function initUploadPage() {
    const dropZone    = document.getElementById("drop-zone");
    const fileInput   = document.getElementById("file-input");
    const previewZone = document.getElementById("preview-zone");
    const previewImg  = document.getElementById("preview-img");
    const previewName = document.getElementById("preview-name");
    const previewSize = document.getElementById("preview-size");
    const analyzeBtn  = document.getElementById("analyze-btn");
    const resetBtn    = document.getElementById("reset-btn");
    const progressPanel = document.getElementById("progress-panel");
    const progressLabel = document.getElementById("progress-label");
    const errorBanner = document.getElementById("error-banner");
    const errorMsg    = document.getElementById("error-msg");
    const errorClose  = document.getElementById("error-close");

    let selectedFile = null;

    /* ── Drop zone drag events ─────────────────────────── */
    ["dragenter", "dragover"].forEach(ev =>
      dropZone.addEventListener(ev, e => { e.preventDefault(); dropZone.classList.add("drag-over"); })
    );
    ["dragleave", "drop"].forEach(ev =>
      dropZone.addEventListener(ev, e => { e.preventDefault(); dropZone.classList.remove("drag-over"); })
    );
    dropZone.addEventListener("drop", e => {
      const file = e.dataTransfer.files[0];
      if (file) setFile(file);
    });
    dropZone.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", () => {
      if (fileInput.files[0]) setFile(fileInput.files[0]);
    });

    /* ── File set ──────────────────────────────────────── */
    function setFile(file) {
      if (!isImage(file)) {
        showError("Please select an image file (PNG, JPG, JPEG, BMP, TIFF).");
        return;
      }
      selectedFile = file;
      const reader = new FileReader();
      reader.onload = e => { previewImg.src = e.target.result; };
      reader.readAsDataURL(file);
      previewName.textContent = file.name;
      previewSize.textContent = formatBytes(file.size);
      dropZone.classList.add("hidden");
      previewZone.classList.remove("hidden");
      hideError();
    }

    function isImage(file) {
      return /\.(png|jpe?g|bmp|tiff?)$/i.test(file.name);
    }

    /* ── Reset ─────────────────────────────────────────── */
    resetBtn.addEventListener("click", () => {
      selectedFile = null;
      fileInput.value = "";
      previewImg.src = "";
      previewZone.classList.add("hidden");
      dropZone.classList.remove("hidden");
      progressPanel.classList.add("hidden");
      hideError();
    });

    /* ── Analyze ───────────────────────────────────────── */
    analyzeBtn.addEventListener("click", () => {
      if (!selectedFile) return;
      uploadAndAnalyze(selectedFile);
    });

    errorClose.addEventListener("click", hideError);

    /* ── Progress steps data ───────────────────────────── */
    const steps = [
      { id: "p-validate", label: "Validating image…" },
      { id: "p-enhance",  label: "Applying CLAHE enhancement…" },
      { id: "p-detect",   label: "Running YOLO detection…" },
      { id: "p-classify", label: "Running CNN classification ensemble…" },
      { id: "p-segment",  label: "Running Attention U-Net segmentation…" },
      { id: "p-vlm",      label: "Generating clinical report…" },
    ];

    function animateProgress(durationMs) {
      const interval = durationMs / steps.length;
      steps.forEach((s, i) => {
        setTimeout(() => {
          // Mark previous as done
          if (i > 0) {
            const prev = document.getElementById(steps[i - 1].id);
            if (prev) { prev.classList.remove("active"); prev.classList.add("done"); }
          }
          const el = document.getElementById(s.id);
          if (el) el.classList.add("active");
          progressLabel.textContent = s.label;
        }, interval * i);
      });
    }

    function markAllDone() {
      steps.forEach(s => {
        const el = document.getElementById(s.id);
        if (el) { el.classList.remove("active"); el.classList.add("done"); }
      });
      progressLabel.textContent = "Analysis complete — loading results…";
    }

    /* ── Upload & fetch ────────────────────────────────── */
    function uploadAndAnalyze(file) {
      previewZone.classList.add("hidden");
      progressPanel.classList.remove("hidden");
      hideError();

      // Start progress animation assuming ~20–40 s pipeline
      const estimatedMs = 25000;
      animateProgress(estimatedMs);

      const formData = new FormData();
      formData.append("image", file);

      fetch("/analyze", { method: "POST", body: formData })
        .then(res => res.json())
        .then(data => {
          markAllDone();
          setTimeout(() => {
            if (data.error && !data.validation) {
              // Hard error (not validation failure)
              progressPanel.classList.add("hidden");
              showError(data.error);
              previewZone.classList.remove("hidden");
              return;
            }
            // Store result and go to results page
            sessionStorage.setItem("polypnet_result", JSON.stringify(data));
            window.location.href = "/results";
          }, 600);
        })
        .catch(err => {
          progressPanel.classList.add("hidden");
          showError("Network error: " + err.message);
          previewZone.classList.remove("hidden");
          // Reset all progress steps
          steps.forEach(s => {
            const el = document.getElementById(s.id);
            if (el) el.classList.remove("active","done");
          });
        });
    }

    function showError(msg) {
      errorMsg.textContent = msg;
      errorBanner.classList.remove("hidden");
    }
    function hideError() { errorBanner.classList.add("hidden"); }
  }


  /* ═══════════════════════════════════════════════════════
     RESULTS PAGE
     ═══════════════════════════════════════════════════════ */
  function initResultsPage() {
    const loadingState  = document.getElementById("loading-state");
    const resultsGrid   = document.getElementById("results-grid");
    const invalidState  = document.getElementById("invalid-state");
    const rhBadge       = document.getElementById("rh-badge");
    const rhSub         = document.getElementById("rh-sub");

    // Retrieve stored result
    const raw = sessionStorage.getItem("polypnet_result");
    if (!raw) {
      loadingState.innerHTML = "<p>No analysis data found. <a href='/'>Run an analysis first →</a></p>";
      return;
    }

    const data = JSON.parse(raw);
    sessionStorage.removeItem("polypnet_result");

    // Brief delay for dramatic effect
    setTimeout(() => renderResults(data), 500);

    function renderResults(d) {
      loadingState.classList.add("hidden");

      /* ── Validation failed ──────────────────────────────── */
      if (!d.validation || !d.validation.valid) {
        invalidState.classList.remove("hidden");
        const reason = d.validation ? d.validation.reason : (d.error || "Unknown error.");
        document.getElementById("invalid-reason").textContent = reason;
        rhBadge.textContent = "❌ Invalid Image";
        rhBadge.style.background = "rgba(255,71,87,0.15)";
        rhBadge.style.borderColor = "rgba(255,71,87,0.4)";
        rhBadge.style.color = "var(--red)";
        return;
      }

      /* ── Header ─────────────────────────────────────────── */
      const label = d.classification ? d.classification.label : "—";
      if (label === "Malignant") {
        rhBadge.textContent = "⚠ Malignant Detected";
        rhBadge.style.background  = "rgba(255,71,87,0.15)";
        rhBadge.style.borderColor = "rgba(255,71,87,0.4)";
        rhBadge.style.color       = "var(--red)";
      } else if (label === "Benign") {
        rhBadge.textContent = "✅ Benign";
        rhBadge.style.background  = "rgba(34,197,94,0.15)";
        rhBadge.style.borderColor = "rgba(34,197,94,0.4)";
        rhBadge.style.color       = "var(--green)";
      } else {
        rhBadge.textContent = "ℹ No Polyp Detected";
        rhBadge.style.background  = "rgba(148,163,184,0.15)";
        rhBadge.style.borderColor = "rgba(148,163,184,0.4)";
        rhBadge.style.color       = "var(--text-2)";
      }

      rhSub.textContent = `Polyps detected: ${d.polyp_count ?? 0}  ·  Coverage: ${d.coverage_pct ?? 0}%`;

      /* ── Images ─────────────────────────────────────────── */
      setImg("img-original", d.original_image);
      setImg("img-enhanced", d.enhanced_image);
      setImg("img-detection", d.detection_image);
      setImg("img-mask",    d.mask_path);
      setImg("img-overlay", d.overlay_path);

      /* ── Detection stats ────────────────────────────────── */
      const detStats = document.getElementById("detect-stats");
      if (d.detections && d.detections.length > 0) {
        detStats.innerHTML = d.detections.map((det, i) =>
          `<span class="det-pill">Region ${i+1}: ${(det.confidence * 100).toFixed(1)}% confidence</span>`
        ).join("");
      } else {
        detStats.innerHTML = `<span style="color:var(--text-2);font-size:.85rem;">No polyp regions detected by YOLO.</span>`;
      }

      /* ── Classification ─────────────────────────────────── */
      const cls         = d.classification || {};
      const verdictEl   = document.getElementById("classify-verdict");
      const barsEl      = document.getElementById("model-bars");
      const isNoPolyp   = cls.label === "No Polyp Detected";
      const isMalignant = cls.label === "Malignant";

      if (isNoPolyp) {
        verdictEl.className = "classify-verdict benign";
        verdictEl.innerHTML = `
          <span class="verdict-icon">ℹ️</span>
          <div>
            <div class="verdict-label" style="color:var(--text-2)">No Polyp Detected</div>
            <div class="verdict-conf">YOLO and segmentation found no polyp region.</div>
          </div>`;
        barsEl.innerHTML = `<p style="color:var(--text-2);font-size:.85rem">Classification skipped — no polyp evidence found.</p>`;
      } else {
        verdictEl.className = "classify-verdict " + (isMalignant ? "malignant" : "benign");
        verdictEl.innerHTML = `
          <span class="verdict-icon">${isMalignant ? "⚠️" : "✅"}</span>
          <div>
            <div class="verdict-label" style="color:${isMalignant ? "var(--red)" : "var(--green)"}">${cls.label || "—"}</div>
          </div>`;

        if (cls.model_scores) {
          const malignantClass = isMalignant ? "malignant-fill" : "";
          barsEl.innerHTML = Object.entries(cls.model_scores)
            .map(([model, score]) => `
              <div class="model-bar-row">
                <div class="model-bar-label">
                  <span class="model-bar-name">${model}</span>
                  <span class="model-bar-pct">${pct(score)}</span>
                </div>
                <div class="bar-track">
                  <div class="bar-fill ${malignantClass}" style="width:0%" data-target="${(score*100).toFixed(1)}%"></div>
                </div>
              </div>`).join("");

          requestAnimationFrame(() => {
            document.querySelectorAll(".bar-fill").forEach(bar => {
              bar.style.width = bar.dataset.target;
            });
          });
        }
      }


      /* ── Segmentation note ──────────────────────────────── */
      const segNote = document.getElementById("seg-note");
      segNote.textContent = `Polyp area coverage: ${d.coverage_pct ?? 0}% of image`;

      /* ── Clinical text ──────────────────────────────────── */
      const clinicalEl = document.getElementById("clinical-text");
      clinicalEl.textContent = d.clinical_description || "No clinical description generated.";

      /* ── Show grid ──────────────────────────────────────── */
      resultsGrid.classList.remove("hidden");
    }
  }


  /* ── Utilities ──────────────────────────────────────────── */
  function setImg(id, relPath) {
    const el = document.getElementById(id);
    if (!el) return;
    if (relPath) {
      el.src = "/static/" + relPath;
      el.alt = id;
    }
  }

  function pct(val) {
    if (val == null) return "—";
    return (val * 100).toFixed(1) + "%";
  }

  function formatBytes(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  }

})();
