const API_RAW_URL = "/predict_file";
const API_PROCESSED_URL = "/predict_processed_file";

const screenUpload = document.getElementById("screenUpload");
const screenProcessing = document.getElementById("screenProcessing");
const screenReport = document.getElementById("screenReport");

const stepUpload = document.getElementById("stepUpload");
const stepProcess = document.getElementById("stepProcess");
const stepReport = document.getElementById("stepReport");

const predictBtn = document.getElementById("predictBtn");
const newCaseBtn = document.getElementById("newCaseBtn");
const downloadReportBtn = document.getElementById("downloadReportBtn");

const fileInput = document.getElementById("fileInput");
const patientIdInput = document.getElementById("patientId");
const clinicianNameInput = document.getElementById("clinicianName");
const uploadHint = document.getElementById("uploadHint");
const processingText = document.getElementById("processingText");

const reportMeta = document.getElementById("reportMeta");
const decisionText = document.getElementById("decisionText");
const clinicalInterpretation = document.getElementById("clinicalInterpretation");
const probCalEl = document.getElementById("probCal");
const probRawEl = document.getElementById("probRaw");
const qualityEl = document.getElementById("quality");
const abstainedEl = document.getElementById("abstained");
const modelVersionEl = document.getElementById("modelVersion");
const inferenceMsEl = document.getElementById("inferenceMs");
const reasonsEl = document.getElementById("reasons");
const eventsEl = document.getElementById("events");
const followupText = document.getElementById("followupText");

let latestResult = null;
let latestContext = null;

function currentMode() {
  return document.querySelector('input[name="inputMode"]:checked')?.value ?? "raw";
}

function showScreen(name) {
  const screens = [
    ["upload", screenUpload, stepUpload],
    ["processing", screenProcessing, stepProcess],
    ["report", screenReport, stepReport],
  ];

  screens.forEach(([key, section, step]) => {
    const active = key === name;
    section.classList.toggle("active", active);
    step.classList.toggle("active", active);
  });
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function fmt(value) {
  return Number(value ?? 0).toFixed(3);
}

function isPositiveDecision(decision) {
  return String(decision || "").endsWith("CHANCES OF ASD");
}

function positiveStage(decision) {
  const d = String(decision || "");
  if (d.startsWith("HIGH")) return "HIGH";
  if (d.startsWith("MEDIUM")) return "MEDIUM";
  if (d.startsWith("LOW")) return "LOW";
  return "";
}

function shouldShowEventEvidence(decision) {
  return isPositiveDecision(decision) || decision === "NEEDS RECHECKING";
}

function interpretationFromDecision(decision, abstained) {
  if (abstained || decision === "LOW QUALITY VIDEO/FALSE VIDEO UPLOAD") {
    return "Insufficient quality for definitive risk screening.";
  }
  if (isPositiveDecision(decision)) {
    const stage = positiveStage(decision);
    if (stage === "HIGH") return "High positive ASD-associated behavioral signal.";
    if (stage === "MEDIUM") return "Moderate positive ASD-associated behavioral signal.";
    return "Low positive ASD-associated behavioral signal.";
  }
  if (decision === "NEEDS RECHECKING") {
    return "Intermediate signal; findings require clinical correlation.";
  }
  return "Low ASD-associated behavioral signal in this sample.";
}

function followupFromDecision(decision, abstained) {
  if (abstained || decision === "LOW QUALITY VIDEO/FALSE VIDEO UPLOAD") {
    return "Acquire a higher-quality recording and repeat screening. If concern persists, proceed with formal developmental assessment.";
  }
  if (isPositiveDecision(decision)) {
    const stage = positiveStage(decision);
    if (stage === "HIGH") {
      return "Recommend urgent referral for comprehensive multidisciplinary ASD evaluation and detailed developmental history review.";
    }
    if (stage === "MEDIUM") {
      return "Recommend timely referral for multidisciplinary ASD evaluation and focused developmental assessment.";
    }
    return "Recommend structured follow-up with targeted ASD assessment, with escalation if concerns persist.";
  }
  if (decision === "NEEDS RECHECKING") {
    return "Recommend repeat sampling and targeted clinical observation, with escalation to full diagnostic evaluation if concerns continue.";
  }
  return "Continue routine developmental surveillance and reassess if new social-communication concerns are observed.";
}

function clearList(el) {
  el.innerHTML = "";
}

function addListItems(el, items) {
  clearList(el);
  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    el.appendChild(li);
  });
}

function renderReport(data, context) {
  const decision = String(data.decision ?? "-").toUpperCase();
  const abstained = Boolean(data.abstained);
  const interpretation = interpretationFromDecision(decision, abstained);
  const followup = followupFromDecision(decision, abstained);

  decisionText.textContent = decision;
  decisionText.className = isPositiveDecision(decision) ? "decision-badge positive"
    : decision === "NEEDS RECHECKING" ? "decision-badge borderline"
    : decision === "LOW QUALITY VIDEO/FALSE VIDEO UPLOAD" ? "decision-badge abstain"
    : "decision-badge negative";

  clinicalInterpretation.textContent = interpretation;
  probCalEl.textContent = fmt(data.prob_calibrated);
  probRawEl.textContent = fmt(data.prob_raw);
  qualityEl.textContent = fmt(data.quality_score);
  abstainedEl.textContent = abstained ? "Yes" : "No";
  modelVersionEl.textContent = data.model_version || "N/A";
  inferenceMsEl.textContent = Number(data.inference_ms ?? 0).toString();
  followupText.textContent = followup;

  const stamp = new Date().toLocaleString();
  const patient = context.patientId ? `Patient: ${context.patientId}` : "Patient: Not specified";
  const clinician = context.clinicianName ? `Reviewer: ${context.clinicianName}` : "Reviewer: Not specified";
  const modeLabel = context.mode === "processed" ? "Landmarks-Only Video" : "Standard Video";
  reportMeta.textContent = `${patient} | ${clinician} | Mode: ${modeLabel} | Generated: ${stamp}`;

  const reasons = Array.isArray(data.reasons) && data.reasons.length
    ? data.reasons
    : ["No rationale signals were returned by the backend."];
  addListItems(reasonsEl, reasons);

  if (shouldShowEventEvidence(decision)) {
    const events = Array.isArray(data.events) ? data.events : [];
    if (events.length === 0) {
      addListItems(eventsEl, ["No event-token evidence was available for this case."]);
    } else {
      addListItems(
        eventsEl,
        events.map((evt) => {
          const eventName = evt.event || "unknown_event";
          const mean = fmt(evt.mean_confidence);
          const count = Number(evt.count ?? 0);
          return `${eventName} | mean confidence ${mean} | observations ${count}`;
        }),
      );
    }
  } else {
    addListItems(eventsEl, ["Event evidence display is limited to Positive or Needs Rechecking outcomes."]);
  }
}

function pdfEscape(str) {
  const ascii = String(str ?? "").replace(/[^\x20-\x7E]/g, "?");
  return ascii.replaceAll("\\", "\\\\").replaceAll("(", "\\(").replaceAll(")", "\\)");
}

function wrapText(text, maxChars) {
  const src = String(text ?? "").trim();
  if (!src) return [""];
  const words = src.split(/\s+/);
  const lines = [];
  let cur = "";

  words.forEach((word) => {
    if (!cur) {
      cur = word;
      return;
    }
    const next = `${cur} ${word}`;
    if (next.length <= maxChars) {
      cur = next;
    } else {
      lines.push(cur);
      cur = word;
    }
  });

  if (cur) lines.push(cur);
  return lines;
}

function makePdf(result, context) {
  const now = new Date().toLocaleString();
  const decision = String(result.decision ?? "-").toUpperCase();
  const abstained = Boolean(result.abstained);
  const interpretation = interpretationFromDecision(decision, abstained);
  const followup = followupFromDecision(decision, abstained);
  const reasons = Array.isArray(result.reasons) ? result.reasons : [];
  const events = Array.isArray(result.events) ? result.events : [];
  const modeLabel = context.mode === "processed" ? "Landmarks-Only Video" : "Standard Video";
  const pageWidth = 595.28;
  const pageHeight = 841.89;
  const marginX = 40;
  const bodyWidth = pageWidth - marginX * 2;
  const bottomY = 50;

  const palette = {
    bg: [1.0, 1.0, 1.0],
    navy: [0.08, 0.21, 0.37],
    softBlue: [0.94, 0.97, 1.0],
    border: [0.76, 0.82, 0.89],
    text: [0.12, 0.15, 0.2],
    muted: [0.34, 0.4, 0.48],
    white: [1.0, 1.0, 1.0],
  };

  function decisionColor() {
    if (decision === "HIGH CHANCES OF ASD") return [0.73, 0.2, 0.14];
    if (decision === "MEDIUM CHANCES OF ASD") return [0.79, 0.43, 0.11];
    if (decision === "LOW CHANCES OF ASD") return [0.78, 0.56, 0.12];
    if (decision === "NEEDS RECHECKING") return [0.66, 0.47, 0.16];
    if (decision === "LOW QUALITY VIDEO/FALSE VIDEO UPLOAD") return [0.52, 0.17, 0.24];
    return [0.19, 0.48, 0.3];
  }

  function lineHeight(size) {
    return size * 1.35;
  }

  function charsForWidth(width, size) {
    const approxCharWidth = Math.max(4.5, size * 0.53);
    return Math.max(12, Math.floor(width / approxCharWidth));
  }

  function wrapForWidth(text, width, size) {
    return wrapText(text, charsForWidth(width, size));
  }

  let y = 0;
  let ops = [];
  const pages = [];

  function rgbFill(color) {
    return `${color[0].toFixed(3)} ${color[1].toFixed(3)} ${color[2].toFixed(3)} rg`;
  }

  function rgbStroke(color) {
    return `${color[0].toFixed(3)} ${color[1].toFixed(3)} ${color[2].toFixed(3)} RG`;
  }

  function drawRect(x, top, w, h, fillColor = null, strokeColor = null, strokeWidth = 1) {
    const bottom = top - h;
    if (fillColor) ops.push(rgbFill(fillColor));
    if (strokeColor) {
      ops.push(rgbStroke(strokeColor));
      ops.push(`${strokeWidth.toFixed(2)} w`);
    }
    const paint = fillColor && strokeColor ? "B" : fillColor ? "f" : "S";
    ops.push(`${x.toFixed(2)} ${bottom.toFixed(2)} ${w.toFixed(2)} ${h.toFixed(2)} re ${paint}`);
  }

  function drawText(x, baseline, text, size = 10, bold = false, color = palette.text) {
    const font = bold ? "F2" : "F1";
    ops.push(rgbFill(color));
    ops.push(`BT /${font} ${size.toFixed(2)} Tf ${x.toFixed(2)} ${baseline.toFixed(2)} Td (${pdfEscape(text)}) Tj ET`);
  }

  function beginPage(firstPage = false) {
    if (ops.length) pages.push(ops.join("\n"));
    ops = [];

    const headerTop = pageHeight - 24;
    if (firstPage) {
      const headerHeight = 92;
      drawRect(marginX, headerTop, bodyWidth, headerHeight, palette.navy, palette.navy, 1);
      drawText(marginX + 14, headerTop - 30, "ASDMOTION CLINICAL SCREENING REPORT", 15, true, palette.white);
      drawText(marginX + 14, headerTop - 47, "Confidential Medical Decision-Support Summary", 9, false, [0.9, 0.94, 0.99]);
      drawText(marginX + 14, headerTop - 63, "For clinical review only - not a standalone diagnosis", 8.5, false, [0.86, 0.91, 0.97]);

      const badgeWidth = 205;
      const badgeHeight = 40;
      const badgeTop = headerTop - 18;
      const badgeX = marginX + bodyWidth - badgeWidth - 14;
      drawRect(badgeX, badgeTop, badgeWidth, badgeHeight, decisionColor(), null, 0);
      const badgeLines = wrapForWidth(decision, badgeWidth - 14, 8.5);
      drawText(badgeX + 8, badgeTop - 17, "SCREENING DECISION", 7.2, true, [0.95, 0.95, 0.95]);
      if (badgeLines[0]) drawText(badgeX + 8, badgeTop - 29, badgeLines[0], 8.7, true, palette.white);
      if (badgeLines[1]) drawText(badgeX + 8, badgeTop - 38, badgeLines[1], 8.7, true, palette.white);
      y = headerTop - headerHeight - 14;
    } else {
      const bandHeight = 32;
      drawRect(marginX, headerTop, bodyWidth, bandHeight, palette.softBlue, palette.border, 0.9);
      drawText(marginX + 10, headerTop - 20, "ASDMotion Clinical Screening Report (continued)", 10, true, palette.navy);
      y = headerTop - bandHeight - 12;
    }
  }

  function ensureSpace(heightNeeded) {
    if (y - heightNeeded < bottomY) {
      beginPage(false);
    }
  }

  function drawSection(title, lines) {
    const headerHeight = 24;
    const innerPad = 12;
    let bodyHeight = 0;
    lines.forEach((ln) => {
      bodyHeight += lineHeight(ln.size || 10);
    });
    const sectionHeight = headerHeight + innerPad + bodyHeight + 8;

    ensureSpace(sectionHeight + 10);

    const top = y;
    drawRect(marginX, top, bodyWidth, sectionHeight, palette.bg, palette.border, 0.9);
    drawRect(marginX, top, bodyWidth, headerHeight, palette.softBlue, palette.border, 0.9);
    drawText(marginX + 10, top - 16, title, 10, true, palette.navy);

    let cursor = top - headerHeight - 10;
    lines.forEach((ln) => {
      const size = ln.size || 10;
      const indent = ln.indent || 0;
      drawText(marginX + 12 + indent, cursor, ln.text, size, Boolean(ln.bold), ln.color || palette.text);
      cursor -= lineHeight(size);
    });

    y = top - sectionHeight - 10;
  }

  function drawKeyValueGrid(title, items, columns = 2) {
    const headerHeight = 24;
    const innerPad = 10;
    const rowHeight = 28;
    const rowCount = Math.ceil(items.length / columns);
    const gridHeight = rowCount * rowHeight;
    const sectionHeight = headerHeight + innerPad + gridHeight + innerPad;

    ensureSpace(sectionHeight + 10);

    const top = y;
    drawRect(marginX, top, bodyWidth, sectionHeight, palette.bg, palette.border, 0.9);
    drawRect(marginX, top, bodyWidth, headerHeight, palette.softBlue, palette.border, 0.9);
    drawText(marginX + 10, top - 16, title, 10, true, palette.navy);

    const gridTop = top - headerHeight - innerPad;
    const cellWidth = (bodyWidth - 16) / columns;
    let idx = 0;
    for (let r = 0; r < rowCount; r += 1) {
      for (let c = 0; c < columns; c += 1) {
        if (idx >= items.length) break;
        const item = items[idx];
        const x = marginX + 8 + c * cellWidth;
        const cellTop = gridTop - r * rowHeight;
        drawRect(x, cellTop, cellWidth - 6, rowHeight - 2, r % 2 === 0 ? [0.985, 0.99, 1.0] : [0.972, 0.982, 0.995], palette.border, 0.5);
        drawText(x + 7, cellTop - 12, item.label, 8.1, true, palette.muted);
        const valueLines = wrapForWidth(item.value, cellWidth - 18, 9.1);
        drawText(x + 7, cellTop - 22, valueLines[0] || "-", 9.1, false, palette.text);
        idx += 1;
      }
    }

    y = top - sectionHeight - 10;
  }

  function drawEventTable(title, rows) {
    if (!rows.length) {
      drawSection(title, [
        { text: "No event-token evidence was available for this case.", size: 10 },
      ]);
      return;
    }

    const headerHeight = 24;
    const innerPad = 10;
    const tableHeaderHeight = 20;
    const rowHeight = 18;
    const sectionHeight = headerHeight + innerPad + tableHeaderHeight + rows.length * rowHeight + innerPad;

    ensureSpace(sectionHeight + 10);

    const top = y;
    drawRect(marginX, top, bodyWidth, sectionHeight, palette.bg, palette.border, 0.9);
    drawRect(marginX, top, bodyWidth, headerHeight, palette.softBlue, palette.border, 0.9);
    drawText(marginX + 10, top - 16, title, 10, true, palette.navy);

    const tableTop = top - headerHeight - innerPad;
    const col1 = bodyWidth * 0.62;
    const col2 = bodyWidth * 0.2;
    const col3 = bodyWidth - col1 - col2;

    drawRect(marginX + 1, tableTop, bodyWidth - 2, tableHeaderHeight, [0.925, 0.95, 0.985], palette.border, 0.6);
    drawText(marginX + 8, tableTop - 13, "Event", 8.4, true, palette.navy);
    drawText(marginX + col1 + 5, tableTop - 13, "Mean Conf.", 8.4, true, palette.navy);
    drawText(marginX + col1 + col2 + 5, tableTop - 13, "Count", 8.4, true, palette.navy);

    rows.forEach((row, i) => {
      const rowTop = tableTop - tableHeaderHeight - i * rowHeight;
      const fill = i % 2 === 0 ? [0.988, 0.992, 0.998] : [0.976, 0.985, 0.996];
      drawRect(marginX + 1, rowTop, bodyWidth - 2, rowHeight, fill, palette.border, 0.3);
      drawText(marginX + 8, rowTop - 12, row.event, 8.8, false, palette.text);
      drawText(marginX + col1 + 6, rowTop - 12, row.mean, 8.8, false, palette.text);
      drawText(marginX + col1 + col2 + 6, rowTop - 12, row.count, 8.8, false, palette.text);
    });

    y = top - sectionHeight - 10;
  }

  beginPage(true);

  const summaryLines = [];
  summaryLines.push({ text: "Decision", size: 8.6, bold: true, color: palette.muted });
  wrapForWidth(decision, bodyWidth - 24, 14).forEach((line) => {
    summaryLines.push({ text: line, size: 14, bold: true, color: decisionColor() });
  });
  summaryLines.push({ text: "Clinical Interpretation", size: 8.6, bold: true, color: palette.muted });
  wrapForWidth(interpretation, bodyWidth - 24, 10).forEach((line) => {
    summaryLines.push({ text: line, size: 10, color: palette.text });
  });
  drawSection("Clinical Screening Summary", summaryLines);

  drawKeyValueGrid("Case Information", [
    { label: "Patient ID", value: context.patientId || "Not specified" },
    { label: "Reviewer", value: context.clinicianName || "Not specified" },
    { label: "Generated", value: now },
    { label: "Input Mode", value: modeLabel },
    { label: "Model Version", value: result.model_version || "N/A" },
    { label: "Inference Time (ms)", value: String(Number(result.inference_ms ?? 0)) },
  ], 2);

  drawKeyValueGrid("Quantitative Measures", [
    { label: "Calibrated Probability", value: fmt(result.prob_calibrated) },
    { label: "Raw Probability", value: fmt(result.prob_raw) },
    { label: "Quality Score", value: fmt(result.quality_score) },
    { label: "Abstained", value: abstained ? "Yes" : "No" },
  ], 2);

  const rationaleLines = [];
  if (reasons.length) {
    reasons.forEach((reason) => {
      const wrapped = wrapForWidth(reason, bodyWidth - 30, 10);
      wrapped.forEach((line, idx) => {
        rationaleLines.push({ text: idx === 0 ? `- ${line}` : `  ${line}`, size: 10 });
      });
    });
  } else {
    rationaleLines.push({ text: "- No rationale signals were returned by the backend.", size: 10 });
  }
  drawSection("Clinical Rationale", rationaleLines);

  if (shouldShowEventEvidence(decision)) {
    const eventRows = events.slice(0, 8).map((evt) => ({
      event: String(evt.event || "unknown_event"),
      mean: fmt(evt.mean_confidence),
      count: String(Number(evt.count ?? 0)),
    }));
    drawEventTable("Behavioral Event Evidence", eventRows);
  } else {
    drawSection("Behavioral Event Evidence", [
      { text: "Event evidence display is limited to Positive or Needs Rechecking outcomes.", size: 10 },
    ]);
  }

  const followupLines = [];
  wrapForWidth(followup, bodyWidth - 24, 10).forEach((line) => {
    followupLines.push({ text: line, size: 10 });
  });
  drawSection("Recommended Clinical Follow-up", followupLines);

  const noticeLines = [];
  wrapForWidth(
    "This screening output is decision-support only and must not be used as a standalone diagnosis.",
    bodyWidth - 24,
    9.5,
  ).forEach((line) => noticeLines.push({ text: line, size: 9.5, color: palette.muted }));
  wrapForWidth(
    "Comprehensive clinical assessment remains required, including formal developmental evaluation.",
    bodyWidth - 24,
    9.5,
  ).forEach((line) => noticeLines.push({ text: line, size: 9.5, color: palette.muted }));
  drawSection("Clinical Use Notice", noticeLines);

  ensureSpace(52);
  drawRect(marginX, y, bodyWidth, 42, [0.985, 0.988, 0.994], palette.border, 0.8);
  drawText(marginX + 10, y - 14, "Clinician Signature: ____________________", 9.5, false, palette.muted);
  drawText(marginX + bodyWidth / 2 + 8, y - 14, "Date: ____________________", 9.5, false, palette.muted);
  y -= 52;

  pages.push(ops.join("\n"));

  const objects = {};
  const nPages = pages.length;
  const firstPageObj = 3;
  const firstContentObj = firstPageObj + nPages;
  const fontRegularObj = firstContentObj + nPages;
  const fontBoldObj = fontRegularObj + 1;
  const maxObj = fontBoldObj;

  objects[1] = "<< /Type /Catalog /Pages 2 0 R >>";
  const kids = [];
  for (let i = 0; i < nPages; i += 1) {
    kids.push(`${firstPageObj + i} 0 R`);
  }
  objects[2] = `<< /Type /Pages /Kids [${kids.join(" ")}] /Count ${nPages} >>`;

  for (let i = 0; i < nPages; i += 1) {
    const pageObj = firstPageObj + i;
    const contentObj = firstContentObj + i;
    const pageNo = i + 1;
    const footer = [
      rgbFill(palette.muted),
      `BT /F1 8 Tf ${marginX.toFixed(2)} 24 Td (Confidential Clinical Report - ASDMotion) Tj ET`,
      `BT /F1 8 Tf ${(pageWidth - marginX - 62).toFixed(2)} 24 Td (Page ${pageNo} of ${nPages}) Tj ET`,
    ].join("\n");
    const content = `${pages[i]}\n${footer}`;
    const contentLen = new TextEncoder().encode(content).length;

    objects[pageObj] = `<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${pageWidth.toFixed(2)} ${pageHeight.toFixed(2)}] /Resources << /Font << /F1 ${fontRegularObj} 0 R /F2 ${fontBoldObj} 0 R >> >> /Contents ${contentObj} 0 R >>`;
    objects[contentObj] = `<< /Length ${contentLen} >>\nstream\n${content}\nendstream`;
  }

  objects[fontRegularObj] = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>";
  objects[fontBoldObj] = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>";

  let pdf = "%PDF-1.4\n";
  const offsets = new Array(maxObj + 1).fill(0);

  for (let i = 1; i <= maxObj; i += 1) {
    offsets[i] = new TextEncoder().encode(pdf).length;
    pdf += `${i} 0 obj\n${objects[i]}\nendobj\n`;
  }

  const xrefOffset = new TextEncoder().encode(pdf).length;
  pdf += `xref\n0 ${maxObj + 1}\n`;
  pdf += "0000000000 65535 f \n";
  for (let i = 1; i <= maxObj; i += 1) {
    const off = String(offsets[i]).padStart(10, "0");
    pdf += `${off} 00000 n \n`;
  }
  pdf += `trailer\n<< /Size ${maxObj + 1} /Root 1 0 R >>\nstartxref\n${xrefOffset}\n%%EOF`;

  return new Blob([new TextEncoder().encode(pdf)], { type: "application/pdf" });
}

async function runPrediction() {
  const mode = currentMode();
  const file = fileInput.files[0];
  if (!file) {
    uploadHint.textContent = "Please select a video file before starting.";
    return;
  }

  uploadHint.textContent = "Uploading and running inference...";
  const modeLabel = mode === "processed" ? "landmarks-only video" : "standard video";
  processingText.textContent = `Analyzing ${modeLabel} and preparing clinical report.`;
  showScreen("processing");

  const context = {
    mode,
    fileName: file.name,
    patientId: patientIdInput.value.trim(),
    clinicianName: clinicianNameInput.value.trim(),
  };

  try {
    const formData = new FormData();
    formData.append("file", file);

    const endpoint = mode === "processed" ? API_PROCESSED_URL : API_RAW_URL;
    const res = await fetch(endpoint, { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data?.error || "Prediction failed. Check backend logs.");
    }

    latestResult = data;
    latestContext = context;
    renderReport(data, context);
    showScreen("report");
  } catch (err) {
    alert(err.message || "Prediction failed. Check backend logs.");
    showScreen("upload");
  } finally {
    uploadHint.textContent = "Select one video file to continue.";
  }
}

function startNewCase() {
  latestResult = null;
  latestContext = null;
  fileInput.value = "";
  showScreen("upload");
}

function downloadReport() {
  if (!latestResult || !latestContext) {
    alert("No report available to download.");
    return;
  }

  const blob = makePdf(latestResult, latestContext);
  const url = URL.createObjectURL(blob);

  const stamp = new Date().toISOString().slice(0, 19).replaceAll(":", "-");
  const pid = latestContext.patientId ? latestContext.patientId.replace(/[^A-Za-z0-9_-]+/g, "_") : "case";
  const filename = `asd_clinical_report_${pid}_${stamp}.pdf`;

  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

predictBtn.addEventListener("click", runPrediction);
newCaseBtn.addEventListener("click", startNewCase);
downloadReportBtn.addEventListener("click", downloadReport);

showScreen("upload");
