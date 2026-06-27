// Generate a one-page defense cheat sheet (US Letter, print-ready).
// Run: NODE_PATH="D:/Documents Dirk/Thesis/node_modules" node scripts/make_cheat_sheet.js
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, WidthType, BorderStyle, ShadingType, LevelFormat,
} = require("docx");

const OUT = "D:/PythonProject1/docs/defense/Defense_Cheat_Sheet.docx";
const NAVY = "1F4E79", GREY = "555555", RED = "A6192E", GREEN = "107C41";
const CW = 10512; // content width @ 0.6" margins on US Letter

const thinB = { style: BorderStyle.SINGLE, size: 2, color: "BBBBBB" };
const cellBorders = { top: thinB, bottom: thinB, left: thinB, right: thinB };
const noBorder = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const noBorders = { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder };

// Shaded full-width callout box (single cell with text).
function calloutBox(fill, runs) {
  return new Table({
    width: { size: CW, type: WidthType.DXA },
    columnWidths: [CW],
    rows: [new TableRow({ children: [new TableCell({
      borders: noBorders,
      width: { size: CW, type: WidthType.DXA },
      shading: { fill, type: ShadingType.CLEAR },
      margins: { top: 90, bottom: 90, left: 160, right: 160 },
      children: [new Paragraph({ spacing: { after: 0 }, children: runs })],
    })] })],
  });
}

function sectionHeader(text) {
  return new Paragraph({
    spacing: { before: 90, after: 46 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: NAVY, space: 1 } },
    children: [new TextRun({ text, bold: true, size: 22, color: NAVY })],
  });
}

// A Q/A block: bold question, indented answer.
function qa(q, a) {
  return [
    new Paragraph({ spacing: { before: 34, after: 6 },
      children: [new TextRun({ text: q, bold: true, size: 19 })] }),
    new Paragraph({ spacing: { after: 12 }, indent: { left: 180 },
      children: [new TextRun({ text: a, size: 19 })] }),
  ];
}

function numRow(label, value, shade) {
  const mk = (txt, bold, color) => new TableCell({
    borders: cellBorders,
    width: { size: txt === label ? 6200 : CW - 6200, type: WidthType.DXA },
    shading: shade ? { fill: "EEF3F8", type: ShadingType.CLEAR } : undefined,
    margins: { top: 40, bottom: 40, left: 110, right: 110 },
    children: [new Paragraph({ spacing: { after: 0 },
      children: [new TextRun({ text: txt, size: 18, bold, color })] })],
  });
  return new TableRow({ children: [mk(label, true, "000000"), mk(value, false, NAVY)] });
}

const NUMS = [
  ["Champion", "LightGBM (selected by rolling-origin PR-AUC)"],
  ["ROC-AUC / PR-AUC / F1 (test)", "0.863 / 0.759 (range 0.70–0.76) / 0.736"],
  ["Precision / Recall (max-F1 point)", "0.625 / 0.895"],
  ["Calibration error ECE (raw → isotonic)", "0.062 → 0.031"],
  ["LightGBM vs Logistic Regression", "+0.005 PR-AUC, p = 0.177 (NOT significant)"],
  ["LightGBM vs Decision Tree / Random Forest", "+0.246 / +0.032 (both significant)"],
  ["Cost: no-model / intervene-all / model", "€2,322,794 / €111,240 / €71,136"],
  ["Saving vs intervene-all", "€40,105  =  36%"],
  ["Cost threshold / recall / % flagged", "0.06 / 0.991 / ~72% of bookings"],
  ["False-positive / false-negative cost", "€15 / proportional to revenue at risk"],
  ["Duplicates in data", "~27%; champion unchanged after de-dup"],
  ["Philippine pilot", "193 rows, test n = 20, ±15pp CI (directional)"],
  ["Top SHAP feature (both markets)", "deposit_type"],
];

const children = [];

// Title
children.push(new Paragraph({ spacing: { after: 0 }, alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: "DEFENSE CHEAT SHEET", bold: true, size: 30, color: NAVY })] }));
children.push(new Paragraph({ spacing: { after: 100 }, alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: "Predicting Hotel Booking Cancellations — A Strategic Business Intelligence Approach (LightGBM)", italics: true, size: 18, color: GREY })] }));

// Pitch box
children.push(calloutBox("EAF1FA", [
  new TextRun({ text: "30-SECOND PITCH:  ", bold: true, size: 19, color: NAVY }),
  new TextRun({ text: "We don’t claim a better cancellation ", size: 19 }),
  new TextRun({ text: "model", italics: true, size: 19 }),
  new TextRun({ text: " — we deliver a calibrated, cost-aware decision ", size: 19 }),
  new TextRun({ text: "system", bold: true, size: 19 }),
  new TextRun({ text: " on an honest, time-aware evaluation, and show the method transfers to a real Philippine resort. The contribution is the system and the rigor, not the algorithm.", size: 19 }),
]));

// Killer answers
children.push(sectionHeader("THREE KILLER ANSWERS"));
qa("Q: If logistic regression ties LightGBM (p = 0.177), why gradient boosting?",
   "Concede the tie. Kept LightGBM for highest recall (0.895), best calibration (the cost thresholds depend on it), 3× faster than XGBoost, and it won the pre-specified selection criterion — not a leaderboard claim.").forEach(p => children.push(p));
qa("Q: Defend the €15 cost — isn’t the cost policy just “warn everyone”?",
   "Concede: at threshold 0.06 it flags ~72% of bookings. Bound it: sensitivity analysis (Fig 4.13); the real discrimination value is at the max-F1 point (precision 0.625 / recall 0.895). €15 is a tested assumption, not a fixed truth.").forEach(p => children.push(p));
qa("Q: Why Dynamic Capability Theory if it doesn’t drive the modeling?",
   "Own it as the organizing lens (sense → seize → transform), not an analytical input. It makes this a business-intelligence study rather than a pure benchmark; the technical work stands on its own.").forEach(p => children.push(p));
qa("Q: 27% of the data are duplicates — isn’t PR-AUC 0.759 inflated?",
   "Disclosed. No train/test leakage. De-duplicating gives 0.703, so the headline is a range 0.70–0.76. Per-record is primary only because booking IDs aren’t available to tell genuine repeats from artifacts.").forEach(p => children.push(p));

// Numbers
children.push(sectionHeader("NUMBERS TO KNOW COLD"));
children.push(new Table({
  width: { size: CW, type: WidthType.DXA },
  columnWidths: [6200, CW - 6200],
  rows: NUMS.map((r, i) => numRow(r[0], r[1], i % 2 === 1)),
}));

// Golden rule + traps
children.push(sectionHeader("IF YOU GET HIT"));
children.push(calloutBox("EAF6EE", [
  new TextRun({ text: "GOLDEN RULE:  ", bold: true, size: 19, color: GREEN }),
  new TextRun({ text: "Agree → Bound → Pivot.  ", bold: true, size: 19 }),
  new TextRun({ text: "Your honesty is your strongest card — leaning into a limitation strengthens you, because the panel has already read the caveats.", size: 19 }),
]));
children.push(new Paragraph({ spacing: { before: 80, after: 30 },
  children: [new TextRun({ text: "NEVER say (traps):", bold: true, size: 19, color: RED })] }));
[
  "“LightGBM is the most accurate.”  → It’s statistically tied with logistic regression.",
  "“The model saves 36% of costs.”  → Say: reduces expected cost 36% vs intervene-all, under our cost assumptions.",
  "“It works in the Philippines.”  → Say: directional pilot (n = 20), not a benchmark.",
].forEach(t => children.push(new Paragraph({
  numbering: { reference: "traps", level: 0 }, spacing: { after: 20 },
  children: [new TextRun({ text: t, size: 18 })] })));

const doc = new Document({
  numbering: { config: [{ reference: "traps", levels: [{ level: 0, format: LevelFormat.BULLET, text: "✗",
    alignment: AlignmentType.LEFT, style: { run: { color: RED }, paragraph: { indent: { left: 360, hanging: 220 } } } }] }] },
  styles: { default: { document: { run: { font: "Arial", size: 19 } } } },
  sections: [{
    properties: { page: {
      size: { width: 12240, height: 15840 },
      margin: { top: 864, right: 864, bottom: 720, left: 864 },
    } },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("WROTE " + OUT + " (" + buf.length + " bytes)"); });
