/* WSP swarm dashboard: reads the JSON that scripts/export_dashboard.py writes to data/
   and draws the race, the agent log, the program inspector, the fit chart, the budget,
   the real runs and the parts. Plain JavaScript, no build step, no libraries. */

"use strict";

(function () {
  var DATA_DIR = "data/";
  var SVG_NS = "http://www.w3.org/2000/svg";
  var PAGE_SIZE = 150;

  // The same fixed order as scripts/plot_curves.py CONDITION_ORDER, so slot N is always the same arm.
  var CONDITION_ORDER = ["swarm", "evolve", "swarm_llm", "random", "random_part", "greedy", "lora", "full_ft"];

  var ARM_INFO = {
    swarm: "The agent swarm: an analyst maps which parts help, planners propose programs and a rule judge keeps the best.",
    evolve: "Regularized evolution over grafts, the primary baseline.",
    swarm_llm: "The swarm with LLM planners.",
    random: "Uniform random programs, the sanity floor.",
    random_part: "The evolution baseline with every graft's units made random: it measures what placement and heal gain on their own.",
    greedy: "Greedy add-if-better over single grafts, then depth shifts.",
    lora: "LoRA on the host, trained on the same heal data.",
    full_ft: "Full fine-tuning of the host on the same heal data."
  };

  var STATUS_INFO = {
    ok: "ok",
    collapsed: "collapsed",
    invalid: "invalid",
    rejected_fit: "rejected by the fit gate",
    compile_error: "compile error",
    cache_hit: "cache hit",
    infra_error: "infra error",
    budget_exhausted: "budget exhausted",
    control: "control"
  };

  var state = {
    index: null,
    cache: {},
    ds: null,
    armsOn: {},
    selected: null,
    logLimit: PAGE_SIZE,
    smoke: null,
    final: null,
    parts: null
  };

  // DOM helpers.

  function el(tag, attrs, children) {
    var node = document.createElement(tag);
    setAttrs(node, attrs);
    appendAll(node, children);
    return node;
  }

  function setAttrs(node, attrs) {
    if (!attrs) {
      return;
    }
    Object.keys(attrs).forEach(function (key) {
      var value = attrs[key];
      if (value === null || value === undefined || value === false) {
        return;
      }
      if (key === "class") {
        node.setAttribute("class", value);
      } else if (key === "text") {
        node.textContent = value;
      } else if (key === "style") {
        node.setAttribute("style", value);
      } else if (key.indexOf("on") === 0 && typeof value === "function") {
        node.addEventListener(key.slice(2), value);
      } else {
        node.setAttribute(key, value === true ? "" : String(value));
      }
    });
  }

  function appendAll(node, children) {
    if (children === null || children === undefined) {
      return;
    }
    if (!Array.isArray(children)) {
      children = [children];
    }
    children.forEach(function (child) {
      if (child === null || child === undefined || child === false) {
        return;
      }
      if (typeof child === "string" || typeof child === "number") {
        node.appendChild(document.createTextNode(String(child)));
      } else {
        node.appendChild(child);
      }
    });
  }

  function svgEl(tag, attrs, parent) {
    var node = document.createElementNS(SVG_NS, tag);
    if (attrs) {
      Object.keys(attrs).forEach(function (key) {
        if (attrs[key] === null || attrs[key] === undefined) {
          return;
        }
        if (key === "text") {
          node.textContent = attrs[key];
        } else {
          node.setAttribute(key, String(attrs[key]));
        }
      });
    }
    if (parent) {
      parent.appendChild(node);
    }
    return node;
  }

  function clear(node) {
    while (node.firstChild) {
      node.removeChild(node.firstChild);
    }
  }

  function byId(id) {
    return document.getElementById(id);
  }

  // Formatting.

  function isNumber(value) {
    return typeof value === "number" && isFinite(value);
  }

  function fmtScore(value) {
    return isNumber(value) ? value.toFixed(4) : "none";
  }

  function fmtShort(value, digits) {
    if (!isNumber(value)) {
      return "none";
    }
    return value.toFixed(digits === undefined ? 3 : digits);
  }

  function fmtFee(value) {
    if (!isNumber(value)) {
      return "none";
    }
    return String(Math.round(value * 100) / 100);
  }

  function fmtInt(value) {
    if (!isNumber(value)) {
      return "none";
    }
    return Math.round(value).toLocaleString("en-US");
  }

  function fmtCompact(value) {
    if (!isNumber(value)) {
      return "";
    }
    var abs = Math.abs(value);
    if (abs >= 1e6) {
      return trimZeros((value / 1e6).toFixed(1)) + "M";
    }
    if (abs >= 1e3) {
      return trimZeros((value / 1e3).toFixed(abs >= 1e4 ? 0 : 1)) + "k";
    }
    return trimZeros(value.toFixed(abs < 10 ? 2 : 0));
  }

  function trimZeros(text) {
    if (text.indexOf(".") < 0) {
      return text;
    }
    return text.replace(/0+$/, "").replace(/\.$/, "");
  }

  function fmtParams(value) {
    if (!isNumber(value)) {
      return "none";
    }
    if (value >= 1e9) {
      return trimZeros((value / 1e9).toFixed(2)) + "B";
    }
    if (value >= 1e6) {
      return trimZeros((value / 1e6).toFixed(1)) + "M";
    }
    if (value >= 1e3) {
      return trimZeros((value / 1e3).toFixed(0)) + "k";
    }
    return String(value);
  }

  function shortPart(id) {
    if (!id) {
      return "part";
    }
    return String(id).replace(/^(stub|pilot)_/, "").replace(/_/g, ".");
  }

  function slotOf(condition) {
    return CONDITION_ORDER.indexOf(condition);
  }

  function armColor(condition) {
    var slot = slotOf(condition);
    return slot >= 0 ? "var(--s" + slot + ")" : "var(--other-arm)";
  }

  function swatch(condition) {
    return el("span", { class: "swatch", style: "background:" + armColor(condition), "aria-hidden": "true" });
  }

  function labelChips(labels) {
    var wrap = el("span", { class: "labels" });
    (labels || []).forEach(function (label) {
      wrap.appendChild(el("span", { class: "label label-" + label, text: label }));
    });
    return wrap;
  }

  function statusBadge(status) {
    var safe = String(status || "none").replace(/[^a-z_]/g, "");
    return el("span", { class: "status" }, [
      el("span", { class: "dot " + safe, "aria-hidden": "true" }),
      STATUS_INFO[status] || String(status)
    ]);
  }

  function plural(count, word) {
    return fmtInt(count) + " " + word + (count === 1 ? "" : "s");
  }

  function median(values) {
    var sorted = values.filter(isNumber).slice().sort(function (a, b) { return a - b; });
    if (sorted.length === 0) {
      return null;
    }
    var middle = Math.floor(sorted.length / 2);
    if (sorted.length % 2 === 1) {
      return sorted[middle];
    }
    return (sorted[middle - 1] + sorted[middle]) / 2;
  }

  function minOf(values) {
    var numbers = values.filter(isNumber);
    return numbers.length ? Math.min.apply(null, numbers) : null;
  }

  function maxOf(values) {
    var numbers = values.filter(isNumber);
    return numbers.length ? Math.max.apply(null, numbers) : null;
  }

  // Tooltip.

  function showTooltip(event, content) {
    var tip = byId("tooltip");
    clear(tip);
    appendAll(tip, content);
    tip.hidden = false;
    var x = event.clientX + 14;
    var y = event.clientY + 14;
    var width = tip.offsetWidth;
    var height = tip.offsetHeight;
    if (x + width > window.innerWidth - 8) {
      x = Math.max(8, event.clientX - width - 14);
    }
    if (y + height > window.innerHeight - 8) {
      y = Math.max(8, event.clientY - height - 14);
    }
    tip.style.left = x + "px";
    tip.style.top = y + "px";
  }

  function hideTooltip() {
    byId("tooltip").hidden = true;
  }

  // Loading.

  function fetchJson(name) {
    return fetch(DATA_DIR + name, { cache: "no-cache" }).then(function (response) {
      if (!response.ok) {
        throw new Error("could not load " + name + " (HTTP " + response.status + ")");
      }
      return response.json();
    });
  }

  function init() {
    fetchJson("index.json").then(function (index) {
      state.index = index;
      renderBanner();
      renderSnapshot();
      renderFooter();
      setupDatasetSelect();
      setupLogFilters();
      selectDataset(index.default_dataset);
      loadSide();
    }).catch(function (error) {
      var banner = byId("data-banner");
      banner.textContent = "The data snapshot did not load: " + error.message + ". Run scripts/export_dashboard.py.";
    });
    var timer = null;
    window.addEventListener("resize", function () {
      clearTimeout(timer);
      timer = setTimeout(redrawCharts, 150);
    });
  }

  function loadSide() {
    var files = state.index.files;
    fetchJson(files.smoke).then(function (smoke) {
      state.smoke = smoke;
      renderSmoke();
    }).catch(function (error) {
      byId("smoke-panel").textContent = "The smoke results did not load: " + error.message;
    });
    fetchJson(files.final).then(function (final) {
      state.final = final;
      renderFinal();
    }).catch(function (error) {
      byId("final-panel").textContent = "The compiled model data did not load: " + error.message;
    });
    fetchJson(files.parts).then(function (parts) {
      state.parts = parts;
      renderParts();
    }).catch(function (error) {
      byId("parts-panel").textContent = "The parts catalog did not load: " + error.message;
    });
  }

  function datasetEntry(id) {
    var entries = state.index.datasets;
    for (var i = 0; i < entries.length; i++) {
      if (entries[i].id === id) {
        return entries[i];
      }
    }
    return entries[0];
  }

  function selectDataset(id) {
    var entry = datasetEntry(id);
    if (!entry) {
      byId("dataset-summary").textContent = "No datasets in this snapshot.";
      return;
    }
    byId("dataset-select").value = entry.id;
    byId("dataset-summary").textContent = "Loading " + entry.title + ".";
    var cached = state.cache[entry.id];
    var load = cached ? Promise.resolve(cached) : fetchJson(entry.file);
    load.then(function (raw) {
      state.cache[entry.id] = raw;
      state.ds = prepareDataset(raw);
      state.armsOn = {};
      state.ds.raw.conditions.forEach(function (condition) { state.armsOn[condition] = true; });
      state.logLimit = PAGE_SIZE;
      resetLogFilters();
      renderAll();
      selectRow(defaultRow(state.ds), false);
    }).catch(function (error) {
      byId("dataset-summary").textContent = "This dataset did not load: " + error.message;
    });
  }

  function prepareDataset(raw) {
    var runs = raw.runs;
    var rows = [];
    var rowsById = {};
    raw.log.forEach(function (compact) {
      var row = Object.assign({}, raw.row_defaults || {}, compact);
      var run = runs[row.run];
      row.condition = run.condition;
      row.seed = run.seed;
      row.method = row.method || run.condition;
      var entry = raw.programs[row.program_hash] || {};
      row.describe = entry.describe || "";
      row.searchText = [row.describe, row.hypothesis, row.program_hash, row.agent, row.status, run.run_id]
        .join(" ").toLowerCase();
      rows.push(row);
      rowsById[row.i] = row;
    });
    var events = raw.events.map(function (event) {
      var item = Object.assign({}, event);
      var run = runs[item.run];
      item.condition = run.condition;
      item.seed = run.seed;
      item.searchText = [item.text, item.kind, item.agent].join(" ").toLowerCase();
      return item;
    });
    return { raw: raw, runs: runs, rows: rows, rowsById: rowsById, events: events };
  }

  function programFor(ds, row) {
    var entry = ds.raw.programs[row.program_hash];
    if (!entry) {
      return null;
    }
    var program = entry.empty_program ? {} : Object.assign({}, ds.raw.program_base || {}, entry.program);
    var grafts = (program.grafts || []).map(function (graft, index) {
      return Object.assign({}, graft, ds.raw.parts[graft.id] || {}, entry.grafts[index] || {});
    });
    return { program: program, describe: entry.describe, grafts: grafts, training: entry.training || null };
  }

  function isCurveRow(row) {
    return row.split === "search" && (row.fidelity === "full" || row.fidelity === null) && row.status === "ok" &&
      isNumber(row.combined);
  }

  function defaultRow(ds) {
    var firstArm = ds.raw.arms[0];
    if (!firstArm) {
      return ds.rows[ds.rows.length - 1] || null;
    }
    var runIndex = firstArm.runs[0];
    var best = null;
    ds.rows.forEach(function (row) {
      if (row.run === runIndex && isCurveRow(row) && (best === null || row.combined > best.combined)) {
        best = row;
      }
    });
    return best || ds.rows[ds.rows.length - 1] || null;
  }

  function renderAll() {
    renderDatasetSummary();
    renderArmToggles();
    renderRaceCharts();
    renderTimesFewer();
    renderRaceTable();
    populateLogFilters();
    renderLog();
    renderFitChart();
    renderBudget();
  }

  function redrawCharts() {
    if (!state.ds) {
      return;
    }
    renderRaceCharts();
    renderFitChart();
    if (state.selected) {
      renderInspector(state.selected);
    }
  }

  // Header, snapshot and footer.

  function renderBanner() {
    var banner = byId("data-banner");
    clear(banner);
    if (state.index.all_search_stub) {
      appendAll(banner, [
        el("strong", { text: "STUB DATA. None of the search numbers on this page are results." }),
        el("p", { text: "Every score in the race, the agent log, the inspector, the fit chart and the budget comes " +
          "from the stub backend, a deterministic simulator that tests the plumbing and the fairness rules. The " +
          "only real model scores are in Real runs, and those are a 20-item SMOKE INTERIM check." })
      ]);
    } else {
      banner.classList.add("mixed");
      appendAll(banner, [
        el("strong", { text: "Mixed data. Read the labels on every number." }),
        el("p", { text: "STUB numbers come from the simulator, SEARCH numbers from the search split, HELD-OUT " +
          "numbers from the split scored once at the end, and INTERIM numbers from the interim scorer." })
      ]);
    }
  }

  function renderSnapshot() {
    var snapshot = state.index.snapshot || {};
    var parts = ["Data snapshot " + (snapshot.generated_utc || "time unknown").replace("T", " ").replace("Z", " UTC")];
    if (snapshot.git_commit) {
      var commit = "from commit " + snapshot.git_commit.slice(0, 7);
      if (snapshot.git_dirty) {
        commit += " (the working tree had uncommitted changes)";
      }
      parts.push(commit);
    }
    parts.push(plural(state.index.datasets.length, "dataset"));
    byId("snapshot").textContent = parts.join(", ") + ".";
  }

  function renderFooter() {
    var index = state.index;
    var footer = byId("footer");
    clear(footer);
    var size = isNumber(index.total_bytes) ? (index.total_bytes / 1e6).toFixed(2) + " MB" : "unknown size";
    appendAll(footer, [
      el("p", { text: "This page is a static snapshot written by scripts/export_dashboard.py from the run logs. " +
        "Refresh it with: bash scripts/deploy_dashboard.sh" }),
      el("p", { text: "Snapshot data: " + plural(Object.keys(index.sizes || {}).length + 1, "file") + ", " + size + ". " +
        (index.trimmed && index.trimmed.length ? "Trimmed to fit the size cap:" : "Nothing was trimmed to fit the size cap.") })
    ]);
    if (index.trimmed && index.trimmed.length) {
      footer.appendChild(el("ul", null, index.trimmed.map(function (item) { return el("li", { text: item }); })));
    }
    if (index.notes && index.notes.length) {
      footer.appendChild(el("p", { text: "Notes from the export:" }));
      footer.appendChild(el("ul", null, index.notes.map(function (item) { return el("li", { text: item }); })));
    }
  }

  // Dataset selector.

  function setupDatasetSelect() {
    var select = byId("dataset-select");
    clear(select);
    state.index.datasets.forEach(function (entry) {
      var kind = entry.kind === "compare" ? "comparison" : "single run";
      var text = entry.title + " (" + kind + ", " + plural(entry.n_runs, "run") + (entry.stub ? ", STUB" : "") + ")";
      select.appendChild(el("option", { value: entry.id, text: text }));
    });
    select.addEventListener("change", function () {
      selectDataset(select.value);
    });
  }

  function renderDatasetSummary() {
    var raw = state.ds.raw;
    var budgets = raw.budgets || {};
    var seeds = raw.seeds.length ? "seeds " + raw.seeds.join(", ") : "no seed list";
    var text = plural(raw.conditions.length, "arm") + " (" + raw.conditions.join(", ") + "), " + seeds + ". ";
    if (isNumber(budgets.eval_budget)) {
      text += "Each run had " + fmtFee(budgets.eval_budget) + " FEE and " + fmtInt(budgets.compute_budget_s) +
        " GPU-seconds for search";
      if (isNumber(budgets.heldout_eval_budget)) {
        text += ", then " + fmtFee(budgets.heldout_eval_budget) + " FEE and " + fmtInt(budgets.heldout_compute_budget_s) +
          " GPU-seconds for held-out";
      }
      text += ". ";
    }
    text += raw.cascade ? "Cascade judge on: cheap low-fidelity evals first, full evals for the top third. " :
      "Cascade judge off: every eval is full fidelity. ";
    text += "Source: " + raw.source + ".";
    var summary = byId("dataset-summary");
    clear(summary);
    appendAll(summary, [text, " ", labelChips(raw.labels)]);
  }

  // Race.

  function renderArmToggles() {
    var wrap = byId("arm-toggles");
    clear(wrap);
    state.ds.raw.conditions.forEach(function (condition) {
      var button = el("button", {
        class: "chip",
        type: "button",
        "aria-pressed": state.armsOn[condition] ? "true" : "false",
        title: ARM_INFO[condition] || condition
      }, [swatch(condition), condition]);
      button.addEventListener("click", function () {
        state.armsOn[condition] = !state.armsOn[condition];
        button.setAttribute("aria-pressed", state.armsOn[condition] ? "true" : "false");
        renderRaceCharts();
        renderFitChart();
      });
      wrap.appendChild(button);
    });
  }

  function visibleArms() {
    return state.ds.raw.arms.filter(function (arm) { return state.armsOn[arm.condition]; });
  }

  function niceStep(span, count) {
    if (!(span > 0)) {
      return 1;
    }
    var raw = span / Math.max(1, count);
    var power = Math.pow(10, Math.floor(Math.log10(raw)));
    var scaled = raw / power;
    var nice = scaled < 1.5 ? 1 : scaled < 3 ? 2 : scaled < 7 ? 5 : 10;
    return nice * power;
  }

  function tickValues(min, max, step) {
    var out = [];
    var start = Math.ceil(min / step - 1e-9) * step;
    for (var value = start; value <= max + step * 1e-6; value += step) {
      out.push(Number(value.toFixed(10)));
    }
    return out;
  }

  function decimalsFor(step) {
    return Math.max(0, Math.min(4, -Math.floor(Math.log10(step) + 1e-9)));
  }

  function valueAt(xs, ys, x) {
    var value = null;
    for (var i = 0; i < xs.length; i++) {
      if (xs[i] <= x) {
        value = ys[i];
      } else {
        break;
      }
    }
    return value;
  }

  function stepPath(points, scaleX, scaleY) {
    // points: [[x, y], ...] in x order; each value holds until the next x ("post" steps).
    var d = "";
    for (var i = 0; i < points.length; i++) {
      var px = scaleX(points[i][0]);
      var py = scaleY(points[i][1]);
      if (i === 0) {
        d += "M" + px.toFixed(1) + " " + py.toFixed(1);
      } else {
        d += "H" + px.toFixed(1) + "V" + py.toFixed(1);
      }
    }
    return d;
  }

  function bandPolygons(band, scaleX, scaleY) {
    var shapes = [];
    var start = null;
    for (var i = 0; i <= band.x.length; i++) {
      var defined = i < band.x.length && band.low[i] !== null && band.high[i] !== null;
      if (defined && start === null) {
        start = i;
      }
      if (!defined && start !== null) {
        shapes.push(bandShape(band, start, i - 1, scaleX, scaleY));
        start = null;
      }
    }
    return shapes;
  }

  function bandShape(band, first, last, scaleX, scaleY) {
    var upper = [];
    var lower = [];
    for (var i = first; i < last; i++) {
      upper.push([band.x[i], band.high[i]], [band.x[i + 1], band.high[i]]);
    }
    upper.push([band.x[last], band.high[last]]);
    lower.push([band.x[last], band.low[last]]);
    for (var j = last - 1; j >= first; j--) {
      lower.push([band.x[j + 1], band.low[j]], [band.x[j], band.low[j]]);
    }
    var points = upper.concat(lower).map(function (point) {
      return scaleX(point[0]).toFixed(1) + "," + scaleY(point[1]).toFixed(1);
    });
    return points.join(" ");
  }

  function medianPoints(band) {
    var points = [];
    for (var i = 0; i < band.x.length; i++) {
      if (band.median[i] !== null) {
        points.push([band.x[i], band.median[i]]);
      }
    }
    return points;
  }

  function seedPoints(curve) {
    var points = curve.steps.slice();
    var last = points[points.length - 1];
    if (curve.end > last[0]) {
      points.push([curve.end, last[1]]);
    }
    return points;
  }

  function frame(container, height, margin) {
    var width = Math.max(280, container.clientWidth || 600);
    var root = svgEl("svg", { viewBox: "0 0 " + width + " " + height, width: width, height: height, role: "img" });
    return {
      svg: root,
      width: width,
      height: height,
      margin: margin,
      innerW: width - margin.left - margin.right,
      innerH: height - margin.top - margin.bottom
    };
  }

  function drawAxes(f, xs, ys, options) {
    var grid = svgEl("g", { class: "grid" }, f.svg);
    ys.ticks.forEach(function (value) {
      var y = ys.scale(value);
      svgEl("line", { x1: f.margin.left, x2: f.width - f.margin.right, y1: y, y2: y }, grid);
      svgEl("text", { x: f.margin.left - 6, y: y + 4, "text-anchor": "end", text: value.toFixed(ys.decimals) }, f.svg);
    });
    xs.ticks.forEach(function (value) {
      var x = xs.scale(value);
      svgEl("text", { x: x, y: f.height - f.margin.bottom + 16, "text-anchor": "middle", text: options.formatX(value) }, f.svg);
    });
    svgEl("line", {
      class: "baseline", x1: f.margin.left, x2: f.width - f.margin.right,
      y1: f.height - f.margin.bottom, y2: f.height - f.margin.bottom
    }, f.svg);
    svgEl("text", {
      class: "axis-title", x: f.margin.left + f.innerW / 2, y: f.height - 6, "text-anchor": "middle", text: options.xTitle
    }, f.svg);
    var yTitle = svgEl("text", {
      class: "axis-title", x: 0, y: 0, "text-anchor": "middle", text: options.yTitle,
      transform: "translate(12," + (f.margin.top + f.innerH / 2) + ") rotate(-90)"
    }, f.svg);
    return yTitle;
  }

  function linearScale(min, max, from, to) {
    var span = max - min || 1;
    return function (value) {
      return from + (value - min) / span * (to - from);
    };
  }

  function yAxis(min, max, f, count) {
    if (min === max) {
      min -= 0.05;
      max += 0.05;
    }
    var pad = (max - min) * 0.06;
    var step = niceStep(max - min + 2 * pad, count);
    var low = Math.floor((min - pad) / step) * step;
    var high = Math.ceil((max + pad) / step) * step;
    return {
      min: low,
      max: high,
      ticks: tickValues(low, high, step),
      decimals: decimalsFor(step),
      scale: linearScale(low, high, f.height - f.margin.bottom, f.margin.top)
    };
  }

  function xAxis(min, max, f, count) {
    var step = niceStep(max - min, count);
    var high = Math.max(max, min + step);
    return {
      min: min,
      max: high,
      ticks: tickValues(min, high, step),
      scale: linearScale(min, high, f.margin.left, f.width - f.margin.right)
    };
  }

  function addWatermark(f) {
    if (!state.ds.raw.stub) {
      return;
    }
    svgEl("text", {
      class: "watermark",
      x: f.margin.left + f.innerW / 2,
      y: f.margin.top + f.innerH / 2 + 20,
      "text-anchor": "middle",
      transform: "rotate(-16 " + (f.margin.left + f.innerW / 2) + " " + (f.margin.top + f.innerH / 2) + ")",
      text: "STUB",
      "aria-hidden": "true"
    }, f.svg);
  }

  function spreadLabels(items, gap) {
    var order = items.slice().sort(function (a, b) { return a.y - b.y; });
    var previous = null;
    order.forEach(function (item) {
      if (previous !== null && item.y < previous + gap) {
        item.y = previous + gap;
      }
      previous = item.y;
    });
  }

  function renderRaceCharts() {
    drawRaceChart(byId("chart-evals"), "evals");
    drawRaceChart(byId("chart-compute"), "compute");
  }

  function drawRaceChart(container, axis) {
    clear(container);
    var ds = state.ds;
    var stubTag = ds.raw.stub ? " [SEARCH] [STUB]" : " [SEARCH]";
    byId(axis === "evals" ? "chart-evals-title" : "chart-compute-title").textContent =
      (axis === "evals" ? "Best combined score against evals spent" : "Best combined score against GPU-seconds") + stubTag;
    var series = [];
    visibleArms().forEach(function (arm) {
      var seeds = [];
      arm.runs.forEach(function (runIndex) {
        var curve = ds.runs[runIndex].curves[axis];
        if (curve && curve.steps.length) {
          seeds.push(curve);
        }
      });
      if (seeds.length && arm.band[axis]) {
        series.push({ arm: arm, seeds: seeds, band: arm.band[axis] });
      }
    });
    if (series.length === 0) {
      container.appendChild(el("p", { class: "chart-empty", text: "No arm with scored full-fidelity search rows is switched on." }));
      return;
    }
    var narrow = (container.clientWidth || 600) < 480;
    var labelRoom = series.length <= 4 ? 84 : 14;
    var height = narrow ? 250 : 300;
    var f = frame(container, height, { top: 12, right: labelRoom, bottom: 44, left: 52 });
    var xMax = 0;
    var yValues = [];
    series.forEach(function (item) {
      item.seeds.forEach(function (curve) {
        xMax = Math.max(xMax, curve.end);
        curve.steps.forEach(function (point) { yValues.push(point[1]); });
      });
    });
    var xs = xAxis(0, xMax, f, narrow ? 4 : 6);
    var ys = yAxis(minOf(yValues), maxOf(yValues), f, 5);
    var formatX = axis === "evals" ? function (v) { return trimZeros(v.toFixed(1)); } : fmtCompact;
    drawAxes(f, xs, ys, {
      formatX: formatX,
      xTitle: axis === "evals" ? "Evals spent, FEE (search budget)" : "GPU-seconds (search budget)",
      yTitle: "Best combined score (search)"
    });
    addWatermark(f);
    if (ys.min < 0 && ys.max > 0) {
      var zeroY = ys.scale(0);
      svgEl("line", { class: "zero-line", x1: f.margin.left, x2: f.width - f.margin.right, y1: zeroY, y2: zeroY }, f.svg);
      svgEl("text", { class: "halo", x: f.width - f.margin.right - 4, y: zeroY + 14, "text-anchor": "end",
        text: "0 = host score" }, f.svg);
    }
    var plot = svgEl("g", null, f.svg);
    series.forEach(function (item) {
      bandPolygons(item.band, xs.scale, ys.scale).forEach(function (points) {
        svgEl("polygon", { points: points, style: "fill:" + armColor(item.arm.condition) + ";opacity:0.13" }, plot);
      });
    });
    series.forEach(function (item) {
      item.seeds.forEach(function (curve) {
        svgEl("path", {
          d: stepPath(seedPoints(curve), xs.scale, ys.scale),
          style: "fill:none;stroke:" + armColor(item.arm.condition) + ";stroke-width:1;opacity:0.5"
        }, plot);
      });
    });
    var ends = [];
    series.forEach(function (item) {
      var points = medianPoints(item.band);
      if (points.length === 0) {
        return;
      }
      svgEl("path", {
        d: stepPath(points, xs.scale, ys.scale),
        style: "fill:none;stroke:" + armColor(item.arm.condition) + ";stroke-width:2.5;stroke-linejoin:round"
      }, plot);
      var last = points[points.length - 1];
      ends.push({ name: item.arm.condition, x: xs.scale(last[0]), y: ys.scale(last[1]), value: last[1] });
    });
    if (series.length <= 4) {
      spreadLabels(ends, 14);
      ends.forEach(function (end) {
        svgEl("text", { x: end.x + 6, y: end.y + 4, text: end.name, style: "fill:var(--text)" }, f.svg);
      });
    }
    var summary = ends.map(function (end) { return end.name + " " + fmtShort(end.value); }).join(", ");
    f.svg.setAttribute("aria-label", "Step chart of the best combined score so far against " +
      (axis === "evals" ? "evals spent" : "GPU-seconds") + ". Median best at the end: " + summary + "." +
      (ds.raw.stub ? " STUB data." : ""));
    addHover(f, xs, ys, series, axis, formatX);
    container.appendChild(f.svg);
  }

  function addHover(f, xs, ys, series, axis, formatX) {
    var line = svgEl("line", {
      class: "crosshair", x1: 0, x2: 0, y1: f.margin.top, y2: f.height - f.margin.bottom, visibility: "hidden"
    }, f.svg);
    var overlay = svgEl("rect", {
      x: f.margin.left, y: f.margin.top, width: f.innerW, height: f.innerH, fill: "transparent", style: "cursor:crosshair"
    }, f.svg);
    function move(event) {
      var box = f.svg.getBoundingClientRect();
      var scale = f.width / box.width;
      var px = (event.clientX - box.left) * scale;
      var x = xs.min + (px - f.margin.left) / f.innerW * (xs.max - xs.min);
      if (x < xs.min || x > xs.max) {
        return;
      }
      line.setAttribute("x1", px);
      line.setAttribute("x2", px);
      line.setAttribute("visibility", "visible");
      var rows = [el("div", { class: "tip-title", text: "At " + formatX(x) + (axis === "evals" ? " FEE" : " GPU-seconds") })];
      series.forEach(function (item) {
        var mid = valueAt(item.band.x, item.band.median, x);
        var low = valueAt(item.band.x, item.band.low, x);
        var high = valueAt(item.band.x, item.band.high, x);
        var lastX = item.band.x[item.band.x.length - 1];
        var text = mid === null ? "no score yet in every seed" :
          fmtShort(mid) + " median (" + fmtShort(low) + " to " + fmtShort(high) + ")";
        if (mid !== null && x > lastX) {
          text += ", stopped at " + formatX(lastX);
        }
        rows.push(el("div", { class: "tip-row" }, [swatch(item.arm.condition), item.arm.condition + ": " + text]));
      });
      rows.push(el("div", { class: "meta", text: state.ds.raw.stub ? "SEARCH, STUB" : "SEARCH" }));
      showTooltip(event, rows);
    }
    overlay.addEventListener("pointermove", move);
    overlay.addEventListener("pointerdown", move);
    overlay.addEventListener("pointerleave", function () {
      line.setAttribute("visibility", "hidden");
      hideTooltip();
    });
  }

  function renderTimesFewer() {
    var box = byId("times-fewer");
    clear(box);
    var fewer = state.ds.raw.times_fewer || {};
    var names = Object.keys(fewer);
    var tag = state.ds.raw.stub ? ["SEARCH", "STUB"] : ["SEARCH"];
    if (names.length === 0) {
      box.textContent = "This dataset has no swarm and evolve pair, so there is no N-times-fewer number.";
      return;
    }
    names.forEach(function (name) {
      var result = fewer[name];
      var parts = [];
      var compute = result.compute;
      var evals = result.evals;
      var head = name + " against evolve's median final best (" + fmtShort(result.target, 4) + "): ";
      if (isNumber(compute.ratio)) {
        parts.push(name + " needed " + fmtInt(compute.fast_median) + " GPU-seconds to reach it and evolve needed " +
          fmtInt(compute.slow_median) + ", " + compute.ratio.toFixed(2) + " times fewer.");
      } else {
        parts.push("the median " + name + " seed did not reach it, so there is no GPU-seconds ratio.");
      }
      if (isNumber(evals.ratio)) {
        parts.push(" On evals, " + fmtFee(evals.fast_median) + " FEE against " + fmtFee(evals.slow_median) + ", " +
          evals.ratio.toFixed(2) + " times fewer.");
      }
      appendAll(box, [el("span", { text: head + parts.join("") }), " ", labelChips(tag), el("br")]);
    });
  }

  function renderRaceTable() {
    var ds = state.ds;
    var wrap = byId("race-table");
    clear(wrap);
    var table = el("table", { class: "data stack" });
    table.appendChild(el("thead", null, el("tr", null, ["Arm", "Final best per seed", "Median final best",
      "Evals used (median)", "GPU-seconds used (median)", "Labels"].map(function (name, index) {
      return el("th", { class: index >= 2 && index <= 4 ? "num" : null, text: name });
    }))));
    var body = el("tbody");
    ds.raw.arms.forEach(function (arm) {
      var runs = arm.runs.map(function (i) { return ds.runs[i]; });
      var finals = runs.map(function (run) { return run.final_best; });
      var perSeed = runs.map(function (run) { return "s" + run.seed + " " + fmtShort(run.final_best, 4); }).join(", ");
      body.appendChild(el("tr", null, [
        el("td", { "data-label": "Arm" }, el("span", { class: "arm-tag" }, [swatch(arm.condition), arm.condition])),
        el("td", { "data-label": "Final best per seed", class: "mono", text: perSeed }),
        el("td", { "data-label": "Median final best", class: "num", text: fmtShort(median(finals), 4) }),
        el("td", { "data-label": "Evals used (median)", class: "num",
          text: fmtFee(median(runs.map(function (run) { return run.budget.eval_used; }))) }),
        el("td", { "data-label": "GPU-seconds used (median)", class: "num",
          text: fmtInt(median(runs.map(function (run) { return run.budget.compute_used_s; }))) }),
        el("td", { "data-label": "Labels" }, labelChips(ds.raw.stub ? ["SEARCH", "STUB"] : ["SEARCH"]))
      ]));
    });
    table.appendChild(body);
    wrap.appendChild(table);
  }

  // Agent log.

  function setupLogFilters() {
    ["filter-arm", "filter-seed", "filter-status", "filter-agent", "filter-events"].forEach(function (id) {
      byId(id).addEventListener("change", function () {
        state.logLimit = PAGE_SIZE;
        renderLog();
      });
    });
    var timer = null;
    byId("filter-text").addEventListener("input", function () {
      clearTimeout(timer);
      timer = setTimeout(function () {
        state.logLimit = PAGE_SIZE;
        renderLog();
      }, 150);
    });
    byId("log-more").addEventListener("click", function () {
      state.logLimit += PAGE_SIZE;
      renderLog();
    });
  }

  function resetLogFilters() {
    byId("filter-text").value = "";
    byId("filter-events").checked = false;
  }

  function fillSelect(select, allText, values, labelFor) {
    var previous = select.value;
    clear(select);
    select.appendChild(el("option", { value: "", text: allText }));
    values.forEach(function (value) {
      select.appendChild(el("option", { value: String(value), text: labelFor ? labelFor(value) : String(value) }));
    });
    select.value = values.map(String).indexOf(previous) >= 0 ? previous : "";
  }

  function populateLogFilters() {
    var ds = state.ds;
    var statuses = {};
    var agents = {};
    ds.rows.forEach(function (row) {
      statuses[row.status] = true;
      if (row.agent) {
        agents[row.agent] = true;
      }
    });
    ds.events.forEach(function (event) {
      if (event.agent) {
        agents[event.agent] = true;
      }
    });
    fillSelect(byId("filter-arm"), "All arms", ds.raw.conditions);
    fillSelect(byId("filter-seed"), "All seeds", ds.raw.seeds, function (seed) { return "seed " + seed; });
    fillSelect(byId("filter-status"), "All statuses", Object.keys(statuses).sort(), function (status) {
      return STATUS_INFO[status] || status;
    });
    fillSelect(byId("filter-agent"), "All agents", Object.keys(agents).sort());
    byId("filter-arm").value = "";
    byId("filter-seed").value = "";
    byId("filter-status").value = "";
    byId("filter-agent").value = "";
  }

  function readFilters() {
    return {
      arm: byId("filter-arm").value,
      seed: byId("filter-seed").value,
      status: byId("filter-status").value,
      agent: byId("filter-agent").value,
      text: byId("filter-text").value.trim().toLowerCase(),
      events: byId("filter-events").checked
    };
  }

  function matchesCommon(item, filters) {
    if (filters.arm && item.condition !== filters.arm) {
      return false;
    }
    if (filters.seed !== "" && String(item.seed) !== filters.seed) {
      return false;
    }
    if (filters.agent && item.agent !== filters.agent) {
      return false;
    }
    if (filters.text && item.searchText.indexOf(filters.text) < 0) {
      return false;
    }
    return true;
  }

  function renderLog() {
    var ds = state.ds;
    var filters = readFilters();
    var items = ds.rows.filter(function (row) {
      return matchesCommon(row, filters) && (!filters.status || row.status === filters.status);
    }).map(function (row) { return { kind: "row", t: row.t || "", order: row.i, item: row }; });
    var eventCount = 0;
    if (filters.events && !filters.status) {
      ds.events.forEach(function (event, index) {
        if (matchesCommon(event, filters)) {
          items.push({ kind: "event", t: event.t || "", order: -1 - index, item: event });
          eventCount += 1;
        }
      });
    }
    items.sort(function (a, b) {
      if (a.t !== b.t) {
        return a.t < b.t ? 1 : -1;
      }
      return b.order - a.order;
    });
    var list = byId("log-list");
    clear(list);
    var shown = items.slice(0, state.logLimit);
    shown.forEach(function (entry) {
      list.appendChild(entry.kind === "row" ? logRowElement(entry.item) : logEventElement(entry.item));
    });
    if (items.length === 0) {
      list.appendChild(el("p", { class: "chart-empty", text: "No rows match these filters." }));
    }
    var rowCount = items.length - eventCount;
    byId("log-count").textContent = "Showing " + fmtInt(shown.length) + " of " + plural(rowCount, "eval row") +
      (eventCount ? " and " + plural(eventCount, "note") : "") + " that match, newest first. The dataset has " +
      plural(ds.rows.length, "eval row") + ".";
    byId("log-more").hidden = shown.length >= items.length;
  }

  function fitChips(row) {
    var program = programFor(state.ds, row);
    var wrap = el("span", { class: "log-fits" });
    if (!program) {
      return wrap;
    }
    program.grafts.forEach(function (graft, index) {
      var fit = row.fit_r2 && row.fit_r2[index];
      var layer = graft.host_layer !== null && graft.host_layer !== undefined ? graft.host_layer : graft.at;
      wrap.appendChild(el("span", {
        class: "fit-chip",
        text: (graft.control ? "rand(" + shortPart(graft.id) + ")" : shortPart(graft.id)) + "@" + layer + " fit " +
          (isNumber(fit) ? fit.toFixed(2) : "skipped")
      }));
    });
    return wrap;
  }

  function logRowElement(row) {
    var run = state.ds.runs[row.run];
    var where = row.agent ? row.agent + (row.round !== null && row.round !== undefined ? ", round " + row.round : "") : "";
    var scoreText = isNumber(row.combined) ? fmtScore(row.combined) : "no score";
    var fidelity = row.fidelity && row.fidelity !== "full" ? row.fidelity + " fidelity" : null;
    var button = el("button", { class: "log-row", type: "button", "data-i": row.i,
      "aria-current": state.selected && state.selected.i === row.i ? "true" : "false" }, [
      el("span", { class: "log-line" }, [
        statusBadge(row.status),
        el("span", { class: "score", text: scoreText }),
        labelChips(row.labels),
        fidelity ? el("span", { text: fidelity }) : null,
        el("span", { text: "eval " + fmtFee(row.eval_count) + " FEE" }),
        el("span", { class: "arm-tag" }, [swatch(row.condition), row.condition + " s" + run.seed]),
        where ? el("span", { text: where }) : null
      ]),
      el("span", { class: "log-program", style: "display:block", text: row.describe || "(no program)" }),
      row.hypothesis ? el("span", { class: "log-why", style: "display:block", text: row.hypothesis }) : null,
      fitChips(row)
    ]);
    button.addEventListener("click", function () {
      selectRow(row, true);
    });
    return el("div", { role: "listitem" }, button);
  }

  function logEventElement(event) {
    var kind = String(event.kind).replace(/_/g, " ");
    return el("div", { class: "log-event", role: "listitem" }, [
      el("strong", { text: event.agent ? event.agent + " (" + kind + ")" : kind }),
      " " + event.condition + " s" + event.seed + ", eval " + fmtFee(event.eval_count) + " FEE: " + event.text + " ",
      labelChips(event.labels)
    ]);
  }

  function selectRow(row, scroll) {
    if (!row) {
      return;
    }
    state.selected = row;
    var buttons = byId("log-list").querySelectorAll(".log-row");
    for (var i = 0; i < buttons.length; i++) {
      buttons[i].setAttribute("aria-current", Number(buttons[i].getAttribute("data-i")) === row.i ? "true" : "false");
    }
    renderInspector(row);
    if (scroll && window.innerWidth <= 900) {
      byId("inspector").scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  // Program inspector.

  function renderInspector(row) {
    var body = byId("inspector-body");
    clear(body);
    var ds = state.ds;
    var run = ds.runs[row.run];
    var info = programFor(ds, row);
    var where = [row.condition + " seed " + run.seed, row.agent, row.round !== null && row.round !== undefined ?
      "round " + row.round : null].filter(Boolean).join(", ");
    appendAll(body, [
      el("p", { class: "meta", text: "Eval " + fmtFee(row.eval_count) + " FEE in " + run.run_id + ". " + where + "." }),
      el("p", { class: "log-line" }, [statusBadge(row.status), el("span", { class: "score", text: isNumber(row.combined) ?
        "combined " + fmtScore(row.combined) : "no score" }), labelChips(row.labels),
        row.fidelity && row.fidelity !== "full" ? el("span", { text: row.fidelity + " fidelity" }) : null]),
      el("p", { class: "describe", text: info ? info.describe : "(program not in this snapshot)" })
    ]);
    if (row.hypothesis) {
      appendAll(body, [el("h4", { text: "Hypothesis" }), el("p", { text: row.hypothesis })]);
    }
    if (row.citations && row.citations.length) {
      body.appendChild(el("h4", { text: "Citations" }));
      body.appendChild(el("ul", null, row.citations.map(function (citation) {
        return el("li", { text: (citation.title || "untitled") + (citation.url ? " (" + citation.url + ")" : "") });
      })));
    }
    if (!info) {
      return;
    }
    body.appendChild(el("h4", { text: "Where the grafts sit in the host" }));
    var strip = el("div", { class: "host-strip" });
    body.appendChild(strip);
    drawHostStrip(strip, info.grafts, ds.raw.host_layers || 16, ds.raw.host);
    if (info.training) {
      body.appendChild(el("p", { text: "No grafts: this arm trains the host instead (" +
        (info.training.describe || info.training.method) + " on " + (info.training.data || "the heal data") + ", " +
        fmtParams(info.training.trainable_params) + " trainable parameters" +
        (row.step ? ", checkpoint at step " + fmtInt(row.step) + " of " + fmtInt(info.training.planned_steps) : "") + ")." }));
    } else if (info.grafts.length === 0) {
      body.appendChild(el("p", { text: "No grafts: this program is the host alone (a no-graft control keeps only the heal)." }));
    }
    if (info.grafts.length) {
      var cards = el("div", { class: "graft-cards" });
      info.grafts.forEach(function (graft, index) {
        cards.appendChild(graftCard(graft, index, row));
      });
      body.appendChild(cards);
    }
    renderHeal(body, info, row);
    renderScores(body, row);
    renderCompute(body, row, run);
    renderLineage(body, row);
    body.appendChild(el("h4", { text: "Program JSON" }));
    body.appendChild(el("pre", { class: "json", text: JSON.stringify(info.program, null, 2) }));
  }

  function drawHostStrip(container, grafts, layers, host) {
    var width = Math.max(260, container.clientWidth || 360);
    var laneH = 28;
    var top = 6;
    var stripTop = top + grafts.length * laneH + (grafts.length ? 6 : 0);
    var cellH = 28;
    var height = stripTop + cellH + 34;
    var left = 4;
    var cellW = (width - left * 2) / layers;
    var root = svgEl("svg", { viewBox: "0 0 " + width + " " + height, width: width, height: height, role: "img" });
    container.appendChild(root);
    var hit = {};
    grafts.forEach(function (graft) {
      var layer = graft.host_layer !== null && graft.host_layer !== undefined ? graft.host_layer : graft.at;
      if (isNumber(layer)) {
        hit[layer] = true;
      }
    });
    for (var i = 0; i < layers; i++) {
      var x = left + i * cellW;
      svgEl("rect", { class: "layer" + (hit[i] ? " hit" : ""), x: x + 1, y: stripTop, width: Math.max(2, cellW - 2),
        height: cellH, rx: 4 }, root);
      svgEl("text", { x: x + cellW / 2, y: stripTop + cellH / 2 + 4, "text-anchor": "middle", text: String(i) }, root);
    }
    svgEl("text", { x: left, y: stripTop + cellH + 16, text: "layer 0, near the input" }, root);
    svgEl("text", { x: width - left, y: stripTop + cellH + 16, "text-anchor": "end", text: "near the output" }, root);
    svgEl("text", { x: width / 2, y: stripTop + cellH + 30, "text-anchor": "middle",
      text: (host || "host") + ", " + layers + " layers" }, root);
    var labels = [];
    grafts.forEach(function (graft, index) {
      var layer = graft.host_layer !== null && graft.host_layer !== undefined ? graft.host_layer : graft.at;
      if (!isNumber(layer)) {
        return;
      }
      var cx = left + (layer + 0.5) * cellW;
      var cy = top + index * laneH + laneH / 2;
      svgEl("line", { class: "connector", x1: cx, x2: cx, y1: cy + 9, y2: stripTop }, root);
      svgEl("circle", { class: "pill" + (graft.control ? " control" : ""), cx: cx, cy: cy, r: 9 }, root);
      svgEl("text", { class: "pill-text" + (graft.control ? " control" : ""), x: cx, y: cy + 4, "text-anchor": "middle",
        text: String(index + 1) }, root);
      var roomRight = width - cx - 16;
      var roomLeft = cx - 16;
      var right = roomRight >= roomLeft;
      var label = shortPart(graft.id) + (graft.kind ? " (" + graft.kind.replace(/_/g, " ") + ")" : "") +
        (graft.control ? ", random part" : "");
      var text = svgEl("text", { class: "graft-label", x: right ? cx + 14 : cx - 14, y: cy + 4,
        "text-anchor": right ? "start" : "end", text: label }, root);
      fitLabel(text, shortPart(graft.id) + (graft.control ? " (random)" : ""), right ? roomRight : roomLeft);
      labels.push("graft " + (index + 1) + " " + label + " at host layer " + layer);
    });
    root.setAttribute("aria-label", "The host's " + layers + " layers as a strip. " +
      (labels.length ? labels.join("; ") + "." : "No grafts."));
  }

  function fitLabel(node, short, room) {
    // Shorten a graft label that would run off the strip; the card below always has the whole name.
    if (node.getComputedTextLength() <= room) {
      return;
    }
    node.textContent = short;
    var text = short;
    while (node.getComputedTextLength() > room && text.length > 4) {
      text = text.slice(0, -1);
      node.textContent = text + "...";
    }
  }

  function graftCard(graft, index, row) {
    var fit = row.fit_r2 && row.fit_r2[index];
    var layer = graft.host_layer !== null && graft.host_layer !== undefined ? graft.host_layer : graft.at;
    var placement = String(layer);
    if (graft.at === null || graft.at === undefined) {
      placement += " (default placement)";
    }
    var donor = graft.donor ? graft.donor + (isNumber(graft.donor_layer) ? ", layer " + graft.donor_layer : "") : "not in the report";
    var items = [
      ["Part id", el("span", { class: "mono", text: graft.id })],
      ["Donor", donor],
      ["Kind", graft.kind ? graft.kind.replace(/_/g, " ") : "unknown"],
      ["Host layer (at)", placement],
      ["Glue rank", isNumber(graft.glue_rank) ? String(graft.glue_rank) : "full rank"],
      ["Control", graft.control ? (graft.control === "random" ? "random part of the same shape" : graft.control) +
        (isNumber(graft.control_seed) ? ", seed " + graft.control_seed : "") : "no, the real part"],
      ["fit_r2", (isNumber(fit) ? fit.toFixed(4) : "skipped") + (graft.fit ? " (" + graft.fit + ")" : "")]
    ];
    if (isNumber(graft.part_params) || isNumber(graft.glue_params)) {
      items.push(["Parameters", "part " + fmtParams(graft.part_params) + ", glue " + fmtParams(graft.glue_params)]);
    }
    if (isNumber(graft.n_units)) {
      items.push(["Units", fmtInt(graft.n_units)]);
    }
    if (isNumber(graft.d_donor) && isNumber(graft.d_host)) {
      items.push(["Widths", "donor " + graft.d_donor + " to host " + graft.d_host]);
    }
    var list = el("dl", { class: "kv" });
    items.forEach(function (item) {
      list.appendChild(el("dt", { text: item[0] }));
      list.appendChild(el("dd", null, item[1]));
    });
    return el("div", { class: "graft-card" }, [
      el("h4", null, [el("span", { class: "graft-num", text: String(index + 1) }), shortPart(graft.id),
        graft.control ? el("span", { class: "control-tag", text: "CONTROL" }) : null]),
      list
    ]);
  }

  function renderHeal(body, info, row) {
    var heal = info.program.heal;
    body.appendChild(el("h4", { text: "Heal recipe" }));
    if (info.training) {
      body.appendChild(el("p", { text: "Not used: training arms have no graft glue to heal." }));
      return;
    }
    if (!heal) {
      body.appendChild(el("p", { text: "No heal pass." }));
    } else {
      var parts = [fmtInt(heal.steps) + " steps on " + heal.data];
      if (isNumber(heal.lr)) {
        parts.push("learning rate " + heal.lr);
      }
      if (isNumber(heal.batch_size)) {
        parts.push("batch " + heal.batch_size);
      }
      if (isNumber(heal.seq_len)) {
        parts.push("sequence length " + heal.seq_len);
      }
      body.appendChild(el("p", { text: parts.join(", ") + (heal.steps === 0 ? " (an unhealed graft)" : "") + "." }));
    }
    var curve = row.heal !== null && row.heal !== undefined ? state.ds.raw.heal_curves[row.heal] : null;
    if (curve && curve.length > 1) {
      var spark = svgEl("svg", { class: "spark", viewBox: "0 0 280 44", role: "img",
        "aria-label": "Heal loss from " + curve[0] + " to " + curve[curve.length - 1] });
      var low = minOf(curve);
      var high = maxOf(curve);
      var d = "";
      curve.forEach(function (value, index) {
        var x = 4 + index / (curve.length - 1) * 272;
        var y = 40 - (high === low ? 0.5 : (value - low) / (high - low)) * 36;
        d += (index === 0 ? "M" : "L") + x.toFixed(1) + " " + y.toFixed(1);
      });
      svgEl("path", { d: d }, spark);
      body.appendChild(spark);
      body.appendChild(el("p", { class: "meta" }, ["Heal loss " + curve[0] + " to " + curve[curve.length - 1] +
        " (" + curve.length + " points shown) ", labelChips(row.stub ? ["STUB"] : [])]));
    }
  }

  function renderScores(body, row) {
    body.appendChild(el("h4", { text: "Scores" }));
    var scores = row.scores || {};
    var names = Object.keys(scores);
    if (names.length === 0) {
      body.appendChild(el("p", { text: "No eval ran for this row (" + (STATUS_INFO[row.status] || row.status) + ")." }));
      return;
    }
    var list = el("dl", { class: "kv" });
    names.forEach(function (name) {
      list.appendChild(el("dt", { text: name }));
      list.appendChild(el("dd", { text: fmtScore(scores[name]) }));
    });
    list.appendChild(el("dt", { text: "combined" }));
    list.appendChild(el("dd", null, [fmtScore(row.combined) + " ", labelChips(row.labels)]));
    body.appendChild(list);
    body.appendChild(el("p", { class: "meta", text: "Combined is the share of the host's headroom on the target task " +
      "that the program gains, minus a penalty if a guard task drops more than 10 percent below the host." }));
  }

  function renderCompute(body, row, run) {
    body.appendChild(el("h4", { text: "Compute for this eval" }));
    var phases = row.compute_s || {};
    var names = Object.keys(phases);
    if (names.length === 0) {
      body.appendChild(el("p", { text: "No compute charged." }));
      return;
    }
    var high = maxOf(names.map(function (name) { return phases[name]; })) || 1;
    var total = 0;
    var bars = el("div", { class: "phase-bars" });
    names.forEach(function (name) {
      total += phases[name];
      bars.appendChild(el("div", { class: "phase-row" }, [
        el("span", { text: name }),
        el("span", null, el("span", { class: "bar", style: "display:block;width:" + Math.max(1, phases[name] / high * 100) + "%" })),
        el("span", { class: "num", text: fmtInt(phases[name]) + " s" })
      ]));
    });
    body.appendChild(bars);
    var budget = row.split === "heldout" ? run.budget.heldout.compute_budget_s : run.budget.compute_budget_s;
    body.appendChild(el("p", { class: "meta", text: fmtInt(total) + " GPU-seconds for this eval. The run had used " +
      fmtInt(row.compute_used_s) + " of its " + fmtInt(budget) + " " + (row.split === "heldout" ? "held-out" : "search") +
      " GPU-seconds by then." + (row.stub ? " STUB cost model." : "") }));
  }

  function findProgramRow(row, programHash) {
    var ds = state.ds;
    var found = null;
    ds.rows.forEach(function (candidate) {
      if (candidate.run === row.run && candidate.program_hash === programHash && candidate.i !== row.i) {
        if (found === null || (candidate.split === "search" && found.split !== "search")) {
          found = candidate;
        }
      }
    });
    return found;
  }

  function renderLineage(body, row) {
    var links = [];
    if (row.parent) {
      var parent = findProgramRow(row, row.parent);
      links.push(["Parent program", parent, row.parent]);
    }
    if (row.control_of) {
      var original = findProgramRow(row, row.control_of);
      links.push([row.control === "no_graft" ? "No-graft control of" : "Random-part control of", original, row.control_of]);
    }
    if (links.length === 0) {
      return;
    }
    body.appendChild(el("h4", { text: "Lineage" }));
    links.forEach(function (link) {
      var target = link[1];
      if (target) {
        var button = el("button", { class: "link-button", type: "button", text: target.describe });
        button.addEventListener("click", function () {
          selectRow(target, true);
        });
        body.appendChild(el("p", null, [link[0] + ": ", button]));
      } else {
        body.appendChild(el("p", { text: link[0] + ": program " + link[2] + " (not in this log)." }));
      }
    });
  }

  // Fit chart.

  function fitGroup(condition) {
    if (condition === "swarm" || condition === "evolve") {
      return condition;
    }
    return "other";
  }

  function renderFitChart() {
    var container = byId("fit-chart");
    clear(container);
    var ds = state.ds;
    var fit = ds.raw.fit;
    var tag = fit.labels || [];
    byId("fit-chart-title").textContent = "fit_r2 against combined score: Spearman rho " +
      (isNumber(fit.spearman) ? fit.spearman.toFixed(3) : "n/a") + " over " + plural(fit.n, "graft") +
      " [" + tag.join("] [") + "]";
    var points = fit.points.filter(function (point) {
      var arm = ds.raw.arms[point[2]];
      return arm && state.armsOn[arm.condition];
    });
    renderFitLegend();
    if (points.length === 0) {
      container.appendChild(el("p", { class: "chart-empty", text: "No scored grafts for the arms switched on." }));
      return;
    }
    var narrow = (container.clientWidth || 600) < 480;
    var f = frame(container, narrow ? 260 : 320, { top: 12, right: 14, bottom: 44, left: 52 });
    var fits = points.map(function (point) { return point[0]; });
    var scores = points.map(function (point) { return point[1]; });
    var xs = xAxis(Math.min(0, minOf(fits)), 1, f, narrow ? 4 : 5);
    var ys = yAxis(minOf(scores), maxOf(scores), f, 5);
    drawAxes(f, xs, ys, {
      formatX: function (v) { return v.toFixed(1); },
      xTitle: "fit_r2 before heal (per graft)",
      yTitle: "Combined score (search)"
    });
    addWatermark(f);
    var layers = { other: svgEl("g", null, f.svg), evolve: svgEl("g", null, f.svg), swarm: svgEl("g", null, f.svg) };
    var placed = [];
    points.forEach(function (point) {
      var arm = ds.raw.arms[point[2]];
      var group = fitGroup(arm.condition);
      var color = group === "other" ? "var(--other-arm)" : armColor(arm.condition);
      var cx = xs.scale(point[0]);
      var cy = ys.scale(point[1]);
      svgEl("circle", { cx: cx.toFixed(1), cy: cy.toFixed(1), r: 4,
        style: "fill:" + color + ";opacity:" + (group === "other" ? 0.45 : 0.75) + ";stroke:var(--surface);stroke-width:1" },
      layers[group]);
      placed.push({ x: cx, y: cy, point: point, arm: arm });
    });
    f.svg.setAttribute("aria-label", "Scatter of fit_r2 against combined score for " + plural(points.length, "graft") +
      ". Spearman rho " + (isNumber(fit.spearman) ? fit.spearman.toFixed(3) : "n/a") + "." + (ds.raw.stub ? " STUB data." : ""));
    var ring = svgEl("circle", { r: 8, visibility: "hidden", style: "fill:none;stroke:var(--text);stroke-width:2" }, f.svg);
    var overlay = svgEl("rect", { x: f.margin.left, y: f.margin.top, width: f.innerW, height: f.innerH, fill: "transparent" }, f.svg);
    var current = null;
    function nearest(event) {
      var box = f.svg.getBoundingClientRect();
      var scale = f.width / box.width;
      var px = (event.clientX - box.left) * scale;
      var py = (event.clientY - box.top) * scale;
      var best = null;
      var bestDistance = 14 * 14;
      placed.forEach(function (item) {
        var dx = item.x - px;
        var dy = item.y - py;
        var distance = dx * dx + dy * dy;
        if (distance < bestDistance) {
          bestDistance = distance;
          best = item;
        }
      });
      return best;
    }
    overlay.addEventListener("pointermove", function (event) {
      current = nearest(event);
      if (!current) {
        ring.setAttribute("visibility", "hidden");
        overlay.style.cursor = "default";
        hideTooltip();
        return;
      }
      overlay.style.cursor = "pointer";
      ring.setAttribute("cx", current.x);
      ring.setAttribute("cy", current.y);
      ring.setAttribute("visibility", "visible");
      var row = ds.rowsById[current.point[3]];
      showTooltip(event, [
        el("div", { class: "tip-title" }, [swatch(current.arm.condition), " " + current.arm.condition +
          (row ? " s" + row.seed + ", eval " + fmtFee(row.eval_count) : "")]),
        el("div", { text: "fit_r2 " + current.point[0].toFixed(3) + ", combined " + fmtScore(current.point[1]) }),
        row ? el("div", { class: "mono", text: row.describe }) : null,
        el("div", { class: "meta", text: (tag.join(", ")) + (row ? ". Click to inspect." : "") })
      ]);
    });
    overlay.addEventListener("pointerleave", function () {
      ring.setAttribute("visibility", "hidden");
      hideTooltip();
    });
    overlay.addEventListener("click", function (event) {
      var item = nearest(event);
      if (!item) {
        return;
      }
      var row = ds.rowsById[item.point[3]];
      if (row) {
        selectRow(row, false);
        byId("inspector").scrollIntoView({ behavior: "smooth", block: "start" });
      }
    });
    container.appendChild(f.svg);
    var perArm = ds.raw.arms.filter(function (arm) { return arm.fit && arm.fit.n > 0; }).map(function (arm) {
      return arm.condition + " " + (isNumber(arm.fit.spearman) ? arm.fit.spearman.toFixed(3) : "n/a") + " (" +
        plural(arm.fit.n, "graft") + ")";
    });
    byId("fit-arms").textContent = perArm.length ? "Spearman rho per arm: " + perArm.join(", ") + ". Rows the fit gate " +
      "rejected have no score, so they are not plotted." : "";
  }

  function renderFitLegend() {
    var legend = byId("fit-legend");
    clear(legend);
    var conditions = state.ds.raw.conditions;
    ["swarm", "evolve"].forEach(function (condition) {
      if (conditions.indexOf(condition) >= 0) {
        legend.appendChild(el("span", { class: "legend-item" }, [swatch(condition), condition]));
      }
    });
    var others = conditions.filter(function (condition) { return condition !== "swarm" && condition !== "evolve"; });
    if (others.length) {
      legend.appendChild(el("span", { class: "legend-item" }, [
        el("span", { class: "swatch", style: "background:var(--other-arm)", "aria-hidden": "true" }),
        "other arms (" + others.join(", ") + ")"
      ]));
    }
  }

  // Budget.

  function rangeText(values, format) {
    var mid = median(values);
    var low = minOf(values);
    var high = maxOf(values);
    if (mid === null) {
      return "none";
    }
    if (low === high) {
      return format(mid);
    }
    return format(mid) + " [" + format(low) + " to " + format(high) + "]";
  }

  function meter(value, cap) {
    var share = isNumber(value) && isNumber(cap) && cap > 0 ? Math.min(1, value / cap) : 0;
    return el("div", { class: "meter", role: "img", "aria-label": Math.round(share * 100) + " percent of the cap" },
      el("span", { style: "width:" + (share * 100).toFixed(1) + "%" }));
  }

  function renderBudget() {
    var ds = state.ds;
    var check = ds.raw.budget_check || {};
    var budgets = ds.raw.budgets || {};
    var line = byId("budget-equal");
    clear(line);
    if (check.equal_caps) {
      var text = "Equal caps: every run in this dataset had " + fmtFee(budgets.eval_budget) + " FEE and " +
        fmtInt(budgets.compute_budget_s) + " GPU-seconds for search";
      if (isNumber(budgets.heldout_eval_budget)) {
        text += ", and " + fmtFee(budgets.heldout_eval_budget) + " FEE and " + fmtInt(budgets.heldout_compute_budget_s) +
          " GPU-seconds for held-out";
      }
      text += ".";
      if (check.compare_check && check.compare_check.equal) {
        text += " compare.py's own budget check passed too.";
      }
      appendAll(line, [el("span", { class: "icon-ok", "aria-hidden": "true", text: "✓" }), text]);
    } else {
      appendAll(line, [el("span", { class: "icon-bad", "aria-hidden": "true", text: "!" }),
        "Unequal caps: the runs in this dataset did not all have the same budgets."]);
    }
    var wrap = byId("budget-table");
    clear(wrap);
    var table = el("table", { class: "data stack" });
    table.appendChild(el("thead", null, el("tr", null, ["Arm", "Evals used (FEE)", "GPU-seconds used",
      "Where the GPU-seconds went (median)", "Probes", "Overshoot"].map(function (name) {
      return el("th", { text: name });
    }))));
    var body = el("tbody");
    var hasTraining = false;
    ds.raw.arms.forEach(function (arm) {
      var runs = arm.runs.map(function (i) { return ds.runs[i]; });
      var evals = runs.map(function (run) { return run.budget.eval_used; });
      var compute = runs.map(function (run) { return run.budget.compute_used_s; });
      var phaseNames = {};
      runs.forEach(function (run) {
        Object.keys(run.budget.phases_s || {}).forEach(function (name) { phaseNames[name] = true; });
      });
      if (phaseNames.train) {
        hasTraining = true;
      }
      var phases = Object.keys(phaseNames).map(function (name) {
        return { name: name, value: median(runs.map(function (run) { return (run.budget.phases_s || {})[name] || 0; })) };
      }).sort(function (a, b) { return b.value - a.value; }).map(function (item) {
        return item.name + " " + fmtInt(item.value) + " s";
      }).join(", ");
      var evalCap = runs[0] ? runs[0].budget.eval_budget : null;
      var computeCap = runs[0] ? runs[0].budget.compute_budget_s : null;
      var overshoot = maxOf(runs.map(function (run) { return run.budget.overshoot_s; }));
      body.appendChild(el("tr", null, [
        el("td", { "data-label": "Arm" }, el("span", { class: "arm-tag", title: ARM_INFO[arm.condition] || "" },
          [swatch(arm.condition), arm.condition])),
        el("td", { "data-label": "Evals used (FEE)" }, [rangeText(evals, fmtFee) + " of " + fmtFee(evalCap), meter(median(evals), evalCap)]),
        el("td", { "data-label": "GPU-seconds used" }, [rangeText(compute, fmtInt) + " of " + fmtInt(computeCap),
          meter(median(compute), computeCap)]),
        el("td", { "data-label": "Where the GPU-seconds went (median)", text: phases || "none" }),
        el("td", { "data-label": "Probes", class: "num", text: rangeText(runs.map(function (run) { return run.budget.probes; }), fmtInt) }),
        el("td", { "data-label": "Overshoot", class: "num", text: isNumber(overshoot) && overshoot > 0 ?
          fmtInt(overshoot) + " s (logged)" : "none" })
      ]));
    });
    table.appendChild(body);
    wrap.appendChild(table);
    var notes = ["An arm stops when either axis runs out, so arms end at different points of the other axis. " +
      "Overshoot is how far in-flight work ran past the compute cap; it is logged and reported, never refunded."];
    if (hasTraining) {
      notes.push("LoRA and full fine-tuning spend their GPU-seconds on training and only a few evals on checkpoints, " +
        "so they are compared on the GPU-seconds axis.");
    }
    notes.forEach(function (note) {
      wrap.appendChild(el("p", { class: "meta", text: note }));
    });
    wrap.appendChild(el("p", { class: "meta" }, ["All budget numbers in this dataset carry ", labelChips(ds.raw.stub ?
      ["STUB"] : []), ds.raw.stub ? " because the stub cost model produced them." : ""]));
  }

  // Real runs.

  function renderSmoke() {
    var panel = byId("smoke-panel");
    clear(panel);
    var runs = (state.smoke && state.smoke.runs) || [];
    if (runs.length === 0) {
      panel.appendChild(el("p", { class: "meta", text: "No Modal smoke results in this snapshot." }));
      return;
    }
    panel.appendChild(smokeCard(runs[0], true));
    if (runs.length > 1) {
      var more = el("details", { class: "more" }, el("summary", { text: "Earlier smoke runs (" + (runs.length - 1) + ")" }));
      runs.slice(1).forEach(function (run) {
        more.appendChild(smokeCard(run, false));
      });
      panel.appendChild(more);
    }
  }

  function smokeCard(run, latest) {
    var card = el("div", { class: "subcard" });
    var when = run.started ? run.started.replace("T", " ").replace("Z", " UTC") : "time unknown";
    appendAll(card, [
      el("h3", null, [(latest ? "Latest Modal smoke" : "Modal smoke") + ", " + when + " ", labelChips(run.labels)]),
      el("p", { text: "Not a result. A plumbing check of the real compile, heal and eval on " + plural(run.eval_limit || 20, "GSM8K search item") +
        ", scored with the interim scorer. With 20 items the standard error of a score near 0.25 is about 0.1, so these " +
        "three scores cannot be told apart." })
    ]);
    var grid = el("div", { class: "smoke-grid" });
    (run.models || []).forEach(function (model) {
      grid.appendChild(smokeTile(model));
    });
    card.appendChild(grid);
    var graft = null;
    (run.models || []).forEach(function (model) {
      if (model.role === "graft" && model.graft) {
        graft = model;
      }
    });
    var facts = [];
    if (graft) {
      var g = graft.graft;
      facts.push(["Graft", el("span", { class: "mono", text: g.id })]);
      facts.push(["Donor", (g.donor || "unknown") + ", layer " + g.donor_layer + ", " + String(g.kind || "").replace(/_/g, " ") +
        (isNumber(g.n_units) ? " of " + fmtInt(g.n_units) + " units" : "")]);
      facts.push(["Placed at", "host layer " + g.host_layer + " (" + fmtParams(g.glue_params) + " glue parameters)"]);
      facts.push(["fit_r2", fmtShort(g.fit_r2, 3) + " before heal (" + g.fit + ", " + fmtInt(g.fit_pairs) + " pairs)"]);
      if (graft.heal) {
        facts.push(["Heal", graft.heal.steps + " steps on " + graft.heal.data + ", learning rate " + graft.heal.lr +
          ", batch " + graft.heal.batch_size]);
      }
    }
    facts.push(["GPU", String(run.gpu || "unknown") + (run.ledger && run.ledger.devices ? " (ledger: " + run.ledger.devices.join(", ") + ")" : "")]);
    if (run.ledger && isNumber(run.ledger.compute_used_s)) {
      facts.push(["Charged", fmtShort(run.ledger.compute_used_s, 1) + " GPU-seconds over " + run.ledger.spends + " token spends"]);
    }
    if (isNumber(run.cost_estimate_usd)) {
      facts.push(["Cost estimate", "$" + run.cost_estimate_usd.toFixed(3)]);
    }
    facts.push(["Source", run.source]);
    var list = el("dl", { class: "kv" });
    facts.forEach(function (fact) {
      list.appendChild(el("dt", { text: fact[0] }));
      list.appendChild(el("dd", null, fact[1]));
    });
    card.appendChild(list);
    var checkNames = Object.keys(run.checks || {});
    var passed = checkNames.filter(function (name) { return run.checks[name]; });
    var checks = el("details", { class: "more" }, el("summary", {
      text: passed.length + " of " + checkNames.length + " plumbing checks passed"
    }));
    checks.appendChild(el("ul", { class: "check-list" }, checkNames.map(function (name) {
      return el("li", { text: (run.checks[name] ? "pass: " : "FAIL: ") + name.replace(/_/g, " ") });
    })));
    card.appendChild(checks);
    return card;
  }

  function smokeTile(model) {
    var score = model.scores ? model.scores.gsm8k : null;
    var items = model.n_items || 20;
    var right = isNumber(score) ? Math.round(score * items) : null;
    var seconds = model.seconds || {};
    var phaseText = Object.keys(seconds).filter(function (name) { return seconds[name] > 0; }).map(function (name) {
      return name + " " + fmtShort(seconds[name], 1) + " s";
    }).join(", ");
    if (model.cache_hit) {
      phaseText = "cached score" + (phaseText ? ", " + phaseText : "");
    }
    var tile = el("div", { class: "tile" }, [
      el("div", { class: "tile-title", text: model.title }),
      el("div", { class: "tile-value", text: isNumber(score) ? score.toFixed(2) : "none" }),
      el("div", { class: "tile-sub", text: "GSM8K " + (right !== null ? right + " of " + items + " right" : "not scored") }),
      el("div", { class: "tile-sub" }, labelChips(model.labels)),
      el("div", { class: "tile-sub", text: (model.gpu || "") + (phaseText ? ", " + phaseText : "") })
    ]);
    if (model.sample) {
      tile.appendChild(el("details", null, [el("summary", { class: "meta", text: "First answer (cut to 280 characters)" }),
        el("div", { class: "sample", text: model.sample })]));
    }
    return tile;
  }

  function renderFinal() {
    var panel = byId("final-panel");
    clear(panel);
    var final = state.final || { available: false, runs: [] };
    if (!final.available || !final.runs.length) {
      panel.appendChild(el("div", { class: "coming-next" }, [
        el("h3", { text: "Compiled model: coming next" }),
        el("p", { text: "When a run finishes, finalize recompiles the judge's best program, scores it once on the " +
          "held-out split next to its random-part and no-graft controls, and writes outputs/final/<run_id>/final.json. " +
          "This card then shows the model: its grafts, where they sit, its held-out score against the host and both " +
          "controls, and what it cost." })
      ]));
      return;
    }
    final.runs.forEach(function (run) {
      finalCard(run, panel);
    });
  }

  function finalCard(run, panel) {
    var data = run.final || {};
    var card = el("div", { class: "subcard" });
    panel.appendChild(card);
    card.appendChild(el("h3", null, ["Compiled model: " + run.run_id + " ", labelChips(data.labels || [])]));
    var program = data.program || (data.best && data.best.program) || null;
    if (program && program.grafts) {
      var strip = el("div", { class: "host-strip" });
      card.appendChild(strip);
      drawHostStrip(strip, program.grafts.map(function (graft) {
        return Object.assign({ host_layer: graft.at }, graft);
      }), 16, program.host || program.base);
    }
    var list = el("dl", { class: "kv" });
    Object.keys(data).forEach(function (key) {
      var value = data[key];
      if (value === null || typeof value === "object") {
        return;
      }
      list.appendChild(el("dt", { text: key.replace(/_/g, " ") }));
      list.appendChild(el("dd", { text: String(value) }));
    });
    card.appendChild(list);
    if (run.card_md) {
      card.appendChild(el("details", { class: "more" }, [el("summary", { text: "Model card (markdown)" }),
        el("pre", { class: "json", text: run.card_md })]));
    }
    card.appendChild(el("details", { class: "more" }, [el("summary", { text: "final.json" }),
      el("pre", { class: "json", text: JSON.stringify(data, null, 2) })]));
    card.appendChild(el("p", { class: "meta", text: "Source: " + run.source }));
  }

  // Parts.

  function renderParts() {
    var panel = byId("parts-panel");
    clear(panel);
    var catalogs = (state.parts && state.parts.catalogs) || [];
    if (catalogs.length === 0) {
      panel.appendChild(el("p", { class: "meta", text: "No pilot catalog in this snapshot." }));
      return;
    }
    catalogs.forEach(function (catalog) {
      panel.appendChild(el("h3", { text: catalog.title + " (" + plural(catalog.parts.length, "part") + ")" }));
      panel.appendChild(el("p", { class: "meta", text: "From " + catalog.file + "." }));
      var table = el("table", { class: "data stack" });
      table.appendChild(el("thead", null, el("tr", null, ["Part", "Donor", "Layer", "Kind", "Units",
        "How it was chosen", "Effect vs random"].map(function (name) {
        return el("th", { text: name });
      }))));
      var body = el("tbody");
      catalog.parts.forEach(function (part) {
        var effect = isNumber(part.mean_effect) ? fmtShort(part.mean_effect, 3) +
          (isNumber(part.random_mean_effect) ? " vs " + fmtShort(part.random_mean_effect, 3) + " for random units" : "") : "not measured";
        var how = part.method ? part.method + (part.function ? ": " + part.function : "") : (part.trained_on || "");
        body.appendChild(el("tr", null, [
          el("td", { "data-label": "Part", class: "mono", text: part.id }),
          el("td", { "data-label": "Donor", text: part.model }),
          el("td", { "data-label": "Layer", class: "num", text: String(part.layer) }),
          el("td", { "data-label": "Kind", text: String(part.type || "").replace(/_/g, " ") }),
          el("td", { "data-label": "Units", class: "num", text: isNumber(part.n_units) ? String(part.n_units) : "?" }),
          el("td", { "data-label": "How it was chosen", text: how }),
          el("td", { "data-label": "Effect vs random", text: effect })
        ]));
      });
      table.appendChild(body);
      panel.appendChild(table);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
