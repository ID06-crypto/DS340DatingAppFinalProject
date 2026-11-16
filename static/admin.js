// Draws admin charts from data attributes, so pages need no inline scripts.
function hatch(ctx, color, variant) {
  var tile = document.createElement("canvas");
  tile.width = tile.height = 7;
  var t = tile.getContext("2d");
  t.strokeStyle = color;
  t.lineWidth = 1.2;
  t.beginPath();
  if (variant % 2) { t.moveTo(0, 0); t.lineTo(7, 7); } else { t.moveTo(0, 7); t.lineTo(7, 0); }
  t.stroke();
  return ctx.createPattern(tile, "repeat");
}

document.addEventListener("DOMContentLoaded", function () {
  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  // Ink and pencil greys only, to match the rest of the site.
  var palette = ["#1a1a1a", "#77746d", "#4a4a47", "#a3a097", "#2f2f2d"];
  if (window.Chart) {
    Chart.defaults.font.family = '"Inter Variable", system-ui, sans-serif';
    Chart.defaults.color = "#4a4a47";
    Chart.defaults.borderColor = "#cdc8ba";
  }
  var dashes = [[], [6, 4], [2, 3], [10, 3, 2, 3], [1, 1]];  // Lines differ by more than colour.
  document.querySelectorAll("canvas[data-chart]").forEach(function (canvas) {
    var spec = JSON.parse(canvas.getAttribute("data-chart"));
    var ctx = canvas.getContext("2d");
    spec.data.datasets.forEach(function (dataset, i) {
      dataset.borderColor = palette[i % palette.length];
      if (spec.type === "bar") {
        // Bars are outlined in ink and filled with pencil hatching, like the rest of the site.
        dataset.backgroundColor = hatch(ctx, palette[i % palette.length], i);
        dataset.borderWidth = 1.5;
      } else {
        dataset.backgroundColor = palette[i % palette.length];
        dataset.borderDash = dashes[i % dashes.length];
      }
    });
    spec.options = spec.options || {};
    spec.options.animation = reduceMotion ? false : spec.options.animation;
    spec.options.maintainAspectRatio = false;
    // Sideways bar charts get taller with more rows, so every label fits.
    if (spec.rowHeight) {
      canvas.parentElement.style.height = (spec.data.labels.length * spec.rowHeight + 90) + "px";
    }
    var views = spec.views;
    delete spec.views;
    delete spec.rowHeight;
    var chart = new Chart(canvas, spec);
    // A switch between versions of the same chart, like all ratings or only some.
    var toggle = canvas.id && views && document.querySelector('[data-switch-for="' + canvas.id + '"]');
    if (toggle) {
      var caption = document.querySelector('[data-caption-for="' + canvas.id + '"]');
      toggle.addEventListener("change", function (event) {
        var view = views[event.target.value];
        if (!view) { return; }
        chart.data.datasets[0].data = view.data;
        chart.update();
        if (caption) { caption.textContent = view.caption; }
      });
    }
  });
});

// Heat maps, drawn on a canvas from a data-heatmap attribute: {labels, groups, values, max}.
// Paper for 0 through ink for the maximum, hatching where there's no score yet, the
// diagonal (people and themselves) left blank, and lines between gender groups.
(function () {
  var PAPER = [246, 243, 234];
  var INK = [26, 26, 26];

  function shade(t) {
    t = Math.max(0, Math.min(1, t));
    var c = PAPER.map(function (p, i) { return Math.round(p + (INK[i] - p) * t); });
    return "rgb(" + c.join(",") + ")";
  }

  function hatchPattern(ctx) {
    var tile = document.createElement("canvas");
    tile.width = tile.height = 6;
    var t = tile.getContext("2d");
    t.fillStyle = "#f6f3ea"; t.fillRect(0, 0, 6, 6);
    t.strokeStyle = "#b3afa4"; t.lineWidth = 1;
    t.beginPath(); t.moveTo(0, 6); t.lineTo(6, 0); t.stroke();
    return ctx.createPattern(tile, "repeat");
  }

  function draw(canvas) {
    var spec = JSON.parse(canvas.getAttribute("data-heatmap"));
    var n = spec.labels.length;
    var wrap = canvas.parentElement;
    if (!n) { return; }
    var margin = 30, legend = 52;
    var cell = Math.max(3, Math.min(18, Math.floor((wrap.clientWidth - margin - 12) / n)));
    var size = cell * n;
    var width = margin + size + 12, height = margin + size + legend;
    var ratio = window.devicePixelRatio || 1;
    canvas.width = width * ratio; canvas.height = height * ratio;
    canvas.style.width = width + "px"; canvas.style.height = height + "px";
    var ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    var hatch = hatchPattern(ctx);

    for (var i = 0; i < n; i++) {
      for (var j = 0; j < n; j++) {
        var v = spec.values[i][j];
        var x = margin + j * cell, y = margin + i * cell;
        if (i === j) { ctx.fillStyle = "#ece7da"; }
        else if (v === null) { ctx.fillStyle = hatch; }
        else { ctx.fillStyle = shade(v / spec.max); }
        ctx.fillRect(x, y, cell, cell);
      }
    }

    // Gender groups: a line where each changes, and its letter along both edges.
    ctx.strokeStyle = "#1a1a1a"; ctx.fillStyle = "#1a1a1a";
    ctx.font = "600 11px ui-monospace, Menlo, monospace";
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    var start = 0;
    for (var k = 1; k <= n; k++) {
      if (k === n || spec.groups[k] !== spec.groups[start]) {
        var mid = margin + ((start + k) / 2) * cell;
        ctx.fillText(spec.groups[start], mid, margin / 2);
        ctx.fillText(spec.groups[start], margin / 2, mid);
        if (k < n) {
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.moveTo(margin + k * cell, margin - 4); ctx.lineTo(margin + k * cell, margin + size);
          ctx.moveTo(margin - 4, margin + k * cell); ctx.lineTo(margin + size, margin + k * cell);
          ctx.stroke();
        }
        start = k;
      }
    }
    ctx.lineWidth = 1.5;
    ctx.strokeRect(margin, margin, size, size);

    // Legend: the scale, and what hatching means.
    var ly = margin + size + 18, lw = Math.min(220, size);
    for (var s = 0; s < lw; s++) { ctx.fillStyle = shade(s / (lw - 1)); ctx.fillRect(margin + s, ly, 1, 10); }
    ctx.strokeRect(margin, ly, lw, 10);
    ctx.fillStyle = "#4a4a47"; ctx.font = "500 10px ui-monospace, Menlo, monospace"; ctx.textBaseline = "top";
    ctx.textAlign = "left"; ctx.fillText("0", margin, ly + 14);
    ctx.textAlign = "right"; ctx.fillText(String(spec.max), margin + lw, ly + 14);
    var hx = margin + lw + 24;
    if (hx + 120 < width) {
      ctx.fillStyle = hatch; ctx.fillRect(hx, ly, 14, 10); ctx.strokeRect(hx, ly, 14, 10);
      ctx.fillStyle = "#4a4a47"; ctx.textAlign = "left"; ctx.textBaseline = "middle";
      ctx.fillText("not rated yet", hx + 20, ly + 5);
    }

    canvas._heat = { spec: spec, margin: margin, cell: cell, n: n };
  }

  function describe(canvas, event) {
    var h = canvas._heat;
    if (!h) { return null; }
    var box = canvas.getBoundingClientRect();
    var j = Math.floor((event.clientX - box.left - h.margin) / h.cell);
    var i = Math.floor((event.clientY - box.top - h.margin) / h.cell);
    if (i < 0 || j < 0 || i >= h.n || j >= h.n || i === j) { return null; }
    var ids = h.spec.ids || [];
    function name(k) { return h.spec.labels[k] + (ids[k] ? " (" + ids[k] + ")" : ""); }
    var a = name(i), b = name(j), v = h.spec.values[i][j];
    if (canvas.getAttribute("data-kind") === "reciprocal") {
      return v === null ? a + " and " + b + ": not both rated yet" : a + " and " + b + ": " + v.toFixed(2) + " mutual";
    }
    return v === null ? a + " hasn't rated " + b + " yet" : a + " rated " + b + ": " + v.toFixed(2);
  }

  document.addEventListener("DOMContentLoaded", function () {
    var canvases = document.querySelectorAll("canvas[data-heatmap]");
    canvases.forEach(function (canvas) {
      draw(canvas);
      var tip = document.createElement("div");
      tip.className = "heat-tip";
      tip.hidden = true;
      canvas.parentElement.appendChild(tip);
      canvas.addEventListener("mousemove", function (event) {
        var text = describe(canvas, event);
        if (!text) { tip.hidden = true; return; }
        tip.textContent = text;
        tip.hidden = false;
        var box = canvas.parentElement.getBoundingClientRect();
        tip.style.left = (event.clientX - box.left + canvas.parentElement.scrollLeft + 14) + "px";
        tip.style.top = (event.clientY - box.top + 14) + "px";
      });
      canvas.addEventListener("mouseleave", function () { tip.hidden = true; });
    });
    var timer = null;
    window.addEventListener("resize", function () {
      clearTimeout(timer);
      timer = setTimeout(function () { canvases.forEach(draw); }, 150);
    });
  });
})();
