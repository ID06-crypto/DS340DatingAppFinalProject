// Draws admin charts from data attributes, so pages need no inline scripts.
document.addEventListener("DOMContentLoaded", function () {
  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var palette = ["#1d4ed8", "#b45309", "#047857", "#7c3aed", "#be123c"];
  var dashes = [[], [6, 4], [2, 3], [10, 3, 2, 3], [1, 1]];  // Lines differ by more than colour.
  document.querySelectorAll("canvas[data-chart]").forEach(function (canvas) {
    var spec = JSON.parse(canvas.getAttribute("data-chart"));
    spec.data.datasets.forEach(function (dataset, i) {
      dataset.borderColor = palette[i % palette.length];
      dataset.backgroundColor = palette[i % palette.length];
      dataset.borderDash = dashes[i % dashes.length];
    });
    spec.options = spec.options || {};
    spec.options.animation = reduceMotion ? false : spec.options.animation;
    spec.options.maintainAspectRatio = false;
    new Chart(canvas, spec);
  });
});
