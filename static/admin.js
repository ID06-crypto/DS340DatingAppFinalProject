// Draws admin charts from data attributes, so pages need no inline scripts.
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
