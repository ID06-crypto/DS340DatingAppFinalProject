// The rating control has no default: submitting is only possible after the person sets a
// score, with either the slider or the number box, which stay in sync.
function setUpRating(root) {
  root.querySelectorAll("form.rating").forEach(function (form) {
    var slider = form.querySelector('input[type="range"]');
    var number = form.querySelector('input[type="number"]');
    var submit = form.querySelector('button[type="submit"]');
    function allow(ok) { submit.disabled = !ok; }
    slider.addEventListener("input", function () {
      slider.classList.remove("untouched");
      number.value = Number(slider.value).toFixed(1);
      allow(true);
    });
    number.addEventListener("input", function () {
      var value = parseFloat(number.value);
      var ok = !isNaN(value) && value >= 1 && value <= 10;
      if (ok) { slider.value = value; slider.classList.remove("untouched"); }
      allow(ok);
    });
  });
}

document.addEventListener("DOMContentLoaded", function () { setUpRating(document); });

// After HTMX swaps in a new card, set it up and move focus to it for screen readers.
document.addEventListener("htmx:afterSwap", function (event) {
  setUpRating(event.detail.target);
  var focusTarget = event.detail.target.querySelector("[data-focus]");
  if (focusTarget) { focusTarget.focus(); }
});
