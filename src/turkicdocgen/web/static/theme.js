(function () {
  let theme = null;
  const match = document.cookie.match(/(?:^|; )tdg-theme=([^;]*)/);
  if (match) {
    theme = match[1];
  }
  if (!theme) {
    theme = localStorage.getItem("tdg-theme");
  }
  if (theme !== "light" && theme !== "dark") {
    theme = "light";
  }
  document.documentElement.dataset.theme = theme;
  localStorage.setItem("tdg-theme", theme);
})();
