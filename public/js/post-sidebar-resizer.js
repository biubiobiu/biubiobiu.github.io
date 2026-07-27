"use strict";

(function () {
  var desktopQuery = window.matchMedia("(min-width: 1025px)");
  var sidebar = document.getElementById("sidebar-section");
  var toggle = document.getElementById("post-sidebar-toggle");
  var resizer = document.getElementById("post-sidebar-resizer");

  if (!sidebar || !toggle || !resizer || !sidebar.classList.contains("post-sidebar-resizable")) {
    return;
  }

  var widthKey = "toha-post-sidebar-width";
  var collapsedKey = "toha-post-sidebar-collapsed";
  var defaultWidth = Math.round(window.innerWidth * 0.2);
  var currentWidth = defaultWidth;

  function limits() {
    return {
      min: 220,
      max: Math.min(520, Math.round(window.innerWidth * 0.38))
    };
  }

  function clampWidth(width) {
    var range = limits();
    return Math.max(range.min, Math.min(range.max, Math.round(width)));
  }

  function setWidth(width, persist) {
    currentWidth = clampWidth(width);
    sidebar.style.setProperty("--post-sidebar-width", currentWidth + "px");
    resizer.setAttribute("aria-valuemax", String(limits().max));
    resizer.setAttribute("aria-valuenow", String(currentWidth));
    if (persist) {
      localStorage.setItem(widthKey, String(currentWidth));
    }
  }

  function setCollapsed(collapsed, persist) {
    sidebar.classList.toggle("is-collapsed", collapsed);
    document.body.classList.toggle("post-sidebar-collapsed", collapsed);
    toggle.setAttribute("aria-expanded", String(!collapsed));
    toggle.setAttribute("aria-label", collapsed ? "展开左侧目录" : "收起左侧目录");

    var icon = toggle.querySelector("i");
    var label = toggle.querySelector("span");
    if (icon) {
      icon.classList.toggle("fa-chevron-left", !collapsed);
      icon.classList.toggle("fa-chevron-right", collapsed);
    }
    if (label) {
      label.textContent = collapsed ? "展开目录" : "收起目录";
    }
    if (persist) {
      localStorage.setItem(collapsedKey, collapsed ? "true" : "false");
    }
  }

  function restorePreferences() {
    var storedWidth = Number(localStorage.getItem(widthKey));
    if (Number.isFinite(storedWidth) && storedWidth > 0) {
      setWidth(storedWidth, false);
    } else {
      setWidth(defaultWidth, false);
    }
    setCollapsed(localStorage.getItem(collapsedKey) === "true", false);
  }

  toggle.addEventListener("click", function () {
    if (!desktopQuery.matches) {
      return;
    }
    setCollapsed(!sidebar.classList.contains("is-collapsed"), true);
  });

  resizer.addEventListener("pointerdown", function (event) {
    if (!desktopQuery.matches || sidebar.classList.contains("is-collapsed")) {
      return;
    }

    event.preventDefault();
    resizer.setPointerCapture(event.pointerId);
    document.body.classList.add("post-sidebar-is-resizing");

    function resize(pointerEvent) {
      setWidth(pointerEvent.clientX, false);
    }

    function stop(pointerEvent) {
      if (resizer.hasPointerCapture(pointerEvent.pointerId)) {
        resizer.releasePointerCapture(pointerEvent.pointerId);
      }
      document.body.classList.remove("post-sidebar-is-resizing");
      setWidth(currentWidth, true);
      resizer.removeEventListener("pointermove", resize);
      resizer.removeEventListener("pointerup", stop);
      resizer.removeEventListener("pointercancel", stop);
    }

    resizer.addEventListener("pointermove", resize);
    resizer.addEventListener("pointerup", stop);
    resizer.addEventListener("pointercancel", stop);
  });

  resizer.addEventListener("keydown", function (event) {
    if (!desktopQuery.matches || sidebar.classList.contains("is-collapsed")) {
      return;
    }

    var distance = event.shiftKey ? 30 : 10;
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      setWidth(currentWidth - distance, true);
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      setWidth(currentWidth + distance, true);
    } else if (event.key === "Home") {
      event.preventDefault();
      setWidth(limits().min, true);
    } else if (event.key === "End") {
      event.preventDefault();
      setWidth(limits().max, true);
    }
  });

  resizer.addEventListener("dblclick", function () {
    setWidth(window.innerWidth * 0.2, true);
  });

  window.addEventListener("resize", function () {
    if (desktopQuery.matches) {
      setWidth(currentWidth, false);
    }
  });

  restorePreferences();
})();

(function () {
  var desktopQuery = window.matchMedia("(min-width: 1025px)");
  var toc = document.getElementById("toc-section");
  var toggle = document.getElementById("post-toc-toggle");

  if (!toc || !toggle || !toc.classList.contains("post-toc-collapsible")) {
    return;
  }

  var collapsedKey = "toha-post-toc-collapsed";

  function setCollapsed(collapsed, persist) {
    toc.classList.toggle("is-collapsed", collapsed);
    document.body.classList.toggle("post-toc-collapsed", collapsed);
    toggle.setAttribute("aria-expanded", String(!collapsed));
    toggle.setAttribute("aria-label", collapsed ? "展开文章目录" : "收起文章目录");

    var icon = toggle.querySelector("i");
    var label = toggle.querySelector("span");
    if (icon) {
      icon.classList.toggle("fa-chevron-right", !collapsed);
      icon.classList.toggle("fa-chevron-left", collapsed);
    }
    if (label) {
      label.textContent = collapsed ? "展开目录" : "收起目录";
    }
    if (persist) {
      localStorage.setItem(collapsedKey, collapsed ? "true" : "false");
    }
  }

  toggle.addEventListener("click", function () {
    if (!desktopQuery.matches) {
      return;
    }
    setCollapsed(!toc.classList.contains("is-collapsed"), true);
  });

  setCollapsed(localStorage.getItem(collapsedKey) === "true", false);
})();
