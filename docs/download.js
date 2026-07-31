// The four cards in the HTML are the real, working fallback: they are what
// crawlers index and what a visitor without JavaScript uses. This file only
// collapses them into one button when it is sure which platform is asking.
(function () {
  "use strict";

  var BASE = "https://github.com/giancarlobrusca/local-img/releases/latest/download/";

  // Pure: everything it needs arrives as arguments, so it can be called with
  // fabricated inputs from a console. An iPad in desktop mode claims to be a
  // Mac, and offering it a .dmg would be a lie; no Mac reports touch points.
  function isMobile(env) {
    var ua = env.ua || "";
    var platform = env.platform || "";
    var touch = env.maxTouchPoints || 0;
    var subject = platform + " " + ua;
    if (/Android|iPhone|iPad|iPod/i.test(subject)) return true;
    if (/Mac/i.test(subject) && touch > 1) return true;
    return false;
  }

  // Pure, same contract as before: a platform key, or null when it declines.
  // Callers who need to know *why* it declined use isMobile(env) instead of
  // re-testing navigator.userAgent themselves.
  function pickPlatform(env) {
    if (isMobile(env)) return null;

    var ua = env.ua || "";
    var platform = env.platform || "";
    var subject = platform + " " + ua;

    if (/Mac/i.test(subject)) return "mac";
    if (/Win/i.test(subject)) return "windows";
    // Android is already gone, so Linux here means desktop Linux.
    if (/Linux|X11/i.test(subject)) return "linux";
    return null;
  }

  window.localImgPickPlatform = pickPlatform;
  window.localImgIsMobile = isMobile;

  var CARDS = {
    mac:     { file: "local-img-macos-arm64.dmg",     label: "macOS" },
    windows: { file: "local-img-windows-x64.msi",     label: "Windows" },
    linux:   { file: "local-img-linux-x64.AppImage",  label: "Linux" }
  };

  var grid = document.getElementById("downloads");
  if (!grid) return;

  var data = grid.dataset;
  var uaData = navigator.userAgentData;
  var env = {
    ua: navigator.userAgent,
    platform: (uaData && uaData.platform) || navigator.platform,
    maxTouchPoints: navigator.maxTouchPoints
  };
  var picked = pickPlatform(env);

  if (!picked) {
    // Unknown desktop, or a phone/tablet. Keep all four cards; say why if it
    // is mobile, using the same signal (UA + touch points) pickPlatform
    // used, because "download" on a phone otherwise promises something the
    // visitor cannot use — and an iPad in desktop mode deserves the same
    // explanation, not silence.
    if (isMobile(env)) {
      var note = document.createElement("p");
      note.className = "dl-mobile";
      note.textContent = data.mobile || "";
      grid.parentNode.insertBefore(note, grid);
    }
    return;
  }

  var card = CARDS[picked];
  var hero = document.createElement("a");
  hero.className = "dl-hero";
  hero.href = BASE + card.file;

  var heroLine = document.createElement("span");
  heroLine.className = "dl-hero-line";
  heroLine.textContent = (data.verb || "") + " " + card.label;

  var heroMeta = document.createElement("span");
  heroMeta.className = "dl-hero-meta";
  heroMeta.textContent = data[picked] || "";

  hero.appendChild(heroLine);
  hero.appendChild(heroMeta);

  var others = document.createElement("p");
  others.className = "dl-others";
  others.textContent = data.others || "";
  var links = grid.querySelectorAll("a.dl");
  for (var i = 0; i < links.length; i++) {
    if (links[i].href.indexOf(card.file) !== -1) continue;
    var a = document.createElement("a");
    a.href = links[i].href;
    a.textContent = links[i].querySelector(".os").textContent + " " +
                    links[i].querySelector(".file").textContent;
    others.appendChild(document.createTextNode(" "));
    others.appendChild(a);
  }

  grid.parentNode.insertBefore(hero, grid);
  grid.parentNode.insertBefore(others, grid);
  grid.remove();
})();
