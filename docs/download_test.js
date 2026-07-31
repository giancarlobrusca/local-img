// Table-driven checks for the platform detection in download.js.
//
// download.js is a classic browser script wrapped in an IIFE that touches
// `document` and `navigator` at load time. This loads the real file (not a
// copy of its logic) into a minimal fake window/document so the IIFE runs
// to completion, then calls the pure functions it exposes on `window`.
//
// Run: node docs/download_test.js

"use strict";

const fs = require("fs");
const path = require("path");
const vm = require("vm");

const SRC = path.join(__dirname, "download.js");

function loadPickers() {
  const fakeWindow = {};
  // No #downloads element, so the IIFE returns right after exposing the
  // pure functions on window — it never touches the rest of the DOM.
  const fakeDocument = {
    getElementById: function () { return null; }
  };
  const sandbox = {
    window: fakeWindow,
    document: fakeDocument,
    navigator: { userAgent: "", userAgentData: null, maxTouchPoints: 0 }
  };
  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync(SRC, "utf8"), sandbox, { filename: SRC });
  return {
    pickPlatform: fakeWindow.localImgPickPlatform,
    isMobile: fakeWindow.localImgIsMobile
  };
}

const { pickPlatform, isMobile } = loadPickers();

if (typeof pickPlatform !== "function") {
  console.log("FAIL  download.js did not expose window.localImgPickPlatform");
  process.exit(1);
}
if (typeof isMobile !== "function") {
  console.log("FAIL  download.js did not expose window.localImgIsMobile");
  process.exit(1);
}

const MAC_UA =
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 " +
  "(KHTML, like Gecko) Version/17.0 Safari/605.1.15";
const WIN_UA =
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 " +
  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36";
const LINUX_UA =
  "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 " +
  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36";
const IPHONE_UA =
  "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) " +
  "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1";
const ANDROID_UA =
  "Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 " +
  "(KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36";
const BOT_UA = "SomeCrawler/1.0 (+https://example.com/bot)";

// Each case: [label, env, expected pickPlatform() result, expected isMobile()]
const CASES = [
  ["macOS desktop",
    { ua: MAC_UA, platform: "MacIntel", maxTouchPoints: 0 },
    "mac", false],

  ["Windows desktop",
    { ua: WIN_UA, platform: "Win32", maxTouchPoints: 0 },
    "windows", false],

  ["Linux desktop",
    { ua: LINUX_UA, platform: "Linux x86_64", maxTouchPoints: 0 },
    "linux", false],

  ["iPhone",
    { ua: IPHONE_UA, platform: "iPhone", maxTouchPoints: 5 },
    null, true],

  ["iPad in desktop mode (Mac UA, touch points)",
    { ua: MAC_UA, platform: "MacIntel", maxTouchPoints: 5 },
    null, true],

  ["Android",
    { ua: ANDROID_UA, platform: "Linux armv8l", maxTouchPoints: 5 },
    null, true],

  ["unrecognised UA",
    { ua: BOT_UA, platform: "", maxTouchPoints: 0 },
    null, false]
];

let failures = 0;

for (const [label, env, wantPlatform, wantMobile] of CASES) {
  const gotPlatform = pickPlatform(env);
  const platformOk = gotPlatform === wantPlatform;
  console.log(
    (platformOk ? "ok  " : "FAIL") +
    "  pickPlatform(" + label + ") = " + JSON.stringify(gotPlatform) +
    " (want " + JSON.stringify(wantPlatform) + ")"
  );
  if (!platformOk) failures++;

  const gotMobile = isMobile(env);
  const mobileOk = gotMobile === wantMobile;
  console.log(
    (mobileOk ? "ok  " : "FAIL") +
    "  isMobile(" + label + ") = " + JSON.stringify(gotMobile) +
    " (want " + JSON.stringify(wantMobile) + ")"
  );
  if (!mobileOk) failures++;
}

console.log("");
if (failures > 0) {
  console.log(failures + " check(s) failed");
  process.exit(1);
}
console.log("all checks passed");
