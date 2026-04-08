const statusEl = document.getElementById("status");
const startAudioBtn = document.getElementById("start-audio");
const stopBtn = document.getElementById("stop");
const delaySliderEl = document.getElementById("delay-slider");
const delayValueEl = document.getElementById("delay-value");
const BRIDGE_BASE_CANDIDATES = [
  "http://127.0.0.1:39091",
  "http://localhost:39091",
];
const AUDIO_DELAY_STORAGE_KEY = "audioDelayMs";
const DEFAULT_AUDIO_DELAY_MS = 95;

function clampAudioDelayMs(value) {
  const delayMs = Number(value ?? DEFAULT_AUDIO_DELAY_MS);
  if (!Number.isFinite(delayMs)) {
    return DEFAULT_AUDIO_DELAY_MS;
  }
  return Math.max(0, Math.min(500, Math.round(delayMs)));
}

function renderDelayValue(value) {
  const delayMs = clampAudioDelayMs(value);
  delayValueEl.textContent = `${delayMs} ms`;
  delaySliderEl.value = String(delayMs);
  return delayMs;
}

async function loadDelayPreference() {
  try {
    const stored = await chrome.storage.local.get([AUDIO_DELAY_STORAGE_KEY]);
    return renderDelayValue(stored?.[AUDIO_DELAY_STORAGE_KEY]);
  } catch (_error) {
    return renderDelayValue(DEFAULT_AUDIO_DELAY_MS);
  }
}

async function setLiveDelay(value) {
  const delayMs = clampAudioDelayMs(value);
  renderDelayValue(delayMs);
  try {
    await chrome.runtime.sendMessage({
      target: "background",
      type: "set-audio-delay-ms",
      audioDelayMs: delayMs,
    });
  } catch (_error) {}
  return delayMs;
}

async function persistDelayPreference(value) {
  const delayMs = clampAudioDelayMs(value);
  try {
    await chrome.storage.local.set({ [AUDIO_DELAY_STORAGE_KEY]: delayMs });
  } catch (_error) {}
  return delayMs;
}

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.className = isError ? "" : "muted";
}

async function probeBridgeHealth() {
  let lastError = null;
  for (const base of BRIDGE_BASE_CANDIDATES) {
    try {
      const response = await fetch(`${base}/health`);
      if (!response.ok) {
        lastError = new Error(`bridge responded with ${response.status}`);
        continue;
      }
      const payload = await response.json();
      return { base, payload };
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError || new Error("HDRTVNet++ local bridge is unreachable.");
}

async function refreshBridgeStatus() {
  try {
    const { base, payload } = await probeBridgeHealth();
    const sessionCount = Number(payload.session_count || 0);
    setStatus(
      `Local bridge ready at ${payload.bridge_url || base}\nActive sessions: ${sessionCount}`
    );
  } catch (error) {
    setStatus(
      `HDRTVNet++ local bridge is not reachable.\nRun run_gui.bat or py src/gui.py first, then try again.\n\n${error.message || error}`,
      true
    );
  }
}

startAudioBtn.addEventListener("click", async () => {
  setStatus("Starting Chrome Audio Sync ...");
  try {
    await probeBridgeHealth();
    const audioDelayMs = await setLiveDelay(delaySliderEl.value);
    await persistDelayPreference(audioDelayMs);
    const response = await chrome.runtime.sendMessage({
      target: "background",
      type: "start-audio-current-tab",
      audioDelayMs,
    });
    if (!response || !response.ok) {
      throw new Error(response?.error || "capture start failed");
    }
    setStatus(
      `Sharing "${response.title || "Current Tab"}"\n${
        response.hasAudio === false
          ? "No tab-audio track was detected."
          : `Chrome is now replaying delayed tab audio locally at ${response.audioDelayMs || audioDelayMs} ms.`
      }\nReturn to HDRTVNet++ and press Play. HDRTVNet++ will stop this session automatically when playback stops or the app closes.`
    );
  } catch (error) {
    setStatus(`Could not start Chrome Audio Sync:\n${error.message || error}`, true);
  }
});

stopBtn.addEventListener("click", async () => {
  setStatus("Stopping Chrome Audio Sync ...");
  try {
    await chrome.runtime.sendMessage({
      target: "background",
      type: "stop-capture",
    });
    setStatus("Chrome Audio Sync stopped.");
  } catch (error) {
    setStatus(`Could not stop capture:\n${error.message || error}`, true);
  }
});

delaySliderEl.addEventListener("input", () => {
  void setLiveDelay(delaySliderEl.value);
});

delaySliderEl.addEventListener("change", () => {
  void persistDelayPreference(delaySliderEl.value);
});

void loadDelayPreference();
refreshBridgeStatus();
