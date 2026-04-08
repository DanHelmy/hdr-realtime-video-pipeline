const OFFSCREEN_DOCUMENT_PATH = "offscreen.html";

async function hasOffscreenDocument() {
  if (!chrome.runtime.getContexts) {
    return false;
  }
  const contexts = await chrome.runtime.getContexts({
    contextTypes: ["OFFSCREEN_DOCUMENT"],
    documentUrls: [chrome.runtime.getURL(OFFSCREEN_DOCUMENT_PATH)],
  });
  return contexts.length > 0;
}

async function ensureOffscreenDocument() {
  if (await hasOffscreenDocument()) {
    return;
  }
  await chrome.offscreen.createDocument({
    url: OFFSCREEN_DOCUMENT_PATH,
    reasons: [chrome.offscreen.Reason.USER_MEDIA],
    justification: "Capture active browser-tab audio, delay it locally in Chrome, and expose a sync session to HDRTVNet++.",
  });
}

async function getActiveTab() {
  const tabs = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
  return tabs[0] || null;
}

function inferBrowserNames() {
  const ua = navigator.userAgent || "";
  if (ua.includes("Edg/")) {
    return { browserName: "Microsoft Edge", processName: "msedge.exe" };
  }
  return { browserName: "Google Chrome", processName: "chrome.exe" };
}

async function startAudioSyncForActiveTab(audioDelayMs = 95) {
  const tab = await getActiveTab();
  if (!tab || typeof tab.id !== "number") {
    return { ok: false, error: "no active tab found" };
  }
  await ensureOffscreenDocument();
  const streamId = await chrome.tabCapture.getMediaStreamId({ targetTabId: tab.id });
  const browserInfo = inferBrowserNames();
  const result = await chrome.runtime.sendMessage({
    target: "offscreen",
    type: "offscreen-start-audio-sync",
    streamId,
    title: tab.title || "Browser Tab",
    sourceUrl: tab.url || "",
    browserName: browserInfo.browserName,
    processName: browserInfo.processName,
    audioDelayMs,
  });
  return {
    ok: !!result?.ok,
    error: result?.error || "",
    title: tab.title || "Browser Tab",
    hasAudio: result?.hasAudio !== false,
    audioDelayMs: Number(result?.audioDelayMs || audioDelayMs || 95),
  };
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (!message || typeof message !== "object" || message.target !== "background") {
    return false;
  }

  (async () => {
    if (message.type === "start-audio-current-tab") {
      return startAudioSyncForActiveTab(message.audioDelayMs);
    }

    if (message.type === "set-audio-delay-ms") {
      await ensureOffscreenDocument();
      const result = await chrome.runtime.sendMessage({
        target: "offscreen",
        type: "offscreen-set-audio-delay-ms",
        audioDelayMs: Number(message.audioDelayMs || 0),
      });
      return result || { ok: true };
    }

    if (message.type === "stop-capture") {
      await ensureOffscreenDocument();
      const result = await chrome.runtime.sendMessage({
        target: "offscreen",
        type: "offscreen-stop-capture",
      });
      return result || { ok: true };
    }

    return { ok: false, error: `unknown message type: ${message.type}` };
  })()
    .then((payload) => sendResponse(payload))
    .catch((error) => sendResponse({ ok: false, error: error?.message || String(error) }));
  return true;
});
