const BRIDGE_BASE_CANDIDATES = [
  "http://127.0.0.1:39091",
  "http://localhost:39091",
];
const DEFAULT_AUDIO_DELAY_MS = 95;
const MAX_AUDIO_DELAY_S = 2.0;
const SESSION_KEEPALIVE_INTERVAL_MS = 1000;
const SESSION_KEEPALIVE_BACKOFF_MAX_MS = 5000;

const state = {
  stream: null,
  sessionId: "",
  running: false,
  metadata: null,
  audioContext: null,
  audioSource: null,
  audioDelayNode: null,
  audioOutputGain: null,
  audioDelayMs: DEFAULT_AUDIO_DELAY_MS,
  bridgeFailureCount: 0,
  bridgeFailureSincePerf: 0,
  lastBridgeFailureLogPerf: 0,
  bridgeBase: "",
  stopPromise: null,
  keepaliveTimerId: 0,
};

function clampAudioDelayMs(value) {
  const delayMs = Number(value ?? DEFAULT_AUDIO_DELAY_MS);
  if (!Number.isFinite(delayMs)) {
    return DEFAULT_AUDIO_DELAY_MS;
  }
  return Math.max(0, Math.min(Math.round(MAX_AUDIO_DELAY_S * 1000.0), Math.round(delayMs)));
}

function clearSessionKeepalive() {
  if (state.keepaliveTimerId) {
    try {
      clearTimeout(state.keepaliveTimerId);
    } catch (_error) {}
  }
  state.keepaliveTimerId = 0;
}

function applyAudioDelayMs(delayMs, { immediate = false } = {}) {
  const clamped = clampAudioDelayMs(delayMs);
  state.audioDelayMs = clamped;
  const delayNode = state.audioDelayNode;
  const audioContext = state.audioContext;
  if (!delayNode || !audioContext) {
    return clamped;
  }
  const nextDelaySec = clamped / 1000.0;
  try {
    const param = delayNode.delayTime;
    const now = audioContext.currentTime;
    param.cancelScheduledValues(now);
    if (immediate) {
      param.setValueAtTime(nextDelaySec, now);
    } else {
      param.setValueAtTime(param.value, now);
      param.linearRampToValueAtTime(nextDelaySec, now + 0.04);
    }
  } catch (_error) {
    try {
      delayNode.delayTime.value = nextDelaySec;
    } catch (_ignored) {}
  }
  return clamped;
}

async function probeBridgeBase() {
  let lastError = null;
  for (const base of BRIDGE_BASE_CANDIDATES) {
    try {
      const response = await fetch(`${base}/health`);
      if (response.ok) {
        state.bridgeBase = base;
        return base;
      }
      lastError = new Error(`bridge health check failed with ${response.status}`);
    } catch (error) {
      lastError = error;
    }
  }
  state.bridgeBase = "";
  throw lastError || new Error("HDRTVNet++ local bridge is unreachable.");
}

async function bridgeFetch(path, init = {}, allowRetry = true) {
  const base = state.bridgeBase || await probeBridgeBase();
  try {
    return await fetch(`${base}${path}`, init);
  } catch (error) {
    if (!allowRetry) {
      throw error;
    }
    state.bridgeBase = "";
    const fallbackBase = await probeBridgeBase();
    return fetch(`${fallbackBase}${path}`, init);
  }
}

async function postJson(path, payload) {
  const response = await bridgeFetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const error = new Error(`${path} failed with ${response.status}`);
    error.bridgeStatus = response.status;
    throw error;
  }
  return response.json();
}

async function ensureBridgeReachable() {
  const base = await probeBridgeBase();
  const response = await bridgeFetch("/health", {}, false);
  if (!response.ok) {
    throw new Error(`bridge health check failed with ${response.status}`);
  }
  state.bridgeBase = base;
}

function resetBridgeFailureState() {
  state.bridgeFailureCount = 0;
  state.bridgeFailureSincePerf = 0;
  state.lastBridgeFailureLogPerf = 0;
}

async function handleBridgeFailure(scope, error) {
  const detail = error?.message || String(error || "unknown bridge error");
  const bridgeStatus = Number(error?.bridgeStatus || 0);
  if (bridgeStatus === 410) {
    console.warn(`[HDRTVNet++] ${scope} could not refresh the old HDRTVNet++ session. Chrome Audio Sync will keep running until you stop it manually.`);
    return;
  }
  if (bridgeStatus >= 400 && bridgeStatus < 500) {
    console.error(`[HDRTVNet++] ${scope} hit a bridge error, but Chrome Audio Sync will keep running: ${detail}`);
    return;
  }
  const now = performance.now();
  state.bridgeFailureCount += 1;
  if (state.bridgeFailureSincePerf <= 0) {
    state.bridgeFailureSincePerf = now;
  }
  if (
    state.bridgeFailureCount <= 2
    || (now - state.lastBridgeFailureLogPerf) >= 1500
  ) {
    state.lastBridgeFailureLogPerf = now;
    console.warn(
      `[HDRTVNet++] ${scope} stalled for ${Math.round(now - state.bridgeFailureSincePerf)} ms`,
      error
    );
  }
}

function clearAudioGraph() {
  if (state.audioOutputGain) {
    try {
      state.audioOutputGain.disconnect();
    } catch (_error) {}
  }
  if (state.audioDelayNode) {
    try {
      state.audioDelayNode.disconnect();
    } catch (_error) {}
  }
  if (state.audioSource) {
    try {
      state.audioSource.disconnect();
    } catch (_error) {}
  }
  if (state.audioContext) {
    try {
      state.audioContext.close();
    } catch (_error) {}
  }
  state.audioContext = null;
  state.audioSource = null;
  state.audioDelayNode = null;
  state.audioOutputGain = null;
}

async function stopCapture(options = {}) {
  if (state.stopPromise) {
    return state.stopPromise;
  }
  state.stopPromise = (async () => {
    const notifyBridge = options.notifyBridge !== false;
    state.running = false;
    clearSessionKeepalive();
    if (state.stream) {
      for (const track of state.stream.getTracks()) {
        track.stop();
      }
    }
    clearAudioGraph();
    state.stream = null;
    const sessionId = state.sessionId;
    state.sessionId = "";
    state.metadata = null;
    resetBridgeFailureState();
    if (notifyBridge && sessionId) {
      try {
        await postJson(`/session/${sessionId}/stop`, {});
      } catch (_error) {
        // Ignore bridge shutdown races.
      }
    }
    return { ok: true };
  })();
  try {
    return await state.stopPromise;
  } finally {
    state.stopPromise = null;
  }
}

async function ensureBridgeSession(metadata) {
  const payload = {
    session_id: state.sessionId || "",
    title: metadata.title || "Browser Tab",
    source_url: metadata.sourceUrl || "",
    browser_name: metadata.browserName || "",
    process_name: metadata.processName || "",
    width: metadata.width || 0,
    height: metadata.height || 0,
    fps: metadata.captureFps || 30,
    has_audio: !!metadata.hasAudio,
    audio_sample_rate: metadata.audioSampleRate || 0,
    audio_channels: metadata.audioChannels || 0,
    audio_bits_per_sample: 16,
  };
  const response = await postJson("/session/start", payload);
  state.sessionId = String(response.session_id || "");
  if (!state.sessionId) {
    throw new Error("bridge did not return a session id");
  }
  resetBridgeFailureState();
}

async function postSessionKeepalive(metadata) {
  const response = await bridgeFetch(`/session/${state.sessionId}/keepalive`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      title: metadata?.title || "Browser Tab",
      source_url: metadata?.sourceUrl || "",
      browser_name: metadata?.browserName || "",
      process_name: metadata?.processName || "",
      width: metadata?.width || 0,
      height: metadata?.height || 0,
      fps: metadata?.captureFps || 30,
      has_audio: !!metadata?.hasAudio,
      audio_sample_rate: metadata?.audioSampleRate || 0,
      audio_channels: metadata?.audioChannels || 0,
      audio_bits_per_sample: 16,
    }),
  });
  if (!response.ok) {
    const error = new Error(`session keepalive failed with ${response.status}`);
    error.bridgeStatus = response.status;
    throw error;
  }
}

function scheduleSessionKeepalive() {
  clearSessionKeepalive();
  if (!state.running || !state.sessionId) {
    return;
  }
  const retryDelayMs = Math.min(
    SESSION_KEEPALIVE_BACKOFF_MAX_MS,
    SESSION_KEEPALIVE_INTERVAL_MS * Math.max(1, state.bridgeFailureCount || 1),
  );
  state.keepaliveTimerId = setTimeout(async () => {
    state.keepaliveTimerId = 0;
    if (!state.running || !state.sessionId) {
      return;
    }
    try {
      await postSessionKeepalive(state.metadata || {});
      resetBridgeFailureState();
    } catch (error) {
      await handleBridgeFailure("session keepalive", error);
    } finally {
      if (state.running && state.sessionId) {
        scheduleSessionKeepalive();
      }
    }
  }, retryDelayMs);
}

async function setupAudioCapture(stream) {
  const audioTracks = stream.getAudioTracks();
  if (!audioTracks.length) {
    return { hasAudio: false, audioSampleRate: 0, audioChannels: 0 };
  }

  const audioContext = new AudioContext({ latencyHint: "interactive" });
  await audioContext.resume();

  const source = audioContext.createMediaStreamSource(new MediaStream(audioTracks));
  const delayNode = audioContext.createDelay(MAX_AUDIO_DELAY_S);
  const outputGain = audioContext.createGain();
  outputGain.gain.value = 1.0;
  source.connect(delayNode);
  delayNode.connect(outputGain);
  outputGain.connect(audioContext.destination);

  state.audioContext = audioContext;
  state.audioSource = source;
  state.audioDelayNode = delayNode;
  state.audioOutputGain = outputGain;
  applyAudioDelayMs(state.audioDelayMs, { immediate: true });

  return {
    hasAudio: true,
    audioSampleRate: Math.max(1, Number(audioContext.sampleRate || 48000)),
    audioChannels: 2,
  };
}

async function startAudioSync(message) {
  await stopCapture();
  await ensureBridgeReachable();
  state.audioDelayMs = clampAudioDelayMs(message?.audioDelayMs);

  const mediaConstraints = {
    audio: {
      mandatory: {
        chromeMediaSource: "tab",
        chromeMediaSourceId: message.streamId,
      },
    },
  };

  const stream = await navigator.mediaDevices.getUserMedia(mediaConstraints);

  try {
    state.stream = stream;
    state.running = true;
    state.metadata = {
      title: message.title || "Browser Tab",
      sourceUrl: message.sourceUrl || "",
      browserName: message.browserName || "",
      processName: message.processName || "",
      width: 0,
      height: 0,
      captureFps: 30,
      hasAudio: false,
      audioSampleRate: 0,
      audioChannels: 0,
    };

    const audioInfo = await setupAudioCapture(stream);
    state.metadata.hasAudio = !!audioInfo.hasAudio;
    state.metadata.audioSampleRate = Number(audioInfo.audioSampleRate || 0);
    state.metadata.audioChannels = Number(audioInfo.audioChannels || 0);

    await ensureBridgeSession(state.metadata);
    scheduleSessionKeepalive();
    return {
      ok: true,
      sessionId: state.sessionId,
      hasAudio: !!state.metadata.hasAudio,
      audioDelayMs: state.audioDelayMs,
    };
  } catch (error) {
    await stopCapture({ notifyBridge: false });
    throw error;
  }
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (!message || typeof message !== "object" || message.target !== "offscreen") {
    return false;
  }

  (async () => {
    if (message?.type === "offscreen-start-audio-sync") {
      return startAudioSync(message);
    }
    if (message?.type === "offscreen-set-audio-delay-ms") {
      const audioDelayMs = applyAudioDelayMs(message?.audioDelayMs, { immediate: false });
      return { ok: true, audioDelayMs };
    }
    if (message?.type === "offscreen-stop-capture") {
      return stopCapture();
    }
    return { ok: false, error: `unknown offscreen message: ${message?.type}` };
  })()
    .then((payload) => sendResponse(payload))
    .catch((error) => sendResponse({ ok: false, error: error?.message || String(error) }));
  return true;
});
