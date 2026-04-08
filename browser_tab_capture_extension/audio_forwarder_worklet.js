const PCM_CHUNK_FRAMES = 2048;

class HDRTVNetTabAudioForwarder extends AudioWorkletProcessor {
  constructor() {
    super();
    this._framesSent = 0;
    this._buffer = new Int16Array(PCM_CHUNK_FRAMES * 2);
    this._bufferedFrames = 0;
  }

  _clampPcmSample(value) {
    const v = Math.max(-1.0, Math.min(1.0, Number(value || 0)));
    if (v <= -1.0) {
      return -32768;
    }
    if (v >= 1.0) {
      return 32767;
    }
    return v < 0 ? Math.round(v * 32768.0) : Math.round(v * 32767.0);
  }

  _flushBuffered() {
    if (this._bufferedFrames <= 0) {
      return;
    }
    const samples = this._bufferedFrames * 2;
    const payload = this._buffer.slice(0, samples);
    const ptsMs = (this._framesSent / sampleRate) * 1000.0;
    this._framesSent += this._bufferedFrames;
    this._bufferedFrames = 0;
    this.port.postMessage(
      {
        type: "audio-chunk",
        ptsMs,
        sampleRate,
        channels: 2,
        buffer: payload.buffer,
      },
      [payload.buffer]
    );
  }

  process(inputs, outputs) {
    const input = inputs[0];
    const output = outputs[0];

    if (output) {
      for (let ch = 0; ch < output.length; ch += 1) {
        output[ch].fill(0);
      }
    }

    if (!input || input.length === 0 || !input[0] || input[0].length === 0) {
      return true;
    }

    const left = input[0];
    const right = input[1] || left;
    const frames = left.length;
    if (frames <= 0) {
      return true;
    }

    let srcIndex = 0;
    while (srcIndex < frames) {
      const space = PCM_CHUNK_FRAMES - this._bufferedFrames;
      const take = Math.min(space, frames - srcIndex);
      const dstBase = this._bufferedFrames * 2;
      for (let i = 0; i < take; i += 1) {
        const src = srcIndex + i;
        const dst = dstBase + (i * 2);
        this._buffer[dst + 0] = this._clampPcmSample(left[src]);
        this._buffer[dst + 1] = this._clampPcmSample(right[src]);
      }
      this._bufferedFrames += take;
      srcIndex += take;
      if (this._bufferedFrames >= PCM_CHUNK_FRAMES) {
        this._flushBuffered();
      }
    }
    return true;
  }
}

registerProcessor("hdrtvnet-tab-audio-forwarder", HDRTVNetTabAudioForwarder);
