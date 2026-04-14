// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

// This file runs in the AudioWorklet global scope (audio thread), not the main
// thread. It is loaded via URL.createObjectURL from the useAudioCapture hook.

// ---------------------------------------------------------------------------
// Global type declarations for the AudioWorklet scope
// (Not available in standard DOM lib for the main thread)
// ---------------------------------------------------------------------------

/* eslint-disable @typescript-eslint/no-explicit-any */
declare const sampleRate: number;

/** Minimal AudioWorkletProcessor interface for the processor global scope. */
declare abstract class AudioWorkletProcessor {
    readonly port: MessagePort;
    abstract process(
        inputs: Float32Array[][],
        outputs: Float32Array[][],
        parameters: Record<string, Float32Array>,
    ): boolean;
}

declare function registerProcessor(name: string, ctor: new (...args: any[]) => AudioWorkletProcessor): void;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const TARGET_SAMPLE_RATE = 16000;

// ---------------------------------------------------------------------------
// PcmCaptureProcessor
//
// Captures audio from the microphone at the AudioContext sample rate and
// downsamples to TARGET_SAMPLE_RATE (16kHz) using linear interpolation.
// Posts Float32Array chunks back to the main thread via this.port.
// ---------------------------------------------------------------------------

class PcmCaptureProcessor extends AudioWorkletProcessor {
    /**
     * Called for each 128-sample render quantum by the audio engine.
     * Downsamples the first channel of the first input to 16 kHz and
     * posts the result to the main thread.
     */
    override process(inputs: Float32Array[][]): boolean {
        const channelData = inputs[0]?.[0];
        if (!channelData || channelData.length === 0) {
            return true;
        }

        if (sampleRate === TARGET_SAMPLE_RATE) {
            // Already at target rate — post directly (no copy needed)
            this.port.postMessage(channelData);
        } else {
            // Linear interpolation downsampling
            const ratio = sampleRate / TARGET_SAMPLE_RATE;
            const outputLength = Math.floor(channelData.length / ratio);
            const output = new Float32Array(outputLength);

            for (let i = 0; i < outputLength; i++) {
                const srcIndex = i * ratio;
                const lo = Math.floor(srcIndex);
                const hi = Math.min(lo + 1, channelData.length - 1);
                const frac = srcIndex - lo;
                output[i] = channelData[lo]! * (1 - frac) + channelData[hi]! * frac;
            }

            this.port.postMessage(output);
        }

        return true; // Keep processor alive
    }
}

registerProcessor('pcm-capture-processor', PcmCaptureProcessor);
