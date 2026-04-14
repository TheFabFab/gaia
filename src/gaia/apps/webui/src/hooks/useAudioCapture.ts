// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

import { useState, useRef, useCallback } from 'react';
// Import AudioWorklet processor source as raw text to be loaded as a Blob URL.
// Vite resolves `?raw` imports to the file contents at build/test time.
import workletSrc from '../services/audioWorklet.ts?raw';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface UseAudioCaptureOptions {
    /** Called for each PCM chunk posted from the AudioWorklet. */
    onAudioData: (chunk: Float32Array) => void;
}

export interface UseAudioCaptureResult {
    /** Start microphone capture. Requests user permission on first call. */
    start: () => Promise<void>;
    /** Stop capture and release all browser resources. */
    stop: () => void;
    /** True while audio is being captured. */
    isCapturing: boolean;
    /** Non-null when the last `start()` call failed (e.g. permission denied). */
    error: Error | null;
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

/**
 * Manages microphone permission, AudioContext creation, and AudioWorklet
 * lifecycle for real-time PCM audio capture at 16 kHz.
 *
 * @param options - Configuration including the `onAudioData` callback.
 * @returns Controls and state for audio capture.
 */
export function useAudioCapture({ onAudioData }: UseAudioCaptureOptions): UseAudioCaptureResult {
    const [isCapturing, setIsCapturing] = useState(false);
    const [error, setError] = useState<Error | null>(null);

    const audioContextRef = useRef<AudioContext | null>(null);
    const workletNodeRef = useRef<AudioWorkletNode | null>(null);
    const sourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null);
    const streamRef = useRef<MediaStream | null>(null);

    /**
     * Requests microphone access, creates a 16 kHz AudioContext, loads the
     * PCM capture AudioWorklet, and begins forwarding audio chunks via
     * `onAudioData`.
     *
     * @throws Never — errors are stored in the `error` state instead.
     */
    const start = useCallback(async () => {
        setError(null);
        try {
            // 1. Request microphone permission
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            streamRef.current = stream;

            // 2. Create AudioContext at target sample rate
            const ctx = new AudioContext({ sampleRate: 16000 });
            audioContextRef.current = ctx;

            // 3. Load AudioWorklet processor from a Blob URL
            const blob = new Blob([workletSrc], { type: 'application/javascript' });
            const moduleUrl = URL.createObjectURL(blob);
            await ctx.audioWorklet.addModule(moduleUrl);

            // 4. Create AudioWorkletNode for the PCM processor
            const workletNode = new AudioWorkletNode(ctx, 'pcm-capture-processor');
            workletNodeRef.current = workletNode;

            // 5. Forward audio chunks to caller
            workletNode.port.onmessage = (e: MessageEvent<Float32Array>) => {
                onAudioData(e.data);
            };

            // 6. Connect microphone → worklet node
            const source = ctx.createMediaStreamSource(stream);
            sourceNodeRef.current = source;
            source.connect(workletNode);

            setIsCapturing(true);
        } catch (err) {
            const wrapped = err instanceof Error ? err : new Error(String(err));
            setError(wrapped);
        }
    }, [onAudioData]);

    /**
     * Disconnects the audio graph, stops all MediaStream tracks, and closes
     * the AudioContext.
     */
    const stop = useCallback(() => {
        // Stop all microphone tracks
        streamRef.current?.getTracks().forEach((t) => t.stop());
        streamRef.current = null;

        // Disconnect audio graph
        sourceNodeRef.current?.disconnect();
        sourceNodeRef.current = null;
        workletNodeRef.current?.disconnect();
        workletNodeRef.current = null;

        // Close AudioContext
        void audioContextRef.current?.close();
        audioContextRef.current = null;

        setIsCapturing(false);
    }, []);

    return { start, stop, isCapturing, error };
}
