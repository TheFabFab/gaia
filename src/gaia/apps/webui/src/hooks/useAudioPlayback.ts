// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

import { useRef, useState, useCallback } from 'react';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface UseAudioPlaybackResult {
    /**
     * Decodes and queues a WAV/PCM ArrayBuffer for sequential playback.
     *
     * @param data - Raw audio bytes returned by Lemonade TTS (WAV format).
     */
    queueAudio: (data: ArrayBuffer) => Promise<void>;
    /**
     * Stops the currently playing buffer immediately and discards the queue.
     */
    interrupt: () => void;
    /** True while at least one audio buffer is currently being played. */
    isPlaying: boolean;
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

/**
 * Manages an AudioContext, a decoded-buffer queue, and sequential playback.
 *
 * The AudioContext is created lazily on the first `queueAudio` call to comply
 * with browser autoplay policies that require a user gesture before any audio
 * context is created.
 *
 * Buffers are played one after another. When the current source ends, the
 * next queued buffer starts automatically. `interrupt()` stops the active
 * source immediately and clears the pending queue.
 *
 * @returns Controls and state for TTS audio playback.
 */
export function useAudioPlayback(): UseAudioPlaybackResult {
    const [isPlaying, setIsPlaying] = useState(false);

    const audioCtxRef = useRef<AudioContext | null>(null);
    /** Queue of decoded AudioBuffers waiting to be played. */
    const queueRef = useRef<AudioBuffer[]>([]);
    /** The currently playing BufferSourceNode, if any. */
    const activeSourceRef = useRef<AudioBufferSourceNode | null>(null);

    // ---------------------------------------------------------------------------
    // Internal helpers
    // ---------------------------------------------------------------------------

    /**
     * Gets or creates the shared AudioContext.
     * Resumes it if suspended (browser autoplay policy).
     */
    const getAudioContext = useCallback(async (): Promise<AudioContext> => {
        if (!audioCtxRef.current) {
            audioCtxRef.current = new AudioContext();
        }
        if (audioCtxRef.current.state === 'suspended') {
            await audioCtxRef.current.resume();
        }
        return audioCtxRef.current;
    }, []);

    /**
     * Pulls the next buffer from the queue and starts playing it.
     * Called when the previous source ends naturally.
     * Must only be invoked from within the hook (not re-entrant safe).
     */
    const playNext = useCallback((ctx: AudioContext) => {
        const nextBuffer = queueRef.current.shift();
        if (nextBuffer === undefined) {
            // Queue exhausted — mark as idle
            activeSourceRef.current = null;
            setIsPlaying(false);
            return;
        }

        const source = ctx.createBufferSource();
        source.buffer = nextBuffer;
        source.connect(ctx.destination);
        source.onended = () => {
            // Only chain to next if this source is still the active one
            // (interrupt() sets activeSourceRef.current to null)
            if (activeSourceRef.current === source) {
                activeSourceRef.current = null;
                playNext(ctx);
            }
        };
        activeSourceRef.current = source;
        source.start();
    }, []);

    // ---------------------------------------------------------------------------
    // Public API
    // ---------------------------------------------------------------------------

    /**
     * Decodes `data` and adds the resulting AudioBuffer to the playback queue.
     * If nothing is currently playing, starts playback immediately.
     *
     * @param data - WAV/PCM audio bytes from Lemonade TTS.
     */
    const queueAudio = useCallback(async (data: ArrayBuffer): Promise<void> => {
        const ctx = await getAudioContext();
        // decodeAudioData consumes the buffer; clone defensively
        const copy = data.slice(0);
        const decoded = await ctx.decodeAudioData(copy);

        queueRef.current.push(decoded);

        // Start playback if nothing is currently playing
        if (activeSourceRef.current === null) {
            setIsPlaying(true);
            playNext(ctx);
        }
    }, [getAudioContext, playNext]);

    /**
     * Stops the currently playing source immediately and empties the queue.
     */
    const interrupt = useCallback(() => {
        // Clear queue so playNext doesn't chain after stop
        queueRef.current = [];

        const source = activeSourceRef.current;
        if (source !== null) {
            // Prevent the onended handler from chaining to a new buffer
            activeSourceRef.current = null;
            try {
                source.stop();
            } catch {
                // stop() throws if the source hasn't started — safe to ignore
            }
        }

        setIsPlaying(false);
    }, []);

    return { queueAudio, interrupt, isPlaying };
}
