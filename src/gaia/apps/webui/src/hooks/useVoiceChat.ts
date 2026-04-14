// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

import { useState, useRef, useEffect, useCallback } from 'react';
import { useAudioPlayback } from './useAudioPlayback';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const WS_PATH = '/api/voice/stream';
const BACKOFF_INITIAL_MS = 1000;
const BACKOFF_MAX_MS = 30_000;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface VoiceCapabilities {
    asr: {
        available: boolean;
        defaultModel: string | null;
        websocketPort: number | null;
    };
    tts: {
        available: boolean;
    };
}

export interface UseVoiceChatOptions {
    /** Called when a transcript event arrives from the server. */
    onTranscript?: (text: string) => void;
    /** Called when an LLM response event arrives from the server. */
    onResponse?: (text: string) => void;
    /** Called when TTS audio is about to start. */
    onTtsStart?: () => void;
    /** Called when TTS audio has finished. */
    onTtsEnd?: () => void;
    /** Called with each binary TTS audio frame. */
    onTtsAudio?: (data: ArrayBuffer) => void;
    /** Called when the server sends an error message. */
    onError?: (msg: string) => void;
}

export interface UseVoiceChatResult {
    /** Latest voice capabilities reported by the server, or null before first status. */
    capabilities: VoiceCapabilities | null;
    /** True while the WebSocket connection is open. */
    isConnected: boolean;
    /**
     * Sends a `start` message to the server to begin a voice session.
     *
     * @param sessionId - The chat session ID to associate with this voice session.
     * @param model - Optional ASR model name override.
     */
    start: (sessionId: string, model?: string) => void;
    /** Sends a `stop` message and closes the WebSocket cleanly. */
    stop: () => void;
    /**
     * Forwards a raw PCM ArrayBuffer to the server as a binary WebSocket frame.
     *
     * @param buffer - PCM audio data from the AudioWorklet.
     */
    sendAudio: (buffer: ArrayBuffer) => void;
    /** Sends an `interrupt` message to cancel ongoing TTS playback. */
    interrupt: () => void;
    /** Non-null when a WebSocket error has occurred. */
    error: string | null;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Determine the WebSocket URL relative to the current page origin. */
function resolveWsUrl(): string {
    if (typeof window === 'undefined') return `ws://localhost:4200${WS_PATH}`;
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.protocol === 'file:'
        ? 'localhost:4200'
        : window.location.host;
    return `${proto}//${host}${WS_PATH}`;
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

/**
 * Manages a WebSocket connection to the GAIA voice streaming endpoint.
 *
 * Connects immediately on mount, reconnects with exponential backoff on
 * abnormal close, and exposes controls for starting/stopping voice sessions
 * and sending PCM audio frames.
 *
 * @param options - Callbacks for transcript, response, TTS, and error events.
 * @returns State and controls for the voice WebSocket session.
 */
export function useVoiceChat(options: UseVoiceChatOptions): UseVoiceChatResult {
    const {
        onTranscript,
        onResponse,
        onTtsStart,
        onTtsEnd,
        onTtsAudio,
        onError,
    } = options;

    const [capabilities, setCapabilities] = useState<VoiceCapabilities | null>(null);
    const [isConnected, setIsConnected] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const audioPlayback = useAudioPlayback();

    const wsRef = useRef<WebSocket | null>(null);
    const backoffRef = useRef(BACKOFF_INITIAL_MS);
    const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    // When true, onclose should not trigger a reconnect attempt.
    const intentionalCloseRef = useRef(false);

    // Stable refs for callbacks to avoid re-running the effect on every render.
    const onTranscriptRef = useRef(onTranscript);
    const onResponseRef = useRef(onResponse);
    const onTtsStartRef = useRef(onTtsStart);
    const onTtsEndRef = useRef(onTtsEnd);
    const onTtsAudioRef = useRef(onTtsAudio);
    const onErrorRef = useRef(onError);

    useEffect(() => { onTranscriptRef.current = onTranscript; }, [onTranscript]);
    useEffect(() => { onResponseRef.current = onResponse; }, [onResponse]);
    useEffect(() => { onTtsStartRef.current = onTtsStart; }, [onTtsStart]);
    useEffect(() => { onTtsEndRef.current = onTtsEnd; }, [onTtsEnd]);
    useEffect(() => { onTtsAudioRef.current = onTtsAudio; }, [onTtsAudio]);
    useEffect(() => { onErrorRef.current = onError; }, [onError]);

    const connect = useCallback(() => {
        const url = resolveWsUrl();
        const ws = new WebSocket(url);
        wsRef.current = ws;

        ws.onopen = () => {
            setIsConnected(true);
            setError(null);
            backoffRef.current = BACKOFF_INITIAL_MS; // Reset backoff on successful connect
        };

        ws.onclose = (ev) => {
            setIsConnected(false);
            wsRef.current = null;

            // Only reconnect on abnormal close and when not stopped intentionally
            if (!intentionalCloseRef.current && !ev.wasClean) {
                const delay = backoffRef.current;
                backoffRef.current = Math.min(delay * 2, BACKOFF_MAX_MS);
                reconnectTimerRef.current = setTimeout(() => {
                    reconnectTimerRef.current = null;
                    connect();
                }, delay);
            }
        };

        ws.onerror = () => {
            setError('WebSocket connection error');
            onErrorRef.current?.('WebSocket connection error');
        };

        ws.onmessage = (ev: MessageEvent) => {
            // Binary frame → TTS audio
            if (ev.data instanceof ArrayBuffer) {
                onTtsAudioRef.current?.(ev.data);
                void audioPlayback.queueAudio(ev.data);
                return;
            }
            if (typeof ev.data !== 'string') return;

            let msg: Record<string, unknown>;
            try {
                msg = JSON.parse(ev.data) as Record<string, unknown>;
            } catch {
                return;
            }

            const type = msg['type'];
            switch (type) {
                case 'status': {
                    const asr = msg['asr'] as { available: boolean; defaultModel: string | null; websocketPort: number | null } | undefined;
                    const tts = msg['tts'] as { available: boolean } | undefined;
                    if (asr !== undefined && tts !== undefined) {
                        setCapabilities({
                            asr: {
                                available: asr.available,
                                defaultModel: asr.defaultModel ?? null,
                                websocketPort: asr.websocketPort ?? null,
                            },
                            tts: { available: tts.available },
                        });
                    }
                    break;
                }
                case 'transcript':
                    onTranscriptRef.current?.(String(msg['text'] ?? ''));
                    break;
                case 'response':
                    onResponseRef.current?.(String(msg['text'] ?? ''));
                    break;
                case 'tts_start':
                    onTtsStartRef.current?.();
                    break;
                case 'tts_end':
                    onTtsEndRef.current?.();
                    break;
                case 'error':
                    setError(String(msg['message'] ?? 'Unknown server error'));
                    onErrorRef.current?.(String(msg['message'] ?? 'Unknown server error'));
                    break;
                default:
                    break;
            }
        };
    }, []); // eslint-disable-line react-hooks/exhaustive-deps

    // Connect on mount, disconnect on unmount.
    useEffect(() => {
        intentionalCloseRef.current = false;
        connect();

        return () => {
            intentionalCloseRef.current = true;
            if (reconnectTimerRef.current !== null) {
                clearTimeout(reconnectTimerRef.current);
                reconnectTimerRef.current = null;
            }
            wsRef.current?.close();
            wsRef.current = null;
        };
    }, [connect]);

    // ---------------------------------------------------------------------------
    // Public API
    // ---------------------------------------------------------------------------

    /**
     * Sends a `start` message to the server to begin ASR for a session.
     *
     * @param sessionId - Chat session ID to associate with the voice session.
     * @param model - Optional ASR model name override.
     */
    const start = useCallback((sessionId: string, model?: string) => {
        wsRef.current?.send(JSON.stringify({ type: 'start', sessionId, model }));
    }, []);

    /**
     * Sends a `stop` message and closes the WebSocket without triggering
     * a reconnect attempt.
     */
    const stop = useCallback(() => {
        intentionalCloseRef.current = true;
        wsRef.current?.send(JSON.stringify({ type: 'stop' }));
        wsRef.current?.close();
    }, []);

    /**
     * Forwards raw PCM audio data to the server as a binary WebSocket frame.
     *
     * @param buffer - ArrayBuffer containing PCM audio from the AudioWorklet.
     */
    const sendAudio = useCallback((buffer: ArrayBuffer) => {
        wsRef.current?.send(buffer);
    }, []);

    /**
     * Stops local TTS playback immediately and sends an `interrupt` message
     * to cancel ongoing TTS streaming on the server.
     */
    const interrupt = useCallback(() => {
        audioPlayback.interrupt();
        wsRef.current?.send(JSON.stringify({ type: 'interrupt' }));
    }, [audioPlayback]);

    return { capabilities, isConnected, start, stop, sendAudio, interrupt, error };
}
