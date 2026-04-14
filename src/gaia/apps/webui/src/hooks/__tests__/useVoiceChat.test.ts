// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useVoiceChat } from '../useVoiceChat';

// ---------------------------------------------------------------------------
// WebSocket mock factory
// ---------------------------------------------------------------------------

/** Tracks all created WebSocket instances so tests can control them */
const wsInstances: MockWebSocket[] = [];

class MockWebSocket {
    url: string;
    readyState: number = WebSocket.CONNECTING;
    onopen: ((e: Event) => void) | null = null;
    onclose: ((e: CloseEvent) => void) | null = null;
    onerror: ((e: Event) => void) | null = null;
    onmessage: ((e: MessageEvent) => void) | null = null;
    send = vi.fn();
    close = vi.fn().mockImplementation(() => {
        this.readyState = WebSocket.CLOSED;
    });

    static CONNECTING = 0;
    static OPEN = 1;
    static CLOSING = 2;
    static CLOSED = 3;

    constructor(url: string) {
        this.url = url;
        wsInstances.push(this);
    }

    /** Helper: trigger open event */
    simulateOpen() {
        this.readyState = WebSocket.OPEN;
        this.onopen?.(new Event('open'));
    }

    /** Helper: trigger a JSON message from server */
    simulateMessage(data: unknown) {
        this.onmessage?.(
            new MessageEvent('message', { data: JSON.stringify(data) })
        );
    }

    /** Helper: trigger a binary message */
    simulateBinaryMessage(data: ArrayBuffer) {
        this.onmessage?.(new MessageEvent('message', { data }));
    }

    /** Helper: trigger close event */
    simulateClose(code = 1000, reason = '') {
        this.readyState = WebSocket.CLOSED;
        this.onclose?.(new CloseEvent('close', { code, reason, wasClean: code === 1000 }));
    }

    /** Helper: trigger error */
    simulateError() {
        this.onerror?.(new Event('error'));
    }
}

beforeEach(() => {
    wsInstances.length = 0;
    vi.stubGlobal('WebSocket', MockWebSocket);
    vi.useFakeTimers();

    // AudioContext mock — needed because useVoiceChat now calls useAudioPlayback
    // which lazily constructs an AudioContext when binary frames arrive.
    const mockAudioBuffer = { duration: 0.1, length: 160, numberOfChannels: 1, sampleRate: 24000 };
    const mockSourceNode = {
        buffer: null as AudioBuffer | null,
        onended: null as (() => void) | null,
        start: vi.fn(),
        stop: vi.fn(),
        connect: vi.fn(),
    };
    const mockAudioCtx = {
        state: 'running' as AudioContextState,
        destination: {},
        resume: vi.fn().mockResolvedValue(undefined),
        decodeAudioData: vi.fn().mockResolvedValue(mockAudioBuffer),
        createBufferSource: vi.fn().mockReturnValue(mockSourceNode),
    };
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    vi.stubGlobal('AudioContext', vi.fn().mockImplementation(function(this: any) { return mockAudioCtx; }));
});

afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
});

// ---------------------------------------------------------------------------
// Helper: get the latest WebSocket instance
// ---------------------------------------------------------------------------
function latestWs(): MockWebSocket {
    const ws = wsInstances[wsInstances.length - 1];
    if (!ws) throw new Error('No WebSocket instance created');
    return ws;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('useVoiceChat', () => {
    it('establishes WebSocket connection to /api/voice/stream', async () => {
        const { result } = renderHook(() => useVoiceChat({}));

        // useVoiceChat should open the WebSocket immediately on mount
        expect(wsInstances.length).toBeGreaterThan(0);
        expect(latestWs().url).toContain('/api/voice/stream');
    });

    it('parses JSON status message and updates capability state', async () => {
        const { result } = renderHook(() => useVoiceChat({}));

        act(() => {
            const ws = latestWs();
            ws.simulateOpen();
            ws.simulateMessage({
                type: 'status',
                asr: { available: true, models: [{ name: 'Whisper-Small', recipe: 'whispercpp' }], defaultModel: 'Whisper-Small', websocketPort: 8765 },
                tts: { available: false, models: [], defaultModel: null },
            });
        });

        expect(result.current.capabilities?.asr.available).toBe(true);
        expect(result.current.capabilities?.tts.available).toBe(false);
    });

    it('sends start message with session ID when start() is called', async () => {
        const { result } = renderHook(() => useVoiceChat({}));

        act(() => {
            latestWs().simulateOpen();
        });

        act(() => {
            result.current.start('test-session-123');
        });

        expect(latestWs().send).toHaveBeenCalledWith(
            JSON.stringify({ type: 'start', sessionId: 'test-session-123', model: undefined })
        );
    });

    it('forwards PCM ArrayBuffers to WebSocket as binary', async () => {
        const { result } = renderHook(() => useVoiceChat({}));

        act(() => {
            latestWs().simulateOpen();
        });

        const buffer = new Float32Array([0.1, -0.2]).buffer;
        act(() => {
            result.current.sendAudio(buffer);
        });

        expect(latestWs().send).toHaveBeenCalledWith(buffer);
    });

    it('sends stop message on stop()', async () => {
        const { result } = renderHook(() => useVoiceChat({}));

        act(() => {
            latestWs().simulateOpen();
        });

        act(() => {
            result.current.stop();
        });

        expect(latestWs().send).toHaveBeenCalledWith(
            JSON.stringify({ type: 'stop' })
        );
    });

    it('reconnects on WebSocket close with exponential backoff', async () => {
        renderHook(() => useVoiceChat({}));
        expect(wsInstances.length).toBe(1);

        // Simulate open then unexpected close (not clean, code 1006)
        act(() => {
            const ws = latestWs();
            ws.simulateOpen();
            ws.simulateClose(1006, 'abnormal');
        });

        // Should not reconnect immediately — must wait for backoff
        expect(wsInstances.length).toBe(1);

        // Advance timers by the initial backoff (1 second)
        act(() => {
            vi.advanceTimersByTime(1100);
        });

        expect(wsInstances.length).toBe(2);
        expect(latestWs().url).toContain('/api/voice/stream');
    });

    it('reports error state on WebSocket error', async () => {
        const { result } = renderHook(() => useVoiceChat({}));

        act(() => {
            latestWs().simulateOpen();
            latestWs().simulateError();
        });

        expect(result.current.error).toBeTruthy();
    });

    it('parses transcript messages and calls onTranscript callback', async () => {
        const onTranscript = vi.fn();
        const { result } = renderHook(() => useVoiceChat({ onTranscript }));

        act(() => {
            latestWs().simulateOpen();
            latestWs().simulateMessage({ type: 'transcript', text: 'Hello world' });
        });

        expect(onTranscript).toHaveBeenCalledWith('Hello world');
        // result.current should still be connected
        expect(result.current.isConnected).toBe(true);
    });

    it('parses response messages and calls onResponse callback', async () => {
        const onResponse = vi.fn();
        const { result } = renderHook(() => useVoiceChat({ onResponse }));

        act(() => {
            latestWs().simulateOpen();
            latestWs().simulateMessage({ type: 'response', text: 'Hi there!' });
        });

        expect(onResponse).toHaveBeenCalledWith('Hi there!');
        expect(result.current.isConnected).toBe(true);
    });

    it('routes binary WebSocket frames to onTtsAudio callback', async () => {
        const onTtsAudio = vi.fn();
        renderHook(() => useVoiceChat({ onTtsAudio }));

        const audioData = new ArrayBuffer(512);
        await act(async () => {
            latestWs().simulateOpen();
            latestWs().simulateBinaryMessage(audioData);
        });

        expect(onTtsAudio).toHaveBeenCalledWith(audioData);
    });

    it('sends interrupt JSON message when interrupt() is called', async () => {
        const { result } = renderHook(() => useVoiceChat({}));

        act(() => {
            latestWs().simulateOpen();
        });

        act(() => {
            result.current.interrupt();
        });

        expect(latestWs().send).toHaveBeenCalledWith(
            JSON.stringify({ type: 'interrupt' })
        );
    });
});
