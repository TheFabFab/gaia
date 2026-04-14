// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useAudioCapture } from '../useAudioCapture';

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

/** Minimal AudioWorkletNode mock */
function makeWorkletNode() {
    const port = {
        onmessage: null as ((e: MessageEvent) => void) | null,
        postMessage: vi.fn(),
    };
    return {
        port,
        connect: vi.fn(),
        disconnect: vi.fn(),
    };
}

/** Minimal MediaStreamAudioSourceNode mock */
function makeSourceNode() {
    return { connect: vi.fn(), disconnect: vi.fn() };
}

/** Build a mock AudioContext */
function makeAudioContext(sampleRate = 16000) {
    const workletNode = makeWorkletNode();
    const sourceNode = makeSourceNode();
    return {
        sampleRate,
        state: 'running',
        close: vi.fn().mockResolvedValue(undefined),
        audioWorklet: {
            addModule: vi.fn().mockResolvedValue(undefined),
        },
        createMediaStreamSource: vi.fn().mockReturnValue(sourceNode),
        _workletNode: workletNode,
        _sourceNode: sourceNode,
        // AudioWorkletNode constructor is called with (ctx, name) — we return our mock
        __workletNodeFactory: vi.fn().mockReturnValue(workletNode),
    };
}

/** Build a mock MediaStream with one audio track */
function makeMediaStream() {
    const track = { stop: vi.fn(), kind: 'audio', enabled: true };
    return {
        getTracks: vi.fn().mockReturnValue([track]),
        _track: track,
    };
}

let mockAudioContextInst: ReturnType<typeof makeAudioContext>;
let mockMediaStream: ReturnType<typeof makeMediaStream>;
let MockAudioWorkletNode: ReturnType<typeof vi.fn>;

beforeEach(() => {
    mockMediaStream = makeMediaStream();
    mockAudioContextInst = makeAudioContext();

    // Mock navigator.mediaDevices.getUserMedia
    vi.stubGlobal('navigator', {
        mediaDevices: {
            getUserMedia: vi.fn().mockResolvedValue(mockMediaStream),
        },
    });

    // Mock AudioContext constructor
    // Uses a regular function (not arrow) so Reflect.construct works in vitest 4.x.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const AudioContextMock = vi.fn().mockImplementation(function(this: any) { return mockAudioContextInst; });
    vi.stubGlobal('AudioContext', AudioContextMock);

    // Mock AudioWorkletNode constructor — must be global because useAudioCapture news it up
    // Uses a regular function (not arrow) so Reflect.construct works in vitest 4.x.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    MockAudioWorkletNode = vi.fn().mockImplementation(function(this: any) { return mockAudioContextInst._workletNode; });
    vi.stubGlobal('AudioWorkletNode', MockAudioWorkletNode);

    // Mock URL.createObjectURL (needed for worklet module URL)
    vi.stubGlobal('URL', {
        createObjectURL: vi.fn().mockReturnValue('blob:mock-url'),
    });
});

afterEach(() => {
    vi.unstubAllGlobals();
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('useAudioCapture', () => {
    it('requests microphone permission via navigator.mediaDevices.getUserMedia', async () => {
        const onAudioData = vi.fn();
        const { result } = renderHook(() => useAudioCapture({ onAudioData }));

        await act(async () => {
            await result.current.start();
        });

        expect(navigator.mediaDevices.getUserMedia).toHaveBeenCalledWith({ audio: true });
    });

    it('creates AudioContext at 16kHz sample rate', async () => {
        const onAudioData = vi.fn();
        const { result } = renderHook(() => useAudioCapture({ onAudioData }));

        await act(async () => {
            await result.current.start();
        });

        // AudioContext should have been constructed with sampleRate 16000
        expect(AudioContext).toHaveBeenCalledWith({ sampleRate: 16000 });
    });

    it('registers AudioWorkletProcessor for PCM capture', async () => {
        const onAudioData = vi.fn();
        const { result } = renderHook(() => useAudioCapture({ onAudioData }));

        await act(async () => {
            await result.current.start();
        });

        // addModule should have been called on the AudioWorklet
        expect(mockAudioContextInst.audioWorklet.addModule).toHaveBeenCalled();
        // AudioWorkletNode constructor should have been called with the processor name
        expect(AudioWorkletNode).toHaveBeenCalledWith(
            mockAudioContextInst,
            'pcm-capture-processor'
        );
    });

    it('calls onAudioData callback with Float32Array chunks', async () => {
        const onAudioData = vi.fn();
        const { result } = renderHook(() => useAudioCapture({ onAudioData }));

        await act(async () => {
            await result.current.start();
        });

        // Simulate a message from the AudioWorklet port
        const chunk = new Float32Array([0.1, -0.2, 0.3]);
        const port = mockAudioContextInst._workletNode.port;
        act(() => {
            if (port.onmessage) {
                port.onmessage(new MessageEvent('message', { data: chunk }));
            }
        });

        expect(onAudioData).toHaveBeenCalledWith(chunk);
    });

    it('stops MediaStream tracks on cleanup (stop)', async () => {
        const onAudioData = vi.fn();
        const { result } = renderHook(() => useAudioCapture({ onAudioData }));

        await act(async () => {
            await result.current.start();
        });

        act(() => {
            result.current.stop();
        });

        expect(mockMediaStream._track.stop).toHaveBeenCalled();
    });

    it('reports permission denied error when getUserMedia rejects', async () => {
        // Override getUserMedia to throw PermissionDeniedError
        vi.stubGlobal('navigator', {
            mediaDevices: {
                getUserMedia: vi.fn().mockRejectedValue(
                    Object.assign(new Error('Permission denied'), { name: 'NotAllowedError' })
                ),
            },
        });

        const onAudioData = vi.fn();
        const { result } = renderHook(() => useAudioCapture({ onAudioData }));

        await act(async () => {
            await result.current.start();
        });

        expect(result.current.error).toBeTruthy();
        expect(result.current.error?.message).toMatch(/Permission denied/i);
    });
});
