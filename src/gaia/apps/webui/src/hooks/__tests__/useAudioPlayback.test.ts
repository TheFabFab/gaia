// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useAudioPlayback } from '../useAudioPlayback';

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

/** Represents a scheduled AudioBufferSourceNode */
interface MockSourceNode {
    buffer: AudioBuffer | null;
    onended: (() => void) | null;
    start: ReturnType<typeof vi.fn>;
    stop: ReturnType<typeof vi.fn>;
    connect: ReturnType<typeof vi.fn>;
    /** Test helper: simulate natural playback completion */
    simulateEnded: () => void;
}

function makeSourceNode(): MockSourceNode {
    const node: MockSourceNode = {
        buffer: null,
        onended: null,
        start: vi.fn(),
        stop: vi.fn(),
        connect: vi.fn(),
        simulateEnded: () => { node.onended?.(); },
    };
    return node;
}

/** Tracks created AudioContext instances */
let mockAudioContextInst: ReturnType<typeof makeAudioContext>;

function makeAudioContext(state: AudioContextState = 'running') {
    const sourceNodes: MockSourceNode[] = [];

    const ctx = {
        state,
        destination: {},
        resume: vi.fn().mockResolvedValue(undefined),
        close: vi.fn().mockResolvedValue(undefined),
        decodeAudioData: vi.fn().mockImplementation((buffer: ArrayBuffer) => {
            // Return a minimal AudioBuffer-like object
            const audioBuffer = {
                duration: 0.1,
                length: 160,
                numberOfChannels: 1,
                sampleRate: 24000,
            } as unknown as AudioBuffer;
            return Promise.resolve(audioBuffer);
        }),
        createBufferSource: vi.fn().mockImplementation(() => {
            const node = makeSourceNode();
            sourceNodes.push(node);
            return node;
        }),
        _sourceNodes: sourceNodes,
    };
    return ctx;
}

beforeEach(() => {
    mockAudioContextInst = makeAudioContext();

    // Mock AudioContext constructor — regular function for Reflect.construct compat (vitest 4.x)
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const AudioContextMock = vi.fn().mockImplementation(function(this: any) {
        return mockAudioContextInst;
    });
    vi.stubGlobal('AudioContext', AudioContextMock);
});

afterEach(() => {
    vi.unstubAllGlobals();
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('useAudioPlayback', () => {
    it('creates AudioContext for playback when queueAudio is first called', async () => {
        const { result } = renderHook(() => useAudioPlayback());

        const data = new ArrayBuffer(100);
        await act(async () => {
            await result.current.queueAudio(data);
        });

        // AudioContext should have been constructed
        expect(AudioContext).toHaveBeenCalledTimes(1);
    });

    it('queues audio buffers and plays sequentially', async () => {
        const { result } = renderHook(() => useAudioPlayback());

        const buf1 = new ArrayBuffer(100);
        const buf2 = new ArrayBuffer(200);

        await act(async () => {
            await result.current.queueAudio(buf1);
        });

        // First buffer should start playing
        expect(mockAudioContextInst.createBufferSource).toHaveBeenCalledTimes(1);
        const node1 = mockAudioContextInst._sourceNodes[0];
        expect(node1.start).toHaveBeenCalledTimes(1);

        // Queue second buffer while first is playing
        await act(async () => {
            await result.current.queueAudio(buf2);
        });

        // Second buffer should NOT start yet (first is still playing)
        expect(mockAudioContextInst.createBufferSource).toHaveBeenCalledTimes(1);

        // Simulate first buffer ending → second should auto-start
        act(() => {
            node1.simulateEnded();
        });

        expect(mockAudioContextInst.createBufferSource).toHaveBeenCalledTimes(2);
        const node2 = mockAudioContextInst._sourceNodes[1];
        expect(node2.start).toHaveBeenCalledTimes(1);
    });

    it('stops playback immediately on interrupt', async () => {
        const { result } = renderHook(() => useAudioPlayback());

        const buf1 = new ArrayBuffer(100);
        const buf2 = new ArrayBuffer(200);

        await act(async () => {
            await result.current.queueAudio(buf1);
            await result.current.queueAudio(buf2);
        });

        const node1 = mockAudioContextInst._sourceNodes[0];
        expect(node1.start).toHaveBeenCalledTimes(1);

        // Interrupt: current source should be stopped, queue cleared
        act(() => {
            result.current.interrupt();
        });

        expect(node1.stop).toHaveBeenCalledTimes(1);

        // After interrupt, even if old node fires onended, no new node should start
        act(() => {
            node1.simulateEnded();
        });

        // No second node should have been created after interrupt
        expect(mockAudioContextInst.createBufferSource).toHaveBeenCalledTimes(1);
    });

    it('handles empty audio queue gracefully (interrupt with nothing playing)', () => {
        const { result } = renderHook(() => useAudioPlayback());

        // Should not throw when interrupting with nothing queued
        expect(() => {
            act(() => {
                result.current.interrupt();
            });
        }).not.toThrow();
    });

    it('resumes suspended AudioContext before playing', async () => {
        // Override to return a suspended AudioContext
        mockAudioContextInst = makeAudioContext('suspended');
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        vi.stubGlobal('AudioContext', vi.fn().mockImplementation(function(this: any) {
            return mockAudioContextInst;
        }));

        const { result } = renderHook(() => useAudioPlayback());

        const data = new ArrayBuffer(100);
        await act(async () => {
            await result.current.queueAudio(data);
        });

        // resume() should have been called because state is 'suspended'
        expect(mockAudioContextInst.resume).toHaveBeenCalledTimes(1);
        // Playback should still start
        expect(mockAudioContextInst.createBufferSource).toHaveBeenCalledTimes(1);
    });
});
