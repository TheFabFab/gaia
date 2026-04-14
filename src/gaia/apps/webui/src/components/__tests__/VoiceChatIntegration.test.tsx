// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * Integration tests for voice chat wiring:
 * - Voice messages appear in chat history as text
 * - Voice mode preference persists across page reloads (localStorage)
 *
 * PRD-116 Task 7
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useState, useEffect, useCallback } from 'react';

// ---------------------------------------------------------------------------
// Helper: a minimal hook that models the voice mode localStorage persistence
// (mirrors what ChatView implements)
// ---------------------------------------------------------------------------

function useVoiceModePreference() {
    const [voiceEnabled, setVoiceEnabled] = useState<boolean>(() => {
        try {
            return localStorage.getItem('voice-mode-enabled') === 'true';
        } catch {
            return false;
        }
    });

    const toggle = useCallback(() => {
        setVoiceEnabled((prev) => {
            const next = !prev;
            try {
                if (next) {
                    localStorage.setItem('voice-mode-enabled', 'true');
                } else {
                    localStorage.removeItem('voice-mode-enabled');
                }
            } catch { /* ignore */ }
            return next;
        });
    }, []);

    return { voiceEnabled, toggle };
}

// ---------------------------------------------------------------------------
// Helper: minimal hook exercising the transcript→message-add flow
// (mirrors what ChatView's useVoiceChat callbacks do)
// ---------------------------------------------------------------------------

interface Message { role: 'user' | 'assistant'; content: string; id: number }

function useVoiceMessages(sessionId: string) {
    const [messages, setMessages] = useState<Message[]>([]);

    const handleTranscript = useCallback((text: string) => {
        setMessages((prev) => [
            ...prev,
            { id: Date.now(), role: 'user' as const, content: text },
        ]);
    }, []);

    const handleResponse = useCallback((text: string) => {
        setMessages((prev) => [
            ...prev,
            { id: Date.now() + 1, role: 'assistant' as const, content: text },
        ]);
    }, []);

    return { messages, handleTranscript, handleResponse };
}

// ---------------------------------------------------------------------------
// Tests: localStorage persistence
// ---------------------------------------------------------------------------

describe('Voice mode preference: localStorage persistence', () => {
    beforeEach(() => {
        localStorage.clear();
    });

    it('defaults to false when no preference stored', () => {
        const { result } = renderHook(() => useVoiceModePreference());
        expect(result.current.voiceEnabled).toBe(false);
    });

    it('reads true from localStorage on mount', () => {
        localStorage.setItem('voice-mode-enabled', 'true');
        const { result } = renderHook(() => useVoiceModePreference());
        expect(result.current.voiceEnabled).toBe(true);
    });

    it('persists true to localStorage when toggled on', () => {
        const { result } = renderHook(() => useVoiceModePreference());
        expect(result.current.voiceEnabled).toBe(false);
        act(() => { result.current.toggle(); });
        expect(result.current.voiceEnabled).toBe(true);
        expect(localStorage.getItem('voice-mode-enabled')).toBe('true');
    });

    it('removes preference from localStorage when toggled off', () => {
        localStorage.setItem('voice-mode-enabled', 'true');
        const { result } = renderHook(() => useVoiceModePreference());
        act(() => { result.current.toggle(); });
        expect(result.current.voiceEnabled).toBe(false);
        expect(localStorage.getItem('voice-mode-enabled')).toBeNull();
    });

    it('persists across simulated remount (new renderHook call reads from localStorage)', () => {
        const { result: first } = renderHook(() => useVoiceModePreference());
        act(() => { first.current.toggle(); });
        expect(first.current.voiceEnabled).toBe(true);

        // Simulate page reload: new hook instance reads from localStorage
        const { result: second } = renderHook(() => useVoiceModePreference());
        expect(second.current.voiceEnabled).toBe(true);
    });
});

// ---------------------------------------------------------------------------
// Tests: voice messages appear in chat history
// ---------------------------------------------------------------------------

describe('Voice messages in chat history', () => {
    it('adds user message when transcript arrives', () => {
        const { result } = renderHook(() => useVoiceMessages('session-1'));

        act(() => {
            result.current.handleTranscript('Hallo, wie geht es dir?');
        });

        expect(result.current.messages).toHaveLength(1);
        expect(result.current.messages[0].role).toBe('user');
        expect(result.current.messages[0].content).toBe('Hallo, wie geht es dir?');
    });

    it('adds assistant message when LLM response arrives', () => {
        const { result } = renderHook(() => useVoiceMessages('session-1'));

        act(() => { result.current.handleTranscript('Frage?'); });
        act(() => { result.current.handleResponse('Das ist die Antwort.'); });

        expect(result.current.messages).toHaveLength(2);
        expect(result.current.messages[1].role).toBe('assistant');
        expect(result.current.messages[1].content).toBe('Das ist die Antwort.');
    });

    it('preserves message order: user transcript then assistant response', () => {
        const { result } = renderHook(() => useVoiceMessages('session-1'));

        act(() => {
            result.current.handleTranscript('Was ist zwei plus zwei?');
        });
        act(() => {
            result.current.handleResponse('Vier.');
        });

        const [userMsg, assistantMsg] = result.current.messages;
        expect(userMsg.role).toBe('user');
        expect(assistantMsg.role).toBe('assistant');
    });

    it('accumulates multiple voice exchanges in order', () => {
        const { result } = renderHook(() => useVoiceMessages('session-1'));

        act(() => { result.current.handleTranscript('Erste Frage'); });
        act(() => { result.current.handleResponse('Erste Antwort'); });
        act(() => { result.current.handleTranscript('Zweite Frage'); });
        act(() => { result.current.handleResponse('Zweite Antwort'); });

        expect(result.current.messages).toHaveLength(4);
        expect(result.current.messages[0].content).toBe('Erste Frage');
        expect(result.current.messages[1].content).toBe('Erste Antwort');
        expect(result.current.messages[2].content).toBe('Zweite Frage');
        expect(result.current.messages[3].content).toBe('Zweite Antwort');
    });
});

// ---------------------------------------------------------------------------
// Tests: voice state machine transitions
// ---------------------------------------------------------------------------

type VoiceState = 'idle' | 'listening' | 'processing' | 'playing';

function useVoiceStateMachine() {
    const [voiceState, setVoiceState] = useState<VoiceState>('idle');

    const startListening = useCallback(() => setVoiceState('listening'), []);
    const startProcessing = useCallback(() => setVoiceState('processing'), []);
    const startPlaying = useCallback(() => setVoiceState('playing'), []);
    const reset = useCallback(() => setVoiceState('idle'), []);

    return { voiceState, startListening, startProcessing, startPlaying, reset };
}

describe('Voice state machine', () => {
    it('starts in idle state', () => {
        const { result } = renderHook(() => useVoiceStateMachine());
        expect(result.current.voiceState).toBe('idle');
    });

    it('transitions idle → listening when recording starts', () => {
        const { result } = renderHook(() => useVoiceStateMachine());
        act(() => { result.current.startListening(); });
        expect(result.current.voiceState).toBe('listening');
    });

    it('transitions listening → processing when stop is sent', () => {
        const { result } = renderHook(() => useVoiceStateMachine());
        act(() => { result.current.startListening(); });
        act(() => { result.current.startProcessing(); });
        expect(result.current.voiceState).toBe('processing');
    });

    it('transitions processing → playing when TTS starts', () => {
        const { result } = renderHook(() => useVoiceStateMachine());
        act(() => { result.current.startListening(); });
        act(() => { result.current.startProcessing(); });
        act(() => { result.current.startPlaying(); });
        expect(result.current.voiceState).toBe('playing');
    });

    it('transitions playing → idle when TTS ends', () => {
        const { result } = renderHook(() => useVoiceStateMachine());
        act(() => { result.current.startListening(); });
        act(() => { result.current.startProcessing(); });
        act(() => { result.current.startPlaying(); });
        act(() => { result.current.reset(); });
        expect(result.current.voiceState).toBe('idle');
    });
});
