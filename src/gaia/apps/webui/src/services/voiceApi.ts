// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * REST client for the GAIA voice capability status endpoint.
 */

const API_BASE =
    window.location.protocol === 'file:'
        ? 'http://localhost:4200/api'
        : '/api';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface VoiceModelInfo {
    /** Display name of the model (e.g. "Whisper-Small"). */
    name: string;
    /** Model recipe identifier (e.g. "whispercpp"). */
    recipe: string;
}

export interface AsrStatus {
    /** Whether ASR (speech-to-text) is available. */
    available: boolean;
    /** All loaded ASR models. */
    models: VoiceModelInfo[];
    /** Name of the default ASR model, or null if none. */
    defaultModel: string | null;
    /** Lemonade realtime WebSocket port, or null if unavailable. */
    websocketPort: number | null;
}

export interface TtsStatus {
    /** Whether TTS (text-to-speech) is available. */
    available: boolean;
    /** All loaded TTS models. */
    models: VoiceModelInfo[];
    /** Name of the default TTS model, or null if none. */
    defaultModel: string | null;
}

export interface VoiceStatusResponse {
    asr: AsrStatus;
    tts: TtsStatus;
}

// ---------------------------------------------------------------------------
// API call
// ---------------------------------------------------------------------------

/**
 * Fetches voice capability status from the GAIA backend.
 *
 * @returns A promise resolving to ASR and TTS availability information.
 * @throws An error when the HTTP request fails.
 */
export async function fetchVoiceStatus(): Promise<VoiceStatusResponse> {
    const res = await fetch(`${API_BASE}/voice/status`);
    if (!res.ok) {
        throw new Error(`HTTP ${res.status}: failed to fetch voice status`);
    }
    return res.json() as Promise<VoiceStatusResponse>;
}
