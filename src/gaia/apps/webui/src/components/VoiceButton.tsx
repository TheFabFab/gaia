// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

import './VoiceButton.css';

/** Props for the VoiceButton component. */
interface VoiceButtonProps {
    /** Whether microphone is actively capturing audio. */
    isRecording: boolean;
    /** Whether captured audio is being processed server-side. */
    isProcessing: boolean;
    /** Whether voice mode is available (ASR model loaded). */
    voiceAvailable: boolean;
    /** Callback invoked when the button is clicked to toggle voice mode. */
    onToggle: () => void;
}

/**
 * Circular toggle button for voice interaction.
 *
 * - Idle: displays a microphone emoji icon.
 * - Recording (`isRecording=true`): displays a pulsing red dot.
 * - Processing (`isProcessing=true`): displays a spinning indicator.
 * - Disabled (`voiceAvailable=false`): button is disabled with a tooltip.
 */
export function VoiceButton({
    isRecording,
    isProcessing,
    voiceAvailable,
    onToggle,
}: VoiceButtonProps) {
    let extraClass = '';
    if (isRecording) extraClass = 'voice-button--recording';
    else if (isProcessing) extraClass = 'voice-button--processing';

    const title = voiceAvailable ? undefined : 'Voice mode requires ASR model';

    return (
        <button
            className={`voice-button ${extraClass}`.trim()}
            onClick={onToggle}
            disabled={!voiceAvailable}
            title={title}
            aria-label={
                isRecording
                    ? 'Stop recording'
                    : isProcessing
                      ? 'Processing voice'
                      : 'Start voice mode'
            }
            type="button"
        >
            {isProcessing ? (
                <span className="voice-button__spinner" aria-hidden="true" />
            ) : isRecording ? (
                <span className="voice-button__recording" aria-hidden="true" />
            ) : (
                <span className="voice-button__mic" aria-hidden="true">🎙️</span>
            )}
        </button>
    );
}
