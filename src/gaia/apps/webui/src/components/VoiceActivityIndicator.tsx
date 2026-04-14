// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

import './VoiceActivityIndicator.css';

/** Props for the VoiceActivityIndicator component. */
interface VoiceActivityIndicatorProps {
    /** Whether voice activity (microphone input) is currently detected. */
    active: boolean;
}

/**
 * Animated waveform indicator shown while the microphone is active.
 *
 * Renders five bars with staggered CSS animations when `active=true`.
 * Renders `null` when `active=false`.
 */
export function VoiceActivityIndicator({ active }: VoiceActivityIndicatorProps) {
    if (!active) return null;

    return (
        <div className="voice-activity-indicator" aria-label="Voice activity" role="status">
            <span className="voice-activity-indicator__bar" aria-hidden="true" />
            <span className="voice-activity-indicator__bar" aria-hidden="true" />
            <span className="voice-activity-indicator__bar" aria-hidden="true" />
            <span className="voice-activity-indicator__bar" aria-hidden="true" />
            <span className="voice-activity-indicator__bar" aria-hidden="true" />
        </div>
    );
}
