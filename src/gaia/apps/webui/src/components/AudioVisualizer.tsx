// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

import './AudioVisualizer.css';

/** Props for the AudioVisualizer component. */
interface AudioVisualizerProps {
    /** Whether TTS audio is currently playing. */
    isPlaying: boolean;
    /** Normalised audio level in the range 0–1. Defaults to 0. */
    audioLevel?: number;
}

/**
 * Level-meter visualizer shown during TTS audio playback.
 *
 * Renders a horizontal progress bar whose width reflects `audioLevel` when
 * `isPlaying=true`. Renders `null` when `isPlaying=false`.
 *
 * @param isPlaying - Whether audio is currently playing.
 * @param audioLevel - Normalised audio amplitude (0–1). Defaults to `0`.
 */
export function AudioVisualizer({ isPlaying, audioLevel = 0 }: AudioVisualizerProps) {
    if (!isPlaying) return null;

    const clampedLevel = Math.max(0, Math.min(1, audioLevel));
    const widthPct = `${Math.round(clampedLevel * 100)}%`;

    return (
        <div className="audio-visualizer" aria-label="Audio playing" role="status">
            <div className="audio-visualizer__track">
                <div
                    className="audio-visualizer__level"
                    style={{ width: widthPct }}
                    aria-valuenow={Math.round(clampedLevel * 100)}
                    aria-valuemin={0}
                    aria-valuemax={100}
                    role="progressbar"
                />
            </div>
            <span className="audio-visualizer__label" aria-hidden="true">▶</span>
        </div>
    );
}
