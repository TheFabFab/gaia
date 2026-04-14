// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

import { describe, it, expect } from 'vitest';
import { render } from '@testing-library/react';
import { AudioVisualizer } from '../AudioVisualizer';

describe('AudioVisualizer', () => {
    it('renders level meter during playback', () => {
        const { container } = render(
            <AudioVisualizer isPlaying={true} audioLevel={0.7} />
        );
        const visualizer = container.querySelector('.audio-visualizer');
        expect(visualizer).not.toBeNull();
        const levelBar = container.querySelector('.audio-visualizer__level');
        expect(levelBar).not.toBeNull();
    });

    it('hidden when not playing', () => {
        const { container } = render(
            <AudioVisualizer isPlaying={false} audioLevel={0} />
        );
        expect(container.firstChild).toBeNull();
    });
});
