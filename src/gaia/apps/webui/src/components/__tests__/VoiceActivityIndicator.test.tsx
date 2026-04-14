// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { VoiceActivityIndicator } from '../VoiceActivityIndicator';

describe('VoiceActivityIndicator', () => {
    it('renders animated waveform when active=true', () => {
        const { container } = render(<VoiceActivityIndicator active={true} />);
        const wrapper = container.querySelector('.voice-activity-indicator');
        expect(wrapper).not.toBeNull();
        // Should have waveform bars
        const bars = container.querySelectorAll('.voice-activity-indicator__bar');
        expect(bars.length).toBeGreaterThanOrEqual(3);
    });

    it('renders nothing when active=false', () => {
        const { container } = render(<VoiceActivityIndicator active={false} />);
        expect(container.firstChild).toBeNull();
    });
});
