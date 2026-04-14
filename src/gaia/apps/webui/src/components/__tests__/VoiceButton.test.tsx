// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { VoiceButton } from '../VoiceButton';

describe('VoiceButton', () => {
    it('renders microphone icon in idle state', () => {
        render(
            <VoiceButton
                isRecording={false}
                isProcessing={false}
                voiceAvailable={true}
                onToggle={vi.fn()}
            />
        );
        const btn = screen.getByRole('button');
        expect(btn).toBeInTheDocument();
        // Idle state should not show recording or processing indicators
        expect(btn.querySelector('.voice-button__recording')).toBeNull();
        expect(btn.querySelector('.voice-button__spinner')).toBeNull();
    });

    it('shows recording animation when isRecording=true', () => {
        render(
            <VoiceButton
                isRecording={true}
                isProcessing={false}
                voiceAvailable={true}
                onToggle={vi.fn()}
            />
        );
        const btn = screen.getByRole('button');
        expect(btn.querySelector('.voice-button__recording')).not.toBeNull();
    });

    it('shows processing spinner when isProcessing=true', () => {
        render(
            <VoiceButton
                isRecording={false}
                isProcessing={true}
                voiceAvailable={true}
                onToggle={vi.fn()}
            />
        );
        const btn = screen.getByRole('button');
        expect(btn.querySelector('.voice-button__spinner')).not.toBeNull();
    });

    it('calls onToggle when clicked', async () => {
        const onToggle = vi.fn();
        render(
            <VoiceButton
                isRecording={false}
                isProcessing={false}
                voiceAvailable={true}
                onToggle={onToggle}
            />
        );
        const btn = screen.getByRole('button');
        await userEvent.click(btn);
        expect(onToggle).toHaveBeenCalledOnce();
    });

    it('is disabled when voiceAvailable=false', () => {
        render(
            <VoiceButton
                isRecording={false}
                isProcessing={false}
                voiceAvailable={false}
                onToggle={vi.fn()}
            />
        );
        const btn = screen.getByRole('button');
        expect(btn).toBeDisabled();
    });

    it('shows tooltip explaining voice unavailability when disabled', () => {
        render(
            <VoiceButton
                isRecording={false}
                isProcessing={false}
                voiceAvailable={false}
                onToggle={vi.fn()}
            />
        );
        const btn = screen.getByRole('button');
        expect(btn).toHaveAttribute('title', 'Voice mode requires ASR model');
    });
});
