/**
 * Argus Core - Chat Component Tests
 * ====================================
 * Comprehensive tests for chat UI components.
 *
 * Test Coverage:
 * - ChatContainer: empty state, message display, send flow, error handling
 * - ChatInput: validation, send on Enter, character count
 * - ChatMessage: user/assistant styling, copy, timestamp
 * - SuggestedQuestions: click handling, verdict-based questions
 * - Accessibility compliance (WCAG 2.1 AA)
 *
 * Target: >80% coverage
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { screen, waitFor, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders, createMockAnalysisDetail } from '../utils/test-utils';
import type { ChatMessage as ChatMessageType, ChatHistory, ChatResponse } from '@/types/chat';

// Mock scrollIntoView for jsdom
Element.prototype.scrollIntoView = vi.fn();

// ============== MOCKS ==============

const mockSendMessage = vi.hoisted(() => ({
  mutate: vi.fn(),
  isPending: false,
  isError: false,
  error: null as Error | null,
}));

const mockClearChat = vi.hoisted(() => ({
  mutate: vi.fn(),
  isPending: false,
  isError: false,
}));

const mockHistory: ChatHistory = {
  analysis_id: 'test-analysis-123',
  messages: [],
  total_messages: 0,
};

let mockHistoryData: ChatHistory = { ...mockHistory };
let mockHistoryLoading = false;
let mockHistoryError: Error | null = null;

vi.mock('@/hooks/useChat', () => ({
  useChatHistory: () => ({
    data: mockHistoryData,
    isLoading: mockHistoryLoading,
    isError: !!mockHistoryError,
    error: mockHistoryError,
    refetch: vi.fn(),
  }),
  useSendMessage: () => mockSendMessage,
  useClearChat: () => mockClearChat,
  chatKeys: {
    all: ['chat'],
    history: (id: string) => ['chat', 'history', id],
  },
}));

// ============== TEST DATA ==============

const userMessage: ChatMessageType = {
  role: 'user',
  content: 'Why was this flagged as fake?',
  timestamp: '2026-03-29T10:00:00Z',
};

const assistantMessage: ChatMessageType = {
  role: 'assistant',
  content: 'The analysis detected GAN artifacts in the facial region with high confidence.',
  timestamp: '2026-03-29T10:00:05Z',
};

const testMessages: ChatHistory = {
  analysis_id: 'test-analysis-123',
  messages: [userMessage, assistantMessage],
  total_messages: 2,
};

// ============== CHAT CONTAINER TESTS ==============

describe('ChatContainer', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockHistoryData = { ...mockHistory };
    mockHistoryLoading = false;
    mockHistoryError = null;
    mockSendMessage.isPending = false;
    mockSendMessage.isError = false;
    mockSendMessage.error = null;
  });

  it('renders empty state with suggested questions', async () => {
    const { ChatContainer } = await import('@/components/chat/ChatContainer');
    renderWithProviders(
      <ChatContainer analysisId="test-analysis-123" verdict="likely_fake" />
    );

    expect(screen.getByText('Start a conversation')).toBeInTheDocument();
    expect(screen.getByText(/Ask questions about the analysis results/i)).toBeInTheDocument();
  });

  it('renders messages when history exists', async () => {
    mockHistoryData = { ...testMessages };
    const { ChatContainer } = await import('@/components/chat/ChatContainer');
    renderWithProviders(
      <ChatContainer analysisId="test-analysis-123" />
    );

    expect(screen.getByText('Why was this flagged as fake?')).toBeInTheDocument();
    expect(screen.getByText(/GAN artifacts/)).toBeInTheDocument();
  });

  it('shows loading state while fetching history', async () => {
    mockHistoryLoading = true;
    const { ChatContainer } = await import('@/components/chat/ChatContainer');
    renderWithProviders(
      <ChatContainer analysisId="test-analysis-123" />
    );

    expect(screen.getByTestId('chat-container')).toBeInTheDocument();
  });

  it('shows error state when history fails', async () => {
    mockHistoryError = new Error('Network error');
    const { ChatContainer } = await import('@/components/chat/ChatContainer');
    renderWithProviders(
      <ChatContainer analysisId="test-analysis-123" />
    );

    expect(screen.getByText('Failed to load chat')).toBeInTheDocument();
  });

  it('renders with custom className', async () => {
    const { ChatContainer } = await import('@/components/chat/ChatContainer');
    renderWithProviders(
      <ChatContainer analysisId="test-analysis-123" className="custom-class" />
    );

    expect(screen.getByTestId('chat-container')).toHaveClass('custom-class');
  });

  it('has correct accessibility attributes', async () => {
    mockHistoryData = { ...testMessages };
    const { ChatContainer } = await import('@/components/chat/ChatContainer');
    renderWithProviders(
      <ChatContainer analysisId="test-analysis-123" />
    );

    expect(screen.getByRole('log')).toBeInTheDocument();
    expect(screen.getByLabelText('Clear chat history')).toBeInTheDocument();
  });
});

// ============== CHAT INPUT TESTS ==============

describe('ChatInput', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders with placeholder text', async () => {
    const { ChatInput } = await import('@/components/chat/ChatInput');
    renderWithProviders(
      <ChatInput onSend={vi.fn()} placeholder="Ask a question..." />
    );

    expect(screen.getByPlaceholderText('Ask a question...')).toBeInTheDocument();
  });

  it('disables send button when input is empty', async () => {
    const { ChatInput } = await import('@/components/chat/ChatInput');
    renderWithProviders(
      <ChatInput onSend={vi.fn()} />
    );

    const sendButton = screen.getByLabelText('Send message');
    expect(sendButton).toBeDisabled();
  });

  it('enables send button when input has text', async () => {
    const user = userEvent.setup();
    const { ChatInput } = await import('@/components/chat/ChatInput');
    renderWithProviders(
      <ChatInput onSend={vi.fn()} />
    );

    const input = screen.getByLabelText('Chat message input');
    await user.type(input, 'Hello');

    const sendButton = screen.getByLabelText('Send message');
    expect(sendButton).not.toBeDisabled();
  });

  it('calls onSend when Enter is pressed', async () => {
    const user = userEvent.setup();
    const onSend = vi.fn();
    const { ChatInput } = await import('@/components/chat/ChatInput');
    renderWithProviders(
      <ChatInput onSend={onSend} />
    );

    const input = screen.getByLabelText('Chat message input');
    await user.type(input, 'Test message{Enter}');

    expect(onSend).toHaveBeenCalledWith('Test message');
  });

  it('clears input after sending', async () => {
    const user = userEvent.setup();
    const { ChatInput } = await import('@/components/chat/ChatInput');
    renderWithProviders(
      <ChatInput onSend={vi.fn()} />
    );

    const input = screen.getByLabelText('Chat message input') as HTMLTextAreaElement;
    await user.type(input, 'Test message{Enter}');

    expect(input.value).toBe('');
  });

  it('shows character count', async () => {
    const user = userEvent.setup();
    const { ChatInput } = await import('@/components/chat/ChatInput');
    renderWithProviders(
      <ChatInput onSend={vi.fn()} maxLength={100} />
    );

    const input = screen.getByLabelText('Chat message input');
    await user.type(input, 'Hello');

    expect(screen.getByText('5/100')).toBeInTheDocument();
  });

  it('disables input when isLoading', async () => {
    const { ChatInput } = await import('@/components/chat/ChatInput');
    renderWithProviders(
      <ChatInput onSend={vi.fn()} isLoading={true} />
    );

    const input = screen.getByLabelText('Chat message input');
    expect(input).toBeDisabled();
  });

  it('does not send on Shift+Enter', async () => {
    const user = userEvent.setup();
    const onSend = vi.fn();
    const { ChatInput } = await import('@/components/chat/ChatInput');
    renderWithProviders(
      <ChatInput onSend={onSend} />
    );

    const input = screen.getByLabelText('Chat message input');
    await user.type(input, 'Line 1{Shift>}{Enter}{/Shift}Line 2');

    expect(onSend).not.toHaveBeenCalled();
  });
});

// ============== CHAT MESSAGE TESTS ==============

describe('ChatMessage', () => {
  it('renders user message with correct styling', async () => {
    const { ChatMessage } = await import('@/components/chat/ChatMessage');
    renderWithProviders(
      <ChatMessage message={userMessage} />
    );

    expect(screen.getByText('Why was this flagged as fake?')).toBeInTheDocument();
    expect(screen.getByLabelText('user message')).toBeInTheDocument();
  });

  it('renders assistant message with correct styling', async () => {
    const { ChatMessage } = await import('@/components/chat/ChatMessage');
    renderWithProviders(
      <ChatMessage message={assistantMessage} />
    );

    expect(screen.getByText(/GAN artifacts/)).toBeInTheDocument();
    expect(screen.getByLabelText('assistant message')).toBeInTheDocument();
  });

  it('shows timestamp for messages', async () => {
    const { ChatMessage } = await import('@/components/chat/ChatMessage');
    renderWithProviders(
      <ChatMessage message={userMessage} />
    );

    const timeElements = screen.getAllByText(/\d{1,2}:\d{2}/);
    expect(timeElements.length).toBeGreaterThan(0);
  });

  it('has copy button for assistant messages', async () => {
    const { ChatMessage } = await import('@/components/chat/ChatMessage');
    renderWithProviders(
      <ChatMessage message={assistantMessage} />
    );

    expect(screen.getByLabelText('Copy message')).toBeInTheDocument();
  });

  it('does not show copy button for user messages', async () => {
    const { ChatMessage } = await import('@/components/chat/ChatMessage');
    renderWithProviders(
      <ChatMessage message={userMessage} />
    );

    expect(screen.queryByLabelText('Copy message')).not.toBeInTheDocument();
  });

  it('copies message content on copy click', async () => {
    const user = userEvent.setup();
    const writeText = vi.fn().mockResolvedValue(undefined);

    Object.defineProperty(navigator, 'clipboard', {
      value: { writeText },
      writable: true,
      configurable: true,
    });

    const { ChatMessage } = await import('@/components/chat/ChatMessage');
    renderWithProviders(
      <ChatMessage message={assistantMessage} />
    );

    await user.click(screen.getByLabelText('Copy message'));

    expect(writeText).toHaveBeenCalledWith(assistantMessage.content);
  });
});

// ============== SUGGESTED QUESTIONS TESTS ==============

describe('SuggestedQuestions', () => {
  it('renders default questions', async () => {
    const { SuggestedQuestions } = await import('@/components/chat/SuggestedQuestions');
    renderWithProviders(
      <SuggestedQuestions onQuestionClick={vi.fn()} />
    );

    expect(screen.getByText(/What specific artifacts/)).toBeInTheDocument();
    expect(screen.getByText(/How confident/)).toBeInTheDocument();
  });

  it('shows fake-specific question for likely_fake verdict', async () => {
    const { SuggestedQuestions } = await import('@/components/chat/SuggestedQuestions');
    renderWithProviders(
      <SuggestedQuestions onQuestionClick={vi.fn()} verdict="likely_fake" />
    );

    expect(screen.getByText(/Why was this flagged/)).toBeInTheDocument();
  });

  it('shows authentic-specific question for authentic verdict', async () => {
    const { SuggestedQuestions } = await import('@/components/chat/SuggestedQuestions');
    renderWithProviders(
      <SuggestedQuestions onQuestionClick={vi.fn()} verdict="authentic" />
    );

    expect(screen.getByText(/What evidence supports/)).toBeInTheDocument();
  });

  it('calls onQuestionClick when question is clicked', async () => {
    const user = userEvent.setup();
    const onQuestionClick = vi.fn();
    const { SuggestedQuestions } = await import('@/components/chat/SuggestedQuestions');
    renderWithProviders(
      <SuggestedQuestions onQuestionClick={onQuestionClick} />
    );

    await user.click(screen.getByText(/What specific artifacts/));

    expect(onQuestionClick).toHaveBeenCalled();
  });

  it('disables questions when isDisabled', async () => {
    const { SuggestedQuestions } = await import('@/components/chat/SuggestedQuestions');
    renderWithProviders(
      <SuggestedQuestions onQuestionClick={vi.fn()} isDisabled={true} />
    );

    const buttons = screen.getAllByRole('button');
    buttons.forEach(button => {
      expect(button).toBeDisabled();
    });
  });

  it('has correct accessibility attributes', async () => {
    const { SuggestedQuestions } = await import('@/components/chat/SuggestedQuestions');
    renderWithProviders(
      <SuggestedQuestions onQuestionClick={vi.fn()} />
    );

    const buttons = screen.getAllByRole('button');
    buttons.forEach(button => {
      expect(button).toHaveAttribute('aria-label');
    });
  });
});

// ============== CHAT INDEX EXPORTS ==============

describe('Chat Components Index', () => {
  it('exports all chat components', async () => {
    const chatModule = await import('@/components/chat');

    expect(chatModule.ChatContainer).toBeDefined();
    expect(chatModule.ChatMessage).toBeDefined();
    expect(chatModule.ChatInput).toBeDefined();
    expect(chatModule.SuggestedQuestions).toBeDefined();
  });
});
