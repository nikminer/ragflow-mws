import { requestChatCompletionStream } from './chat-completion-stream';

jest.mock('@/utils/api', () => ({
  __esModule: true,
  default: { completionUrl: '/api/v1/chat/completions' },
}));
jest.mock('@/utils/authorization-util', () => ({
  getAuthorization: () => 'test-token',
}));

const originalFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = originalFetch;
});

test.each(['high', 'none', 'default'] as const)(
  'sends model effort %s independently of the RAG reasoning mode',
  (effort) => {
    const fetch = jest.fn().mockResolvedValue({});
    globalThis.fetch = fetch;
    const signal = new AbortController().signal;
    requestChatCompletionStream(
      {
        chatId: 'chat-id',
        sessionId: 'session-id',
        messages: [],
        enableThinking: '3',
        llmSetting: { reasoning_effort: effort },
      },
      signal,
    );
    const [url, request] = fetch.mock.calls[0];
    expect(url).toBe('/api/v1/chat/completions');
    expect(request.signal).toBe(signal);
    expect(JSON.parse(request.body)).toMatchObject({
      reasoning_effort: effort,
      reasoning: 3,
    });
  },
);

test('omits effort when no model override is configured', () => {
  const fetch = jest.fn().mockResolvedValue({});
  globalThis.fetch = fetch;
  requestChatCompletionStream(
    { sessionId: 'session-id', messages: [] },
    new AbortController().signal,
  );
  expect(JSON.parse(fetch.mock.calls[0][1].body)).not.toHaveProperty(
    'reasoning_effort',
  );
});
