'use client';

import React, { useMemo, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

type ChatMessageRole = 'system' | 'user' | 'assistant';

interface ChatMessage {
  role: ChatMessageRole;
  content: string;
}

interface ChatEntry {
  role: 'user' | 'bot';
  content: string;
}

export default function ClientSideChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    { role: 'system', content: 'You are a friendly assistant that loves using emojis.' },
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  const chatLog = useMemo<ChatEntry[]>(
    () =>
      messages
        .filter((msg) => msg.role !== 'system')
        .map((msg) => ({
          role: msg.role === 'user' ? 'user' : 'bot',
          content: msg.content,
        })),
    [messages]
  );

  const sendQuestion = async () => {
    const trimmed = input.trim();
    if (!trimmed) return;
    setLoading(true);
    setInput('');

    const nextMessages: ChatMessage[] = [...messages, { role: 'user', content: trimmed }];
    setMessages(nextMessages);

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: nextMessages }),
      });
      const data = (await res.json()) as {
        message?: string;
        role?: ChatMessageRole;
        error?: string;
      };
      if (res.ok && data.message) {
        const message = data.message;
        setMessages((prev) => [...prev, { role: data.role ?? 'assistant', content: message }]);
      } else {
        setMessages((prev) => [
          ...prev,
          { role: 'assistant', content: data.error || 'Failed to get a valid response.' },
        ]);
      }
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: 'Network error, please try again.' },
      ]);
    }
    setLoading(false);
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      sendQuestion();
    }
  };

  return (
    <div className="flex flex-col h-[calc(100vh-60px)] border rounded shadow bg-white dark:bg-gray-800">
      {/* Chat Messages */}
      <div className="flex-grow p-4 overflow-auto">
        {chatLog.map((entry, index) => (
          <div
            key={index}
            className={`mb-4 flex ${entry.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[75%] p-3 rounded-lg ${
                entry.role === 'user'
                  ? 'bg-blue-500 text-white rounded-tr-none'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-100 rounded-tl-none'
              }`}
            >
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{entry.content}</ReactMarkdown>
            </div>
          </div>
        ))}
        {loading && (
          <div className="mb-4 flex justify-start">
            <div className="max-w-[75%] p-3 bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-100 rounded-lg rounded-tl-none">
              Thinking...
            </div>
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="border-t p-4 bg-gray-50 pl-20 dark:bg-gray-900">
        <div className="flex space-x-2">
          <input
            type="text"
            placeholder="Type your question..."
            className="flex-grow p-2 border border-gray-300 dark:border-gray-600 rounded focus:outline-none bg-white dark:bg-gray-800 text-gray-800 dark:text-gray-100"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
          />
          <button
            onClick={sendQuestion}
            disabled={loading || !input.trim()}
            className="bg-blue-500 text-white px-4 py-2 rounded disabled:bg-gray-400"
          >
            {loading ? 'Sending...' : 'Send'}
          </button>
        </div>
      </div>
    </div>
  );
}
