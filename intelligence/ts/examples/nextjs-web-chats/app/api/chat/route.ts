import { chatWithHistory, chatWithMessages } from '@/lib/chat';
import type { Message } from '@flwr/flwr';

export async function POST(req: Request) {
  try {
    const { question, messages } = (await req.json()) as {
      question?: string;
      messages?: Message[];
    };

    if (Array.isArray(messages) && messages.length > 0) {
      const responseMessage = await chatWithMessages(messages);
      return new Response(
        JSON.stringify({ message: responseMessage.content, role: responseMessage.role }),
        { status: 200 }
      );
    }

    if (!question?.trim()) {
      return new Response(JSON.stringify({ error: 'Missing question.' }), { status: 400 });
    }

    const message = await chatWithHistory(question);
    return new Response(JSON.stringify({ message }), { status: 200 });
  } catch (error) {
    console.error('Error in /api/chat:', error);
    return new Response(
      JSON.stringify({ error: 'Failed to process the request. Please try again later.' }),
      { status: 500 }
    );
  }
}
