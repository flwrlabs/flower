import { ChatResponseResult, FlowerIntelligence, Message } from '@flwr/flwr';

const fi: FlowerIntelligence = FlowerIntelligence.instance;

// Global chat history with an initial system message.
export const history: Message[] = [
  { role: 'system', content: 'You are a friendly assistant that loves using emojis.' },
];

export async function chatWithMessages(messages: Message[]): Promise<Message> {
  try {
    const response: ChatResponseResult = await fi.chat({
      messages,
    });
    if (!response || (response.ok && !response.message)) {
      throw new Error('Invalid response structure from the chat service.');
    }
    if (!response.ok) {
      console.error(response);
      throw new Error('Failed to get a valid response.');
    }

    return response.message;
  } catch (error) {
    console.error('Error in chatWithMessages:', error);
    throw new Error('Failed to get a valid response from the chat service.');
  }
}

export async function chatWithHistory(question: string): Promise<string> {
  try {
    history.push({ role: 'user', content: question });
    const response: ChatResponseResult = await fi.chat({
      messages: history,
    });
    if (!response || (response.ok && !response.message)) {
      throw new Error('Invalid response structure from the chat service.');
    }
    if (!response.ok) {
      console.error(response);
      throw new Error('Failed to get a valid response.');
    }

    history.push(response.message);
    return response.message.content;
  } catch (error) {
    console.error('Error in chatWithHistory:', error);
    throw new Error('Failed to get a valid response from the chat service.');
  }
}
