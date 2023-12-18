import { LLM, ChatMessage, ChatResponse } from './LLM';
import { GoogleGenerativeAI } from '@google/generative-ai';
import _ from "lodash";

export interface GeminiConfig {
  apiKey?: string;
  // add other options as needed
}

export function getGeminiConfigFromEnv(init?: Partial<GeminiConfig>): GeminiConfig {
  return {
    apiKey: init?.apiKey ?? process.env.GEMINI_API_KEY,
    // add other options as needed
  };
}

let defaultGeminiSession: { session: GeminiSession; options: ClientOptions }[] = [];

export class GeminiSession extends LLM {
  client: GoogleGenerativeAI;

  constructor(init?: Partial<GeminiConfig>) {
    const options = getGeminiConfigFromEnv(init);

    if (!options.apiKey) {
      throw new Error("Set Gemini API key in GEMINI_API_KEY env variable");
    }

    this.client = new GoogleGenerativeAI(options.apiKey);
  }

  async chat(
    messages: ChatMessage[],
    parentEvent?: Event,
    streaming?: boolean
  ): Promise<ChatResponse | AsyncGenerator<string, void, unknown>> {
    const baseRequestParams = {
      model: this.model,
      messages: messages.map((message) => ({
        role: message.role,
        content: message.content,
      })),
      // add other parameters as needed
    };

    if (streaming) {
      return this.streamChat(messages, parentEvent);
    }

    const response = await this.client.chat(baseRequestParams);
    const content = response.choices[0].message?.content ?? "";
    return {
      message: { content, role: response.choices[0].message.role },
    };
  }

  async complete(
    prompt: string,
    parentEvent?: Event,
    streaming?: boolean
  ): Promise<ChatResponse | AsyncGenerator<string, void, unknown>> {
    return this.chat(
      [{ content: prompt, role: "user" }],
      parentEvent,
      streaming
    );
  }

  async *streamChat(
    messages: ChatMessage[],
    parentEvent?: Event
  ): AsyncGenerator<string, void, unknown> {
    const baseRequestParams = {
      model: this.model,
      messages: messages.map((message) => ({
        role: message.role,
        content: message.content,
      })),
      // add other parameters as needed
      stream: true,
    };

    const chunkStream = await this.client.chat(baseRequestParams);

    for await (const part of chunkStream) {
      yield part.choices[0].message?.content ?? "";
    }
  }

  streamComplete(
    query: string,
    parentEvent?: Event
  ): AsyncGenerator<string, void, unknown> {
    return this.streamChat([{ content: query, role: "user" }], parentEvent);
  }
}

export function getGeminiSession(init?: Partial<GeminiConfig>) {
  let session = defaultGeminiSession.find((session) => {
    return _.isEqual(session.options, init);
  })?.session;

  if (!session) {
    session = new GeminiSession(init);
    defaultGeminiSession.push({ session, options: init });
  }

  return session;
}

export const ALL_AVAILABLE_GEMINI_MODELS = {
  "gemini-pro": { contextWindow: 30720 },
  "gemini-pro-vision": { contextWindow: 12288 },
  "embedding-001": { contextWindow: 2048 },
  "aqa": { contextWindow: 7168 },
};