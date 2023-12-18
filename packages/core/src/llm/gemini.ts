import { LLM, ChatMessage, ChatResponse } from './LLM';
import { GoogleGenerativeAI } from '@google/generative-ai';

export const ALL_AVAILABLE_GEMINI_MODELS = {
  "gemini-pro": { contextWindow: 30720 },
  "gemini-pro-vision": { contextWindow: 12288 },
  "embedding-001": { contextWindow: 2048 },
  "aqa": { contextWindow: 7168 },
};

interface GeminiConfig {
    model_name: string;
    temperature?: number;
    max_tokens?: number;
    // ... other specific Gemini configuration parameters
}
 
class GeminiSession {
    apiKey?: string;
    private client: any;

    constructor(init?: Partial<GeminiSession>) {
        if (init?.apiKey) {
            this.apiKey = init?.apiKey;
        } else {
            if (typeof process !== undefined) {
                this.apiKey = process.env.GEMINI_API_KEY;
            }
        }
        if (!this.apiKey) {
            throw new Error("Set Gemini API key in GEMINI_API_KEY env variable");
        }
    }

    async getClient() {
        const { default: GeminiClient } = await import('@google/generative-ai');
        if (!this.client) {
            this.client = new GeminiClient(this.apiKey);
        }
        return this.client;
    }
}

export class GeminiAI implements LLM {
    hasStreaming: boolean = false;
    model: keyof typeof ALL_AVAILABLE_GEMINI_MODELS;
    temperature: number;
    maxTokens?: number;
    apiKey?: string;

    private session: GeminiSession;

    constructor(init?: Partial<GeminiAI>) {
        this.model = init?.model ?? "gemini-pro";
        this.temperature = init?.temperature ?? 0.9;
        this.maxTokens = init?.maxTokens ?? undefined;
        this.session = new GeminiSession(init);
    }

    get metadata() {
        return {
            model: this.model,
            temperature: this.temperature,
            maxTokens: this.maxTokens,
            contextWindow: ALL_AVAILABLE_GEMINI_MODELS[this.model].contextWindow,
            tokenizer: undefined,
        };
    }

    tokens(messages: ChatMessage[]): number {
        throw new Error("Method not implemented.");
    }

    async chat<
        T extends boolean | undefined = undefined,
        R = T extends true ? AsyncGenerator<string, void, unknown> : ChatResponse,
    >(messages: ChatMessage[], parentEvent?: Event, streaming?: T): Promise<R> {
        // Streaming
        if (streaming) {
            if (!this.hasStreaming) {
                throw Error("No streaming support for this LLM.");
            }
            return this.streamChat(messages, parentEvent) as R;
        }
        // Non-streaming
        const client = await this.session.getClient();
        const response = await client.chat(this.buildParams(messages));
        const message = response.choices[0].message;
        return {
            message,
        } as R;
    }

    async complete<
        T extends boolean | undefined = undefined,
        R = T extends true ? AsyncGenerator<string, void, unknown> : ChatResponse,
    >(prompt: string, parentEvent?: Event, streaming?: T): Promise<R> {
        return this.chat(
            [{ content: prompt, role: "user" }],
            parentEvent,
            streaming,
        );
    }

    // ... other methods like streaming versions, embeddings, token counting, etc.

    // Utility methods specific to Gemini

    // ... error handling and validation

    // Export metadata or other necessary information
}

export { GeminiAI };
