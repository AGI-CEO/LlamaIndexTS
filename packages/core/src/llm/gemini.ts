import { GoogleGenerativeAI, StartChatParams } from "@google/generative-ai";
import _ from "lodash";
import { CallbackManager, Event } from "../callbacks/CallbackManager";
import { ChatMessage, ChatResponse, LLM } from "./LLM";

import { Tokenizers, globalsHelper } from "../GlobalsHelper";

export interface GeminiConfig {
  apiKey?: string;
  model: keyof typeof ALL_AVAILABLE_GEMINI_MODELS;
  temperature: number;
  topP: number;
  maxTokens?: number;
  callbackManager?: CallbackManager;
}

export function getGeminiConfigFromEnv(
  init?: Partial<GeminiConfig>,
): GeminiConfig {
  return {
    apiKey: init?.apiKey ?? process.env.GEMINI_API_KEY,
    model:
      init?.model ??
      (process.env.GEMINI_MODEL as keyof typeof ALL_AVAILABLE_GEMINI_MODELS),
    temperature:
      init?.temperature ?? parseFloat(process.env.GEMINI_TEMPERATURE || "0.9"),
    topP: init?.topP ?? parseFloat(process.env.GEMINI_TOP_P || "1"),
    // add other options as needed
  };
}

export const ALL_AVAILABLE_GEMINI_MODELS = {
  "gemini-pro": { contextWindow: 30720 },
  "gemini-pro-vision": { contextWindow: 12288 },
  "embedding-001": { contextWindow: 2048 },
  aqa: { contextWindow: 7168 },
};

let defaultGeminiSession: { session: GeminiSession }[] = [];

export class GeminiSession implements LLM {
  hasStreaming: boolean = true;
  client: GoogleGenerativeAI;
  model: keyof typeof ALL_AVAILABLE_GEMINI_MODELS; // Add this line
  temperature: number;
  topP: number;
  maxTokens?: number;
  apiKey?: string;
  callbackManager?: CallbackManager;

  session: GeminiSession;
  buildParams(messages: ChatMessage[]): any {
    return messages.map((message) => ({
      role: message.role,
      content: message.content,
    }));
  }
  constructor(init?: Partial<GeminiConfig>) {
    const options = getGeminiConfigFromEnv(init);

    if (!options.apiKey) {
      throw new Error("Set Gemini API key in GEMINI_API_KEY env variable");
    }

    this.client = new GoogleGenerativeAI(options.apiKey);
    this.model = init?.model ?? "gemini-pro";
    this.temperature = init?.temperature ?? 0.1;
    this.topP = init?.topP ?? 1;
    this.maxTokens = init?.maxTokens ?? undefined;
    this.callbackManager = init?.callbackManager;
    this.session = new GeminiSession(init);
  }

  get metadata() {
    return {
      model: this.model,
      temperature: this.temperature,
      topP: this.topP,
      maxTokens: this.maxTokens,
      contextWindow: ALL_AVAILABLE_GEMINI_MODELS[this.model].contextWindow,
      tokenizer: Tokenizers.CL100K_BASE,
    };
  }

  tokens(messages: ChatMessage[]): number {
    // for latest OpenAI models, see https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    const tokenizer = globalsHelper.tokenizer(this.metadata.tokenizer);
    const tokensPerMessage = 3;
    let numTokens = 0;
    for (const message of messages) {
      numTokens += tokensPerMessage;
      for (const value of Object.values(message)) {
        numTokens += tokenizer(value).length;
      }
    }
    numTokens += 3; // every reply is primed with <|im_start|>assistant<|im_sep|>
    return numTokens;
  }

  async chat<
    T extends boolean | undefined = undefined,
    R = T extends true ? AsyncGenerator<string, void, unknown> : ChatResponse,
  >(messages: ChatMessage[], parentEvent?: Event, streaming?: T): Promise<R> {
    const chatParams: StartChatParams = {
      history: this.buildParams(messages),
      // other parameters required by ModelParams
    };
    const modelParams: GeminiConfig = {
      model: this.model,
      temperature: this.temperature,
      maxTokens: this.maxTokens,
      topP: this.topP,
    };
    // Streaming
    if (streaming) {
      if (!this.hasStreaming) {
        throw Error("No streaming support for this LLM.");
      }
      return this.streamChat(messages, parentEvent) as R;
    }
    // Non-streaming
    const client = this.client.getGenerativeModel(modelParams);

    const chat = client.startChat(chatParams);
    const result = await chat.sendMessage(this.buildParams(messages));
    const response = result.response.text ?? "";
    return {
      message: { response, role: response.length > 0 ? "bot" : "user" },
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

  async *streamChat(
    messages: ChatMessage[],
    parentEvent?: Event,
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
    parentEvent?: Event,
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
