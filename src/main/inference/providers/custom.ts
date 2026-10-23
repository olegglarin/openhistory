import OpenAI from "openai";
import { zodResponseFormat } from "openai/helpers/zod";
import type { z } from "zod";
import { normalizeCustomEndpoint } from "@shared/inference";
import { InferenceOutputError } from "../errors";
import type { InferenceProviderAdapter, StructuredGenerationRequest } from "../contracts";

// OpenAI-compatible servers reject structured output in different words:
// vLLM reports grammar compilation failures, proxies mention response_format,
// some refuse "structured output" outright.
const STRUCTURED_OUTPUT_REJECTION =
  /response_format|json_schema|json grammar|structured output|guided (?:json|decoding)|grammar compilation/i;

// Thinking models (e.g. Qwen3 on vLLM) can spend their entire completion
// budget on internal reasoning and never emit the answer. LiteLLM-style
// proxies merge extra_body into the provider request, where
// chat_template_kwargs.enable_thinking disables the thinking mode.
const THINKING_DISABLE_BODY = { chat_template_kwargs: { enable_thinking: false } };

export class CustomProvider implements InferenceProviderAdapter {
  readonly provider = "custom" as const;
  readonly endpoint: string;
  private readonly client: OpenAI;
  private structuredOutputBroken = false;
  private thinkingDisabled = false;

  constructor(apiKey: string, readonly model: string, endpoint: string, client?: OpenAI) {
    this.endpoint = normalizeCustomEndpoint(endpoint);
    this.client = client ?? new OpenAI({ apiKey, baseURL: this.endpoint });
  }

  async generate<T>(request: StructuredGenerationRequest<T>): Promise<T> {
    if (!this.structuredOutputBroken) {
      try {
        return await this.generateStructured(request);
      } catch (error) {
        if (!isStructuredOutputRejection(error)) throw error;
        this.structuredOutputBroken = true;
      }
    }
    const content = await this.generatePlainContent(request);
    return parseWithRepair(extractJson(content), request.schema);
  }

  private async generateStructured<T>(request: StructuredGenerationRequest<T>): Promise<T> {
    const completion = await this.client.chat.completions.parse({
      model: this.model,
      messages: [
        { role: "system", content: request.instructions },
        { role: "user", content: request.input }
      ],
      response_format: zodResponseFormat(request.schema, request.schemaName),
      max_completion_tokens: request.maxOutputTokens
    });
    const parsed = completion.choices[0]?.message.parsed;
    if (!parsed) throw new InferenceOutputError("invalid_output");
    return request.schema.parse(parsed);
  }

  private async generatePlainContent(request: StructuredGenerationRequest<unknown>): Promise<string> {
    const completion = await this.createPlainCompletion(request, this.thinkingDisabled);
    const content = completion.choices?.[0]?.message?.content ?? "";
    if (this.thinkingDisabled || !isThinkingStarved(completion)) return content;
    // The model spent its whole budget thinking and never answered; retry
    // with thinking disabled and remember it for subsequent requests.
    try {
      const retried = await this.createPlainCompletion(request, true);
      this.thinkingDisabled = true;
      return retried.choices?.[0]?.message?.content ?? "";
    } catch {
      return content;
    }
  }

  private async createPlainCompletion(
    request: StructuredGenerationRequest<unknown>,
    thinkingDisabled: boolean
  ): Promise<PlainCompletion> {
    const base = {
      model: this.model,
      messages: [
        {
          role: "system" as const,
          content: `${request.instructions}\n\nRespond with a single JSON object and nothing else.`
        },
        { role: "user" as const, content: request.input }
      ],
      max_completion_tokens: request.maxOutputTokens
    };
    // extra_body is not part of the SDK types: it passes through in the
    // request body, where LiteLLM-style proxies merge it into the provider
    // parameters (e.g. vLLM chat_template_kwargs).
    const params = thinkingDisabled ? { ...base, extra_body: THINKING_DISABLE_BODY } : base;
    return this.client.chat.completions.create(params as never);
  }
}

interface PlainCompletion {
  choices?: Array<{ finish_reason?: string | null; message?: { content?: string | null } | null }>;
  usage?: { completion_tokens_details?: { reasoning_tokens?: number } | null };
}

function isThinkingStarved(completion: {
  choices?: Array<{ finish_reason?: string | null }>;
  usage?: { completion_tokens_details?: { reasoning_tokens?: number } | null };
}): boolean {
  const choice = completion.choices?.[0];
  const reasoningTokens = completion.usage?.completion_tokens_details?.reasoning_tokens ?? 0;
  return choice?.finish_reason === "length" && reasoningTokens > 0;
}

function isStructuredOutputRejection(error: unknown): boolean {
  return error instanceof Error && STRUCTURED_OUTPUT_REJECTION.test(error.message);
}

function extractJson(content: string): unknown {
  const trimmed = content.trim();
  if (!trimmed) throw new InferenceOutputError("invalid_output");
  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i);
  const candidate = (fenced?.[1] ?? trimmed).trim();
  const start = candidate.indexOf("{");
  const end = candidate.lastIndexOf("}");
  if (start === -1 || end <= start) throw new InferenceOutputError("invalid_output");
  try {
    return JSON.parse(candidate.slice(start, end + 1));
  } catch {
    throw new InferenceOutputError("invalid_output");
  }
}

function parseWithRepair<T>(value: unknown, schema: z.ZodType<T>): T {
  const result = schema.safeParse(value);
  if (result.success) return result.data;
  if (value && typeof value === "object" && !Array.isArray(value)) {
    const repaired = repairPlainDraft(value as Record<string, unknown>, result.error);
    if (repaired !== undefined) {
      const reparsed = schema.safeParse(repaired);
      if (reparsed.success) return reparsed.data;
    }
  }
  return schema.parse(value);
}

// Without enforced structured output, smaller models routinely drop required
// array fields (despite the prompt saying empty arrays are fine) or return a
// bullet list as an array where the schema expects a newline string. Repair
// only these two safe-to-infer shapes; anything else still fails the parse so
// the service retry can handle it.
function repairPlainDraft(value: Record<string, unknown>, error: unknown): Record<string, unknown> | undefined {
  if (!error || typeof error !== "object" || !("issues" in error) || !Array.isArray(error.issues)) return undefined;
  const repaired: Record<string, unknown> = { ...value };
  let changed = false;
  for (const rawIssue of error.issues) {
    if (!rawIssue || typeof rawIssue !== "object") continue;
    const issue = rawIssue as { code?: unknown; path?: unknown; expected?: unknown };
    if (issue.code !== "invalid_type" || !Array.isArray(issue.path) || issue.path.length !== 1) continue;
    const key = issue.path[0];
    if (typeof key !== "string") continue;
    // Zod v4 omits the received type from the issue, so infer it from the
    // value itself: a missing array field becomes [], an array of bullets in
    // a string field is joined into the newline format the schema expects.
    if (issue.expected === "array") {
      if (!(key in repaired)) {
        repaired[key] = [];
        changed = true;
      }
    } else if (issue.expected === "string") {
      const current = repaired[key];
      if (Array.isArray(current) && current.every((item) => typeof item === "string")) {
        repaired[key] = current.join("\n");
        changed = true;
      }
    }
  }
  return changed ? repaired : undefined;
}
