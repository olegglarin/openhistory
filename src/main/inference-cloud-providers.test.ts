import Anthropic from "@anthropic-ai/sdk";
import assert from "node:assert/strict";
import test from "node:test";
import OpenAI from "openai";
import { z } from "zod";
import { AnthropicProvider } from "./inference/providers/anthropic";
import { CustomProvider } from "./inference/providers/custom";
import { KimiProvider } from "./inference/providers/kimi";
import { OpenAIProvider } from "./inference/providers/openai";
import { InferenceOutputError } from "./inference/errors";

const schema = z.object({ title: z.string() });
const request = {
  instructions: "Synthetic instructions",
  input: "Synthetic input",
  schema,
  schemaName: "synthetic_entry",
  maxOutputTokens: 321
};

test("preserves the OpenAI structured response contract", async () => {
  let captured: Record<string, unknown> | undefined;
  const client = {
    responses: {
      parse: async (value: Record<string, unknown>) => {
        captured = value;
        return { output_parsed: { title: "Structured result" } };
      }
    }
  } as unknown as OpenAI;

  const result = await new OpenAIProvider("synthetic-key", "synthetic-openai", client).generate(request);
  assert.deepEqual(result, { title: "Structured result" });
  assert.equal(captured?.model, "synthetic-openai");
  assert.equal(captured?.instructions, request.instructions);
  assert.equal(captured?.input, request.input);
  assert.equal(captured?.max_output_tokens, 321);
  assert.equal(captured?.store, false);
});

test("classifies incomplete and refused OpenAI responses without exposing their text", async () => {
  const incompleteClient = {
    responses: {
      parse: async () => ({
        output_parsed: null,
        status: "incomplete",
        incomplete_details: { reason: "max_output_tokens" },
        output: []
      })
    }
  } as unknown as OpenAI;
  await assert.rejects(
    () => new OpenAIProvider("synthetic-key", "synthetic-openai", incompleteClient).generate(request),
    (error: unknown) => error instanceof InferenceOutputError && error.kind === "incomplete"
  );

  const refusalClient = {
    responses: {
      parse: async () => ({
        output_parsed: null,
        status: "completed",
        incomplete_details: null,
        output: [{ type: "message", content: [{ type: "refusal", refusal: "private refusal text" }] }]
      })
    }
  } as unknown as OpenAI;
  await assert.rejects(
    () => new OpenAIProvider("synthetic-key", "synthetic-openai", refusalClient).generate(request),
    (error: unknown) => error instanceof InferenceOutputError &&
      error.kind === "refusal" && !error.message.includes("private refusal text")
  );
});

test("preserves the Anthropic structured message contract", async () => {
  let captured: Record<string, unknown> | undefined;
  const client = {
    messages: {
      parse: async (value: Record<string, unknown>) => {
        captured = value;
        return { parsed_output: { title: "Structured result" } };
      }
    }
  } as unknown as Anthropic;

  const result = await new AnthropicProvider("synthetic-key", "synthetic-anthropic", client).generate(request);
  assert.deepEqual(result, { title: "Structured result" });
  assert.equal(captured?.model, "synthetic-anthropic");
  assert.equal(captured?.system, request.instructions);
  assert.equal(captured?.max_tokens, 321);
  assert.deepEqual(captured?.messages, [{ role: "user", content: request.input }]);
});

test("preserves Kimi compatibility limits and reasoning settings", async () => {
  let captured: Record<string, unknown> | undefined;
  const client = {
    chat: {
      completions: {
        parse: async (value: Record<string, unknown>) => {
          captured = value;
          return { choices: [{ message: { parsed: { title: "Structured result" } } }] };
        }
      }
    }
  } as unknown as OpenAI;

  const result = await new KimiProvider("synthetic-key", "kimi-k3", client).generate(request);
  assert.deepEqual(result, { title: "Structured result" });
  assert.equal(captured?.model, "kimi-k3");
  assert.equal(captured?.max_completion_tokens, 4_000);
  assert.equal(captured?.reasoning_effort, "low");
  assert.deepEqual(captured?.messages, [
    { role: "system", content: request.instructions },
    { role: "user", content: request.input }
  ]);
});

test("preserves the custom endpoint contract and normalizes bare hosts", async () => {
  let captured: Record<string, unknown> | undefined;
  const client = {
    chat: {
      completions: {
        parse: async (value: Record<string, unknown>) => {
          captured = value;
          return { choices: [{ message: { parsed: { title: "Structured result" } } }] };
        }
      }
    }
  } as unknown as OpenAI;

  const provider = new CustomProvider("synthetic-key", "llama3.1", "http://localhost:11434", client);
  const result = await provider.generate(request);
  assert.deepEqual(result, { title: "Structured result" });
  assert.equal(provider.endpoint, "http://localhost:11434/v1");
  assert.equal(captured?.model, "llama3.1");
  assert.equal(captured?.max_completion_tokens, 321);
  assert.deepEqual(captured?.messages, [
    { role: "system", content: request.instructions },
    { role: "user", content: request.input }
  ]);
});

test("rejects invalid custom endpoint URLs at construction", () => {
  assert.throws(() => new CustomProvider("synthetic-key", "llama3.1", "not a url"), /valid http or https/);
  assert.throws(() => new CustomProvider("synthetic-key", "llama3.1", "ftp://localhost:11434"), /valid http or https/);
});

test("falls back to plain completions when the endpoint rejects structured output", async () => {
  let parseCalls = 0;
  let createCalls = 0;
  const client = {
    chat: {
      completions: {
        parse: async () => {
          parseCalls += 1;
          throw Object.assign(
            new Error(
              "400 litellm.BadRequestError: Hosted_vllmException - Failed to compile json grammar: Grammar compilation failed"
            ),
            { status: 400 },
          );
        },
        create: async (value: Record<string, unknown>) => {
          createCalls += 1;
          assert.equal(value.response_format, undefined);
          return {
            choices: [{ message: { content: "Here is the summary:\n```json\n{\"title\":\"Fallback result\"}\n```" } }]
          };
        }
      }
    }
  } as unknown as OpenAI;

  const provider = new CustomProvider("synthetic-key", "llama3.1", "http://localhost:11434", client);
  assert.deepEqual(await provider.generate(request), { title: "Fallback result" });
  assert.equal(parseCalls, 1);
  assert.equal(createCalls, 1);

  await provider.generate(request);
  assert.equal(parseCalls, 1);
  assert.equal(createCalls, 2);
});

test("does not mask unrelated custom endpoint errors as structured output rejections", async () => {
  let createCalls = 0;
  const client = {
    chat: {
      completions: {
        parse: async () => {
          throw Object.assign(new Error("401 Authentication Error, invalid api key"), { status: 401 });
        },
        create: async () => {
          createCalls += 1;
          return { choices: [{ message: { content: "{\"title\":\"wrong\"}" } }] };
        }
      }
    }
  } as unknown as OpenAI;

  const provider = new CustomProvider("synthetic-key", "llama3.1", "http://localhost:11434", client);
  await assert.rejects(() => provider.generate(request), /invalid api key/);
  assert.equal(createCalls, 0);
});

test("retries thinking-starved custom completions with thinking disabled", async () => {
  const seenParams: Array<Record<string, unknown>> = [];
  const client = {
    chat: {
      completions: {
        parse: async () => {
          throw Object.assign(new Error("400 Failed to compile json grammar"), { status: 400 });
        },
        create: async (value: Record<string, unknown>) => {
          seenParams.push(value);
          if (seenParams.length === 1) {
            return {
              choices: [{ finish_reason: "length", message: { content: "" } }],
              usage: { completion_tokens_details: { reasoning_tokens: 321 } }
            };
          }
          return { choices: [{ message: { content: "{\"title\":\"No thinking result\"}" } }] };
        }
      }
    }
  } as unknown as OpenAI;

  const provider = new CustomProvider("synthetic-key", "llama3.1", "http://localhost:11434", client);
  assert.deepEqual(await provider.generate(request), { title: "No thinking result" });
  assert.equal(seenParams.length, 2);
  assert.equal(seenParams[0]?.extra_body, undefined);
  assert.deepEqual(seenParams[1]?.extra_body, { chat_template_kwargs: { enable_thinking: false } });

  await provider.generate(request);
  assert.equal(seenParams.length, 3);
  assert.deepEqual(seenParams[2]?.extra_body, { chat_template_kwargs: { enable_thinking: false } });
});

test("does not remember a failed thinking-disable retry", async () => {
  const seenParams: Array<Record<string, unknown>> = [];
  const client = {
    chat: {
      completions: {
        parse: async () => {
          throw Object.assign(new Error("400 response_format is not supported"), { status: 400 });
        },
        create: async (value: Record<string, unknown>) => {
          seenParams.push(value);
          if (value.extra_body) {
            throw Object.assign(new Error("400 unrecognized argument"), { status: 400 });
          }
          return {
            choices: [{ finish_reason: "length", message: { content: "" } }],
            usage: { completion_tokens_details: { reasoning_tokens: 321 } }
          };
        }
      }
    }
  } as unknown as OpenAI;

  const provider = new CustomProvider("synthetic-key", "llama3.1", "http://localhost:11434", client);
  await assert.rejects(
    () => provider.generate(request),
    (error: unknown) => error instanceof InferenceOutputError && error.kind === "invalid_output"
  );
  await assert.rejects(
    () => provider.generate(request),
    (error: unknown) => error instanceof InferenceOutputError && error.kind === "invalid_output"
  );
  assert.equal(seenParams.length, 4);
  assert.equal(seenParams.filter((value) => value.extra_body === undefined).length, 2);
});

test("rejects custom plain completions without usable JSON", async () => {
  const client = {
    chat: {
      completions: {
        parse: async () => {
          throw Object.assign(new Error("400 response_format is not supported by this model"), { status: 400 });
        },
        create: async () => ({ choices: [{ message: { content: "I cannot produce JSON right now." } }] })
      }
    }
  } as unknown as OpenAI;

  const provider = new CustomProvider("synthetic-key", "llama3.1", "http://localhost:11434", client);
  await assert.rejects(
    () => provider.generate(request),
    (error: unknown) => error instanceof InferenceOutputError && error.kind === "invalid_output"
  );
});

test("repairs dropped array fields and bullet arrays in custom plain completions", async () => {
  const rollupSchema = z.object({
    title: z.string(),
    summary: z.string(),
    themes: z.array(z.string()),
    accomplishments: z.array(z.string()),
    decisions: z.array(z.string()),
    linkReferences: z.array(z.string())
  });
  const rollupRequest = { ...request, schema: rollupSchema, schemaName: "hour_rollup" };
  const client = {
    chat: {
      completions: {
        parse: async () => {
          throw Object.assign(new Error("400 Failed to compile json grammar"), { status: 400 });
        },
        create: async () => ({
          choices: [
            {
              message: {
                content:
                  '{"title":"Reviewed visa request","summary":["- Submitted the visa request.","- Coordinated flights."],"linkReferences":[]}'
              }
            }
          ]
        })
      }
    }
  } as unknown as OpenAI;

  const provider = new CustomProvider("synthetic-key", "llama3.1", "http://localhost:11434", client);
  assert.deepEqual(await provider.generate(rollupRequest), {
    title: "Reviewed visa request",
    summary: "- Submitted the visa request.\n- Coordinated flights.",
    themes: [],
    accomplishments: [],
    decisions: [],
    linkReferences: []
  });
});

test("still rejects custom plain completions when a required field cannot be repaired", async () => {
  const rollupSchema = z.object({
    title: z.string(),
    summary: z.string(),
    themes: z.array(z.string())
  });
  const rollupRequest = { ...request, schema: rollupSchema, schemaName: "hour_rollup" };
  const client = {
    chat: {
      completions: {
        parse: async () => {
          throw Object.assign(new Error("400 Failed to compile json grammar"), { status: 400 });
        },
        create: async () => ({
          choices: [{ message: { content: '{"summary":["- Only a bullet list"]}' } }]
        })
      }
    }
  } as unknown as OpenAI;

  const provider = new CustomProvider("synthetic-key", "llama3.1", "http://localhost:11434", client);
  await assert.rejects(() => provider.generate(rollupRequest), Error);
});
