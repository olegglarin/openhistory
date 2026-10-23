import assert from "node:assert/strict";
import { rmSync, writeFileSync } from "node:fs";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import test, { type TestContext } from "node:test";
import { InferenceSettingsStore } from "./inference-settings-store";

test("defaults to enabled OpenAI inference while honoring environment model defaults", async (context) => {
  const directory = await testDirectory(context);
  const settings = new InferenceSettingsStore(directory, { openai: "custom-openai-model" }).load();
  assert.equal(settings.enabled, true);
  assert.equal(settings.provider, "openai");
  assert.equal(settings.models.openai, "custom-openai-model");
  assert.equal(settings.models.anthropic, "claude-sonnet-5");
  assert.equal(settings.models.kimi, "kimi-k3");
  assert.equal(settings.models.custom, "");
  assert.equal(settings.customEndpoint, undefined);
});

test("seeds the custom endpoint from an environment default", async (context) => {
  const directory = await testDirectory(context);
  const settings = new InferenceSettingsStore(
    directory,
    undefined,
    "http://localhost:11434"
  ).load();
  assert.equal(settings.customEndpoint, "http://localhost:11434");
});

test("persists provider, model, and disabled state", async (context) => {
  const directory = await testDirectory(context);
  const store = new InferenceSettingsStore(directory);
  store.save({
    version: 1,
    enabled: false,
    provider: "kimi",
    models: {
      apple: "system-default",
      openai: "gpt-5.6-terra",
      anthropic: "claude-haiku-4-5",
      kimi: "kimi-k3",
      custom: ""
    }
  });
  assert.deepEqual(store.load(), {
    version: 1,
    enabled: false,
    provider: "kimi",
    models: {
      apple: "system-default",
      openai: "gpt-5.6-terra",
      anthropic: "claude-haiku-4-5",
      kimi: "kimi-k3",
      custom: ""
    }
  });
});

test("persists a custom endpoint and model selection", async (context) => {
  const directory = await testDirectory(context);
  const store = new InferenceSettingsStore(directory);
  store.save({
    version: 1,
    enabled: true,
    provider: "custom",
    models: {
      apple: "system-default",
      openai: "gpt-5.6-luna",
      anthropic: "claude-sonnet-5",
      kimi: "kimi-k3",
      custom: "llama3.1"
    },
    customEndpoint: "http://localhost:11434/v1"
  });
  assert.deepEqual(store.load(), {
    version: 1,
    enabled: true,
    provider: "custom",
    models: {
      apple: "system-default",
      openai: "gpt-5.6-luna",
      anthropic: "claude-sonnet-5",
      kimi: "kimi-k3",
      custom: "llama3.1"
    },
    customEndpoint: "http://localhost:11434/v1"
  });
});

test("falls back safely when stored inference settings are invalid", async (context) => {
  const directory = await testDirectory(context);
  writeFileSync(resolve(directory, "inference-settings.json"), JSON.stringify({
    version: 1,
    enabled: true,
    provider: "untrusted-provider",
    models: {}
  }));
  assert.equal(new InferenceSettingsStore(directory).load().provider, "openai");
});

async function testDirectory(context: TestContext): Promise<string> {
  const directory = await mkdtemp(resolve(tmpdir(), "openhistory-inference-settings-"));
  context.after(() => rmSync(directory, { recursive: true, force: true }));
  return directory;
}
