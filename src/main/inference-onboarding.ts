import type { InferenceOnboardingSelection } from "@shared/contracts";
import {
  CUSTOM_ENDPOINT_MAX_LENGTH,
  CUSTOM_MODEL_MAX_LENGTH,
  INFERENCE_MODEL_OPTIONS,
  isCloudInferenceProvider,
  isInferenceProvider,
  normalizeCustomEndpoint,
  type AppleInferenceAvailability
} from "@shared/inference";

export function normalizeInferenceOnboardingSelection(
  value: unknown
): InferenceOnboardingSelection {
  if (!value || typeof value !== "object") throw new Error("Invalid model selection");
  const candidate = value as Partial<InferenceOnboardingSelection>;
  if (!isInferenceProvider(candidate.provider)) throw new Error("Invalid inference provider");

  const model = typeof candidate.model === "string" ? candidate.model.trim() : "";
  if (candidate.provider === "custom") {
    if (!model) throw new Error("Enter a model name for the custom endpoint");
    if (model.length > CUSTOM_MODEL_MAX_LENGTH) throw new Error("Model name is too long");
  } else if (!INFERENCE_MODEL_OPTIONS[candidate.provider].some((option) => option.id === model)) {
    throw new Error("Choose a supported model");
  }
  if (candidate.captureEmailActivity !== undefined &&
      typeof candidate.captureEmailActivity !== "boolean") {
    throw new Error("Invalid email capture selection");
  }
  if (candidate.captureMessagingActivity !== undefined &&
      typeof candidate.captureMessagingActivity !== "boolean") {
    throw new Error("Invalid messaging capture selection");
  }
  if (candidate.appPresentationMode !== undefined &&
      candidate.appPresentationMode !== "dock" &&
      candidate.appPresentationMode !== "menuBar") {
    throw new Error("Invalid app presentation selection");
  }
  const captureSelections = {
    captureEmailActivity: candidate.captureEmailActivity === true,
    captureMessagingActivity: candidate.captureMessagingActivity === true,
    appPresentationMode: candidate.appPresentationMode ?? "dock"
  };

  if (!isCloudInferenceProvider(candidate.provider)) {
    return { provider: candidate.provider, model, ...captureSelections };
  }

  const apiKey = typeof candidate.apiKey === "string" ? candidate.apiKey.trim() : "";
  if (!apiKey) throw new Error(`${candidate.provider} requires an API key`);
  if (apiKey.length > 10_000) throw new Error("API key is too long");

  let customEndpoint: string | undefined;
  if (candidate.provider === "custom") {
    customEndpoint = normalizeCustomEndpoint(
      typeof candidate.customEndpoint === "string" ? candidate.customEndpoint : ""
    );
    if (customEndpoint.length > CUSTOM_ENDPOINT_MAX_LENGTH) throw new Error("Endpoint URL is too long");
  }

  return {
    provider: candidate.provider,
    model,
    apiKey,
    ...(customEndpoint ? { customEndpoint } : {}),
    ...captureSelections
  };
}

export function assertInferenceOnboardingAvailability(
  selection: InferenceOnboardingSelection,
  appleAvailability: AppleInferenceAvailability
): void {
  if (selection.provider !== "apple" || appleAvailability.available) return;
  throw new Error(
    appleAvailability.reason ?? "Apple's on-device model is unavailable on this Mac."
  );
}
