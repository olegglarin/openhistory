export const INFERENCE_PROVIDERS = ["apple", "openai", "anthropic", "kimi", "custom"] as const;
export const CLOUD_INFERENCE_PROVIDERS = ["openai", "anthropic", "kimi", "custom"] as const;

export type InferenceProvider = (typeof INFERENCE_PROVIDERS)[number];
export type CloudInferenceProvider = (typeof CLOUD_INFERENCE_PROVIDERS)[number];
export type ApiKeySource = "saved" | "environment" | "none";

export interface InferenceSettings {
  version: 1;
  enabled: boolean;
  provider: InferenceProvider;
  models: Record<InferenceProvider, string>;
  customEndpoint?: string;
}

export interface InferenceState {
  settings: InferenceSettings;
  configured: boolean;
  appleAvailability: AppleInferenceAvailability;
  keySources: Record<InferenceProvider, ApiKeySource>;
}

export interface AppleInferenceAvailability {
  available: boolean;
  reason?: string;
  reasonCode?: AppleInferenceUnavailabilityReason;
}

export type AppleInferenceUnavailabilityReason =
  | "unsupportedOperatingSystem"
  | "appleIntelligenceNotEnabled"
  | "deviceNotEligible"
  | "modelNotReady"
  | "foundationModelsUnavailable"
  | "foundationModelsFrameworkMissing"
  | "unsupportedPlatform"
  | "helperMissing"
  | "workerLaunchFailed"
  | "invalidWorkerResponse";

export interface AppleInferenceAvailabilityGuidance {
  title: string;
  description: string;
  helpLabel?: string;
  helpUrl?: string;
}

export function appleInferenceAvailabilityGuidance(
  availability: AppleInferenceAvailability
): AppleInferenceAvailabilityGuidance {
  switch (availability.reasonCode) {
    case "appleIntelligenceNotEnabled":
      return {
        title: "Turn on Apple Intelligence",
        description: "Apple Intelligence is turned off on this Mac. Turn it on in System Settings, then come back and check again.",
        helpLabel: "How to turn it on",
        helpUrl: "https://support.apple.com/guide/mac-help/mchl46361784/mac"
      };
    case "unsupportedOperatingSystem":
      return {
        title: "Update macOS to use Apple On-Device",
        description: `${availability.reason ?? "Apple On-Device requires macOS 26 or later."} Update macOS, then reopen OpenHistory.`,
        helpLabel: "How to update macOS",
        helpUrl: "https://support.apple.com/guide/mac-help/mchlpx1065/mac"
      };
    case "deviceNotEligible":
      return {
        title: "This Mac is not compatible",
        description: "Apple On-Device requires a Mac that supports Apple Intelligence. Choose a cloud provider to continue."
      };
    case "modelNotReady":
      return {
        title: "Apple Intelligence is still getting ready",
        description: "The on-device model may still be downloading. Keep this Mac connected to power and Wi-Fi, then check again."
      };
    case "foundationModelsFrameworkMissing":
    case "helperMissing":
      return {
        title: "This OpenHistory build cannot use Apple On-Device",
        description: `${availability.reason ?? "The required on-device model component is missing."} Install a newer OpenHistory build or choose a cloud provider.`
      };
    case "unsupportedPlatform":
      return {
        title: "Apple On-Device requires a Mac",
        description: availability.reason ?? "Apple's on-device model is available only on macOS."
      };
    case "workerLaunchFailed":
    case "invalidWorkerResponse":
    case "foundationModelsUnavailable":
    default:
      return {
        title: "Apple On-Device is unavailable",
        description: `${availability.reason ?? "This Mac does not currently meet the on-device model requirements."} Check again or choose a cloud provider.`
      };
  }
}

export interface InferenceModelOption {
  id: string;
  label: string;
  description: string;
}

export const INFERENCE_PROVIDER_LABELS: Record<InferenceProvider, string> = {
  apple: "Apple On-Device (Experimental)",
  openai: "OpenAI",
  anthropic: "Anthropic",
  kimi: "Kimi (Moonshot AI)",
  custom: "Custom"
};

export const INFERENCE_MODEL_OPTIONS: Record<InferenceProvider, InferenceModelOption[]> = {
  apple: [
    { id: "system-default", label: "System Language Model", description: "Private and on-device" }
  ],
  openai: [
    { id: "gpt-5.6-luna", label: "GPT-5.6 Luna", description: "Efficient, high-volume" },
    { id: "gpt-5.6-terra", label: "GPT-5.6 Terra", description: "Balanced" },
    { id: "gpt-5.6-sol", label: "GPT-5.6 Sol", description: "Highest capability" }
  ],
  anthropic: [
    { id: "claude-haiku-4-5", label: "Claude Haiku 4.5", description: "Fastest" },
    { id: "claude-sonnet-5", label: "Claude Sonnet 5", description: "Balanced" },
    { id: "claude-opus-5", label: "Claude Opus 5", description: "Highest capability" }
  ],
  kimi: [
    { id: "kimi-k3", label: "Kimi K3", description: "Flagship model" }
  ],
  custom: []
};

export const DEFAULT_INFERENCE_MODELS: Record<InferenceProvider, string> = {
  apple: "system-default",
  openai: "gpt-5.6-luna",
  anthropic: "claude-sonnet-5",
  kimi: "kimi-k3",
  custom: ""
};

export const CUSTOM_MODEL_MAX_LENGTH = 200;
export const CUSTOM_ENDPOINT_MAX_LENGTH = 2_048;

/**
 * Normalize a user-supplied OpenAI-compatible endpoint into the SDK base URL.
 * Bare hosts (for example "http://localhost:11434") get the conventional "/v1"
 * suffix; paths already present in the URL are preserved as entered.
 */
export function normalizeCustomEndpoint(raw: string): string {
  const trimmed = raw.trim();
  if (!trimmed) throw new Error("Enter an endpoint URL for the custom provider");
  let url: URL;
  try {
    url = new URL(trimmed);
  } catch {
    throw new Error("Enter a valid http or https endpoint URL");
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new Error("Enter a valid http or https endpoint URL");
  }
  const base = `${url.origin}${url.pathname}`.replace(/\/+$/, "");
  return url.pathname.length > 1 ? base : `${base}/v1`;
}

export function isInferenceProvider(value: unknown): value is InferenceProvider {
  return typeof value === "string" && INFERENCE_PROVIDERS.includes(value as InferenceProvider);
}

export function isCloudInferenceProvider(value: unknown): value is CloudInferenceProvider {
  return typeof value === "string" && CLOUD_INFERENCE_PROVIDERS.includes(value as CloudInferenceProvider);
}
