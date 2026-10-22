export const ReasoningEffortValues = [
  'default',
  'none',
  'minimal',
  'low',
  'medium',
  'high',
  'max',
] as const;

export type ReasoningEffort = (typeof ReasoningEffortValues)[number];

// https://mws.ru/docs/cloud-platform/gpt/general/reasoning.html
const MwsReasoningEfforts: Record<string, readonly ReasoningEffort[]> = {
  'qwen3-6-35b-a3b': ['none', 'medium'],
  'glm-5.3': ['low', 'high', 'max'],
  'kimi-k2.6': ['none', 'low', 'high'],
  'gemma-4-31b-it': ['none', 'medium'],
  'gpt-oss-120b': ['low', 'medium', 'high'],
};

export function getMwsReasoningEfforts(
  modelName: string,
): readonly ReasoningEffort[] {
  const name = modelName.split('___')[0].trim().toLowerCase();
  return [
    'default',
    ...(MwsReasoningEfforts[name] ?? ReasoningEffortValues.slice(1)),
  ];
}
