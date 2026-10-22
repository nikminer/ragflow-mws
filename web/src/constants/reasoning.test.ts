import { getMwsReasoningEfforts, ReasoningEffortValues } from './reasoning';

test.each([
  ['gpt-oss-120b', ['default', 'low', 'medium', 'high']],
  [' QWEN3-6-35B-A3B___MWS ', ['default', 'none', 'medium']],
  ['glm-5.3', ['default', 'low', 'high', 'max']],
  ['kimi-k2.6', ['default', 'none', 'low', 'high']],
  ['gemma-4-31b-it', ['default', 'none', 'medium']],
])('offers documented effort values for %s', (name, expected) => {
  expect(getMwsReasoningEfforts(name as string)).toEqual(expected);
});

test('lets custom deployment ids use the supported effort vocabulary', () => {
  expect(getMwsReasoningEfforts('my-deployment')).toEqual(
    ReasoningEffortValues,
  );
});
