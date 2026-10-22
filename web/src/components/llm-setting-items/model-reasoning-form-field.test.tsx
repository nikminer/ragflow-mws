import { useEffect } from 'react';
import { TooltipProvider } from '@/components/ui/tooltip';
import { fireEvent, render, screen, within } from '@testing-library/react';
import { FormProvider, useForm, useWatch } from 'react-hook-form';
import { useFetchAllAddedModels } from '@/hooks/use-llm-request';
import { ModelReasoningFormField } from './model-reasoning-form-field';

jest.mock('@/hooks/use-llm-request', () => ({
  useFetchAllAddedModels: jest.fn(),
}));
jest.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));
jest.mock('../ui/select', () => ({
  Select: ({ value, onValueChange, children }: any) => (
    <select
      value={value}
      onChange={(event) => onValueChange(event.target.value)}
    >
      {children}
    </select>
  ),
  SelectContent: ({ children }: any) => <>{children}</>,
  SelectItem: ({ value, children }: any) => (
    <option value={value}>{children}</option>
  ),
  SelectTrigger: () => null,
  SelectValue: () => null,
}));

function Harness({
  values,
  prefix,
  ownerTenantId,
}: {
  values: Record<string, any>;
  prefix?: string;
  ownerTenantId?: string;
}) {
  const form = useForm({ defaultValues: values });
  const state = useWatch({ control: form.control });
  useEffect(() => form.reset(values), [form, values]);
  return (
    <TooltipProvider>
      <FormProvider {...form}>
        <input aria-label="Model" {...form.register('llm_id')} />
        <ModelReasoningFormField
          prefix={prefix}
          modelFieldName="llm_id"
          ownerTenantId={ownerTenantId}
        />
        <output data-testid="params">{JSON.stringify(state)}</output>
      </FormProvider>
    </TooltipProvider>
  );
}

const readParams = () => JSON.parse(screen.getByTestId('params').textContent!);

beforeEach(() => {
  jest.mocked(useFetchAllAddedModels).mockReturnValue({ data: [] } as any);
});

test('offers model-specific MWS levels and saves the selected effort', () => {
  render(<Harness values={{ llm_id: 'gpt-oss-120b@default@MWS' }} />);
  const select = screen.getByRole('combobox');
  expect(
    within(select)
      .getAllByRole('option')
      .map((item) => item.getAttribute('value')),
  ).toEqual(['default', 'low', 'medium', 'high']);
  fireEvent.change(select, { target: { value: 'high' } });
  expect(readParams().reasoning_effort).toBe('high');
  expect(screen.queryByText('chat.thinkingMode')).not.toBeInTheDocument();
});

test('preserves an existing level and resets it when the model changes', () => {
  render(
    <Harness
      values={{ llm_id: 'gpt-oss-120b@default@MWS', reasoning_effort: 'high' }}
    />,
  );
  expect(screen.getByRole('combobox')).toHaveValue('high');
  fireEvent.change(screen.getByLabelText('Model'), {
    target: { value: 'qwen3-6-35b-a3b@default@MWS' },
  });
  expect(readParams().reasoning_effort).toBe('default');
  expect(
    within(screen.getByRole('combobox'))
      .getAllByRole('option')
      .map((item) => item.getAttribute('value')),
  ).toEqual(['default', 'none', 'medium']);
});

test('resolves opaque model ids using the canvas owner models and keeps nested chat settings', () => {
  jest.mocked(useFetchAllAddedModels).mockReturnValue({
    data: [{ model_id: 'model-id', name: 'glm-5.3', provider_name: 'MWS' }],
  } as any);
  render(
    <Harness
      values={{ llm_id: 'model-id', llm_setting: { reasoning_effort: 'max' } }}
      prefix="llm_setting"
      ownerTenantId="owner-id"
    />,
  );
  expect(screen.getByRole('combobox')).toHaveValue('max');
  expect(useFetchAllAddedModels).toHaveBeenCalledWith(undefined, 'owner-id');
  fireEvent.change(screen.getByRole('combobox'), { target: { value: 'low' } });
  expect(readParams().llm_setting.reasoning_effort).toBe('low');
});

test('keeps the existing thinking control for other providers', () => {
  render(
    <Harness values={{ llm_id: 'qwen@default@Other', thinking: 'enabled' }} />,
  );
  expect(screen.getByText('chat.thinkingMode')).toBeInTheDocument();
  expect(screen.getByRole('combobox')).toHaveValue('enabled');
  fireEvent.change(screen.getByRole('combobox'), {
    target: { value: 'disabled' },
  });
  expect(readParams().thinking).toBe('disabled');
});

test('keeps a valid saved effort when chat settings load after mount', () => {
  const { rerender } = render(<Harness values={{ llm_id: '' }} />);
  rerender(
    <Harness
      values={{ llm_id: 'gpt-oss-120b@default@MWS', reasoning_effort: 'high' }}
    />,
  );
  expect(screen.getByRole('combobox')).toHaveValue('high');
  expect(readParams().reasoning_effort).toBe('high');
});
