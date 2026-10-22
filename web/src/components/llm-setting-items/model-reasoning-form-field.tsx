import { LLMFactory } from '@/constants/llm';
import { getMwsReasoningEfforts } from '@/constants/reasoning';
import { useFetchAllAddedModels } from '@/hooks/use-llm-request';
import { parseModelValue } from '@/utils/llm-util';
import { useEffect, useMemo } from 'react';
import { useFormContext, useWatch } from 'react-hook-form';
import { useTranslation } from 'react-i18next';
import {
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from '../ui/form';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '../ui/select';

export function ModelReasoningFormField({
  prefix,
  modelFieldName,
  ownerTenantId,
}: {
  prefix?: string;
  modelFieldName?: string;
  ownerTenantId?: string;
}) {
  const form = useFormContext();
  const { t } = useTranslation();
  const fieldPrefix = prefix ? `${prefix}.` : '';
  const modelValue = useWatch({
    control: form.control,
    name: modelFieldName ?? `${fieldPrefix}llm_id`,
  });
  const effort = useWatch({
    control: form.control,
    name: `${fieldPrefix}reasoning_effort`,
  });
  const { data: models } = useFetchAllAddedModels(undefined, ownerTenantId);
  const parsed = parseModelValue(modelValue ?? '');
  const model = models.find((item) => item.model_id === modelValue);
  const isMws =
    (model?.provider_name ?? parsed?.model_provider) === LLMFactory.MWS;
  const modelName = model?.name ?? parsed?.model_name ?? '';
  const efforts = useMemo(() => getMwsReasoningEfforts(modelName), [modelName]);

  useEffect(() => {
    if (effort && effort !== 'default' && isMws && !efforts.includes(effort)) {
      form.setValue(`${fieldPrefix}reasoning_effort`, 'default', {
        shouldDirty: true,
      });
    }
  }, [modelValue, effort, isMws, efforts, fieldPrefix, form]);

  const name = `${fieldPrefix}${isMws ? 'reasoning_effort' : 'thinking'}`;
  const options = isMws ? efforts : ['default', 'enabled', 'disabled'];
  return (
    <FormField
      key={name}
      control={form.control}
      name={name}
      render={({ field }) => (
        <FormItem className="flex justify-between items-center">
          <FormLabel
            className="flex-1"
            tooltip={t(isMws ? 'chat.reasoningEffortTip' : 'chat.thinkingTip')}
          >
            {t(isMws ? 'chat.reasoningEffort' : 'chat.thinkingMode')}
          </FormLabel>
          <FormControl>
            <Select
              value={field.value ?? 'default'}
              onValueChange={field.onChange}
            >
              <SelectTrigger className="flex-1 !m-0">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {options.map((value) => (
                  <SelectItem value={value} key={value}>
                    {t(
                      isMws
                        ? `chat.reasoningEffortOptions.${value}`
                        : `chat.thinking${value[0].toUpperCase()}${value.slice(1)}`,
                    )}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </FormControl>
          <FormMessage />
        </FormItem>
      )}
    />
  );
}
