import storage from '@/utils/authorization-util';
import { resolveResendOptions } from './utils';

describe('resolveResendOptions', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it('uses the current thinking selector value instead of the previous request', () => {
    storage.setThinkingLevel('0');

    expect(
      resolveResendOptions({ enableThinking: '3', enableInternet: true }),
    ).toEqual({
      enableThinking: '0',
      enableInternet: true,
    });
  });
});
