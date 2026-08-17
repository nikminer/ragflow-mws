import { NextMessageInputOnPressEnterParameter } from '@/components/message-input/next';
import { EmptyConversationId } from '@/constants/chat';
import { IConversation, IReference } from '@/interfaces/database/chat';
import storage from '@/utils/authorization-util';

/**
 * Regenerate is triggered from the transcript, which has no direct access to
 * the input box controls. Thinking is persisted whenever its selector changes,
 * so read it at resend time; internet is not persisted and keeps the last send.
 */
export function resolveResendOptions(
  lastSendOptions: NextMessageInputOnPressEnterParameter,
): NextMessageInputOnPressEnterParameter {
  const { enableInternet = false } = lastSendOptions;

  return {
    enableThinking: storage.getThinkingLevel(),
    enableInternet,
  };
}

export const isConversationIdExist = (conversationId: string) => {
  return conversationId !== EmptyConversationId && conversationId !== '';
};

export const getDocumentIdsFromConversionReference = (data: IConversation) => {
  const documentIds = data.reference.reduce(
    (pre: Array<string>, cur: IReference) => {
      cur.doc_aggs
        ?.map((x) => x.doc_id)
        .forEach((x) => {
          if (pre.every((y) => y !== x)) {
            pre.push(x);
          }
        });
      return pre;
    },
    [],
  );
  return documentIds.join(',');
};

// Shared fallback so a message without a reference keeps handing MessageItem the
// same object across renders. A fresh literal here would break the item's memo
// on every streaming flush. See useMessageReferences.
export const EmptyReference: IReference = {
  doc_aggs: [],
  chunks: [],
  total: 0,
};
