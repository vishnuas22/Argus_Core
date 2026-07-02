'use client';

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { MessageSquare } from 'lucide-react';
import type { Verdict } from '@/types/analysis';

export interface ChatContainerProps {
  analysisId: string;
  verdict?: Verdict;
  className?: string;
}

export function ChatContainer({
  analysisId,
  verdict,
  className,
}: ChatContainerProps): React.ReactElement {
  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <MessageSquare className="h-5 w-5" />
          AI Analysis Chat
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Alert>
          <AlertDescription>
            Chat with AI about analysis <strong>{analysisId}</strong>
            {verdict ? ` — Verdict: ${verdict}` : ''}
          </AlertDescription>
        </Alert>
      </CardContent>
    </Card>
  );
}
