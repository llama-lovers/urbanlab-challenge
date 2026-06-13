"use client";

import type { SyntaxHighlighterProps } from "@assistant-ui/react-markdown";
import type { FC } from "react";

/**
 * Minimal, dependency-free code renderer used by `markdown-text.tsx`.
 * Renders code blocks verbatim (no colour highlighting). To enable real
 * Shiki highlighting later, replace this with the assistant-ui registry
 * component: `npx shadcn@latest add @assistant-ui/shiki-highlighter`.
 */
export const SyntaxHighlighter: FC<Omit<SyntaxHighlighterProps, "node">> = ({
  components: { Pre, Code },
  code,
}) => {
  return (
    <Pre>
      <Code>{code}</Code>
    </Pre>
  );
};
