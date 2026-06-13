"use client";

import { useState } from "react";
import { CheckIcon, ChevronDownIcon, ChevronUpIcon, WrenchIcon } from "lucide-react";
import type { ToolCallMessagePartComponent } from "@assistant-ui/react";
import { cn } from "@/lib/utils";

/**
 * Minimal renderer for tool calls. This assistant doesn't use tools, so this
 * is only a safe fallback in case a tool-call part ever appears.
 */
export const ToolFallback: ToolCallMessagePartComponent = ({
  toolName,
  argsText,
  result,
}) => {
  const [open, setOpen] = useState(false);
  const isComplete = result !== undefined;

  return (
    <div className="aui-tool-fallback-root mb-4 flex w-full flex-col gap-2 rounded-lg border py-2">
      <div className="aui-tool-fallback-header flex items-center gap-2 px-3">
        {isComplete ? (
          <CheckIcon className="size-4 text-green-600" />
        ) : (
          <WrenchIcon className="size-4" />
        )}
        <span className="flex-grow text-sm">
          Tool: <b>{toolName}</b>
        </span>
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          className="text-muted-foreground hover:text-foreground"
          aria-label={open ? "Collapse" : "Expand"}
        >
          {open ? (
            <ChevronUpIcon className="size-4" />
          ) : (
            <ChevronDownIcon className="size-4" />
          )}
        </button>
      </div>
      {open && (
        <div className={cn("aui-tool-fallback-content flex flex-col gap-2 border-t pt-2")}>
          <div className="px-3">
            <pre className="text-muted-foreground overflow-x-auto text-xs whitespace-pre-wrap">
              {argsText}
            </pre>
          </div>
          {isComplete && (
            <div className="border-t border-dashed px-3 pt-2">
              <pre className="overflow-x-auto text-xs whitespace-pre-wrap">
                {typeof result === "string"
                  ? result
                  : JSON.stringify(result, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
