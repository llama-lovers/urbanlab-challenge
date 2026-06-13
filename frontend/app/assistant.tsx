"use client";

import { useMemo } from "react";
import {
  AssistantRuntimeProvider,
  SimpleImageAttachmentAdapter,
  useLocalRuntime,
} from "@assistant-ui/react";
import { UrbanLabAdapter } from "@/lib/chat-adapter";
import { AuthProvider, AuthControls } from "@/components/auth/auth-gate";
import { Thread } from "@/components/assistant-ui/thread";
import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar";
import { ThreadListSidebar } from "@/components/assistant-ui/threadlist-sidebar";
import { Separator } from "@/components/ui/separator";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbList,
  BreadcrumbPage,
} from "@/components/ui/breadcrumb";

export const Assistant = () => {
  return (
    <AuthProvider>
      <Chat />
    </AuthProvider>
  );
};

const Chat = () => {
  const adapter = useMemo(() => new UrbanLabAdapter(), []);
  const attachments = useMemo(() => new SimpleImageAttachmentAdapter(), []);
  const runtime = useLocalRuntime(adapter, { adapters: { attachments } });

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <SidebarProvider>
        <div className="flex h-dvh w-full pr-0.5">
          <ThreadListSidebar />
          <SidebarInset>
            <header className="flex h-16 shrink-0 items-center gap-2 border-b px-4">
              <SidebarTrigger />
              <Separator orientation="vertical" className="mr-2 h-4" />
              <Breadcrumb>
                <BreadcrumbList>
                  <BreadcrumbItem>
                    <BreadcrumbPage>UrbanLab Assistant</BreadcrumbPage>
                  </BreadcrumbItem>
                </BreadcrumbList>
              </Breadcrumb>
              <div className="ml-auto">
                <AuthControls />
              </div>
            </header>
            <div className="flex-1 overflow-hidden">
              <Thread />
            </div>
          </SidebarInset>
        </div>
      </SidebarProvider>
    </AssistantRuntimeProvider>
  );
};
