'use client'

import { useMemo } from 'react'
import {
  AssistantRuntimeProvider,
  SimpleImageAttachmentAdapter,
  useAui,
  useLocalRuntime,
  useRemoteThreadListRuntime,
} from '@assistant-ui/react'
import { UrbanLabAdapter } from '@/lib/chat-adapter'
import { useBackendThreadListAdapter } from '@/lib/thread-list-adapter'
import { WhisperDictationAdapter } from '@/lib/dictation-adapter'
import { AuthProvider, AuthControls, useAuth } from '@/components/auth/auth-gate'
import { Thread } from '@/components/assistant-ui/thread'
import { SidebarInset, SidebarProvider, SidebarTrigger } from '@/components/ui/sidebar'
import { ThreadListSidebar } from '@/components/assistant-ui/threadlist-sidebar'
import { Separator } from '@/components/ui/separator'
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbList,
  BreadcrumbPage,
} from '@/components/ui/breadcrumb'

export const Assistant = () => {
  return (
    <AuthProvider>
      <Chat />
    </AuthProvider>
  )
}

/**
 * Per-thread runtime. The backend session id is the thread's `remoteId`, which
 * the thread-list runtime resolves (creating a session lazily on the first
 * message via the adapter's `initialize`). `useAui().threadListItem()` gives us
 * that id race-free once initialization has resolved.
 */
const useUrbanLabThreadRuntime = () => {
  const aui = useAui()
  const adapter = useMemo(
    () =>
      new UrbanLabAdapter(async () => {
        const { remoteId } = await aui.threadListItem().initialize()
        return remoteId
      }),
    [aui],
  )
  const attachments = useMemo(() => new SimpleImageAttachmentAdapter(), [])
  const dictation = useMemo(() => new WhisperDictationAdapter(), [])
  return useLocalRuntime(adapter, { adapters: { attachments, dictation } })
}

const Chat = () => {
  const { user } = useAuth()
  // Remount on auth change so the thread list re-fetches with the new token
  // (anonymous → logged-in shows that account's saved sessions).
  return <ChatRuntime key={user?.id ?? 'anon'} />
}

const ChatRuntime = () => {
  const threadListAdapter = useBackendThreadListAdapter()
  const runtime = useRemoteThreadListRuntime({
    runtimeHook: useUrbanLabThreadRuntime,
    adapter: threadListAdapter,
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <SidebarProvider>
        <div className="flex h-dvh w-full pr-0.5">
          <ThreadListSidebar />
          <SidebarInset>
            <header className="flex h-16 shrink-0 items-center gap-2 border-t-4 border-t-primary border-b border-border bg-card px-4">
              <SidebarTrigger />
              <Separator orientation="vertical" className="mr-2 h-4" />
              <Breadcrumb>
                <BreadcrumbList>
                  <BreadcrumbItem>
                    <BreadcrumbPage className="font-semibold tracking-tight">
                      UrbanLab Assistant
                    </BreadcrumbPage>
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
  )
}
