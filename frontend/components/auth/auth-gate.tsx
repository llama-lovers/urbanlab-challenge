"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type FC,
  type PropsWithChildren,
} from "react";
import { LogOutIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import * as api from "@/lib/api";

type AuthState = {
  user: api.AuthUser | null;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string) => Promise<void>;
  logout: () => void;
};

const AuthContext = createContext<AuthState | null>(null);

export const useAuth = (): AuthState => {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within <AuthProvider>");
  return ctx;
};

/**
 * Auth is OPTIONAL — the backend allows anonymous chat. This provider never
 * gates the app; it just tracks an optional logged-in user (for history tied
 * to an account) and is always rendered alongside the chat.
 */
export const AuthProvider: FC<PropsWithChildren> = ({ children }) => {
  const [user, setUser] = useState<api.AuthUser | null>(null);

  useEffect(() => {
    if (!api.getToken()) return;
    api
      .me()
      .then(setUser)
      .catch(() => api.clearToken());
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    await api.login(email, password);
    setUser(await api.me());
  }, []);

  const register = useCallback(async (email: string, password: string) => {
    await api.register(email, password);
    setUser(await api.me());
  }, []);

  const logout = useCallback(() => {
    api.clearToken();
    setUser(null);
  }, []);

  return (
    <AuthContext.Provider value={{ user, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

/** Header control: shows the account + logout, or an optional "Sign in" dialog. */
export const AuthControls: FC = () => {
  const { user, logout } = useAuth();
  const [open, setOpen] = useState(false);

  if (user) {
    return (
      <div className="flex items-center gap-2">
        <span className="text-muted-foreground hidden text-sm sm:inline">
          {user.email}
        </span>
        <Button variant="ghost" size="icon" onClick={logout} aria-label="Wyloguj się">
          <LogOutIcon className="size-4" />
        </Button>
      </div>
    );
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm">
          Zaloguj się
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-sm">
        <LoginForm onSuccess={() => setOpen(false)} />
      </DialogContent>
    </Dialog>
  );
};

const LoginForm: FC<{ onSuccess?: () => void }> = ({ onSuccess }) => {
  const { login, register } = useAuth();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await (mode === "login" ? login : register)(email, password);
      onSuccess?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Coś poszło nie tak");
    } finally {
      setBusy(false);
    }
  };

  return (
    <form onSubmit={onSubmit} className="flex flex-col gap-4">
      <DialogHeader>
        <DialogTitle>
          {mode === "login" ? "Zaloguj się" : "Załóż konto"}
        </DialogTitle>
        <DialogDescription>
          Logowanie jest opcjonalne — pozwala zapisać historię rozmów na koncie.
        </DialogDescription>
      </DialogHeader>

      <Input
        type="email"
        placeholder="email@example.com"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        required
        autoComplete="email"
      />
      <Input
        type="password"
        placeholder="Hasło"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        required
        minLength={6}
        autoComplete={mode === "login" ? "current-password" : "new-password"}
      />

      {error && <p className="text-destructive text-sm">{error}</p>}

      <Button type="submit" disabled={busy}>
        {busy ? "…" : mode === "login" ? "Zaloguj się" : "Zarejestruj się"}
      </Button>

      <button
        type="button"
        className="text-muted-foreground hover:text-foreground text-center text-sm"
        onClick={() => {
          setMode((m) => (m === "login" ? "register" : "login"));
          setError(null);
        }}
      >
        {mode === "login"
          ? "Nie masz konta? Zarejestruj się"
          : "Masz już konto? Zaloguj się"}
      </button>
    </form>
  );
};
