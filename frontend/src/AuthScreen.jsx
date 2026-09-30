import { useState } from "react";
import { login, register } from "./api";

function AuthScreen({ onAuthenticated, notice }) {
  const [mode, setMode] = useState("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const isLogin = mode === "login";

  const submit = async (event) => {
    event.preventDefault();
    setError("");
    setBusy(true);
    try {
      const user = isLogin ? await login(email, password) : await register(email, password);
      onAuthenticated(user);
    } catch (err) {
      setError(err.message || "Something went wrong.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-card">
        <div className="logo-block">
          <div className="logo-icon">S</div>
          <div>
            <h1>STATMIND</h1>
            <p>Data Intelligence OS</p>
          </div>
        </div>

        <div className="auth-copy">
          <h2>{isLogin ? "Welcome back" : "Create your workspace"}</h2>
          <p>
            {isLogin
              ? "Log in to your datasets, models and experiments."
              : "Your datasets and models are private to your account."}
          </p>
        </div>

        {notice && <div className="auth-notice">{notice}</div>}

        <form className="login-form" onSubmit={submit}>
          <label htmlFor="auth-email">Email</label>
          <input
            id="auth-email"
            type="email"
            autoComplete="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />

          <label htmlFor="auth-password">Password</label>
          <input
            id="auth-password"
            type="password"
            autoComplete={isLogin ? "current-password" : "new-password"}
            minLength={isLogin ? undefined : 8}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
          {!isLogin && <span className="auth-hint">At least 8 characters.</span>}

          {error && <div className="auth-error" role="alert">{error}</div>}

          <button type="submit" disabled={busy}>
            {busy ? "Please wait..." : isLogin ? "Log in" : "Create account"}
          </button>
        </form>

        <button
          type="button"
          className="auth-switch"
          onClick={() => {
            setMode(isLogin ? "register" : "login");
            setError("");
          }}
        >
          {isLogin ? "New here? Create an account" : "Already have an account? Log in"}
        </button>
      </div>
    </div>
  );
}

export default AuthScreen;
