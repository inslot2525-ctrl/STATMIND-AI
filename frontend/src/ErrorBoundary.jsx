// Last line of defence: if a component throws while rendering, show a message
// and a way back instead of a blank page.

import { Component } from "react";

export default class ErrorBoundary extends Component {
  state = { error: null };

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    console.error("Render error:", error, info.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;

    return (
      <div className="crash-screen">
        <h2>Something went wrong on this screen</h2>
        <p>Your data and models are safe. Reload to carry on.</p>
        <code>{String(this.state.error.message || this.state.error)}</code>
        <button className="primary" type="button" onClick={() => window.location.reload()}>
          Reload the app
        </button>
      </div>
    );
  }
}
