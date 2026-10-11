// Pointer-following highlight for card borders. polish.css draws the light at
// (--mx, --my); this keeps those two variables on whichever surfaces are under
// the pointer, nested ones included.

const SURFACES = ".card, .upload-card, .studio-card, .summary-card, .chart-card, .hero, .auth-card, .finding";

export function installSpotlight() {
  let frame = 0;
  let lastEvent = null;

  const apply = () => {
    frame = 0;
    let surface = lastEvent.target instanceof Element ? lastEvent.target.closest(SURFACES) : null;

    while (surface) {
      const box = surface.getBoundingClientRect();
      surface.style.setProperty("--mx", `${lastEvent.clientX - box.left}px`);
      surface.style.setProperty("--my", `${lastEvent.clientY - box.top}px`);
      surface = surface.parentElement?.closest(SURFACES);
    }
  };

  // At most one update per frame, however fast the pointer moves
  const onMove = (event) => {
    lastEvent = event;
    if (!frame) frame = requestAnimationFrame(apply);
  };

  document.addEventListener("pointermove", onMove, { passive: true });

  return () => {
    document.removeEventListener("pointermove", onMove);
    if (frame) cancelAnimationFrame(frame);
  };
}
