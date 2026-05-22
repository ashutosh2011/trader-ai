// Minimal vanilla-JS toast notification system shared across pages.
//
// Pages call window.tbToast(message, type) where type is "ok" | "error"
// | "info". A fixed-position container is created on first use and
// recycled afterward. Toasts auto-dismiss after 3.5s (errors after 5s).
(function () {
  const CONTAINER_ID = 'tb-toast-container';

  function ensureContainer() {
    let container = document.getElementById(CONTAINER_ID);
    if (container) return container;
    container = document.createElement('div');
    container.id = CONTAINER_ID;
    container.className = 'fixed top-4 right-4 z-50 flex flex-col gap-2 pointer-events-none';
    document.body.appendChild(container);
    return container;
  }

  function classesFor(type) {
    if (type === 'ok') {
      return 'bg-green-900/80 border-green-700 text-green-100';
    }
    if (type === 'error') {
      return 'bg-red-900/80 border-red-700 text-red-100';
    }
    return 'bg-gray-800 border-gray-700 text-gray-100';
  }

  function show(message, type) {
    const container = ensureContainer();
    const toast = document.createElement('div');
    toast.className =
      'pointer-events-auto rounded border px-3 py-2 text-sm shadow-lg max-w-sm transition-opacity duration-300 ' +
      classesFor(type || 'info');
    toast.style.opacity = '0';
    toast.textContent = String(message);
    container.appendChild(toast);
    requestAnimationFrame(() => { toast.style.opacity = '1'; });
    const lifetime = type === 'error' ? 5000 : 3500;
    setTimeout(() => {
      toast.style.opacity = '0';
      setTimeout(() => { toast.remove(); }, 320);
    }, lifetime);
  }

  window.tbToast = show;
})();
