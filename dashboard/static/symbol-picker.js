// Vanilla JS autocomplete for any <input data-symbol-picker="true">.
//
// Pairs with the Jinja macro in templates/partials/symbol_picker.html and
// is auto-registered on DOMContentLoaded. Supports keyboard nav, click
// outside to dismiss, and writing the selected instrument_token into a
// sibling <input id="<picker-id>-token"> when one exists. A picker that
// auto-fills a token field also dispatches a CustomEvent named
// "symbol-picker:select" so callers can react (e.g. flip a data-source
// toggle to "kite").
(function () {
  const ACTIVE_CLS = 'bg-gray-800';
  const ROW_CLS = 'px-3 py-1.5 text-sm cursor-pointer flex items-baseline gap-2 hover:bg-gray-800';

  function escapeHtml(str) {
    return String(str).replace(/[&<>"']/g, (c) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[c]));
  }

  function setTokenPill(picker, entry) {
    const pillId = picker.input.dataset.tokenPill;
    if (!pillId) return;
    const el = document.getElementById(pillId);
    if (!el) return;
    const valueEl = el.querySelector('[data-role=token-value]');
    if (!valueEl) return;
    if (entry && entry.instrument_token) {
      valueEl.textContent = entry.instrument_token + ' (' + entry.exchange + ')';
      valueEl.classList.remove('text-gray-500');
      valueEl.classList.add('text-blue-300');
    } else {
      valueEl.textContent = '—';
      valueEl.classList.remove('text-blue-300');
      valueEl.classList.add('text-gray-500');
    }
  }

  function writeTokenTarget(picker, entry) {
    const targetId = picker.input.dataset.tokenTarget;
    if (!targetId) return;
    const el = document.getElementById(targetId);
    if (!el) return;
    if (entry && entry.instrument_token) {
      el.value = String(entry.instrument_token);
      el.dispatchEvent(new Event('input', { bubbles: true }));
      el.dispatchEvent(new Event('change', { bubbles: true }));
    }
  }

  function renderResults(picker, results) {
    picker.dropdown.innerHTML = '';
    picker.currentIndex = -1;
    picker.results = results;
    if (!results.length) {
      picker.dropdown.classList.add('hidden');
      return;
    }
    const frag = document.createDocumentFragment();
    results.forEach((entry, idx) => {
      const row = document.createElement('div');
      row.className = ROW_CLS;
      row.setAttribute('role', 'option');
      row.dataset.index = String(idx);
      row.innerHTML =
        '<span class="font-mono font-semibold text-gray-100">' + escapeHtml(entry.symbol) + '</span>' +
        '<span class="text-xs text-gray-500">' + escapeHtml(entry.exchange) + '</span>' +
        (entry.instrument_token ?
          '<span class="ml-auto text-xs text-gray-500 font-mono">' + escapeHtml(String(entry.instrument_token)) + '</span>' :
          '<span class="ml-auto text-xs text-gray-600">no token</span>');
      row.addEventListener('mousedown', (ev) => {
        // mousedown (not click) so the input's blur doesn't fire first and hide us.
        ev.preventDefault();
        choose(picker, entry);
      });
      row.addEventListener('mouseenter', () => setActive(picker, idx));
      frag.appendChild(row);
    });
    picker.dropdown.appendChild(frag);
    picker.dropdown.classList.remove('hidden');
  }

  function setActive(picker, index) {
    const rows = picker.dropdown.querySelectorAll('[role=option]');
    rows.forEach((r) => r.classList.remove(ACTIVE_CLS));
    if (index < 0 || index >= rows.length) {
      picker.currentIndex = -1;
      return;
    }
    picker.currentIndex = index;
    rows[index].classList.add(ACTIVE_CLS);
    rows[index].scrollIntoView({ block: 'nearest' });
  }

  function choose(picker, entry) {
    picker.input.value = entry.symbol;
    picker.dropdown.classList.add('hidden');
    setTokenPill(picker, entry);
    writeTokenTarget(picker, entry);
    picker.input.dispatchEvent(new CustomEvent('symbol-picker:select', {
      bubbles: true,
      detail: entry,
    }));
    picker.input.dispatchEvent(new Event('input', { bubbles: true }));
    picker.input.dispatchEvent(new Event('change', { bubbles: true }));
  }

  async function fetchResults(query) {
    const url = '/api/symbols/search?q=' + encodeURIComponent(query) + '&limit=10';
    const res = await fetch(url, { headers: { 'Accept': 'application/json' } });
    if (!res.ok) return [];
    const data = await res.json().catch(() => ({ results: [] }));
    return Array.isArray(data.results) ? data.results : [];
  }

  function debounce(fn, ms) {
    let handle = null;
    return function (...args) {
      if (handle) clearTimeout(handle);
      handle = setTimeout(() => fn.apply(this, args), ms);
    };
  }

  function attach(input) {
    if (input.dataset.symbolPickerReady === 'true') return;
    input.dataset.symbolPickerReady = 'true';
    const root = input.closest('.symbol-picker-root');
    if (!root) return;
    const dropdown = root.querySelector('.symbol-picker-dropdown');
    if (!dropdown) return;
    const picker = { input, dropdown, results: [], currentIndex: -1 };

    const refresh = async () => {
      const results = await fetchResults(input.value);
      renderResults(picker, results);
    };
    const debouncedRefresh = debounce(refresh, 80);

    input.addEventListener('focus', () => { refresh(); });
    input.addEventListener('input', () => { debouncedRefresh(); });
    input.addEventListener('keydown', (ev) => {
      const rows = picker.dropdown.querySelectorAll('[role=option]');
      if (ev.key === 'ArrowDown') {
        ev.preventDefault();
        if (!rows.length) { refresh(); return; }
        const next = (picker.currentIndex + 1) % rows.length;
        setActive(picker, next);
      } else if (ev.key === 'ArrowUp') {
        ev.preventDefault();
        if (!rows.length) return;
        const prev = (picker.currentIndex - 1 + rows.length) % rows.length;
        setActive(picker, prev);
      } else if (ev.key === 'Enter') {
        if (picker.currentIndex >= 0 && picker.results[picker.currentIndex]) {
          ev.preventDefault();
          choose(picker, picker.results[picker.currentIndex]);
        }
      } else if (ev.key === 'Escape') {
        picker.dropdown.classList.add('hidden');
      }
    });
    input.addEventListener('blur', () => {
      setTimeout(() => picker.dropdown.classList.add('hidden'), 120);
    });
    document.addEventListener('mousedown', (ev) => {
      if (!root.contains(ev.target)) picker.dropdown.classList.add('hidden');
    });
  }

  function init() {
    document.querySelectorAll('input[data-symbol-picker="true"]').forEach(attach);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  window.tbSymbolPicker = { init, attach };
})();
