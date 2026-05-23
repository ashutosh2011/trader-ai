// Modal command-palette symbol picker for the backtest pages.
//
// The form submits ONLY the symbol selected through this component — there
// is no free-text input that posts to the API. The component:
//   * renders a trigger button showing the current selection,
//   * caches the cached NSE instruments dump on first open,
//   * opens a modal with type-to-filter, arrow-key nav, Enter to select.
//
// Auto-attaches on DOMContentLoaded for elements with the
// ``data-symbol-select="true"`` attribute. The selection is readable via
// ``window.tbSymbolSelect.getSelection(rootEl)``.
(function () {
  function escapeHtml(str) {
    return String(str).replace(/[&<>"']/g, (c) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[c]));
  }

  function readEmbeddedOptions(root) {
    const node = root.querySelector('[data-symbol-options]');
    if (!node) return [];
    try {
      const parsed = JSON.parse(node.textContent || '[]');
      return Array.isArray(parsed) ? parsed : [];
    } catch (e) {
      return [];
    }
  }

  async function fetchAllOptions() {
    try {
      const res = await fetch('/api/symbols/search?q=&limit=2000', {
        headers: { 'Accept': 'application/json' },
      });
      if (!res.ok) return [];
      const data = await res.json().catch(() => ({ results: [] }));
      return Array.isArray(data.results) ? data.results : [];
    } catch (e) {
      return [];
    }
  }

  function rankOptions(options, query) {
    const target = (query || '').trim().toUpperCase();
    if (!target) return options.slice();
    const exact = [];
    const prefix = [];
    const substring = [];
    for (const opt of options) {
      const sym = String(opt.symbol || '').toUpperCase();
      if (sym === target) exact.push(opt);
      else if (sym.startsWith(target)) prefix.push(opt);
      else if (sym.includes(target)) substring.push(opt);
    }
    return exact.concat(prefix, substring);
  }

  function renderTrigger(component) {
    const { selection } = component;
    const triggerEl = component.trigger;
    if (!triggerEl) return;
    const symbolEl = triggerEl.querySelector('[data-role=trigger-symbol]');
    const metaEl = triggerEl.querySelector('[data-role=trigger-meta]');
    if (!selection) {
      if (symbolEl) symbolEl.textContent = 'Pick a symbol';
      if (metaEl) metaEl.textContent = 'click to search NSE instruments';
      triggerEl.classList.add('text-slate-500');
      return;
    }
    triggerEl.classList.remove('text-slate-500');
    if (symbolEl) symbolEl.textContent = selection.symbol;
    if (metaEl) {
      const token = selection.instrument_token != null
        ? String(selection.instrument_token)
        : 'no token';
      metaEl.textContent = `${selection.exchange} · ${token}`;
    }
  }

  function writeHidden(component) {
    const { selection } = component;
    const hidden = component.hidden;
    if (!hidden) return;
    hidden.value = selection ? JSON.stringify(selection) : '';
    hidden.dispatchEvent(new CustomEvent('symbol-select:change', {
      bubbles: true,
      detail: selection,
    }));
  }

  function setSelection(component, entry) {
    component.selection = entry ? Object.assign({}, entry) : null;
    renderTrigger(component);
    writeHidden(component);
  }

  function buildRows(component, filtered) {
    const list = component.list;
    list.innerHTML = '';
    component.activeIndex = -1;
    component.filtered = filtered;
    if (!filtered.length) {
      const empty = document.createElement('li');
      empty.className = 'px-4 py-3 text-sm text-slate-500';
      empty.textContent = 'No matches — refresh NSE instruments or adjust the query.';
      list.appendChild(empty);
      return;
    }
    filtered.forEach((entry, idx) => {
      const row = document.createElement('li');
      row.className = (
        'flex items-baseline gap-3 px-4 py-2.5 cursor-pointer ' +
        'text-base border-b border-slate-100 last:border-b-0 ' +
        'text-slate-800 hover:bg-emerald-50'
      );
      row.dataset.index = String(idx);
      const symBadge = `<span class="text-xs font-semibold uppercase tracking-wide text-slate-500">${escapeHtml(entry.exchange || '')}</span>`;
      const tokenStr = entry.instrument_token != null
        ? escapeHtml(String(entry.instrument_token))
        : '—';
      row.innerHTML =
        `<span class="font-mono font-semibold">${escapeHtml(entry.symbol)}</span>` +
        symBadge +
        `<span class="ml-auto text-xs text-slate-400 font-mono">${tokenStr}</span>`;
      row.addEventListener('mousedown', (ev) => {
        ev.preventDefault();
        chooseAndClose(component, entry);
      });
      row.addEventListener('mouseenter', () => setActive(component, idx));
      list.appendChild(row);
    });
    setActive(component, 0);
  }

  function setActive(component, index) {
    const rows = component.list.querySelectorAll('li[data-index]');
    rows.forEach((r) => r.classList.remove('bg-emerald-100', 'text-emerald-900'));
    if (index < 0 || index >= rows.length) {
      component.activeIndex = -1;
      return;
    }
    component.activeIndex = index;
    rows[index].classList.add('bg-emerald-100', 'text-emerald-900');
    rows[index].scrollIntoView({ block: 'nearest' });
  }

  function chooseAndClose(component, entry) {
    setSelection(component, entry);
    closeModal(component);
  }

  function refreshList(component) {
    const filtered = rankOptions(component.options, component.searchEl.value);
    buildRows(component, filtered);
  }

  async function openModal(component) {
    if (component.modal.classList.contains('flex')) return;
    component.modal.classList.remove('hidden');
    component.modal.classList.add('flex');
    document.body.classList.add('overflow-hidden');
    if (!component.optionsLoaded) {
      const embedded = readEmbeddedOptions(component.root);
      const opts = embedded.length ? embedded : await fetchAllOptions();
      component.options = opts.slice();
      component.optionsLoaded = true;
    }
    component.searchEl.value = '';
    refreshList(component);
    setTimeout(() => component.searchEl.focus(), 30);
  }

  function closeModal(component) {
    component.modal.classList.add('hidden');
    component.modal.classList.remove('flex');
    document.body.classList.remove('overflow-hidden');
  }

  function attach(root) {
    if (root.dataset.symbolSelectReady === 'true') return;
    root.dataset.symbolSelectReady = 'true';

    const component = {
      root,
      trigger: root.querySelector('[data-role=trigger]'),
      hidden: root.querySelector('[data-role=hidden]'),
      modal: root.querySelector('[data-role=modal]'),
      searchEl: root.querySelector('[data-role=search]'),
      list: root.querySelector('[data-role=list]'),
      closeBtn: root.querySelector('[data-role=close]'),
      options: [],
      optionsLoaded: false,
      filtered: [],
      activeIndex: -1,
      selection: null,
    };

    const initial = (root.dataset.initialValue || '').trim();
    if (initial) {
      try {
        const parsed = JSON.parse(initial);
        if (parsed && parsed.symbol) {
          setSelection(component, parsed);
        }
      } catch (e) {
        // Treat as a bare symbol string fallback.
        setSelection(component, {
          symbol: initial,
          exchange: '',
          instrument_token: null,
          display_label: initial,
        });
      }
    } else {
      renderTrigger(component);
    }

    if (component.trigger) {
      component.trigger.addEventListener('click', () => openModal(component));
    }
    if (component.closeBtn) {
      component.closeBtn.addEventListener('click', () => closeModal(component));
    }
    component.modal.addEventListener('click', (ev) => {
      if (ev.target === component.modal) closeModal(component);
    });

    component.searchEl.addEventListener('input', () => refreshList(component));
    component.searchEl.addEventListener('keydown', (ev) => {
      const rows = component.list.querySelectorAll('li[data-index]');
      if (ev.key === 'ArrowDown') {
        ev.preventDefault();
        if (!rows.length) return;
        const next = (component.activeIndex + 1) % rows.length;
        setActive(component, next);
      } else if (ev.key === 'ArrowUp') {
        ev.preventDefault();
        if (!rows.length) return;
        const prev = (component.activeIndex - 1 + rows.length) % rows.length;
        setActive(component, prev);
      } else if (ev.key === 'Enter') {
        ev.preventDefault();
        if (component.activeIndex >= 0 && component.filtered[component.activeIndex]) {
          chooseAndClose(component, component.filtered[component.activeIndex]);
        }
      } else if (ev.key === 'Escape') {
        ev.preventDefault();
        closeModal(component);
      }
    });

    document.addEventListener('keydown', (ev) => {
      if (ev.key === 'Escape' && component.modal.classList.contains('flex')) {
        closeModal(component);
      }
    });

    root._tbSymbolSelectComponent = component;
  }

  function getSelection(root) {
    const component = root && root._tbSymbolSelectComponent;
    return component ? (component.selection ? Object.assign({}, component.selection) : null) : null;
  }

  function setSelectionExternal(root, entry) {
    const component = root && root._tbSymbolSelectComponent;
    if (!component) return;
    setSelection(component, entry);
  }

  function init() {
    document.querySelectorAll('[data-symbol-select="true"]').forEach(attach);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  window.tbSymbolSelect = { init, attach, getSelection, setSelection: setSelectionExternal };
})();
