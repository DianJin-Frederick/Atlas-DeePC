(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];

  /* ---- top bar: border on scroll, mobile menu, active link ---- */
  const bar = $('#topbar');
  const onScroll = () => bar.classList.toggle('scrolled', window.scrollY > 8);
  window.addEventListener('scroll', onScroll, { passive: true }); onScroll();

  const nav = $('#nav'), menuBtn = $('#menuBtn');
  menuBtn.addEventListener('click', () => {
    const open = nav.classList.toggle('open');
    menuBtn.setAttribute('aria-expanded', open);
  });
  $$('a', nav).forEach(a => a.addEventListener('click', () => { nav.classList.remove('open'); menuBtn.setAttribute('aria-expanded', false); }));

  const links = new Map($$('a', nav).map(a => [a.getAttribute('href').slice(1), a]));
  const spy = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting) { links.forEach(l => l.classList.remove('active')); links.get(e.target.id)?.classList.add('active'); }
    });
  }, { rootMargin: '-45% 0px -50% 0px' });
  links.forEach((_, id) => { const el = document.getElementById(id); if (el) spy.observe(el); });

  /* ---- videos: play only while visible ---- */
  const tryPlay = v => { const p = v.play(); if (p && p.catch) p.catch(() => {}); };
  const vis = new IntersectionObserver(entries => {
    entries.forEach(e => {
      const v = e.target;
      if (e.isIntersecting && !v.closest('[hidden]')) tryPlay(v); else v.pause();
    });
  }, { threshold: 0.35 });
  $$('video').forEach(v => vis.observe(v));

  /* ---- seed / noise chips ---- */
  $$('[data-player]').forEach(player => {
    const video = $('video', player), note = $('.player-note [data-note]', player);
    $$('.chip', player).forEach(chip => chip.addEventListener('click', () => {
      if (chip.classList.contains('is-active')) return;
      $$('.chip', player).forEach(c => c.classList.toggle('is-active', c === chip));
      video.poster = chip.dataset.poster;
      video.src = chip.dataset.src;
      video.load(); tryPlay(video);
      if (note) note.textContent = chip.dataset.note;
    }));
  });

  /* ---- tabs (each tab group only controls the panels in its own section) ---- */
  $$('.tabs').forEach(group => {
    const scope = group.closest('section') || document;
    const tabs = $$('.tab', group);
    tabs.forEach(tab => tab.addEventListener('click', () => {
      tabs.forEach(t => { const on = t === tab; t.classList.toggle('is-active', on); t.setAttribute('aria-selected', on); });
      $$('.tabpanel', scope).forEach(p => {
        const on = p.id === 'tab-' + tab.dataset.tab;
        p.hidden = !on; p.classList.toggle('is-active', on);
        $$('video', p).forEach(v => on ? (v.getBoundingClientRect().top < innerHeight && tryPlay(v)) : v.pause());
      });
    }));
  });

  /* ---- lightbox ---- */
  const lb = $('#lightbox'), lbImg = $('img', lb);
  $$('.zoomable img').forEach(img => img.addEventListener('click', () => { lbImg.src = img.src; lbImg.alt = img.alt; lb.hidden = false; }));
  const closeLb = () => { lb.hidden = true; lbImg.src = ''; };
  lb.addEventListener('click', closeLb);
  document.addEventListener('keydown', e => { if (e.key === 'Escape' && !lb.hidden) closeLb(); });

  /* ---- copy BibTeX ---- */
  const copyBtn = $('#copyBib');
  copyBtn.addEventListener('click', async () => {
    try { await navigator.clipboard.writeText($('#bibtex').textContent); copyBtn.textContent = 'Copied'; }
    catch { copyBtn.textContent = 'Select & copy'; }
    setTimeout(() => (copyBtn.textContent = 'Copy'), 1600);
  });

  /* ---- comparison charts (Table 2) ---- */
  /* Data comes from the ::: bar-chart block in index.md (data-methods attribute). */
  const COLORS = {
    'accent': 'var(--orange)',
    'accent-light': 'color-mix(in srgb, var(--orange) 50%, white)',
    'link': 'var(--blue)',
    'gray': 'color-mix(in srgb, var(--muted) 55%, white)',
  };
  const chartHost = document.getElementById('charts');
  let methods = [];
  try { methods = JSON.parse(chartHost ? chartHost.dataset.methods || '[]' : '[]'); } catch (e) { methods = []; }
  methods = methods.map(m => ({ ...m, color: COLORS[m.color] || m.color || COLORS.gray }));
  const NS = 'http://www.w3.org/2000/svg';
  const el = (tag, attrs, text) => { const n = document.createElementNS(NS, tag); for (const k in attrs) { const v = String(attrs[k]); if ((k === 'fill' || k === 'stroke') && /var\(|color-mix/.test(v)) n.style.setProperty(k, v); else n.setAttribute(k, v); } if (text != null) n.textContent = text; return n; };

  function barChart({ title, sub, value, fmt, scale, ticks, tickFmt, whisker }) {
    const W = 540, L = 186, R = 58, rowH = 40, top = 6, H = top + methods.length * rowH + 28;
    const x = v => L + scale(v) * (W - L - R);
    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': title });
    ticks.forEach(t => {
      svg.append(el('line', { x1: x(t), x2: x(t), y1: top, y2: H - 24, stroke: 'var(--line)', 'stroke-width': 1 }));
      svg.append(el('text', { x: x(t), y: H - 6, 'text-anchor': 'middle', 'font-size': 12, fill: 'var(--muted)' }, tickFmt(t)));
    });
    methods.forEach((m, i) => {
      const y = top + i * rowH + 8, h = 22, v = value(m);
      svg.append(el('text', { x: L - 12, y: y + h / 2 + 5, 'text-anchor': 'end', 'font-size': 14, fill: 'var(--ink)', 'font-weight': m.name.startsWith('Atlas') ? 600 : 400 }, m.name));
      svg.append(el('rect', { x: L, y, width: Math.max(2, x(v) - L), height: h, rx: 3, fill: m.color }));
      let end = x(v);
      if (whisker && m.p95 != null) {
        const wx = x(m.p95);
        svg.append(el('line', { x1: x(v), x2: wx, y1: y + h / 2, y2: y + h / 2, stroke: 'var(--ink)', 'stroke-width': 1.2, opacity: .55 }));
        svg.append(el('line', { x1: wx, x2: wx, y1: y + 5, y2: y + h - 5, stroke: 'var(--ink)', 'stroke-width': 1.2, opacity: .55 }));
        end = wx;
      }
      svg.append(el('text', { x: end + 7, y: y + h / 2 + 5, 'font-size': 13.5, fill: 'var(--ink-2)', 'font-weight': 600 }, fmt(v)));
    });
    const card = document.createElement('div'); card.className = 'chart-card';
    card.innerHTML = `<h3>${title}</h3><p>${sub}</p>`; card.append(svg); return card;
  }

  const charts = $('#charts');
  if (charts && methods.length) {
    charts.append(barChart({
      title: 'Success rate', sub: '500 Reacher test simulations',
      value: m => m.success, fmt: v => v.toFixed(1) + '%',
      scale: v => v / 100, ticks: [0, 25, 50, 75, 100], tickFmt: t => t + '%'
    }));
    const lo = Math.log10(10), hi = Math.log10(5000);
    charts.append(barChart({
      title: 'Online computation per step', sub: 'Median, whisker to the 95th percentile where reported · log scale',
      value: m => m.med, fmt: v => v >= 1000 ? (v / 1000).toFixed(2) + ' s' : Math.round(v) + ' ms',
      scale: v => (Math.log10(v) - lo) / (hi - lo), ticks: [10, 100, 1000], tickFmt: t => t >= 1000 ? '1 s' : t + ' ms',
      whisker: true
    }));
  }

  /* ---- line charts (data from ::: line-chart blocks in index.md) ----
     Series colors and marker shapes follow a fixed order (never cycled):
     1 accent (Atlas-DeePC), then amber, teal, indigo. */
  const SERIES = [
    { color: 'var(--orange)', marker: 'circle' },
    { color: '#c27e1f', marker: 'square' },
    { color: '#0f9488', marker: 'triangle' },
    { color: '#5b5fc7', marker: 'diamond' },
  ];
  function markerPath(kind, x, y, r) {
    if (kind === 'square') return `M${x - r * .85},${y - r * .85}h${r * 1.7}v${r * 1.7}h${-r * 1.7}z`;
    if (kind === 'triangle') return `M${x},${y - r * 1.1}L${x + r},${y + r * .75}L${x - r},${y + r * .75}z`;
    if (kind === 'diamond') return `M${x},${y - r * 1.15}L${x + r},${y}L${x},${y + r * 1.15}L${x - r},${y}z`;
    return `M${x - r},${y}a${r},${r} 0 1,0 ${2 * r},0a${r},${r} 0 1,0 ${-2 * r},0`;
  }
  $$('.linechart').forEach(host => {
    let c; try { c = JSON.parse(host.dataset.chart); } catch (e) { return; }
    const W = 340, H = 236, L = 44, R = 10, T = 12, B = 48;
    const n = c.x.length, xs = i => L + (n === 1 ? 0 : i * (W - L - R) / (n - 1));
    const [y0, y1] = c.yrange;
    const ys = v => T + (1 - (v - y0) / (y1 - y0)) * (H - T - B);
    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': c.title });
    (c.yticks || []).forEach(t => {
      svg.append(el('line', { x1: L, x2: W - R, y1: ys(t), y2: ys(t), stroke: 'var(--line)', 'stroke-width': 1 }));
      svg.append(el('text', { x: L - 8, y: ys(t) + 4, 'text-anchor': 'end', 'font-size': 13, fill: 'var(--muted)' }, t + (c.unit && c.unit !== '%' ? '' : (c.unit || ''))));
    });
    c.x.forEach((lab, i) => svg.append(el('text', { x: xs(i), y: H - B + 19, 'text-anchor': 'middle', 'font-size': 12.5, fill: 'var(--muted)' }, lab)));
    if (c.xlabel) svg.append(el('text', { x: (L + W - R) / 2, y: H - 6, 'text-anchor': 'middle', 'font-size': 13, fill: 'var(--muted)' }, c.xlabel));
    c.series.forEach((s, k) => {
      const st = SERIES[k] || SERIES[SERIES.length - 1];
      let d = '', pen = false;
      s.values.forEach((v, i) => { if (v == null) { pen = false; return; } d += (pen ? 'L' : 'M') + xs(i) + ',' + ys(v); pen = true; });
      svg.append(el('path', { d, fill: 'none', stroke: st.color, 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }));
      s.values.forEach((v, i) => {
        if (v == null) return;
        const m = el('path', { d: markerPath(st.marker, xs(i), ys(v), 4.6), fill: st.color, stroke: 'var(--card)', 'stroke-width': 1.5 });
        m.append(el('title', {}, `${s.name} · noise ${c.x[i]}: ${v}${c.unit === '%' ? '%' : ' ' + (c.unit || '')}`));
        svg.append(m);
      });
    });
    host.append(svg);
  });

  /* ---- reveal on scroll ---- */
  const rev = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { e.target.classList.add('in'); rev.unobserve(e.target); } }), { rootMargin: '0px 0px -8% 0px' });
  $$('.sec-head, .fig, .player, .step, .chart-card, .table-wrap, .notes').forEach(n => { n.classList.add('reveal'); rev.observe(n); });

  /* ---- KaTeX ---- */
  const renderMath = () => window.renderMathInElement && renderMathInElement(document.body, {
    delimiters: [{ left: '\\[', right: '\\]', display: true }, { left: '\\(', right: '\\)', display: false }],
    throwOnError: false
  });
  if (window.renderMathInElement) renderMath(); else window.addEventListener('load', renderMath);
})();
