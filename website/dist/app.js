// The wordmark, navigation, and section titles are loaded from index.md.
const escapeHtml = text => String(text || '').replace(/[&<>"']/g, character => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[character]);

function sectionSource(markdown, heading) {
  const sections = markdown.split(/^##[ \t]+/m).slice(1);
  const section = sections.find(source => source.split('\n')[0].trim() === heading);
  return section ? section.slice(section.indexOf('\n') + 1).trim() : '';
}

function renderBody(source) {
  return source.split(/\n\s*\n/).filter(Boolean).map(block => {
    const gifLines = block.trim().split('\n').map(line => line.match(/^!\[([^\]]*)\]\((assets\/[^)]+\.gif)\)$/));
    if (gifLines.every(Boolean)) return `<div class="overview-playbacks">${gifLines.map(image => `<figure><img src="${escapeHtml(image[2])}" alt="${escapeHtml(image[1])}"><figcaption>${escapeHtml(image[1])}</figcaption></figure>`).join('')}</div>`;
    const image = block.trim().match(/^!\[([^\]]*)\]\((assets\/[^)]+)\)$/);
    if (image) return `<figure><a href="${escapeHtml(image[2])}" target="_blank" rel="noopener"><img src="${escapeHtml(image[2])}" alt="${escapeHtml(image[1])}" loading="lazy"></a></figure>`;
    return `<p>${escapeHtml(block.trim())}</p>`;
  }).join('');
}

function setting(source, name) {
  const match = source.match(new RegExp(`^- \\*\\*${name}:\\*\\*\\s*(.+)$`, 'mi'));
  return match ? match[1].trim() : '';
}

function contentSections(markdown) {
  return [...markdown.matchAll(/^##[ \t]+(.+)$/gm)]
    .map(match => match[1].trim())
    .filter(title => !['site settings', 'navigation'].includes(title.toLowerCase()));
}

function navigationItems(markdown) {
  const headings = contentSections(markdown);
  const seen = new Set();
  return [...sectionSource(markdown, 'Navigation').matchAll(/^[ \t]*- \[([^\]]+)\]\((#[^)]+)\)[ \t]*$/gm)]
    .map(match => ({ label: match[1], target: match[2] }))
    .filter(item => {
      if (!headings.includes(item.label) || seen.has(item.target)) return false;
      seen.add(item.target);
      return true;
    });
}

fetch('index.md', { cache: 'no-store' })
  .then(response => {
    if (!response.ok) throw new Error('Could not load index.md');
    return response.text();
  })
  .then(markdown => {
    const pageTitle = markdown.match(/^# (.+)$/m)?.[1] || 'Atlas-DeePC';
    const settings = sectionSource(markdown, 'Site settings');
    const items = navigationItems(markdown);
    document.title = setting(settings, 'Browser title') || pageTitle;
    document.querySelector('.menu-toggle span').textContent = setting(settings, 'Menu button') || 'Menu';
    document.querySelector('#primary-nav').innerHTML = items.map(item => `<a href="${escapeHtml(item.target)}">${escapeHtml(item.label)}</a>`).join('');
    const paperTitle = setting(settings, 'Paper title');
    document.querySelector('#content').innerHTML = (paperTitle ? `<h1 class="paper-title">${escapeHtml(paperTitle)}</h1>` : '') + contentSections(markdown).map(label => {
      const item = items.find(entry => entry.label === label) || { label, target: '#' + label.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') };
      const body = renderBody(sectionSource(markdown, item.label));
      return `<section id="${escapeHtml(item.target.slice(1))}" class="${body ? 'has-content' : ''}" tabindex="-1"><h2>${escapeHtml(item.label)}</h2>${body}</section>`;
    }).join('');

    const nav = document.querySelector('#primary-nav');
    const toggle = document.querySelector('.menu-toggle');
    toggle.addEventListener('click', () => {
      const open = toggle.getAttribute('aria-expanded') !== 'true';
      toggle.setAttribute('aria-expanded', String(open));
      nav.classList.toggle('open', open);
    });
    nav.querySelectorAll('a').forEach(link => link.addEventListener('click', () => {
      nav.classList.remove('open');
      toggle.setAttribute('aria-expanded', 'false');
    }));
  })
  .catch(error => {
    document.querySelector('#content').innerHTML = `<p>${escapeHtml(error.message)}</p>`;
  });
