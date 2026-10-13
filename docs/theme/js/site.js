(() => {
  const menu = document.querySelector('.menu-button');
  const sidebar = document.querySelector('.sidebar');
  const setMenu = (open) => {
    document.body.classList.toggle('menu-open', open);
    menu.setAttribute('aria-expanded', String(open));
  };
  menu.addEventListener('click', () => setMenu(menu.getAttribute('aria-expanded') !== 'true'));
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && document.body.classList.contains('menu-open')) {
      setMenu(false);
      menu.focus();
    }
  });
  document.addEventListener('click', (event) => {
    if (!sidebar.contains(event.target) && !menu.contains(event.target)) setMenu(false);
  });

  const dialog = document.querySelector('.search-dialog');
  const input = document.querySelector('#search-input');
  const status = document.querySelector('#search-status');
  const results = document.querySelector('#search-results');
  const searchButton = document.querySelector('.search-button');
  const root = new URL(document.body.dataset.siteRoot.replace(/\/?$/, '/'), location.href);
  let indexPromise;
  let request = 0;
  const loadIndex = () => {
    if (!indexPromise) {
      indexPromise = fetch(document.body.dataset.searchIndex).then((response) => {
        if (!response.ok) throw new Error('Search index could not be loaded.');
        return response.json();
      }).then((index) => index.docs.map((doc) => ({...doc, normalizedTitle: doc.title.toLowerCase(), normalizedText: doc.text.toLowerCase()}))).catch((error) => {
        indexPromise = undefined;
        throw error;
      });
    }
    return indexPromise;
  };
  const openSearch = () => {
    setMenu(false);
    dialog.showModal();
    input.focus();
    if (input.value) search();
  };
  searchButton.addEventListener('click', openSearch);
  document.querySelector('.search-close').addEventListener('click', () => dialog.close());
  dialog.addEventListener('close', () => searchButton.focus());
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && dialog.open) {
      event.preventDefault();
      dialog.close();
      return;
    }
    const editing = event.target.matches('input,textarea,select,[contenteditable="true"]');
    if (event.key === '/' && !editing && !dialog.open && !event.metaKey && !event.ctrlKey && !event.altKey) {
      event.preventDefault();
      openSearch();
    }
  });
  async function search() {
    const current = ++request;
    const query = input.value.trim().toLowerCase();
    results.replaceChildren();
    if (!query) {
      status.textContent = 'Enter a term to search the documentation.';
      return;
    }
    status.textContent = 'Searching…';
    try {
      const docs = await loadIndex();
      if (current !== request) return;
      const words = query.split(/\s+/);
      const matches = docs.filter((doc) => words.every((word) => doc.normalizedTitle.includes(word) || doc.normalizedText.includes(word)))
        .map((doc) => ({...doc, rank: (doc.normalizedTitle.includes(query) ? 20 : 0) + words.reduce((score, word) => score + (doc.normalizedTitle.includes(word) ? 5 : 0), 0)}))
        .sort((a, b) => b.rank - a.rank);
      status.textContent = matches.length ? `${matches.length} results${matches.length > 12 ? '; showing the first 12' : ''}.` : 'No results. Try another term.';
      for (const doc of matches.slice(0, 12)) {
        const item = document.createElement('li');
        const link = document.createElement('a');
        link.href = new URL(doc.location, root).href;
        link.textContent = doc.title;
        const excerpt = document.createElement('p');
        const position = Math.max(0, doc.normalizedText.indexOf(words[0]) - 65);
        excerpt.textContent = `${position ? '…' : ''}${doc.text.slice(position, position + 210)}${doc.text.length > position + 210 ? '…' : ''}`;
        item.append(link, excerpt);
        results.append(item);
      }
    } catch (error) {
      if (current === request) status.textContent = 'Search is unavailable. Use the sidebar or try again.';
    }
  }
  input.addEventListener('input', search);
})();
