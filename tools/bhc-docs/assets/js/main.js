// BHC Documentation - JavaScript

(function() {
    'use strict';

    // Theme management
    const themeToggle = document.querySelector('.theme-toggle');
    const html = document.documentElement;

    function getStoredTheme() {
        return localStorage.getItem('bhc-docs-theme');
    }

    function setTheme(theme) {
        html.setAttribute('data-theme', theme);
        localStorage.setItem('bhc-docs-theme', theme);
    }

    function toggleTheme() {
        const current = html.getAttribute('data-theme') || 'dark';
        setTheme(current === 'dark' ? 'light' : 'dark');
    }

    // Initialize theme
    const storedTheme = getStoredTheme();
    if (storedTheme) {
        setTheme(storedTheme);
    } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) {
        setTheme('light');
    }

    if (themeToggle) {
        themeToggle.addEventListener('click', toggleTheme);
    }

    // Search modal
    const searchModal = document.getElementById('search-modal');
    const searchInput = document.getElementById('search-input');
    const sidebarSearch = document.getElementById('search');
    const searchResults = document.getElementById('search-results');

    function openSearch() {
        if (searchModal) {
            searchModal.hidden = false;
            searchInput.focus();
            document.body.style.overflow = 'hidden';
        }
    }

    function closeSearch() {
        if (searchModal) {
            searchModal.hidden = true;
            document.body.style.overflow = '';
            searchInput.value = '';
            searchResults.innerHTML = '';
        }
    }

    // Open search on sidebar click or '/' key
    if (sidebarSearch) {
        sidebarSearch.addEventListener('focus', function(e) {
            e.preventDefault();
            openSearch();
            this.blur();
        });
    }

    document.addEventListener('keydown', function(e) {
        // Open search with '/'
        if (e.key === '/' && !isInputFocused()) {
            e.preventDefault();
            openSearch();
        }
        // Close with Escape
        if (e.key === 'Escape' && searchModal && !searchModal.hidden) {
            closeSearch();
        }
    });

    // Close modal on background click
    if (searchModal) {
        searchModal.addEventListener('click', function(e) {
            if (e.target === searchModal) {
                closeSearch();
            }
        });
    }

    function isInputFocused() {
        const active = document.activeElement;
        return active && (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA');
    }

    // Search functionality
    let searchIndex = null;

    async function loadSearchIndex() {
        if (searchIndex) return searchIndex;

        try {
            const baseUrl = getBaseUrl();
            const response = await fetch(`${baseUrl}/search-index.json`);
            searchIndex = await response.json();
            return searchIndex;
        } catch (e) {
            console.error('Failed to load search index:', e);
            return [];
        }
    }

    function getBaseUrl() {
        // Get base URL from the page or default to current path
        const baseTag = document.querySelector('base');
        if (baseTag) {
            return baseTag.href.replace(/\/$/, '');
        }
        // Infer from current URL
        const path = window.location.pathname;
        const parts = path.split('/');
        // Remove the current page
        parts.pop();
        return parts.join('/') || '.';
    }

    async function performSearch(query) {
        const index = await loadSearchIndex();
        if (!index || !query.trim()) {
            return [];
        }

        const normalizedQuery = query.toLowerCase();
        const results = [];

        for (const item of index) {
            const score = calculateScore(item, normalizedQuery);
            if (score > 0) {
                results.push({ ...item, score });
            }
        }

        // Sort by score descending
        results.sort((a, b) => b.score - a.score);

        return results.slice(0, 20);
    }

    function calculateScore(item, query) {
        const name = (item.name || '').toLowerCase();
        const module = (item.module || '').toLowerCase();
        const doc = (item.doc || '').toLowerCase();

        let score = 0;

        // Exact name match
        if (name === query) {
            score += 100;
        }
        // Name starts with query
        else if (name.startsWith(query)) {
            score += 80;
        }
        // Name contains query
        else if (name.includes(query)) {
            score += 60;
        }
        // Module match
        if (module.includes(query)) {
            score += 20;
        }
        // Doc contains query
        if (doc.includes(query)) {
            score += 10;
        }

        return score;
    }

    function renderSearchResults(results) {
        if (!searchResults) return;

        if (results.length === 0) {
            searchResults.innerHTML = '<div class="no-results">No results found</div>';
            return;
        }

        const baseUrl = getBaseUrl();
        const html = results.map(item => {
            const anchor = item.kind === 'function' ? 'v' :
                          item.kind === 'type' ? 't' : 'c';
            const modulePath = item.module.replace(/\./g, '/');
            const href = `${baseUrl}/${modulePath}.html#${anchor}:${item.name}`;

            return `
                <div class="result-item">
                    <div class="result-header">
                        <a href="${href}" class="result-link">
                            <span class="result-module">${item.module}.</span><span class="result-name">${item.name}</span>
                        </a>
                        <span class="result-kind">${item.kind}</span>
                    </div>
                    ${item.signature ? `<code class="result-signature">${escapeHtml(item.signature)}</code>` : ''}
                    ${item.doc ? `<p class="result-doc">${truncate(escapeHtml(item.doc), 100)}</p>` : ''}
                </div>
            `;
        }).join('');

        searchResults.innerHTML = html;

        // Handle result clicks
        searchResults.querySelectorAll('a').forEach(link => {
            link.addEventListener('click', () => {
                closeSearch();
            });
        });
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    function truncate(text, length) {
        if (text.length <= length) return text;
        return text.slice(0, length) + '...';
    }

    // Debounced search
    let searchTimeout;
    if (searchInput) {
        searchInput.addEventListener('input', function() {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(async () => {
                const results = await performSearch(this.value);
                renderSearchResults(results);
            }, 150);
        });
    }

    // Keyboard navigation
    document.addEventListener('keydown', function(e) {
        // j/k for navigation (vim-style)
        if (!isInputFocused()) {
            if (e.key === 'j') {
                scrollBy(0, 60);
            } else if (e.key === 'k') {
                scrollBy(0, -60);
            }
        }
    });

    // Smooth scroll to anchors
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                e.preventDefault();
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
                // Update URL
                history.pushState(null, '', this.getAttribute('href'));
            }
        });
    });

    // Highlight current section in sidebar
    function updateActiveSection() {
        const sections = document.querySelectorAll('.decl[id]');
        const navLinks = document.querySelectorAll('.nav-tree a');

        let currentSection = null;
        const scrollPos = window.scrollY + 100;

        sections.forEach(section => {
            if (section.offsetTop <= scrollPos) {
                currentSection = section.id;
            }
        });

        navLinks.forEach(link => {
            const href = link.getAttribute('href');
            if (href && href.includes('#')) {
                const anchor = href.split('#')[1];
                if (anchor === currentSection) {
                    link.classList.add('active');
                } else {
                    link.classList.remove('active');
                }
            }
        });
    }

    let scrollTimeout;
    window.addEventListener('scroll', function() {
        clearTimeout(scrollTimeout);
        scrollTimeout = setTimeout(updateActiveSection, 50);
    });

    // Playground integration
    const runButtons = document.querySelectorAll('.run-btn');
    runButtons.forEach(btn => {
        btn.addEventListener('click', async function() {
            const code = this.dataset.code;
            const example = this.closest('.example');

            this.textContent = 'Running...';
            this.disabled = true;

            try {
                const result = await runInPlayground(code);
                showPlaygroundResult(example, result);
            } catch (e) {
                showPlaygroundError(example, e.message);
            } finally {
                this.textContent = 'Run';
                this.disabled = false;
            }
        });
    });

    async function runInPlayground(code) {
        // Check if playground is available
        if (typeof window.bhcPlayground !== 'undefined') {
            return await window.bhcPlayground.run(code);
        }

        // Fallback: try to load playground
        try {
            const baseUrl = getBaseUrl();
            const script = document.createElement('script');
            script.src = `${baseUrl}/js/playground.js`;
            document.head.appendChild(script);

            return new Promise((resolve, reject) => {
                script.onload = async () => {
                    if (typeof window.bhcPlayground !== 'undefined') {
                        resolve(await window.bhcPlayground.run(code));
                    } else {
                        reject(new Error('Playground not available'));
                    }
                };
                script.onerror = () => reject(new Error('Failed to load playground'));
            });
        } catch (e) {
            throw new Error('Playground not available');
        }
    }

    function showPlaygroundResult(example, result) {
        let output = example.querySelector('.example-output');
        if (!output) {
            output = document.createElement('div');
            output.className = 'example-output';
            example.appendChild(output);
        }
        output.innerHTML = `<pre><code>${escapeHtml(result)}</code></pre>`;
    }

    function showPlaygroundError(example, message) {
        let output = example.querySelector('.example-output');
        if (!output) {
            output = document.createElement('div');
            output.className = 'example-output';
            example.appendChild(output);
        }
        output.innerHTML = `<pre><code class="error">${escapeHtml(message)}</code></pre>`;
    }

    // Mobile sidebar toggle
    const menuToggle = document.querySelector('.menu-toggle');
    const sidebar = document.querySelector('.sidebar');

    if (menuToggle && sidebar) {
        menuToggle.addEventListener('click', function() {
            sidebar.classList.toggle('open');
        });

        // Close sidebar when clicking outside
        document.addEventListener('click', function(e) {
            if (sidebar.classList.contains('open') &&
                !sidebar.contains(e.target) &&
                !menuToggle.contains(e.target)) {
                sidebar.classList.remove('open');
            }
        });
    }

    // Copy code blocks
    document.querySelectorAll('pre code').forEach(block => {
        const pre = block.parentElement;
        const button = document.createElement('button');
        button.className = 'copy-btn';
        button.textContent = 'Copy';
        button.setAttribute('aria-label', 'Copy code');

        button.addEventListener('click', async function() {
            try {
                await navigator.clipboard.writeText(block.textContent);
                this.textContent = 'Copied!';
                setTimeout(() => {
                    this.textContent = 'Copy';
                }, 2000);
            } catch (e) {
                this.textContent = 'Failed';
                setTimeout(() => {
                    this.textContent = 'Copy';
                }, 2000);
            }
        });

        pre.style.position = 'relative';
        pre.appendChild(button);
    });

    // Add copy button styles dynamically
    const style = document.createElement('style');
    style.textContent = `
        .copy-btn {
            position: absolute;
            top: 0.5rem;
            right: 0.5rem;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            cursor: pointer;
            opacity: 0;
            transition: opacity 0.2s;
        }
        pre:hover .copy-btn {
            opacity: 1;
        }
        .copy-btn:hover {
            background: var(--bg-secondary);
        }
        .error {
            color: var(--error);
        }
    `;
    document.head.appendChild(style);

    // TOC sidebar toggle
    const tocSidebar = document.getElementById('toc-sidebar');
    const tocToggle = document.getElementById('toc-toggle');
    const tocContent = document.getElementById('toc-content');

    function getTocState() {
        return localStorage.getItem('bhc-docs-toc-collapsed') === 'true';
    }

    function setTocState(collapsed) {
        localStorage.setItem('bhc-docs-toc-collapsed', collapsed);
        if (tocSidebar) {
            tocSidebar.classList.toggle('collapsed', collapsed);
        }
    }

    // Initialize TOC state
    if (tocSidebar) {
        const isCollapsed = getTocState();
        setTocState(isCollapsed);
    }

    if (tocToggle) {
        tocToggle.addEventListener('click', function() {
            const isCollapsed = tocSidebar.classList.contains('collapsed');
            setTocState(!isCollapsed);
        });
    }

    // Update active item in TOC on scroll
    function updateTocActiveItem() {
        const tocLinks = document.querySelectorAll('.toc-items a');
        const sections = document.querySelectorAll('.decl[id]');

        let currentSection = null;
        const scrollPos = window.scrollY + 120;

        sections.forEach(section => {
            if (section.offsetTop <= scrollPos) {
                currentSection = section.id;
            }
        });

        tocLinks.forEach(link => {
            const href = link.getAttribute('href');
            if (href) {
                const anchor = href.replace('#', '');
                if (anchor === currentSection) {
                    link.classList.add('active');
                } else {
                    link.classList.remove('active');
                }
            }
        });
    }

    // Page header bar hide/show on scroll
    let lastScrollY = window.scrollY;
    let headerHidden = true; // Start hidden until we scroll up
    let headerFixed = false;
    const pageHeaderBar = document.querySelector('.page-header__bar');
    const pageHeader = document.querySelector('.page-header');
    let headerOriginalTop = 0;

    if (pageHeader) {
        headerOriginalTop = pageHeader.offsetTop + pageHeaderBar.offsetHeight;
    }

    function updateHeaderVisibility() {
        if (!pageHeaderBar) return;

        const currentScrollY = window.scrollY;
        const scrollDelta = currentScrollY - lastScrollY;
        const pastHeader = currentScrollY > headerOriginalTop;

        // When scrolled past the original header position
        if (pastHeader) {
            // Make it fixed if not already
            if (!headerFixed) {
                pageHeaderBar.classList.add('page-header__bar--fixed');
                pageHeaderBar.classList.add('hidden');
                headerFixed = true;
                headerHidden = true;
            }

            // Scrolling up - show the fixed header
            if (scrollDelta < -5 && headerHidden) {
                pageHeaderBar.classList.remove('hidden');
                headerHidden = false;
            }
            // Scrolling down - hide the fixed header
            else if (scrollDelta > 10 && !headerHidden) {
                pageHeaderBar.classList.add('hidden');
                headerHidden = true;
            }
        } else {
            // At top - remove fixed positioning, show normally
            if (headerFixed) {
                pageHeaderBar.classList.remove('page-header__bar--fixed');
                pageHeaderBar.classList.remove('hidden');
                headerFixed = false;
                headerHidden = false;
            }
        }

        lastScrollY = currentScrollY;
    }

    // Header visibility needs to update quickly
    window.addEventListener('scroll', function() {
        updateHeaderVisibility();
    }, { passive: true });

    // Combine other scroll handlers with debounce
    window.addEventListener('scroll', function() {
        clearTimeout(scrollTimeout);
        scrollTimeout = setTimeout(() => {
            updateActiveSection();
            updateTocActiveItem();
        }, 50);
    });

    // Version selector
    const versionSelect = document.getElementById('version-select');
    if (versionSelect) {
        versionSelect.addEventListener('change', function() {
            const version = this.value;
            // Navigate to the selected version
            const currentPath = window.location.pathname;
            // Assuming version is in the URL like /v1.0.0/module.html
            const newPath = currentPath.replace(/\/v[\d.]+\//, `/${version}/`);
            if (newPath !== currentPath) {
                window.location.href = newPath;
            }
        });
    }

    // Initialize
    updateActiveSection();
    updateTocActiveItem();
})();
