// Auto-submit when search options change
(function() {
    const form = document.querySelector('.search-bar');
    if (!form) return;
    const q = form.querySelector('input[name=q]');
    form.querySelectorAll('input[type=radio], input[type=date]').forEach(function(el) {
        el.addEventListener('change', function() {
            if (q.value.trim()) form.submit();
        });
    });
})();

// Lightbox
(function() {
    const lightbox = document.getElementById('lightbox');
    const lightboxImg = document.getElementById('lightbox-img');
    let paths = [];
    let currentIdx = -1;

    function collectPaths() {
        const container = document.querySelector('.overlay:not(.hidden) .overlay-body')
            || document.body;
        return Array.from(container.querySelectorAll('.lightbox-link')).map(a => a.dataset.path);
    }

    function show(path) {
        paths = collectPaths();
        currentIdx = paths.indexOf(path);
        lightboxImg.src = '/image?path=' + encodeURIComponent(path);
        lightbox.classList.remove('hidden');
    }

    function close() {
        lightbox.classList.add('hidden');
        lightboxImg.src = '';
    }

    function navigate(delta) {
        if (!paths.length) return;
        currentIdx = (currentIdx + delta + paths.length) % paths.length;
        lightboxImg.src = '/image?path=' + encodeURIComponent(paths[currentIdx]);
    }

    document.addEventListener('click', function(e) {
        const link = e.target.closest('.lightbox-link');
        if (!link) return;
        e.preventDefault();
        show(link.dataset.path);
    });

    lightbox.querySelector('.lightbox-close').addEventListener('click', close);
    lightbox.querySelector('.overlay-backdrop').addEventListener('click', close);
    lightbox.querySelector('.lightbox-prev').addEventListener('click', function() { navigate(-1); });
    lightbox.querySelector('.lightbox-next').addEventListener('click', function() { navigate(1); });

    document.addEventListener('keydown', function(e) {
        if (lightbox.classList.contains('hidden')) return;
        e.stopImmediatePropagation();
        if (e.key === 'Escape') close();
        else if (e.key === 'ArrowLeft') navigate(-1);
        else if (e.key === 'ArrowRight') navigate(1);
    });
})();

// Timeline overlay
(function() {
    const overlay = document.getElementById('timeline-overlay');
    const cards = document.getElementById('timeline-cards');
    let focalId = null;

    function renderCard(r, isFocal) {
        const cls = isFocal ? 'timeline-card timeline-focal' : 'timeline-card timeline-neighbor';
        return `
            <div class="${cls}" data-id="${r.id}">
                <a href="#" class="lightbox-link" data-path="${r.file_path}">
                    <img src="/image?path=${encodeURIComponent(r.file_path)}&thumb=1" loading="lazy" alt="${r.name}">
                </a>
                <div class="timeline-card-date">${r.date}</div>
                <div class="timeline-card-meta"><a href="/image?path=${encodeURIComponent(r.file_path)}" download class="action-link">download</a> &mdash; <a href="#" class="related-link" data-id="${r.id}">related</a></div>
            </div>`;
    }

    function load(id) {
        focalId = id;
        cards.innerHTML = '<p class="loading">Loading...</p>';
        fetch('/timeline/' + id)
            .then(r => r.json())
            .then(data => {
                let html = '';
                data.before.forEach(r => html += renderCard(r, false));
                html += renderCard(data.focal, true);
                data.after.forEach(r => html += renderCard(r, false));
                cards.innerHTML = html;
            });
    }

    function navigate(delta) {
        // Find the neighbor to shift to
        const all = Array.from(cards.querySelectorAll('.timeline-card'));
        const focalIdx = all.findIndex(c => c.classList.contains('timeline-focal'));
        const targetIdx = focalIdx + delta;
        if (targetIdx < 0 || targetIdx >= all.length) return;
        const targetId = all[targetIdx].dataset.id;
        load(targetId);
    }

    document.addEventListener('click', function(e) {
        const link = e.target.closest('.timeline-link');
        if (!link) return;
        e.preventDefault();
        overlay.classList.remove('hidden');
        load(link.dataset.id);
    });

    // Block lightbox on focal card, re-center on neighbor click
    document.addEventListener('click', function(e) {
        const card = e.target.closest('.timeline-card');
        if (!card || !overlay.contains(card)) return;
        const link = e.target.closest('.lightbox-link');
        if (!link) return;
        if (card.classList.contains('timeline-focal')) {
            e.preventDefault();
            e.stopPropagation();
            return;
        }
        if (card.classList.contains('timeline-neighbor')) {
            e.preventDefault();
            e.stopPropagation();
            load(card.dataset.id);
        }
    }, true);

    overlay.querySelector('.timeline-prev').addEventListener('click', function() { navigate(-1); });
    overlay.querySelector('.timeline-next').addEventListener('click', function() { navigate(1); });

    overlay.querySelector('.overlay-close').addEventListener('click', function() {
        overlay.classList.add('hidden');
    });
    overlay.querySelector('.overlay-backdrop').addEventListener('click', function() {
        overlay.classList.add('hidden');
    });

    document.addEventListener('keydown', function(e) {
        if (overlay.classList.contains('hidden')) return;
        if (!document.getElementById('lightbox').classList.contains('hidden')) return;
        if (!document.getElementById('related-overlay').classList.contains('hidden')) return;
        if (e.key === 'Escape') {
            e.stopImmediatePropagation();
            overlay.classList.add('hidden');
        }
        else if (e.key === 'ArrowLeft') navigate(-1);
        else if (e.key === 'ArrowRight') navigate(1);
    });
})();

// Related overlay
(function() {
    const overlay = document.getElementById('related-overlay');
    const body = document.getElementById('related-results');

    function renderCard(r, extra) {
        const cls = extra.source ? 'result result-source' : 'result';
        const simText = extra.similarity != null ? ` &mdash; similarity: ${extra.similarity}` : '';
        return `
            <div class="${cls}">
                <div class="thumb">
                    <a href="#" class="lightbox-link" data-path="${r.file_path}">
                        <img src="/image?path=${encodeURIComponent(r.file_path)}&thumb=1" loading="lazy" alt="${r.name}">
                    </a>
                </div>
                <div class="info">
                    <div class="date">${r.date}</div>
                    <div class="meta">${simText ? `similarity: ${extra.similarity} &mdash; ` : ''}<a href="/image?path=${encodeURIComponent(r.file_path)}" download class="action-link">download</a> &mdash; <a href="#" class="related-link" data-id="${r.id}">related</a> &mdash; <a href="#" class="timeline-link" data-id="${r.id}">timeline</a></div>
                    <div class="meta file-meta">${r.name} &mdash; ${r.width}x${r.height}${r.file_size ? ` &mdash; ${r.file_size}` : ''}</div>
                    <pre class="ocr">${r.ocr_text.replace(/</g, '&lt;')}</pre>
                </div>
            </div>`;
    }

    document.addEventListener('click', function(e) {
        const link = e.target.closest('.related-link');
        if (!link) return;
        e.preventDefault();
        const id = link.dataset.id;
        body.innerHTML = '<p class="loading">Loading...</p>';
        overlay.classList.remove('hidden');

        fetch('/related/' + id)
            .then(r => r.json())
            .then(data => {
                if (!data.related.length) {
                    body.innerHTML = '<p class="loading">No related screenshots found.</p>';
                    return;
                }
                let html = '';
                if (data.source) {
                    html += renderCard(data.source, {source: true});
                }
                html += data.related.map(r => renderCard(r, {similarity: r.similarity})).join('');
                body.innerHTML = html;
            });
    });

    overlay.querySelector('.overlay-close').addEventListener('click', function() {
        overlay.classList.add('hidden');
    });
    overlay.querySelector('.overlay-backdrop').addEventListener('click', function() {
        overlay.classList.add('hidden');
    });
    document.addEventListener('keydown', function(e) {
        if (overlay.classList.contains('hidden')) return;
        if (!document.getElementById('lightbox').classList.contains('hidden')) return;
        if (e.key === 'Escape') {
            e.stopImmediatePropagation();
            overlay.classList.add('hidden');
        }
    });
})();
