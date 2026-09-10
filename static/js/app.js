/**
 * LawWise — Main JavaScript
 * Shared utilities and sidebar toggle
 */

// ── Sidebar Toggle (Mobile) ─────────────────────────────────
const sidebar = document.getElementById('sidebar');
const sidebarOverlay = document.getElementById('sidebarOverlay');
const sidebarToggle = document.getElementById('sidebarToggle');

if (sidebarToggle) {
  sidebarToggle.addEventListener('click', () => {
    sidebar.classList.toggle('open');
    sidebarOverlay.classList.toggle('open');
  });
}

if (sidebarOverlay) {
  sidebarOverlay.addEventListener('click', () => {
    sidebar.classList.remove('open');
    sidebarOverlay.classList.remove('open');
  });
}

// ── HTML Escaping ────────────────────────────────────────────
function escHtml(str) {
  if (typeof str !== 'string') str = String(str || '');
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// ── Markdown-to-HTML (lightweight) ──────────────────────────
function formatMarkdown(text) {
  if (!text) return '';
  let html = escHtml(text);

  // Bold **text**
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  // Italic *text*
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
  // Inline code `code`
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

  // ### H3
  html = html.replace(/^### (.+)$/gm, '<h5 class="md-heading">$1</h5>');
  // ## H2
  html = html.replace(/^## (.+)$/gm, '<h4 class="md-heading">$1</h4>');
  // # H1
  html = html.replace(/^# (.+)$/gm, '<h3 class="md-heading">$1</h3>');

  // Unordered list
  html = html.replace(/^\* (.+)$/gm, '<li>$1</li>');
  html = html.replace(/^- (.+)$/gm, '<li>$1</li>');
  html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');

  // Numbered list
  html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');

  // Horizontal rule
  html = html.replace(/^---+$/gm, '<hr style="border:none;border-top:1px solid #e5e7eb;margin:12px 0;">');

  // Paragraph breaks (double newline)
  html = html.replace(/\n\n+/g, '</p><p>');
  html = html.replace(/\n/g, '<br>');

  // Wrap in paragraph
  html = '<p>' + html + '</p>';

  // Fix heading inside paragraph
  html = html.replace(/<p>(<h[3-5])/g, '$1');
  html = html.replace(/(<\/h[3-5]>)<\/p>/g, '$1');

  return html;
}

// ── Auto-resize textarea ─────────────────────────────────────
document.querySelectorAll('textarea').forEach(ta => {
  ta.addEventListener('input', () => {
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 200) + 'px';
  });
});

// ── Alerts auto-dismiss ──────────────────────────────────────
document.querySelectorAll('.alert').forEach(el => {
  setTimeout(() => {
    if (el.parentNode) el.classList.remove('show');
  }, 5000);
});

console.log('⚖ LawWise AI Legal Aid System loaded.');
