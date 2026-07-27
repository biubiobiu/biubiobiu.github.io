/**
 * annotator.js —— 笔记页轻量批注引擎（自包含，无依赖）
 *
 * 功能：选中文字 → 浮动工具条 → 高亮底纹 / 改文字颜色 / 写批注 / 清除
 *      批注存 localStorage（按文件名区分），支持导出/导入 JSON、显隐开关。
 * 用法：在笔记 HTML 的 </body> 前加 <script defer src="annotator.js"></script>
 * 说明：SVG 图内的文字不支持底纹（SVG 无背景色），高亮会退化为改字色。
 */
(function () {
  'use strict';

  const PAGE_KEY = 'anno:' + (location.pathname.split('/').pop() || 'index');
  const SVG_NS = 'http://www.w3.org/2000/svg';

  const HL_COLORS = [
    ['#ffe867', '黄色底纹'], ['#a7f3c9', '绿色底纹'],
    ['#fbc7d8', '粉色底纹'], ['#bfdcff', '蓝色底纹'],
  ];
  const TXT_COLORS = [
    ['#dc2626', '红字'], ['#2563eb', '蓝字'],
    ['#7c3aed', '紫字'], ['#059669', '绿字'],
  ];

  let items = [];        // {id, type:'hl'|'color'|'note', color?, note?, anchor:{start, exact, prefix, suffix}}
  let visible = true;

  /* ---------------- 存取 ----------------
   * 双存储：localStorage（实时缓存）+ 固化进 HTML 文件的 <script id="anno-baked"> 数据块。
   * 加载时取两者中较新的一份；绑定文件句柄后，每次改动自动写回文件。 */
  function payload() { return { updatedAt: Date.now(), items }; }

  function loadLocal() {
    try {
      const raw = JSON.parse(localStorage.getItem(PAGE_KEY) || 'null');
      if (!raw) return null;
      if (Array.isArray(raw)) return { updatedAt: 0, items: raw };   // 兼容旧格式
      return raw;
    } catch (e) { return null; }
  }
  function loadBaked() {
    const el = document.getElementById('anno-baked');
    if (!el) return null;
    try { return JSON.parse(el.textContent); } catch (e) { return null; }
  }
  function save() {
    localStorage.setItem(PAGE_KEY, JSON.stringify(payload()));
    bakeSoon();   // 已绑定文件则自动写回
  }

  /* ---------------- 固化到文件（File System Access API，Chrome/Edge） ---------------- */
  let fileHandle = null, bakeTimer = null, fsGranted = false, saveBtn = null;

  const idb = {
    open() {
      return new Promise((res, rej) => {
        const r = indexedDB.open('anno-db', 1);
        r.onupgradeneeded = () => r.result.createObjectStore('handles');
        r.onsuccess = () => res(r.result);
        r.onerror = () => rej(r.error);
      });
    },
    async get(key) {
      const db = await this.open();
      return new Promise((res, rej) => {
        const t = db.transaction('handles').objectStore('handles').get(key);
        t.onsuccess = () => res(t.result); t.onerror = () => rej(t.error);
      });
    },
    async set(key, val) {
      const db = await this.open();
      return new Promise((res, rej) => {
        const t = db.transaction('handles', 'readwrite').objectStore('handles').put(val, key);
        t.onsuccess = () => res(); t.onerror = () => rej(t.error);
      });
    },
  };

  function bakedSource(src) {
    const json = JSON.stringify(payload()).replace(/</g, '\\u003c');
    const block = '<script type="application/json" id="anno-baked">' + json + '</scr' + 'ipt>';
    const re = /<script type="application\/json" id="anno-baked">[\s\S]*?<\/script>/;
    if (re.test(src)) return src.replace(re, block);
    if (src.includes('</body>')) return src.replace('</body>', block + '\n</body>');
    return src + '\n' + block;
  }

  async function writeFile() {
    if (!fileHandle) return false;
    try {
      if ((await fileHandle.queryPermission({ mode: 'readwrite' })) !== 'granted') {
        if ((await fileHandle.requestPermission({ mode: 'readwrite' })) !== 'granted') return false;
      }
      fsGranted = true;
      const src = await (await fileHandle.getFile()).text();   // 每次重读磁盘最新内容，只替换数据块
      const w = await fileHandle.createWritable();
      await w.write(bakedSource(src));
      await w.close();
      updateSaveBtn(true);
      return true;
    } catch (e) {
      console.warn('[annotator] 写入文件失败', e);
      updateSaveBtn(false);
      return false;
    }
  }

  function bakeSoon() {
    if (!fileHandle) { updateSaveBtn(false); return; }
    if (!fsGranted) { writeFile(); return; }   // 需要授权时立即写（借用当前点击手势弹授权框）
    clearTimeout(bakeTimer);
    bakeTimer = setTimeout(writeFile, 800);
  }

  async function bindFile() {
    if (!window.showOpenFilePicker) {
      toast('当前浏览器不支持直接写文件（需 Chrome / Edge）。标记仍会保存在浏览器本地，也可用 ⤓ 导出备份。', 5000);
      return;
    }
    try {
      const [h] = await window.showOpenFilePicker({
        types: [{ description: 'HTML 笔记', accept: { 'text/html': ['.html', '.htm'] } }],
      });
      const pageName = decodeURIComponent(location.pathname.split('/').pop() || '');
      if (h.name !== pageName && !confirm('选择的文件是「' + h.name + '」，但当前页面是「' + pageName + '」。\n固化会把本页标记写进所选文件，确定绑定？')) return;
      fileHandle = h;
      await idb.set(PAGE_KEY, h);
      if (await writeFile()) toast('已绑定并固化 ✓ 之后的写写画画会自动存进文件本身');
    } catch (e) { /* 用户取消选择 */ }
  }

  function updateSaveBtn(ok) {
    if (!saveBtn) return;
    const hasMarks = items.length > 0;
    if (fileHandle && ok !== false) {
      saveBtn.classList.remove('anno-attn');
      saveBtn.title = '已绑定文件，改动自动固化；点击可手动立即固化';
    } else {
      saveBtn.classList.toggle('anno-attn', hasMarks);
      saveBtn.title = hasMarks
        ? '标记尚未固化到文件！点击选择本 HTML 文件完成绑定（仅需一次）'
        : '固化到文件：点击选择本 HTML 文件绑定后，标记自动写入文件本身';
    }
  }

  /* ---------------- 文档文本与定位 ---------------- */
  function textNodes() {
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, {
      acceptNode(n) {
        const p = n.parentElement;
        if (!p) return NodeFilter.FILTER_REJECT;
        if (p.closest('[data-anno-ui]')) return NodeFilter.FILTER_REJECT;
        const tag = p.tagName ? p.tagName.toLowerCase() : '';
        if (tag === 'script' || tag === 'style') return NodeFilter.FILTER_REJECT;
        return NodeFilter.FILTER_ACCEPT;
      }
    });
    const arr = [];
    while (walker.nextNode()) arr.push(walker.currentNode);
    return arr;
  }
  function docText() { return textNodes().map(n => n.data).join(''); }

  // 全局偏移 -> Range
  function rangeFromOffsets(start, end) {
    const nodes = textNodes();
    let acc = 0, range = document.createRange(), sSet = false;
    for (const n of nodes) {
      const len = n.data.length;
      if (!sSet && start <= acc + len) { range.setStart(n, Math.max(0, start - acc)); sSet = true; }
      if (sSet && end <= acc + len) { range.setEnd(n, Math.max(0, end - acc)); return range; }
      acc += len;
    }
    return null;
  }

  // 当前选区 -> 全局偏移
  function offsetsFromRange(range) {
    const pre = document.createRange();
    pre.setStart(document.body, 0);
    pre.setEnd(range.startContainer, range.startOffset);
    const start = pre.toString().length;
    return { start, end: start + range.toString().length };
  }

  // 模糊重定位：优先按存储偏移，失配则按 exact + 前后文搜索
  function locate(anchor, text) {
    const { start, exact, prefix, suffix } = anchor;
    if (!exact) return -1;
    if (text.substr(start, exact.length) === exact) return start;
    let idx = text.indexOf(exact), best = -1, bestScore = -Infinity;
    while (idx !== -1) {
      let score = 0;
      if (prefix) score += overlapTail(text.slice(Math.max(0, idx - 32), idx), prefix);
      if (suffix) score += overlapHead(text.slice(idx + exact.length, idx + exact.length + 32), suffix);
      score -= Math.abs(idx - start) / Math.max(text.length, 1);
      if (score > bestScore) { bestScore = score; best = idx; }
      idx = text.indexOf(exact, idx + 1);
    }
    return best;
  }
  function overlapTail(a, b) { let i = 0; while (i < a.length && i < b.length && a[a.length - 1 - i] === b[b.length - 1 - i]) i++; return i; }
  function overlapHead(a, b) { let i = 0; while (i < a.length && i < b.length && a[i] === b[i]) i++; return i; }

  /* ---------------- 包裹与解包 ---------------- */
  function wrapRange(range, item) {
    const nodes = textNodes().filter(n => range.intersectsNode(n));
    const targets = [];
    for (let n of nodes) {
      let s = (n === range.startContainer) ? range.startOffset : 0;
      let e = (n === range.endContainer) ? range.endOffset : n.data.length;
      if (s >= e) continue;
      if (e < n.data.length) n.splitText(e);
      if (s > 0) n = n.splitText(s);
      targets.push(n);
    }
    for (const n of targets) {
      const inSvg = n.parentElement && n.parentElement.namespaceURI === SVG_NS;
      const el = inSvg ? document.createElementNS(SVG_NS, 'tspan') : document.createElement('span');
      el.setAttribute('data-anno-id', item.id);
      el.setAttribute('class', 'anno');
      styleEl(el, item, inSvg);
      n.parentNode.insertBefore(el, n);
      el.appendChild(n);
    }
    return targets.length > 0;
  }

  function styleEl(el, item, inSvg) {
    if (item.type === 'hl') {
      if (inSvg) { el.setAttribute('fill', shade(item.color)); el.setAttribute('font-weight', '700'); }
      else el.style.background = item.color;
    } else if (item.type === 'color') {
      if (inSvg) el.setAttribute('fill', item.color);
      else { el.style.color = item.color; el.style.textDecorationColor = item.color; }
    } else if (item.type === 'note') {
      el.classList.add('anno-note');
      if (inSvg) {
        el.setAttribute('fill', '#b45309'); el.setAttribute('font-weight', '700');
        const t = document.createElementNS(SVG_NS, 'title');
        t.textContent = item.note || '（点击编辑批注）';
        el.appendChild(t);
      } else {
        el.title = item.note || '（点击编辑批注）';
      }
    }
  }
  // SVG 高亮退化色：把浅底纹色加深为可读字色
  function shade(c) {
    const map = { '#ffe867': '#b45309', '#a7f3c9': '#059669', '#fbc7d8': '#db2777', '#bfdcff': '#2563eb' };
    return map[c] || c;
  }

  function unwrap(id) {
    document.querySelectorAll('[data-anno-id="' + id + '"]').forEach(el => {
      const parent = el.parentNode;
      if (!parent) return;
      // 把所有子节点移出（包括嵌套的其他标记 span/tspan，防止叠加标记时误删文字）；
      // 只丢弃我们自己注入的 svg <title> 提示节点。
      [...el.childNodes].forEach(c => {
        const isOurTitle = c.nodeType === 1 && c.tagName && c.tagName.toLowerCase() === 'title';
        if (!isOurTitle) parent.insertBefore(c, el);
      });
      parent.removeChild(el);
      parent.normalize();
    });
  }
  function unwrapAll() { items.forEach(it => unwrap(it.id)); }

  function applyItem(item) {
    const text = docText();
    const pos = locate(item.anchor, text);
    if (pos < 0) return false;
    item.anchor.start = pos;                       // 自愈：更新偏移
    const range = rangeFromOffsets(pos, pos + item.anchor.exact.length);
    if (!range) return false;
    return wrapRange(range, item);
  }
  function applyAll() {
    let fail = 0;
    items.forEach(it => { if (!applyItem(it)) fail++; });
    if (fail) toast(fail + ' 条批注因文本变动未能定位（保留在存储中）');
  }
  function rerender() { unwrapAll(); if (visible) applyAll(); }

  /* ---------------- 动作 ---------------- */
  function captureSelection() {
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed || sel.rangeCount === 0) return null;
    const range = sel.getRangeAt(0);
    const exact = range.toString();
    if (!exact.trim()) return null;
    const { start } = offsetsFromRange(range);
    const text = docText();
    return {
      start,
      exact,
      prefix: text.slice(Math.max(0, start - 32), start),
      suffix: text.slice(start + exact.length, start + exact.length + 32),
    };
  }

  function addItem(type, color) {
    const anchor = captureSelection();
    if (!anchor) return;
    const item = { id: 'a' + Date.now() + Math.random().toString(36).slice(2, 6), type, color: color || null, note: '', anchor };
    items.push(item);
    if (visible) applyItem(item);
    save();
    window.getSelection().removeAllRanges();
    hideToolbar();
    if (type === 'note') openEditor(item);
  }

  function clearSelection() {
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed || sel.rangeCount === 0) return;
    const range = sel.getRangeAt(0);
    const hit = new Set();
    document.querySelectorAll('[data-anno-id]').forEach(el => {
      if (range.intersectsNode(el)) hit.add(el.getAttribute('data-anno-id'));
    });
    if (!hit.size) { toast('选区内没有批注'); return; }
    items = items.filter(it => !hit.has(it.id));
    hit.forEach(unwrap);
    save();
    sel.removeAllRanges();
    hideToolbar();
  }

  /* ---------------- 工具条 ---------------- */
  let toolbar, editor, panel;

  function buildToolbar() {
    toolbar = document.createElement('div');
    toolbar.setAttribute('data-anno-ui', '1');
    toolbar.id = 'anno-toolbar';
    const mkBtn = (html, title, fn) => {
      const b = document.createElement('button');
      b.innerHTML = html; b.title = title;
      b.addEventListener('mousedown', e => e.preventDefault());
      b.addEventListener('click', fn);
      return b;
    };
    HL_COLORS.forEach(([c, name]) => {
      const b = mkBtn('', name, () => addItem('hl', c));
      b.className = 'anno-swatch'; b.style.background = c;
      toolbar.appendChild(b);
    });
    toolbar.appendChild(sep());
    TXT_COLORS.forEach(([c, name]) => {
      const b = mkBtn('A', name, () => addItem('color', c));
      b.className = 'anno-abtn'; b.style.color = c;
      toolbar.appendChild(b);
    });
    toolbar.appendChild(sep());
    toolbar.appendChild(mkBtn('✎', '添加批注', () => addItem('note')));
    toolbar.appendChild(mkBtn('✕', '清除选区内的标记', clearSelection));
    document.body.appendChild(toolbar);
  }
  function sep() { const s = document.createElement('i'); s.className = 'anno-sep'; return s; }

  function showToolbar() {
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed || sel.rangeCount === 0) { hideToolbar(); return; }
    if (!sel.getRangeAt(0).toString().trim()) { hideToolbar(); return; }
    const rect = sel.getRangeAt(0).getBoundingClientRect();
    toolbar.style.display = 'flex';
    const tw = toolbar.offsetWidth;
    let left = rect.left + rect.width / 2 - tw / 2;
    left = Math.max(8, Math.min(left, window.innerWidth - tw - 8));
    let top = rect.top - toolbar.offsetHeight - 8;
    if (top < 4) top = rect.bottom + 8;
    toolbar.style.left = left + 'px';
    toolbar.style.top = top + 'px';
  }
  function hideToolbar() { if (toolbar) toolbar.style.display = 'none'; }

  /* ---------------- 批注编辑弹窗 ---------------- */
  function openEditor(item, anchorEl) {
    closeEditor();
    editor = document.createElement('div');
    editor.setAttribute('data-anno-ui', '1');
    editor.id = 'anno-editor';
    const quote = document.createElement('div');
    quote.className = 'anno-quote';
    quote.textContent = '“' + item.anchor.exact.slice(0, 60) + (item.anchor.exact.length > 60 ? '…' : '') + '”';
    const ta = document.createElement('textarea');
    ta.value = item.note || '';
    ta.placeholder = '写下你的批注…';
    const row = document.createElement('div');
    row.className = 'anno-row';
    const saveBtn = document.createElement('button');
    saveBtn.textContent = '保存';
    saveBtn.className = 'anno-primary';
    saveBtn.onclick = () => {
      item.note = ta.value.trim();
      save();
      unwrap(item.id);
      if (visible) applyItem(item);
      closeEditor();
    };
    const delBtn = document.createElement('button');
    delBtn.textContent = '删除批注';
    delBtn.onclick = () => {
      items = items.filter(i => i.id !== item.id);
      unwrap(item.id); save(); closeEditor();
    };
    row.appendChild(delBtn); row.appendChild(saveBtn);
    editor.appendChild(quote); editor.appendChild(ta); editor.appendChild(row);
    document.body.appendChild(editor);

    let rect = { left: window.innerWidth / 2 - 150, bottom: 120 };
    if (anchorEl && anchorEl.getBoundingClientRect) rect = anchorEl.getBoundingClientRect();
    let left = Math.max(8, Math.min(rect.left, window.innerWidth - 320));
    let top = (rect.bottom || 120) + 8;
    if (top + editor.offsetHeight > window.innerHeight - 8) top = Math.max(8, rect.top - editor.offsetHeight - 8);
    editor.style.left = left + 'px';
    editor.style.top = top + 'px';
    ta.focus();
  }
  function closeEditor() { if (editor) { editor.remove(); editor = null; } }

  /* ---------------- 右上角控制面板 ---------------- */
  function buildPanel() {
    panel = document.createElement('div');
    panel.setAttribute('data-anno-ui', '1');
    panel.id = 'anno-panel';
    const mk = (html, title, fn) => {
      const b = document.createElement('button');
      b.innerHTML = html; b.title = title; b.onclick = fn;
      panel.appendChild(b); return b;
    };
    const eye = mk('👁', '显示 / 隐藏所有标记', () => {
      visible = !visible;
      eye.style.opacity = visible ? 1 : 0.4;
      rerender();
    });
    saveBtn = mk('💾', '固化到文件', async () => {
      if (!fileHandle) { await bindFile(); }
      else if (await writeFile()) toast('已固化到文件 ✓');
    });
    mk('⤓', '导出批注为 JSON 文件', () => {
      const blob = new Blob([JSON.stringify({ page: PAGE_KEY, exported: new Date().toISOString(), items }, null, 2)],
        { type: 'application/json' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = PAGE_KEY.replace('anno:', '').replace(/\.html?$/, '') + '.annotations.json';
      a.click();
      URL.revokeObjectURL(a.href);
    });
    mk('⤒', '从 JSON 文件导入批注（合并）', () => {
      const input = document.createElement('input');
      input.type = 'file'; input.accept = '.json';
      input.onchange = () => {
        const f = input.files[0]; if (!f) return;
        f.text().then(txt => {
          try {
            const data = JSON.parse(txt);
            const incoming = data.items || [];
            const known = new Set(items.map(i => i.id));
            incoming.forEach(i => { if (!known.has(i.id)) items.push(i); });
            save(); rerender();
            toast('已导入 ' + incoming.length + ' 条');
          } catch (e) { toast('导入失败：不是有效的批注 JSON'); }
        });
      };
      input.click();
    });
    mk('🗑', '清空本页所有标记', () => {
      if (!items.length) { toast('本页没有标记'); return; }
      if (confirm('确定清空本页全部 ' + items.length + ' 条标记？（不可恢复，建议先导出）')) {
        unwrapAll(); items = []; save();
      }
    });
    mk('?', '使用说明', () => {
      toast('选中文字 → 工具条：色块=底纹 · A=改字色 · ✎=批注（悬停可见，点击可改）· ✕=清除。首次点 💾 选择本文件绑定后，所有痕迹自动固化进 HTML 文件本身（换浏览器打开也在）。SVG 图内文字不支持底纹，高亮会变为加粗改色。', 7000);
    });
    document.body.appendChild(panel);
  }

  /* ---------------- 杂项 ---------------- */
  let toastEl, toastTimer;
  function toast(msg, ms) {
    if (!toastEl) {
      toastEl = document.createElement('div');
      toastEl.setAttribute('data-anno-ui', '1');
      toastEl.id = 'anno-toast';
      document.body.appendChild(toastEl);
    }
    toastEl.textContent = msg;
    toastEl.style.opacity = 1;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { toastEl.style.opacity = 0; }, ms || 2600);
  }

  function injectCSS() {
    const css = `
#anno-toolbar { position: fixed; z-index: 9999; display: none; gap: 5px; align-items: center;
  background: #1f2933; border-radius: 10px; padding: 7px 10px; box-shadow: 0 6px 24px rgba(0,0,0,.25); }
#anno-toolbar button { border: none; cursor: pointer; border-radius: 6px; width: 24px; height: 24px;
  display: inline-flex; align-items: center; justify-content: center; font-size: 14px; background: transparent; color: #fff; }
#anno-toolbar button:hover { transform: scale(1.15); }
#anno-toolbar .anno-swatch { border: 1px solid rgba(255,255,255,.4); }
#anno-toolbar .anno-abtn { font-weight: 800; font-size: 15px; background: #fff; }
#anno-toolbar .anno-sep { width: 1px; height: 18px; background: rgba(255,255,255,.25); margin: 0 2px; }
#anno-panel { position: fixed; top: 14px; right: 14px; z-index: 9998; display: flex; gap: 6px;
  background: rgba(255,255,255,.92); border: 1px solid #e4e7eb; border-radius: 10px; padding: 6px 8px;
  box-shadow: 0 2px 10px rgba(0,0,0,.08); backdrop-filter: blur(4px); }
#anno-panel button { border: none; background: transparent; cursor: pointer; font-size: 15px;
  width: 26px; height: 26px; border-radius: 6px; position: relative; }
#anno-panel button:hover { background: #eef1f4; }
#anno-panel button.anno-attn::after { content: ''; position: absolute; top: 2px; right: 2px;
  width: 7px; height: 7px; border-radius: 50%; background: #dc2626; }
#anno-editor { position: fixed; z-index: 10000; width: 300px; background: #fff; border: 1px solid #e4e7eb;
  border-radius: 10px; box-shadow: 0 8px 30px rgba(0,0,0,.18); padding: 12px; }
#anno-editor .anno-quote { font-size: 12px; color: #52606d; border-left: 3px solid #b45309;
  padding-left: 8px; margin-bottom: 8px; line-height: 1.5; }
#anno-editor textarea { width: 100%; height: 76px; box-sizing: border-box; border: 1px solid #e4e7eb;
  border-radius: 6px; padding: 6px 8px; font-size: 13px; font-family: inherit; resize: vertical; }
#anno-editor .anno-row { display: flex; justify-content: flex-end; gap: 8px; margin-top: 8px; }
#anno-editor button { border: 1px solid #e4e7eb; background: #fff; border-radius: 6px;
  padding: 4px 12px; font-size: 12.5px; cursor: pointer; }
#anno-editor button.anno-primary { background: #2563eb; color: #fff; border-color: #2563eb; }
#anno-toast { position: fixed; left: 50%; bottom: 28px; transform: translateX(-50%); z-index: 10001;
  background: #1f2933; color: #fff; font-size: 13px; padding: 9px 18px; border-radius: 8px;
  max-width: 72%; line-height: 1.6; opacity: 0; transition: opacity .3s; pointer-events: none; }
span.anno { border-radius: 2px; }
span.anno-note { border-bottom: 2px dotted #b45309; cursor: pointer; }
tspan.anno-note { cursor: pointer; }
`;
    const style = document.createElement('style');
    style.textContent = css;
    document.head.appendChild(style);
  }

  /* ---------------- 事件绑定与启动 ---------------- */
  function init() {
    injectCSS();
    buildToolbar();
    buildPanel();

    // 取 localStorage 与文件内固化数据中较新的一份
    const local = loadLocal(), baked = loadBaked();
    if (local && baked) {
      items = (baked.updatedAt || 0) > (local.updatedAt || 0) ? baked.items : local.items;
    } else {
      items = (local || baked || { items: [] }).items || [];
    }
    localStorage.setItem(PAGE_KEY, JSON.stringify(payload()));
    applyAll();

    // 恢复已绑定的文件句柄（授权在首次改动时按需弹出，一个会话一次）
    idb.get(PAGE_KEY).then(async h => {
      if (h) {
        fileHandle = h;
        try { fsGranted = (await h.queryPermission({ mode: 'readwrite' })) === 'granted'; } catch (e) {}
      }
      updateSaveBtn();
    }).catch(() => updateSaveBtn());

    document.addEventListener('mouseup', e => {
      if (e.target.closest && e.target.closest('[data-anno-ui]')) return;
      setTimeout(showToolbar, 10);
    });
    document.addEventListener('mousedown', e => {
      if (e.target.closest && e.target.closest('[data-anno-ui]')) return;
      hideToolbar();
      if (editor && !editor.contains(e.target)) closeEditor();
    });
    document.addEventListener('keydown', e => {
      if (e.key === 'Escape') { hideToolbar(); closeEditor(); }
    });
    // 点击批注 → 编辑
    document.addEventListener('click', e => {
      const el = e.target.closest && e.target.closest('[data-anno-id]');
      if (!el) return;
      const item = items.find(i => i.id === el.getAttribute('data-anno-id'));
      if (item && item.type === 'note') openEditor(item, el);
    });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
