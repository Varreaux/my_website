(function () {
    'use strict';

    const COLS = 28;
    const ROWS = 16;
    const SQRT2 = Math.SQRT2;

    const COLORS = {
        empty: '#ffffff',
        grid: '#e2e8f0',
        obstacle: '#2d3748',
        start: '#2dd4bf',
        end: '#ef4444',
        visited: 'rgba(124, 92, 255, 0.18)',
        frontier: 'rgba(77, 208, 225, 0.55)',
        path: '#7c5cff',
        pathHalo: 'rgba(124, 92, 255, 0.25)',
    };

    const DIRS = [
        [-1, -1], [0, -1], [1, -1],
        [-1, 0],           [1, 0],
        [-1, 1],  [0, 1],  [1, 1],
    ];

    function key(x, y) { return x + ',' + y; }
    function octile(a, b) {
        const dx = Math.abs(a.x - b.x);
        const dy = Math.abs(a.y - b.y);
        return Math.max(dx, dy) + (SQRT2 - 1) * Math.min(dx, dy);
    }

    function angleBetween(a, b) {
        const dot = a[0] * b[0] + a[1] * b[1];
        const magA = Math.hypot(a[0], a[1]);
        const magB = Math.hypot(b[0], b[1]);
        const cos = Math.max(-1, Math.min(1, dot / (magA * magB)));
        return Math.acos(cos);
    }

    class PQ {
        constructor() { this.items = []; }
        push(item) {
            let lo = 0, hi = this.items.length;
            while (lo < hi) {
                const mid = (lo + hi) >> 1;
                if (this.items[mid].f <= item.f) lo = mid + 1; else hi = mid;
            }
            this.items.splice(lo, 0, item);
        }
        pop() { return this.items.shift(); }
        get size() { return this.items.length; }
    }

    function makeGrid() {
        const g = [];
        for (let y = 0; y < ROWS; y++) {
            g.push(new Array(COLS).fill(0));
        }
        return g;
    }

    function* aStar(grid, start, end, anglePenalty) {
        const open = new PQ();
        const gScore = new Map();
        const cameFrom = new Map();
        const cameDir = new Map();
        const closed = new Set();

        const sk = key(start.x, start.y);
        gScore.set(sk, 0);
        open.push({ x: start.x, y: start.y, f: octile(start, end), dir: null });

        let lastFrontier = [];

        while (open.size > 0) {
            const cur = open.pop();
            const ck = key(cur.x, cur.y);
            if (closed.has(ck)) continue;
            closed.add(ck);

            lastFrontier = open.items.map(n => ({ x: n.x, y: n.y }));
            yield { type: 'expand', node: { x: cur.x, y: cur.y }, frontier: lastFrontier };

            if (cur.x === end.x && cur.y === end.y) {
                const path = [];
                let k = ck;
                while (cameFrom.has(k)) {
                    const [px, py] = cameFrom.get(k).split(',').map(Number);
                    path.unshift({ x: px, y: py });
                    k = key(px, py);
                }
                path.push({ x: end.x, y: end.y });
                yield { type: 'done', path };
                return;
            }

            const curDir = cameDir.get(ck) || null;

            for (const [dx, dy] of DIRS) {
                const nx = cur.x + dx, ny = cur.y + dy;
                if (nx < 0 || nx >= COLS || ny < 0 || ny >= ROWS) continue;
                if (grid[ny][nx] === 1) continue;
                if (dx !== 0 && dy !== 0) {
                    if (grid[cur.y][nx] === 1 && grid[ny][cur.x] === 1) continue;
                }
                const nk = key(nx, ny);
                if (closed.has(nk)) continue;

                const stepCost = (dx !== 0 && dy !== 0) ? SQRT2 : 1;
                let angleCost = 0;
                if (curDir && anglePenalty > 0) {
                    const a = angleBetween(curDir, [dx, dy]);
                    angleCost = (a / Math.PI) * anglePenalty;
                }
                const tentative = gScore.get(ck) + stepCost + angleCost;

                if (!gScore.has(nk) || tentative < gScore.get(nk)) {
                    gScore.set(nk, tentative);
                    cameFrom.set(nk, ck);
                    cameDir.set(nk, [dx, dy]);
                    const f = tentative + octile({ x: nx, y: ny }, end);
                    open.push({ x: nx, y: ny, f, dir: [dx, dy] });
                }
            }
        }
        yield { type: 'fail' };
    }

    function init() {
        const canvas = document.getElementById('astar-canvas');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');

        const dpr = window.devicePixelRatio || 1;
        const cssWidth = canvas.clientWidth;
        const cell = cssWidth / COLS;
        const cssHeight = cell * ROWS;
        canvas.style.height = cssHeight + 'px';
        canvas.width = Math.round(cssWidth * dpr);
        canvas.height = Math.round(cssHeight * dpr);
        ctx.scale(dpr, dpr);

        let grid = makeGrid();
        let start = { x: 2, y: Math.floor(ROWS / 2) };
        let end = { x: COLS - 3, y: Math.floor(ROWS / 2) };
        let visited = new Set();
        let frontier = [];
        let path = [];
        let mode = 'obstacle';
        let isPainting = false;
        let paintValue = 1;
        let running = false;
        let runToken = 0;

        const stats = document.getElementById('astar-stats');
        const messageEl = document.getElementById('astar-message');
        const penaltyInput = document.getElementById('astar-penalty');
        const penaltyVal = document.getElementById('astar-penalty-val');
        const speedInput = document.getElementById('astar-speed');
        const runBtn = document.getElementById('astar-run');
        const clearPathBtn = document.getElementById('astar-clear-path');
        const resetBtn = document.getElementById('astar-reset');
        const modeButtons = document.querySelectorAll('[data-astar-mode]');
        const presetButtons = document.querySelectorAll('[data-astar-preset]');

        function render() {
            ctx.clearRect(0, 0, cssWidth, cssHeight);

            for (let y = 0; y < ROWS; y++) {
                for (let x = 0; x < COLS; x++) {
                    if (grid[y][x] === 1) {
                        ctx.fillStyle = COLORS.obstacle;
                        ctx.fillRect(x * cell, y * cell, cell, cell);
                    }
                }
            }

            ctx.fillStyle = COLORS.visited;
            for (const k of visited) {
                const [vx, vy] = k.split(',').map(Number);
                ctx.fillRect(vx * cell, vy * cell, cell, cell);
            }

            ctx.fillStyle = COLORS.frontier;
            for (const f of frontier) {
                ctx.fillRect(f.x * cell, f.y * cell, cell, cell);
            }

            if (path.length > 1) {
                ctx.lineWidth = Math.max(3, cell * 0.18);
                ctx.lineCap = 'round';
                ctx.lineJoin = 'round';
                ctx.strokeStyle = COLORS.pathHalo;
                ctx.beginPath();
                ctx.moveTo(path[0].x * cell + cell / 2, path[0].y * cell + cell / 2);
                for (let i = 1; i < path.length; i++) {
                    ctx.lineTo(path[i].x * cell + cell / 2, path[i].y * cell + cell / 2);
                }
                ctx.stroke();
                ctx.lineWidth = Math.max(2, cell * 0.12);
                ctx.strokeStyle = COLORS.path;
                ctx.stroke();
            }

            ctx.strokeStyle = COLORS.grid;
            ctx.lineWidth = 1;
            ctx.beginPath();
            for (let x = 0; x <= COLS; x++) {
                ctx.moveTo(x * cell, 0);
                ctx.lineTo(x * cell, cssHeight);
            }
            for (let y = 0; y <= ROWS; y++) {
                ctx.moveTo(0, y * cell);
                ctx.lineTo(cssWidth, y * cell);
            }
            ctx.stroke();

            drawMarker(start, COLORS.start, 'S');
            drawMarker(end, COLORS.end, 'E');
        }

        function drawMarker(pt, color, label) {
            const cx = pt.x * cell + cell / 2;
            const cy = pt.y * cell + cell / 2;
            ctx.beginPath();
            ctx.arc(cx, cy, cell * 0.36, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.fill();
            ctx.fillStyle = '#ffffff';
            ctx.font = '600 ' + Math.round(cell * 0.42) + 'px Inter, system-ui, sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(label, cx, cy + 1);
        }

        function pointerToCell(evt) {
            const rect = canvas.getBoundingClientRect();
            const x = Math.floor((evt.clientX - rect.left) / (rect.width / COLS));
            const y = Math.floor((evt.clientY - rect.top) / (rect.height / ROWS));
            if (x < 0 || x >= COLS || y < 0 || y >= ROWS) return null;
            return { x, y };
        }

        function isStart(c) { return c.x === start.x && c.y === start.y; }
        function isEnd(c) { return c.x === end.x && c.y === end.y; }

        function applyPaint(c) {
            if (running) return;
            if (mode === 'obstacle') {
                if (isStart(c) || isEnd(c)) return;
                grid[c.y][c.x] = paintValue;
            } else if (mode === 'start') {
                if (isEnd(c) || grid[c.y][c.x] === 1) return;
                start = c;
            } else if (mode === 'end') {
                if (isStart(c) || grid[c.y][c.x] === 1) return;
                end = c;
            }
            clearPath();
            render();
        }

        function clearPath() {
            visited = new Set();
            frontier = [];
            path = [];
            updateStats();
            messageEl.textContent = '';
        }

        function resetAll() {
            grid = makeGrid();
            start = { x: 2, y: Math.floor(ROWS / 2) };
            end = { x: COLS - 3, y: Math.floor(ROWS / 2) };
            clearPath();
            render();
        }

        function updateStats() {
            stats.textContent = 'Visited: ' + visited.size + '   ·   Path length: ' + path.length;
        }

        function setMode(next) {
            mode = next;
            modeButtons.forEach(btn => {
                btn.classList.toggle('active', btn.dataset.astarMode === next);
            });
        }

        function speedDelay() {
            const v = Number(speedInput.value);
            return Math.max(0, 60 - v);
        }

        function sleep(ms) {
            return new Promise(res => {
                if (ms <= 0) requestAnimationFrame(res);
                else setTimeout(res, ms);
            });
        }

        async function run() {
            if (running) return;
            running = true;
            runBtn.disabled = true;
            clearPath();
            render();
            messageEl.textContent = 'Searching...';

            const token = ++runToken;
            const penalty = Number(penaltyInput.value);
            const gen = aStar(grid, start, end, penalty);

            let stepCount = 0;
            for (const step of gen) {
                if (token !== runToken) return;
                if (step.type === 'expand') {
                    visited.add(key(step.node.x, step.node.y));
                    frontier = step.frontier;
                    stepCount++;
                    if (stepCount % Math.max(1, Math.round(speedInput.value / 10)) === 0) {
                        updateStats();
                        render();
                        await sleep(speedDelay());
                    }
                } else if (step.type === 'done') {
                    path = step.path;
                    frontier = [];
                    updateStats();
                    render();
                    messageEl.textContent = 'Path found.';
                } else if (step.type === 'fail') {
                    frontier = [];
                    updateStats();
                    render();
                    messageEl.textContent = 'No path found. Try removing some obstacles.';
                }
            }
            running = false;
            runBtn.disabled = false;
        }

        function loadPreset(name) {
            if (running) return;
            grid = makeGrid();
            if (name === 'maze') {
                for (let y = 2; y < ROWS - 2; y += 4) {
                    for (let x = 0; x < COLS; x++) {
                        if (x % 8 !== 3 && x % 8 !== 4) grid[y][x] = 1;
                    }
                }
                for (let y = 4; y < ROWS - 2; y += 4) {
                    for (let x = 0; x < COLS; x++) {
                        if ((x + 4) % 8 !== 3 && (x + 4) % 8 !== 4) grid[y][x] = 1;
                    }
                }
            } else if (name === 'rooms') {
                const wallY1 = Math.floor(ROWS / 3);
                const wallY2 = Math.floor((2 * ROWS) / 3);
                for (let x = 0; x < COLS; x++) {
                    if (x !== 6 && x !== 18) grid[wallY1][x] = 1;
                    if (x !== 10 && x !== 22) grid[wallY2][x] = 1;
                }
            } else if (name === 'diagonal') {
                for (let i = 0; i < Math.min(ROWS, COLS); i++) {
                    if (i > 2 && i < ROWS - 2) {
                        grid[i][i + 2] = 1;
                        grid[i][i + 3] = 1;
                    }
                }
            }
            start = { x: 2, y: Math.floor(ROWS / 2) };
            end = { x: COLS - 3, y: Math.floor(ROWS / 2) };
            if (grid[start.y][start.x] === 1) grid[start.y][start.x] = 0;
            if (grid[end.y][end.x] === 1) grid[end.y][end.x] = 0;
            clearPath();
            render();
        }

        canvas.addEventListener('pointerdown', e => {
            const c = pointerToCell(e);
            if (!c) return;
            canvas.setPointerCapture(e.pointerId);
            isPainting = true;
            if (mode === 'obstacle') {
                paintValue = grid[c.y][c.x] === 1 ? 0 : 1;
            }
            applyPaint(c);
        });
        canvas.addEventListener('pointermove', e => {
            if (!isPainting) return;
            const c = pointerToCell(e);
            if (!c) return;
            applyPaint(c);
        });
        canvas.addEventListener('pointerup', () => { isPainting = false; });
        canvas.addEventListener('pointerleave', () => { isPainting = false; });

        modeButtons.forEach(btn => {
            btn.addEventListener('click', () => setMode(btn.dataset.astarMode));
        });
        presetButtons.forEach(btn => {
            btn.addEventListener('click', () => loadPreset(btn.dataset.astarPreset));
        });

        penaltyInput.addEventListener('input', () => {
            penaltyVal.textContent = Number(penaltyInput.value).toFixed(1);
        });

        runBtn.addEventListener('click', run);
        clearPathBtn.addEventListener('click', () => { clearPath(); render(); });
        resetBtn.addEventListener('click', resetAll);

        penaltyVal.textContent = Number(penaltyInput.value).toFixed(1);
        setMode('obstacle');
        render();
        updateStats();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        setTimeout(init, 0);
    }
})();
