(function () {
    'use strict';

    const CANONICAL_GOAL = [1, 2, 3, 4, 5, 6, 7, 8, 0];

    const PRESETS = {
        p3: {
            name: 'Preset puzzle',
            state: [8, 7, 3, 0, 4, 5, 6, 2, 1],
            goal: CANONICAL_GOAL.slice(),
        },
    };

    function buildGoalPos(goal) {
        const pos = {};
        for (let i = 0; i < 9; i++) {
            if (goal[i] !== 0) pos[goal[i]] = [Math.floor(i / 3), i % 3];
        }
        return pos;
    }

    function manhattan(state, goalPos) {
        let d = 0;
        for (let i = 0; i < 9; i++) {
            const t = state[i];
            if (t === 0) continue;
            const r = Math.floor(i / 3), c = i % 3;
            const [gr, gc] = goalPos[t];
            d += Math.abs(r - gr) + Math.abs(c - gc);
        }
        return d;
    }

    function linearConflicts(state, goalPos) {
        let conf = 0;
        for (let r = 0; r < 3; r++) {
            const filtered = [];
            for (let c = 0; c < 3; c++) {
                const t = state[r * 3 + c];
                if (t === 0) continue;
                const [gr, gc] = goalPos[t];
                if (gr === r) filtered.push(gc);
            }
            for (let i = 0; i < filtered.length; i++) {
                for (let j = i + 1; j < filtered.length; j++) {
                    if (filtered[i] > filtered[j]) conf++;
                }
            }
        }
        for (let c = 0; c < 3; c++) {
            const filtered = [];
            for (let r = 0; r < 3; r++) {
                const t = state[r * 3 + c];
                if (t === 0) continue;
                const [gr, gc] = goalPos[t];
                if (gc === c) filtered.push(gr);
            }
            for (let i = 0; i < filtered.length; i++) {
                for (let j = i + 1; j < filtered.length; j++) {
                    if (filtered[i] > filtered[j]) conf++;
                }
            }
        }
        return conf;
    }

    function inversions(state) {
        let inv = 0;
        const tiles = state.filter(x => x !== 0);
        for (let i = 0; i < tiles.length; i++) {
            for (let j = i + 1; j < tiles.length; j++) {
                if (tiles[i] > tiles[j]) inv++;
            }
        }
        return inv;
    }

    function key(state) {
        let k = 0;
        for (let i = 0; i < 9; i++) k = k * 10 + state[i];
        return k;
    }

    function neighbors(state) {
        const blank = state.indexOf(0);
        const r = Math.floor(blank / 3), c = blank % 3;
        const out = [];
        function swap(nr, nc, action) {
            const ns = state.slice();
            const ni = nr * 3 + nc;
            ns[blank] = ns[ni];
            ns[ni] = 0;
            out.push({ action, state: ns });
        }
        if (c > 0) swap(r, c - 1, 'L');
        if (c < 2) swap(r, c + 1, 'R');
        if (r > 0) swap(r - 1, c, 'U');
        if (r < 2) swap(r + 1, c, 'D');
        return out;
    }

    function applyAction(state, action) {
        const blank = state.indexOf(0);
        const r = Math.floor(blank / 3), c = blank % 3;
        let nr = r, nc = c;
        if (action === 'L') nc = c - 1;
        else if (action === 'R') nc = c + 1;
        else if (action === 'U') nr = r - 1;
        else if (action === 'D') nr = r + 1;
        const ns = state.slice();
        const ni = nr * 3 + nc;
        ns[blank] = ns[ni];
        ns[ni] = 0;
        return ns;
    }

    function isSolvable(state, goal) {
        return (inversions(state) % 2) === (inversions(goal) % 2);
    }

    function scramble(steps, goal) {
        let state = goal.slice();
        let last = null;
        const rev = { L: 'R', R: 'L', U: 'D', D: 'U' };
        for (let i = 0; i < steps; i++) {
            const moves = neighbors(state).filter(m => m.action !== rev[last]);
            const pick = moves[Math.floor(Math.random() * moves.length)];
            state = pick.state;
            last = pick.action;
        }
        return state;
    }

    class MinHeap {
        constructor() { this.items = []; }
        get size() { return this.items.length; }
        push(item) { this.items.push(item); this._up(this.items.length - 1); }
        pop() {
            const top = this.items[0];
            const last = this.items.pop();
            if (this.items.length > 0) {
                this.items[0] = last;
                this._down(0);
            }
            return top;
        }
        _less(a, b) {
            const A = this.items[a], B = this.items[b];
            if (A.f !== B.f) return A.f < B.f;
            return A.id < B.id;
        }
        _up(i) {
            while (i > 0) {
                const p = (i - 1) >> 1;
                if (this._less(i, p)) { [this.items[i], this.items[p]] = [this.items[p], this.items[i]]; i = p; }
                else break;
            }
        }
        _down(i) {
            const n = this.items.length;
            while (true) {
                const l = 2 * i + 1, r = 2 * i + 2;
                let s = i;
                if (l < n && this._less(l, s)) s = l;
                if (r < n && this._less(r, s)) s = r;
                if (s !== i) { [this.items[i], this.items[s]] = [this.items[s], this.items[i]]; i = s; }
                else break;
            }
        }
    }

    async function aStar(initial, goal, heuristic, onProgress) {
        const goalPos = buildGoalPos(goal);
        const h = heuristic === 'h1'
            ? (s) => manhattan(s, goalPos)
            : (s) => manhattan(s, goalPos) + 2 * linearConflicts(s, goalPos);

        const goalKey = key(goal);
        const heap = new MinHeap();
        const gScore = new Map();
        const cameFrom = new Map();
        const closed = new Set();
        let nextId = 0;
        let nodes = 1;

        const expansionLog = [];
        const keyToLogIndex = new Map();

        const initKey = key(initial);
        gScore.set(initKey, 0);
        cameFrom.set(initKey, { prev: null, action: null });
        heap.push({ f: h(initial), id: nextId++, state: initial, key: initKey });

        let lastYield = 0;
        while (heap.size > 0) {
            const cur = heap.pop();
            if (closed.has(cur.key)) continue;
            closed.add(cur.key);

            const cf = cameFrom.get(cur.key);
            const parentKey = cf && cf.prev ? key(cf.prev) : null;
            const parentIdx = parentKey !== null ? keyToLogIndex.get(parentKey) : null;
            keyToLogIndex.set(cur.key, expansionLog.length);
            expansionLog.push({
                state: cur.state,
                f: cur.f,
                g: gScore.get(cur.key),
                action: cf ? cf.action : null,
                parentIdx: parentIdx === undefined ? null : parentIdx,
                key: cur.key,
            });

            if (cur.key === goalKey) {
                const actions = [];
                const states = [];
                let k = cur.key, s = cur.state;
                while (true) {
                    states.unshift(s);
                    const cf2 = cameFrom.get(k);
                    if (cf2.prev === null) break;
                    actions.unshift(cf2.action);
                    s = cf2.prev;
                    k = key(s);
                }
                const fValues = states.map(st => (gScore.get(key(st)) || 0) + h(st));
                const pathKeys = states.map(s => key(s));
                return { actions, fValues, states, depth: actions.length, nodes, heuristic, expansionLog, pathKeys };
            }

            const curG = gScore.get(cur.key);
            for (const { action, state: nbr } of neighbors(cur.state)) {
                const nk = key(nbr);
                if (closed.has(nk)) continue;
                const newG = curG + 1;
                const oldG = gScore.has(nk) ? gScore.get(nk) : Infinity;
                if (newG < oldG) {
                    gScore.set(nk, newG);
                    cameFrom.set(nk, { prev: cur.state, action });
                    heap.push({ f: newG + h(nbr), id: nextId++, state: nbr, key: nk });
                    nodes++;
                }
            }

            if (nodes - lastYield > 2500) {
                lastYield = nodes;
                if (onProgress) onProgress(nodes);
                await new Promise(res => setTimeout(res, 0));
            }
        }
        return null;
    }

    function init() {
        const board = document.getElementById('ep-board');
        if (!board) return;

        const stats = document.getElementById('ep-stats');
        const message = document.getElementById('ep-message');
        const speedInput = document.getElementById('ep-speed');
        const solveBtn = document.getElementById('ep-solve');
        const scrambleBtn = document.getElementById('ep-scramble');
        const resetBtn = document.getElementById('ep-reset');
        const heuristicBtns = document.querySelectorAll('[data-ep-heuristic]');
        const presetBtns = document.querySelectorAll('[data-ep-preset]');

        let heuristic = 'h2';
        let initialState = PRESETS.p3.state.slice();
        let currentGoal = PRESETS.p3.goal.slice();
        let currentState = initialState.slice();
        let running = false;
        let runToken = 0;

        const tileEls = {};
        let cellSize = 64;

        function buildBoard() {
            board.innerHTML = '';
            for (let t = 1; t <= 8; t++) {
                const tile = document.createElement('div');
                tile.className = 'ep-tile';
                tile.textContent = String(t);
                board.appendChild(tile);
                tileEls[t] = tile;
            }
            measureCell();
            renderBoard(currentState, false);
        }

        function measureCell() {
            const rect = board.getBoundingClientRect();
            cellSize = rect.width / 3;
            for (let t = 1; t <= 8; t++) {
                tileEls[t].style.width = cellSize + 'px';
                tileEls[t].style.height = cellSize + 'px';
                tileEls[t].style.fontSize = (cellSize * 0.42) + 'px';
            }
        }

        function renderBoard(state, animate) {
            for (let i = 0; i < 9; i++) {
                const t = state[i];
                if (t === 0) continue;
                const r = Math.floor(i / 3), c = i % 3;
                const tile = tileEls[t];
                tile.style.transition = animate ? 'transform 220ms ease' : 'none';
                tile.style.transform = `translate(${c * cellSize}px, ${r * cellSize}px)`;
            }
            const solved = state.every((v, i) => v === currentGoal[i]);
            board.classList.toggle('solved', solved);
        }

        function renderGoalBoard() {
            const goalBoard = document.getElementById('ep-goal-board');
            if (!goalBoard) return;
            goalBoard.innerHTML = '';
            for (let i = 0; i < 9; i++) {
                const t = currentGoal[i];
                const cell = document.createElement('div');
                cell.className = 'ep-tile static';
                cell.style.gridArea = (Math.floor(i / 3) + 1) + ' / ' + (i % 3 + 1);
                if (t !== 0) {
                    cell.textContent = String(t);
                } else {
                    cell.classList.add('blank');
                }
                goalBoard.appendChild(cell);
            }
        }

        function setHeuristic(h) {
            heuristic = h;
            heuristicBtns.forEach(btn => {
                btn.classList.toggle('active', btn.dataset.epHeuristic === h);
            });
            updateStats();
        }

        function updateStats(extra) {
            const label = heuristic === 'h1' ? 'h1: Manhattan' : 'h2: Manhattan + linear conflict';
            let line = 'Heuristic: <strong>' + label + '</strong>';
            if (extra) {
                if (extra.nodes !== undefined) line += '   ·   Generated: <strong>' + extra.nodes + '</strong>';
                if (extra.expanded !== undefined) line += '   ·   Expanded: <strong>' + extra.expanded + '</strong>';
                if (extra.depth !== undefined) line += '   ·   Solution depth: <strong>' + extra.depth + '</strong>';
            }
            stats.innerHTML = line;
        }

        function loadInitial(state, goal) {
            if (running) return;
            initialState = state.slice();
            currentGoal = (goal || CANONICAL_GOAL).slice();
            currentState = state.slice();
            renderGoalBoard();
            renderBoard(currentState, false);
            message.textContent = '';
            updateStats();
        }

        function speedDelay() {
            const v = Number(speedInput.value);
            return Math.max(40, 700 - v * 6);
        }

        function sleep(ms) {
            return new Promise(res => setTimeout(res, ms));
        }

        const TREE_CAP = 30;
        const treeContainer = document.getElementById('ep-tree-container');

        function layoutTree(visible) {
            const NODE_W = 52;
            const NODE_H = 52;
            const H_GAP = 14;
            const V_GAP = 32;
            const childrenIdx = new Map();
            visible.forEach((node, i) => {
                if (node.parentIdx !== null && node.parentIdx !== undefined) {
                    const list = childrenIdx.get(node.parentIdx) || [];
                    list.push(i);
                    childrenIdx.set(node.parentIdx, list);
                }
            });
            const widths = new Array(visible.length).fill(NODE_W);
            function computeWidth(idx) {
                const ch = childrenIdx.get(idx) || [];
                if (ch.length === 0) { widths[idx] = NODE_W; return NODE_W; }
                let s = 0;
                for (let i = 0; i < ch.length; i++) {
                    s += computeWidth(ch[i]);
                    if (i > 0) s += H_GAP;
                }
                widths[idx] = Math.max(NODE_W, s);
                return widths[idx];
            }
            if (visible.length > 0) computeWidth(0);
            const positions = new Array(visible.length);
            function place(idx, leftX, depth) {
                const myW = widths[idx];
                positions[idx] = {
                    x: leftX + myW / 2 - NODE_W / 2,
                    y: depth * (NODE_H + V_GAP),
                    cx: leftX + myW / 2,
                };
                const ch = childrenIdx.get(idx) || [];
                if (ch.length > 0) {
                    let total = 0;
                    for (let i = 0; i < ch.length; i++) {
                        total += widths[ch[i]];
                        if (i > 0) total += H_GAP;
                    }
                    let cx = leftX + (myW - total) / 2;
                    for (const ci of ch) {
                        place(ci, cx, depth + 1);
                        cx += widths[ci] + H_GAP;
                    }
                }
            }
            if (visible.length > 0) place(0, 0, 0);
            return { positions, totalWidth: widths[0] || NODE_W, nodeW: NODE_W, nodeH: NODE_H };
        }

        function renderTreeOnce(visible, layout, solutionPathSet, moreCount) {
            const NS = 'http://www.w3.org/2000/svg';
            treeContainer.innerHTML = '';
            if (visible.length === 0) return { nodeEls: [], edgeEls: [], stubEls: [] };
            const PAD = 24;
            const stubExtra = moreCount > 0 ? 60 : 0;
            const totalH = Math.max.apply(null, layout.positions.map(p => p.y)) + layout.nodeH + 24 + stubExtra;
            const totalW = layout.totalWidth + PAD * 2;
            const svg = document.createElementNS(NS, 'svg');
            svg.setAttribute('width', totalW);
            svg.setAttribute('height', totalH);
            svg.setAttribute('viewBox', '0 0 ' + totalW + ' ' + totalH);
            svg.classList.add('ep-tree-svg');

            const edgeEls = [];
            for (let i = 0; i < visible.length; i++) {
                if (visible[i].parentIdx === null || visible[i].parentIdx === undefined) {
                    edgeEls.push(null);
                    continue;
                }
                const p = layout.positions[visible[i].parentIdx];
                const c = layout.positions[i];
                const line = document.createElementNS(NS, 'line');
                line.setAttribute('x1', p.cx + PAD);
                line.setAttribute('y1', p.y + layout.nodeH);
                line.setAttribute('x2', c.cx + PAD);
                line.setAttribute('y2', c.y);
                line.classList.add('ep-tree-edge');
                if (solutionPathSet && solutionPathSet.has(visible[i].key) && solutionPathSet.has(visible[visible[i].parentIdx].key)) {
                    line.classList.add('on-path');
                }
                svg.appendChild(line);
                edgeEls.push(line);
            }

            const nodeEls = [];
            for (let i = 0; i < visible.length; i++) {
                const node = visible[i];
                const pos = layout.positions[i];
                const g = document.createElementNS(NS, 'g');
                g.setAttribute('transform', 'translate(' + (pos.x + PAD) + ', ' + pos.y + ')');
                g.classList.add('ep-tree-node');
                if (solutionPathSet && solutionPathSet.has(node.key)) g.classList.add('on-path');

                const rect = document.createElementNS(NS, 'rect');
                rect.setAttribute('width', layout.nodeW);
                rect.setAttribute('height', layout.nodeH);
                rect.setAttribute('rx', 5);
                g.appendChild(rect);

                const cellSize = layout.nodeW / 3;
                for (let p9 = 0; p9 < 9; p9++) {
                    const t = node.state[p9];
                    if (t === 0) continue;
                    const r = Math.floor(p9 / 3), c = p9 % 3;
                    const text = document.createElementNS(NS, 'text');
                    text.setAttribute('x', c * cellSize + cellSize / 2);
                    text.setAttribute('y', r * cellSize + cellSize / 2 + 4);
                    text.setAttribute('text-anchor', 'middle');
                    text.classList.add('ep-tree-tile');
                    text.textContent = String(t);
                    g.appendChild(text);
                }

                const fLabel = document.createElementNS(NS, 'text');
                fLabel.setAttribute('x', layout.nodeW / 2);
                fLabel.setAttribute('y', layout.nodeH + 12);
                fLabel.setAttribute('text-anchor', 'middle');
                fLabel.classList.add('ep-tree-flabel');
                fLabel.textContent = 'f=' + node.f;
                g.appendChild(fLabel);

                svg.appendChild(g);
                nodeEls.push(g);
            }

            const stubEls = [];
            if (moreCount > 0 && visible.length > 0) {
                const lastIdx = visible.length - 1;
                const lastPos = layout.positions[lastIdx];
                const cx = lastPos.cx + PAD;
                const stubY1 = lastPos.y + layout.nodeH + 4;
                const stubY2 = stubY1 + 30;
                const labelY = stubY2 + 16;

                const stubLine = document.createElementNS(NS, 'line');
                stubLine.setAttribute('x1', cx);
                stubLine.setAttribute('y1', stubY1);
                stubLine.setAttribute('x2', cx);
                stubLine.setAttribute('y2', stubY2);
                stubLine.classList.add('ep-tree-edge', 'ep-tree-stub');
                svg.appendChild(stubLine);
                stubEls.push(stubLine);

                const stubLabel = document.createElementNS(NS, 'text');
                stubLabel.setAttribute('x', cx);
                stubLabel.setAttribute('y', labelY);
                stubLabel.setAttribute('text-anchor', 'middle');
                stubLabel.classList.add('ep-tree-stub-label');
                stubLabel.textContent = '+ ' + moreCount + ' more nodes to expand';
                svg.appendChild(stubLabel);
                stubEls.push(stubLabel);
            }

            treeContainer.appendChild(svg);
            return { nodeEls, edgeEls, stubEls };
        }

        function clearTree() {
            if (treeContainer) treeContainer.innerHTML = '';
        }

        async function animateTree(expansionLog, solutionPathSet, stepDelay, getToken, abortToken) {
            if (!treeContainer) return;
            const visible = expansionLog.slice(0, TREE_CAP);
            const layout = layoutTree(visible);
            const moreCount = expansionLog.length - TREE_CAP;
            const { nodeEls, edgeEls, stubEls } = renderTreeOnce(visible, layout, solutionPathSet, moreCount);
            let prev = null;
            for (let i = 0; i < visible.length; i++) {
                if (getToken && getToken() !== abortToken) return;
                nodeEls[i].classList.add('visible');
                if (edgeEls[i]) edgeEls[i].classList.add('visible');
                if (prev) prev.classList.remove('current');
                nodeEls[i].classList.add('current');
                prev = nodeEls[i];
                await sleep(stepDelay);
            }
            if (prev) prev.classList.remove('current');
            if (stubEls && stubEls.length > 0) {
                await sleep(stepDelay);
                stubEls.forEach(el => el.classList.add('visible'));
            }
        }

        async function runSolve() {
            if (running) return;
            if (!isSolvable(initialState, currentGoal)) {
                message.textContent = 'This puzzle is unsolvable (parity check).';
                return;
            }
            running = true;
            solveBtn.disabled = true;
            scrambleBtn.disabled = true;
            resetBtn.disabled = true;
            const token = ++runToken;
            currentState = initialState.slice();
            renderBoard(currentState, false);
            clearTree();
            message.textContent = 'Searching...';
            updateStats({ nodes: 0 });

            const result = await aStar(initialState, currentGoal, heuristic, (n) => {
                if (token !== runToken) return;
                updateStats({ nodes: n });
            });

            if (token !== runToken) return;
            if (!result) {
                message.textContent = 'No solution found.';
                running = false;
                solveBtn.disabled = false;
                scrambleBtn.disabled = false;
                resetBtn.disabled = false;
                return;
            }

            updateStats({ nodes: result.nodes, expanded: result.expansionLog.length, depth: result.depth });
            message.textContent = 'Playing solution: ' + result.actions.length + ' moves.';

            const solutionPathSet = new Set(result.pathKeys);
            const treeNodeCount = Math.min(result.expansionLog.length, TREE_CAP);
            const boardTotalMs = result.actions.length * speedDelay();
            const treeStepDelay = treeNodeCount > 0
                ? Math.max(120, Math.min(boardTotalMs / treeNodeCount, 380))
                : 200;

            const treePromise = animateTree(result.expansionLog, solutionPathSet, treeStepDelay, () => runToken, token);
            const boardPromise = (async () => {
                for (let i = 0; i < result.actions.length; i++) {
                    if (token !== runToken) return;
                    currentState = applyAction(currentState, result.actions[i]);
                    renderBoard(currentState, true);
                    await sleep(speedDelay());
                }
            })();

            await Promise.all([treePromise, boardPromise]);

            if (token !== runToken) return;
            message.textContent = 'Solved in ' + result.depth + ' moves using ' + (heuristic === 'h1' ? 'h1' : 'h2') + '.';
            running = false;
            solveBtn.disabled = false;
            scrambleBtn.disabled = false;
            resetBtn.disabled = false;
        }

        function doScramble() {
            if (running) return;
            const s = scramble(20, CANONICAL_GOAL);
            loadInitial(s, CANONICAL_GOAL);
            message.textContent = 'New scramble loaded (depth ~20).';
        }

        function doReset() {
            if (running) {
                runToken++;
                running = false;
                solveBtn.disabled = false;
                scrambleBtn.disabled = false;
                resetBtn.disabled = false;
            }
            currentState = initialState.slice();
            renderBoard(currentState, false);
            clearTree();
            message.textContent = '';
            updateStats();
        }

        buildBoard();
        setHeuristic('h2');

        heuristicBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                if (running) return;
                setHeuristic(btn.dataset.epHeuristic);
            });
        });
        presetBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                if (running) return;
                presetBtns.forEach(b => b.classList.toggle('active', b === btn));
                const p = PRESETS[btn.dataset.epPreset];
                loadInitial(p.state, p.goal);
            });
        });
        if (presetBtns.length > 0) presetBtns[0].classList.add('active');
        renderGoalBoard();

        solveBtn.addEventListener('click', runSolve);
        scrambleBtn.addEventListener('click', () => {
            if (running) return;
            presetBtns.forEach(b => b.classList.remove('active'));
            doScramble();
        });
        resetBtn.addEventListener('click', doReset);

        let resizeTimer;
        window.addEventListener('resize', () => {
            clearTimeout(resizeTimer);
            resizeTimer = setTimeout(() => {
                measureCell();
                renderBoard(currentState, false);
            }, 100);
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        setTimeout(init, 0);
    }
})();
