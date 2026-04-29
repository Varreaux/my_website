(function () {
    'use strict';

    const PUZZLES = {
        empty: {
            name: 'Pure Kropki',
            board: [
                [0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],
            ],
            rowC: [
                [0,2,0,2,0,0,1,0],
                [1,0,1,0,2,1,0,0],
                [1,2,0,0,0,1,0,0],
                [1,2,0,2,0,2,0,0],
                [0,2,0,0,1,0,0,1],
                [0,0,0,0,0,1,0,0],
                [0,0,0,1,0,0,0,1],
                [0,0,0,0,2,0,0,1],
                [1,0,1,0,0,1,0,0],
            ],
            colC: [
                [0,0,1,0,0,0,1,0,1],
                [0,0,1,1,0,1,1,0,0],
                [0,2,2,0,1,1,2,1,0],
                [1,2,0,0,1,0,0,0,0],
                [2,0,0,0,0,0,1,0,2],
                [0,0,0,1,0,0,1,0,1],
                [0,1,0,0,1,0,0,2,0],
                [0,0,1,0,0,1,1,0,1],
            ],
        },
    };

    function clonePuzzle(p) {
        return {
            name: p.name,
            board: p.board.map(row => row.slice()),
            given: p.board.map(row => row.map(v => v !== 0)),
            rowC: p.rowC.map(row => row.slice()),
            colC: p.colC.map(row => row.slice()),
        };
    }

    function isConsistent(board, rowC, colC, r, c, v) {
        for (let i = 0; i < 9; i++) {
            if (i !== c && board[r][i] === v) return false;
            if (i !== r && board[i][c] === v) return false;
        }
        const br = Math.floor(r / 3) * 3, bc = Math.floor(c / 3) * 3;
        for (let i = br; i < br + 3; i++) {
            for (let j = bc; j < bc + 3; j++) {
                if ((i !== r || j !== c) && board[i][j] === v) return false;
            }
        }
        function checkDot(dot, otherVal) {
            if (otherVal === 0) return true;
            if (dot === 1) return v + 1 === otherVal || v - 1 === otherVal;
            if (dot === 2) return v * 2 === otherVal || v === otherVal * 2;
            return true;
        }
        if (c < 8 && !checkDot(rowC[r][c], board[r][c + 1])) return false;
        if (c > 0 && !checkDot(rowC[r][c - 1], board[r][c - 1])) return false;
        if (r < 8 && !checkDot(colC[r][c], board[r + 1][c])) return false;
        if (r > 0 && !checkDot(colC[r - 1][c], board[r - 1][c])) return false;
        return true;
    }

    function getDomain(state, r, c) {
        const out = [];
        for (let v = 1; v <= 9; v++) {
            if (isConsistent(state.board, state.rowC, state.colC, r, c, v)) out.push(v);
        }
        return out;
    }

    function countUnassignedNeighbors(board, r, c) {
        let n = 0;
        for (let i = 0; i < 9; i++) if (i !== r && board[i][c] === 0) n++;
        for (let j = 0; j < 9; j++) if (j !== c && board[r][j] === 0) n++;
        const br = Math.floor(r / 3) * 3, bc = Math.floor(c / 3) * 3;
        for (let i = br; i < br + 3; i++) {
            for (let j = bc; j < bc + 3; j++) {
                if (board[i][j] === 0 && i !== r && j !== c) n++;
            }
        }
        return n;
    }

    function selectVar(state) {
        let minSize = 10;
        let candidates = [];
        for (let r = 0; r < 9; r++) {
            for (let c = 0; c < 9; c++) {
                if (state.board[r][c] !== 0) continue;
                const sz = getDomain(state, r, c).length;
                if (sz < minSize) { minSize = sz; candidates = [[r, c]]; }
                else if (sz === minSize) candidates.push([r, c]);
            }
        }
        if (candidates.length === 0) return null;
        if (minSize === 0) return { r: candidates[0][0], c: candidates[0][1], dead: true };
        if (candidates.length === 1) return { r: candidates[0][0], c: candidates[0][1] };
        let best = candidates[0], bestDeg = -1;
        for (const [r, c] of candidates) {
            const d = countUnassignedNeighbors(state.board, r, c);
            if (d > bestDeg) { bestDeg = d; best = [r, c]; }
        }
        return { r: best[0], c: best[1] };
    }

    function* solve(state) {
        const stats = { nodes: 0, backtracks: 0 };
        function* recurse() {
            const v = selectVar(state);
            if (!v) {
                yield { type: 'done', stats: { ...stats } };
                return true;
            }
            if (v.dead) {
                return false;
            }
            const { r, c } = v;
            const domain = getDomain(state, r, c);
            yield { type: 'select', r, c, domain: domain.slice(), stats: { ...stats } };
            for (const val of domain) {
                stats.nodes++;
                state.board[r][c] = val;
                yield { type: 'try', r, c, val, stats: { ...stats } };
                const ok = yield* recurse();
                if (ok === true) return true;
                stats.backtracks++;
                state.board[r][c] = 0;
                yield { type: 'undo', r, c, val, stats: { ...stats } };
            }
            return false;
        }
        const ok = yield* recurse();
        if (!ok) yield { type: 'fail', stats: { ...stats } };
    }

    function init() {
        const boardEl = document.getElementById('kropki-board');
        if (!boardEl) return;

        const statsEl = document.getElementById('kropki-stats');
        const messageEl = document.getElementById('kropki-message');
        const speedInput = document.getElementById('kropki-speed');
        const solveBtn = document.getElementById('kropki-solve');
        const resetBtn = document.getElementById('kropki-reset');
        const presetButtons = document.querySelectorAll('[data-kropki-preset]');

        let activePreset = 'empty';
        let state = clonePuzzle(PUZZLES[activePreset]);
        let runToken = 0;
        let running = false;

        const cellEls = [];

        function buildBoard() {
            boardEl.innerHTML = '';
            cellEls.length = 0;
            for (let r = 0; r < 9; r++) {
                cellEls.push([]);
                for (let c = 0; c < 9; c++) {
                    const cell = document.createElement('div');
                    cell.className = 'kk-cell';
                    if (c === 2 || c === 5) cell.classList.add('block-right');
                    if (r === 2 || r === 5) cell.classList.add('block-bottom');

                    const val = document.createElement('span');
                    val.className = 'kk-val';
                    cell.appendChild(val);

                    const cands = document.createElement('span');
                    cands.className = 'kk-cands';
                    cell.appendChild(cands);

                    if (c < 8) {
                        const dot = document.createElement('span');
                        dot.className = 'kk-dot kk-dot-right';
                        cell.appendChild(dot);
                    }
                    if (r < 8) {
                        const dot = document.createElement('span');
                        dot.className = 'kk-dot kk-dot-bottom';
                        cell.appendChild(dot);
                    }

                    boardEl.appendChild(cell);
                    cellEls[r].push(cell);
                }
            }
        }

        function renderState(highlight) {
            for (let r = 0; r < 9; r++) {
                for (let c = 0; c < 9; c++) {
                    const cell = cellEls[r][c];
                    const val = state.board[r][c];
                    const valSpan = cell.querySelector('.kk-val');
                    const candSpan = cell.querySelector('.kk-cands');

                    cell.classList.toggle('given', state.given[r][c]);
                    cell.classList.toggle('filled', val !== 0 && !state.given[r][c]);
                    cell.classList.remove('current', 'flash-bad', 'flash-good');

                    if (val === 0) {
                        valSpan.textContent = '';
                        candSpan.textContent = '';
                    } else {
                        valSpan.textContent = String(val);
                        candSpan.textContent = '';
                    }

                    const rDot = cell.querySelector('.kk-dot-right');
                    if (rDot) {
                        const t = c < 8 ? state.rowC[r][c] : 0;
                        rDot.classList.toggle('white', t === 1);
                        rDot.classList.toggle('black', t === 2);
                    }
                    const bDot = cell.querySelector('.kk-dot-bottom');
                    if (bDot) {
                        const t = r < 8 ? state.colC[r][c] : 0;
                        bDot.classList.toggle('white', t === 1);
                        bDot.classList.toggle('black', t === 2);
                    }
                }
            }
            if (highlight) {
                const { r, c, kind, domain } = highlight;
                const cell = cellEls[r][c];
                if (kind === 'select') {
                    cell.classList.add('current');
                    if (domain) {
                        const candSpan = cell.querySelector('.kk-cands');
                        candSpan.textContent = domain.join(' ');
                    }
                } else if (kind === 'try') {
                    cell.classList.add('current');
                } else if (kind === 'undo') {
                    cell.classList.add('flash-bad');
                }
            }
        }

        function setStats(s) {
            if (!statsEl) return;
            statsEl.innerHTML =
                'Nodes: <strong>' + s.nodes + '</strong>' +
                '   ·   Backtracks: <strong>' + s.backtracks + '</strong>';
        }

        function loadPreset(name) {
            if (running) return;
            activePreset = name;
            state = clonePuzzle(PUZZLES[name]);
            renderState(null);
            setStats({ nodes: 0, backtracks: 0 });
            messageEl.textContent = '';
            presetButtons.forEach(b => b.classList.toggle('active', b.dataset.kropkiPreset === name));
        }

        function speedDelay() {
            const v = Number(speedInput.value);
            return Math.max(0, 220 - v * 2);
        }

        function sleep(ms) {
            return new Promise(res => {
                if (ms <= 0) requestAnimationFrame(res);
                else setTimeout(res, ms);
            });
        }

        async function runSolve() {
            if (running) return;
            running = true;
            solveBtn.disabled = true;
            messageEl.textContent = 'Solving...';
            const token = ++runToken;

            state.board = state.board.map((row, r) => row.map((v, c) => state.given[r][c] ? v : 0));
            renderState(null);

            const gen = solve(state);
            let lastStats = { nodes: 0, backtracks: 0 };

            for (const step of gen) {
                if (token !== runToken) { running = false; solveBtn.disabled = false; return; }
                lastStats = step.stats || lastStats;
                if (step.type === 'select') {
                    renderState({ r: step.r, c: step.c, kind: 'select', domain: step.domain });
                    setStats(step.stats);
                    await sleep(speedDelay() * 0.4);
                } else if (step.type === 'try') {
                    renderState({ r: step.r, c: step.c, kind: 'try' });
                    setStats(step.stats);
                    await sleep(speedDelay());
                } else if (step.type === 'undo') {
                    renderState({ r: step.r, c: step.c, kind: 'undo' });
                    setStats(step.stats);
                    await sleep(speedDelay() * 0.7);
                } else if (step.type === 'done') {
                    renderState(null);
                    setStats(step.stats);
                    messageEl.textContent = 'Solved in ' + step.stats.nodes + ' assignments and ' + step.stats.backtracks + ' backtracks.';
                } else if (step.type === 'fail') {
                    renderState(null);
                    setStats(step.stats);
                    messageEl.textContent = 'No solution found for this puzzle.';
                }
            }

            running = false;
            solveBtn.disabled = false;
        }

        buildBoard();
        renderState(null);
        setStats({ nodes: 0, backtracks: 0 });

        solveBtn.addEventListener('click', runSolve);
        resetBtn.addEventListener('click', () => {
            if (running) { runToken++; running = false; solveBtn.disabled = false; }
            loadPreset(activePreset);
        });

        presetButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                if (running) return;
                loadPreset(btn.dataset.kropkiPreset);
            });
        });
        presetButtons.forEach(b => b.classList.toggle('active', b.dataset.kropkiPreset === activePreset));
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        setTimeout(init, 0);
    }
})();
