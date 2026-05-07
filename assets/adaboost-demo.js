(function () {
    'use strict';

    const COLORS = {
        red: '#ef4444',
        blue: '#2dd4bf',
        boundary: '#1e293b',
        gridLine: '#e2e8f0',
        stumpDim: 'rgba(30, 41, 59, 0.18)',
    };

    function randNormal(mean, std) {
        let u = 0, v = 0;
        while (u === 0) u = Math.random();
        while (v === 0) v = Math.random();
        return mean + std * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }

    function bestStump(points, weights) {
        const n = points.length;
        let best = null;
        let bestErr = Infinity;
        for (let dim = 0; dim < 2; dim++) {
            const order = points.map((_, i) => i).sort((a, b) => {
                const va = dim === 0 ? points[a].x : points[a].y;
                const vb = dim === 0 ? points[b].x : points[b].y;
                return va - vb;
            });
            for (let k = 0; k < n - 1; k++) {
                const v1 = dim === 0 ? points[order[k]].x : points[order[k]].y;
                const v2 = dim === 0 ? points[order[k + 1]].x : points[order[k + 1]].y;
                if (v1 === v2) continue;
                const threshold = (v1 + v2) / 2;
                let errPos = 0;
                let errNeg = 0;
                for (let i = 0; i < n; i++) {
                    const v = dim === 0 ? points[i].x : points[i].y;
                    const predPos = v <= threshold ? 1 : -1;
                    if (predPos !== points[i].label) errPos += weights[i];
                    else errNeg += weights[i];
                }
                if (errPos < bestErr) {
                    bestErr = errPos;
                    best = { dim, threshold, polarity: 1, error: errPos };
                }
                if (errNeg < bestErr) {
                    bestErr = errNeg;
                    best = { dim, threshold, polarity: -1, error: errNeg };
                }
            }
        }
        return best;
    }

    function stumpPredict(stump, x, y) {
        const v = stump.dim === 0 ? x : y;
        const base = v <= stump.threshold ? 1 : -1;
        return stump.polarity === 1 ? base : -base;
    }

    function trainAdaBoost(points, T) {
        const n = points.length;
        if (n === 0) return { stumps: [], finalWeights: [] };
        const labels = new Set(points.map(p => p.label));
        if (labels.size < 2) return { stumps: [], finalWeights: new Array(n).fill(1 / n) };

        const w = new Array(n).fill(1 / n);
        const stumps = [];

        for (let t = 0; t < T; t++) {
            const s = bestStump(points, w);
            if (!s) break;
            const eps = Math.max(1e-10, Math.min(1 - 1e-10, s.error));
            if (eps >= 0.5) break;
            const alpha = 0.5 * Math.log((1 - eps) / eps);
            s.alpha = alpha;
            stumps.push(s);

            let z = 0;
            for (let i = 0; i < n; i++) {
                const pred = stumpPredict(s, points[i].x, points[i].y);
                w[i] = w[i] * Math.exp(-alpha * points[i].label * pred);
                z += w[i];
            }
            if (z > 0) for (let i = 0; i < n; i++) w[i] /= z;
        }

        return { stumps, finalWeights: w };
    }

    function ensembleScore(stumps, x, y, upTo) {
        const limit = upTo === undefined ? stumps.length : Math.min(upTo, stumps.length);
        let s = 0;
        for (let t = 0; t < limit; t++) {
            s += stumps[t].alpha * stumpPredict(stumps[t], x, y);
        }
        return s;
    }

    const PRESETS = {
        separable: () => {
            const pts = [];
            for (let i = 0; i < 14; i++) {
                pts.push({ x: -0.55 + randNormal(0, 0.16), y: -0.55 + randNormal(0, 0.16), label: 1 });
                pts.push({ x: 0.55 + randNormal(0, 0.16), y: 0.55 + randNormal(0, 0.16), label: -1 });
            }
            return pts;
        },
        xor: () => {
            const pts = [];
            for (let i = 0; i < 9; i++) {
                pts.push({ x: -0.55 + randNormal(0, 0.13), y: -0.55 + randNormal(0, 0.13), label: 1 });
                pts.push({ x: 0.55 + randNormal(0, 0.13), y: 0.55 + randNormal(0, 0.13), label: 1 });
                pts.push({ x: -0.55 + randNormal(0, 0.13), y: 0.55 + randNormal(0, 0.13), label: -1 });
                pts.push({ x: 0.55 + randNormal(0, 0.13), y: -0.55 + randNormal(0, 0.13), label: -1 });
            }
            return pts;
        },
        circles: () => {
            const pts = [];
            for (let i = 0; i < 26; i++) {
                const t = (i / 26) * 2 * Math.PI;
                pts.push({ x: 0.75 * Math.cos(t) + randNormal(0, 0.04), y: 0.75 * Math.sin(t) + randNormal(0, 0.04), label: -1 });
            }
            for (let i = 0; i < 14; i++) {
                pts.push({ x: randNormal(0, 0.18), y: randNormal(0, 0.18), label: 1 });
            }
            return pts;
        },
        spiral: () => {
            const pts = [];
            for (let i = 0; i < 30; i++) {
                const t = i / 30 * Math.PI * 2.4;
                const r = 0.15 + 0.32 * t / (Math.PI * 2.4);
                pts.push({ x: r * Math.cos(t) + randNormal(0, 0.025), y: r * Math.sin(t) + randNormal(0, 0.025), label: 1 });
                pts.push({ x: r * Math.cos(t + Math.PI) + randNormal(0, 0.025), y: r * Math.sin(t + Math.PI) + randNormal(0, 0.025), label: -1 });
            }
            return pts;
        },
    };

    function init() {
        const canvas = document.getElementById('ada-canvas');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');

        const dpr = window.devicePixelRatio || 1;
        const cssWidth = canvas.clientWidth;
        const cssHeight = Math.min(cssWidth * 0.7, 480);
        canvas.style.height = cssHeight + 'px';
        canvas.width = Math.round(cssWidth * dpr);
        canvas.height = Math.round(cssHeight * dpr);
        ctx.scale(dpr, dpr);

        const w2x = (x) => ((x + 1.1) / 2.2) * cssWidth;
        const w2y = (y) => ((1.1 - y) / 2.2) * cssHeight;

        const presetBtns = document.querySelectorAll('[data-ada-preset]');
        const classBtns = document.querySelectorAll('[data-ada-class]');
        const roundsInput = document.getElementById('ada-rounds');
        const roundsVal = document.getElementById('ada-rounds-val');
        const clearBtn = document.getElementById('ada-clear');
        const statsEl = document.getElementById('ada-stats');

        let currentClass = 1;
        let points = PRESETS.xor();
        let model = null;
        const MAX_ROUNDS = 50;

        function setClass(c) {
            currentClass = c;
            classBtns.forEach(b => b.classList.toggle('active', Number(b.dataset.adaClass) === c));
        }

        function loadPreset(name) {
            points = PRESETS[name]();
            presetBtns.forEach(b => b.classList.toggle('active', b.dataset.adaPreset === name));
            train();
        }

        function train() {
            if (points.length < 4) {
                model = null;
                renderAll();
                return;
            }
            const labels = new Set(points.map(p => p.label));
            if (labels.size < 2) {
                model = null;
                renderAll();
                return;
            }
            model = trainAdaBoost(points, MAX_ROUNDS);
            renderAll();
        }

        function currentRounds() {
            const t = Math.round(Number(roundsInput.value));
            if (!model) return t;
            return Math.min(t, model.stumps.length);
        }

        function renderAll() {
            renderBackground();
            renderActiveStumps();
            renderPoints();
            renderStats();
        }

        function renderBackground() {
            ctx.clearRect(0, 0, cssWidth, cssHeight);

            if (model && model.stumps.length > 0) {
                const rounds = currentRounds();
                const xs = [-1.1, 1.1];
                const ys = [-1.1, 1.1];
                for (let t = 0; t < rounds; t++) {
                    const s = model.stumps[t];
                    (s.dim === 0 ? xs : ys).push(s.threshold);
                }
                xs.sort((a, b) => a - b);
                ys.sort((a, b) => a - b);
                const xUniq = [];
                const yUniq = [];
                for (const v of xs) if (xUniq.length === 0 || xUniq[xUniq.length - 1] !== v) xUniq.push(v);
                for (const v of ys) if (yUniq.length === 0 || yUniq[yUniq.length - 1] !== v) yUniq.push(v);

                const nCols = xUniq.length - 1;
                const nRows = yUniq.length - 1;
                const scores = new Float64Array(nCols * nRows);
                let maxAbs = 0.001;
                for (let r = 0; r < nRows; r++) {
                    for (let c = 0; c < nCols; c++) {
                        const cx = (xUniq[c] + xUniq[c + 1]) / 2;
                        const cy = (yUniq[r] + yUniq[r + 1]) / 2;
                        const v = ensembleScore(model.stumps, cx, cy, rounds);
                        scores[r * nCols + c] = v;
                        if (Math.abs(v) > maxAbs) maxAbs = Math.abs(v);
                    }
                }
                for (let r = 0; r < nRows; r++) {
                    for (let c = 0; c < nCols; c++) {
                        const v = scores[r * nCols + c];
                        const t = Math.max(-1, Math.min(1, v / maxAbs));
                        let r0, g0, b0;
                        if (t > 0) { r0 = 239; g0 = 68; b0 = 68; }
                        else { r0 = 45; g0 = 212; b0 = 191; }
                        const alpha = Math.min(0.32, Math.abs(t) * 0.45);
                        ctx.fillStyle = 'rgba(' + r0 + ',' + g0 + ',' + b0 + ',' + alpha + ')';
                        const x0 = w2x(xUniq[c]);
                        const x1 = w2x(xUniq[c + 1]);
                        const y0 = w2y(yUniq[r + 1]);
                        const y1 = w2y(yUniq[r]);
                        ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
                    }
                }
            }

            ctx.strokeStyle = COLORS.gridLine;
            ctx.lineWidth = 1;
            ctx.beginPath();
            for (let i = -1; i <= 1.001; i += 0.5) {
                ctx.moveTo(w2x(i), 0);
                ctx.lineTo(w2x(i), cssHeight);
                ctx.moveTo(0, w2y(i));
                ctx.lineTo(cssWidth, w2y(i));
            }
            ctx.stroke();
        }

        function renderActiveStumps() {
            if (!model || model.stumps.length === 0) return;
            const rounds = currentRounds();
            ctx.strokeStyle = COLORS.stumpDim;
            ctx.lineWidth = 1;
            ctx.beginPath();
            for (let t = 0; t < rounds; t++) {
                const s = model.stumps[t];
                if (s.dim === 0) {
                    const sx = w2x(s.threshold);
                    ctx.moveTo(sx, 0);
                    ctx.lineTo(sx, cssHeight);
                } else {
                    const sy = w2y(s.threshold);
                    ctx.moveTo(0, sy);
                    ctx.lineTo(cssWidth, sy);
                }
            }
            ctx.stroke();

            const last = model.stumps[rounds - 1];
            if (last) {
                ctx.strokeStyle = COLORS.boundary;
                ctx.lineWidth = 2;
                ctx.beginPath();
                if (last.dim === 0) {
                    const sx = w2x(last.threshold);
                    ctx.moveTo(sx, 0);
                    ctx.lineTo(sx, cssHeight);
                } else {
                    const sy = w2y(last.threshold);
                    ctx.moveTo(0, sy);
                    ctx.lineTo(cssWidth, sy);
                }
                ctx.stroke();
            }
        }

        function renderPoints() {
            const weights = model ? model.finalWeights : null;
            let maxW = 1;
            if (weights) for (const w of weights) if (w > maxW) maxW = w;
            for (let i = 0; i < points.length; i++) {
                const p = points[i];
                let radius = 6;
                if (weights) {
                    const wn = weights[i] * points.length;
                    radius = 4 + Math.min(7, Math.sqrt(wn) * 2.2);
                }
                ctx.beginPath();
                ctx.arc(w2x(p.x), w2y(p.y), radius, 0, Math.PI * 2);
                ctx.fillStyle = p.label === 1 ? COLORS.red : COLORS.blue;
                ctx.fill();
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 1.5;
                ctx.stroke();
            }
        }

        function renderStats() {
            if (!statsEl) return;
            if (!model || model.stumps.length === 0) {
                statsEl.textContent = 'Add points of both classes to train.';
                return;
            }
            const rounds = currentRounds();
            let correct = 0;
            for (const p of points) {
                const pred = ensembleScore(model.stumps, p.x, p.y, rounds) >= 0 ? 1 : -1;
                if (pred === p.label) correct++;
            }
            const acc = (correct / points.length * 100).toFixed(1);
            const lastErr = model.stumps[rounds - 1] ? model.stumps[rounds - 1].error.toFixed(3) : '-';
            statsEl.innerHTML =
                'Boost rounds in use: <strong>' + rounds + '</strong> / ' + model.stumps.length +
                '   ·   Last stump weighted error: <strong>' + lastErr + '</strong>' +
                '   ·   Training accuracy: <strong>' + acc + '%</strong>' +
                '   ·   Points: <strong>' + points.length + '</strong>';
        }

        canvas.addEventListener('click', e => {
            const rect = canvas.getBoundingClientRect();
            const x = ((e.clientX - rect.left) / rect.width) * 2.2 - 1.1;
            const y = 1.1 - ((e.clientY - rect.top) / rect.height) * 2.2;
            if (Math.abs(x) > 1.1 || Math.abs(y) > 1.1) return;
            points.push({ x, y, label: currentClass });
            train();
        });

        presetBtns.forEach(b => {
            b.addEventListener('click', () => loadPreset(b.dataset.adaPreset));
        });
        classBtns.forEach(b => {
            b.addEventListener('click', () => setClass(Number(b.dataset.adaClass)));
        });
        roundsInput.addEventListener('input', () => {
            roundsVal.textContent = String(Math.round(Number(roundsInput.value)));
            renderAll();
        });
        clearBtn.addEventListener('click', () => {
            points = [];
            presetBtns.forEach(b => b.classList.remove('active'));
            train();
        });

        roundsVal.textContent = String(Math.round(Number(roundsInput.value)));
        setClass(1);
        presetBtns.forEach(b => b.classList.toggle('active', b.dataset.adaPreset === 'xor'));
        train();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        setTimeout(init, 0);
    }
})();
