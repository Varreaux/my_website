(function () {
    'use strict';

    const COLORS = {
        truth: 'rgba(100, 116, 139, 0.55)',
        linear: '#7c5cff',
        forest: '#0d9488',
        point: 'rgba(30, 41, 59, 0.85)',
        gridLine: '#e2e8f0',
        axisLabel: '#94a3b8',
        axisTitle: '#475569',
    };

    function makeRng(seed) {
        let s = seed >>> 0;
        return () => {
            s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
            return s / 4294967296;
        };
    }

    function gaussian(rng) {
        let u = 0, v = 0;
        while (u === 0) u = rng();
        while (v === 0) v = rng();
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }

    const PRESETS = {
        hour: {
            label: 'Hour of day',
            xMin: 0, xMax: 24, yMin: 0, yMax: 2400,
            n: 110,
            noise: 180,
            truth: h => 200
                + 750 * Math.exp(-((h - 8) ** 2) / 3.5)
                + 1500 * Math.exp(-((h - 18) ** 2) / 6)
                + 320 * Math.exp(-((h - 13) ** 2) / 12),
            xTicks: [0, 6, 12, 18, 24],
            yTicks: [0, 600, 1200, 1800, 2400],
            xFormat: v => v + 'h',
        },
        temperature: {
            label: 'Temperature',
            xMin: -20, xMax: 35, yMin: 0, yMax: 2200,
            n: 100,
            noise: 200,
            truth: t => Math.max(0, 1900 - 4.5 * (t - 22) ** 2),
            xTicks: [-20, -10, 0, 10, 20, 30],
            yTicks: [0, 550, 1100, 1650, 2200],
            xFormat: v => v + '°C',
        },
        linear: {
            label: 'Linear feature',
            xMin: 0, xMax: 1, yMin: 0, yMax: 2400,
            n: 90,
            noise: 220,
            truth: x => 200 + 2000 * x,
            xTicks: [0, 0.25, 0.5, 0.75, 1],
            yTicks: [0, 600, 1200, 1800, 2400],
            xFormat: v => v.toFixed(2),
        },
        step: {
            label: 'Day of week (0=Mon)',
            xMin: -0.5, xMax: 6.5, yMin: 0, yMax: 2200,
            n: 100,
            noise: 160,
            truth: d => d < 4.5 ? 1700 : 700,
            xTicks: [0, 1, 2, 3, 4, 5, 6],
            yTicks: [0, 550, 1100, 1650, 2200],
            xFormat: v => String(v),
        },
    };

    function generateData(preset, seed) {
        const rng = makeRng(seed);
        const xs = [], ys = [];
        for (let i = 0; i < preset.n; i++) {
            const x = preset.xMin + rng() * (preset.xMax - preset.xMin);
            const y = preset.truth(x) + gaussian(rng) * preset.noise;
            xs.push(x);
            ys.push(Math.max(0, y));
        }
        return { xs, ys };
    }

    function fitLinear(xs, ys) {
        const n = xs.length;
        let sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (let i = 0; i < n; i++) {
            sx += xs[i]; sy += ys[i];
            sxx += xs[i] * xs[i];
            sxy += xs[i] * ys[i];
        }
        const denom = n * sxx - sx * sx;
        const slope = denom === 0 ? 0 : (n * sxy - sx * sy) / denom;
        const intercept = (sy - slope * sx) / n;
        return { slope, intercept };
    }

    function predictLinear(model, x) {
        return model.slope * x + model.intercept;
    }

    function buildTree(xs, ys, maxDepth, minLeaf) {
        function build(idxs, depth) {
            let sumY = 0, sumYY = 0;
            for (const i of idxs) { sumY += ys[i]; sumYY += ys[i] * ys[i]; }
            const meanY = sumY / idxs.length;
            if (depth >= maxDepth || idxs.length <= minLeaf * 2) {
                return { leaf: true, prediction: meanY };
            }
            const sorted = [...idxs].sort((a, b) => xs[a] - xs[b]);
            let bestImpurity = Infinity;
            let bestThr = null;
            let leftSum = 0, leftSq = 0;
            for (let k = 0; k < sorted.length - 1; k++) {
                const yv = ys[sorted[k]];
                leftSum += yv;
                leftSq += yv * yv;
                if (xs[sorted[k]] === xs[sorted[k + 1]]) continue;
                const leftN = k + 1;
                const rightN = idxs.length - leftN;
                if (leftN < minLeaf || rightN < minLeaf) continue;
                const rightSum = sumY - leftSum;
                const rightSq = sumYY - leftSq;
                const leftSSE = leftSq - (leftSum * leftSum) / leftN;
                const rightSSE = rightSq - (rightSum * rightSum) / rightN;
                const total = leftSSE + rightSSE;
                if (total < bestImpurity) {
                    bestImpurity = total;
                    bestThr = (xs[sorted[k]] + xs[sorted[k + 1]]) / 2;
                }
            }
            if (bestThr === null) {
                return { leaf: true, prediction: meanY };
            }
            const left = [], right = [];
            for (const i of idxs) {
                if (xs[i] <= bestThr) left.push(i); else right.push(i);
            }
            return {
                leaf: false,
                threshold: bestThr,
                left: build(left, depth + 1),
                right: build(right, depth + 1),
            };
        }
        return build(xs.map((_, i) => i), 0);
    }

    function predictTree(node, x) {
        while (!node.leaf) {
            node = x <= node.threshold ? node.left : node.right;
        }
        return node.prediction;
    }

    function buildForest(xs, ys, nTrees, maxDepth, seed) {
        const rng = makeRng(seed);
        const trees = [];
        const n = xs.length;
        for (let t = 0; t < nTrees; t++) {
            const bxs = new Array(n), bys = new Array(n);
            for (let i = 0; i < n; i++) {
                const j = Math.floor(rng() * n);
                bxs[i] = xs[j];
                bys[i] = ys[j];
            }
            trees.push(buildTree(bxs, bys, maxDepth, 2));
        }
        return trees;
    }

    function predictForest(trees, x) {
        let s = 0;
        for (const t of trees) s += predictTree(t, x);
        return s / trees.length;
    }

    function r2(ysTrue, ysPred) {
        const n = ysTrue.length;
        let mean = 0;
        for (let i = 0; i < n; i++) mean += ysTrue[i];
        mean /= n;
        let ssTot = 0, ssRes = 0;
        for (let i = 0; i < n; i++) {
            ssTot += (ysTrue[i] - mean) ** 2;
            ssRes += (ysTrue[i] - ysPred[i]) ** 2;
        }
        return ssTot === 0 ? 0 : 1 - ssRes / ssTot;
    }

    function init() {
        const canvas = document.getElementById('rf-canvas');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');

        const dpr = window.devicePixelRatio || 1;
        const cssWidth = canvas.clientWidth;
        const cssHeight = Math.min(cssWidth * 0.55, 460);
        canvas.style.height = cssHeight + 'px';
        canvas.width = Math.round(cssWidth * dpr);
        canvas.height = Math.round(cssHeight * dpr);
        ctx.scale(dpr, dpr);

        const presetBtns = document.querySelectorAll('[data-rf-preset]');
        const treesInput = document.getElementById('rf-trees');
        const treesVal = document.getElementById('rf-trees-val');
        const depthInput = document.getElementById('rf-depth');
        const depthVal = document.getElementById('rf-depth-val');
        const statsEl = document.getElementById('rf-stats');

        const PAD_L = 56;
        const PAD_R = 18;
        const PAD_T = 16;
        const PAD_B = 38;

        const SEED_DATA = 42;
        const SEED_FOREST = 7;

        let presetName = 'hour';
        let preset = PRESETS[presetName];
        let data = null;
        let linearModel = null;
        let forest = null;
        let stats = { linearR2: 0, forestR2: 0 };

        function w2x(x) {
            return PAD_L + ((x - preset.xMin) / (preset.xMax - preset.xMin)) * (cssWidth - PAD_L - PAD_R);
        }
        function w2y(y) {
            return PAD_T + (1 - (y - preset.yMin) / (preset.yMax - preset.yMin)) * (cssHeight - PAD_T - PAD_B);
        }

        function setPreset(name) {
            presetName = name;
            preset = PRESETS[name];
            presetBtns.forEach(b => b.classList.toggle('active', b.dataset.rfPreset === name));
            data = generateData(preset, SEED_DATA);
            train();
        }

        function train() {
            if (!data) return;
            linearModel = fitLinear(data.xs, data.ys);
            const nTrees = Number(treesInput.value);
            const maxDepth = Number(depthInput.value);
            forest = buildForest(data.xs, data.ys, nTrees, maxDepth, SEED_FOREST);
            const linPred = data.xs.map(x => predictLinear(linearModel, x));
            const forPred = data.xs.map(x => predictForest(forest, x));
            stats.linearR2 = r2(data.ys, linPred);
            stats.forestR2 = r2(data.ys, forPred);
            render();
        }

        function render() {
            ctx.clearRect(0, 0, cssWidth, cssHeight);

            const plotL = PAD_L;
            const plotR = cssWidth - PAD_R;
            const plotT = PAD_T;
            const plotB = cssHeight - PAD_B;

            ctx.strokeStyle = COLORS.gridLine;
            ctx.lineWidth = 1;
            ctx.beginPath();
            for (const xt of preset.xTicks) {
                ctx.moveTo(w2x(xt), plotT);
                ctx.lineTo(w2x(xt), plotB);
            }
            for (const yt of preset.yTicks) {
                ctx.moveTo(plotL, w2y(yt));
                ctx.lineTo(plotR, w2y(yt));
            }
            ctx.stroke();

            ctx.fillStyle = COLORS.axisLabel;
            ctx.font = '10px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            for (const xt of preset.xTicks) {
                ctx.fillText(preset.xFormat(xt), w2x(xt), plotB + 4);
            }
            ctx.textAlign = 'right';
            ctx.textBaseline = 'middle';
            for (const yt of preset.yTicks) {
                ctx.fillText(String(yt), plotL - 6, w2y(yt));
            }

            ctx.fillStyle = COLORS.axisTitle;
            ctx.font = '11px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'alphabetic';
            ctx.fillText(preset.label, (plotL + plotR) / 2, cssHeight - 6);
            ctx.save();
            ctx.translate(14, (plotT + plotB) / 2);
            ctx.rotate(-Math.PI / 2);
            ctx.fillText('Rented bike count', 0, 0);
            ctx.restore();

            const STEPS = 220;

            ctx.strokeStyle = COLORS.truth;
            ctx.lineWidth = 1.6;
            ctx.setLineDash([5, 4]);
            ctx.beginPath();
            for (let s = 0; s <= STEPS; s++) {
                const xv = preset.xMin + (s / STEPS) * (preset.xMax - preset.xMin);
                const yv = preset.truth(xv);
                if (s === 0) ctx.moveTo(w2x(xv), w2y(yv));
                else ctx.lineTo(w2x(xv), w2y(yv));
            }
            ctx.stroke();
            ctx.setLineDash([]);

            ctx.fillStyle = COLORS.point;
            for (let i = 0; i < data.xs.length; i++) {
                ctx.beginPath();
                ctx.arc(w2x(data.xs[i]), w2y(data.ys[i]), 3, 0, 2 * Math.PI);
                ctx.fill();
            }

            ctx.strokeStyle = COLORS.linear;
            ctx.lineWidth = 2.4;
            ctx.beginPath();
            ctx.moveTo(w2x(preset.xMin), w2y(predictLinear(linearModel, preset.xMin)));
            ctx.lineTo(w2x(preset.xMax), w2y(predictLinear(linearModel, preset.xMax)));
            ctx.stroke();

            ctx.strokeStyle = COLORS.forest;
            ctx.lineWidth = 2.4;
            ctx.beginPath();
            for (let s = 0; s <= STEPS; s++) {
                const xv = preset.xMin + (s / STEPS) * (preset.xMax - preset.xMin);
                const yv = predictForest(forest, xv);
                if (s === 0) ctx.moveTo(w2x(xv), w2y(yv));
                else ctx.lineTo(w2x(xv), w2y(yv));
            }
            ctx.stroke();

            if (statsEl) {
                statsEl.innerHTML =
                    'Trees: <strong>' + Number(treesInput.value) + '</strong>' +
                    '   ·   Max depth: <strong>' + Number(depthInput.value) + '</strong>' +
                    '   ·   Linear R²: <strong>' + stats.linearR2.toFixed(2) + '</strong>' +
                    '   ·   Forest R²: <strong>' + stats.forestR2.toFixed(2) + '</strong>' +
                    '   ·   Points: <strong>' + data.xs.length + '</strong>';
            }
        }

        presetBtns.forEach(b => {
            b.addEventListener('click', () => setPreset(b.dataset.rfPreset));
        });
        treesInput.addEventListener('input', () => {
            treesVal.textContent = String(Number(treesInput.value));
            train();
        });
        depthInput.addEventListener('input', () => {
            depthVal.textContent = String(Number(depthInput.value));
            train();
        });

        treesVal.textContent = String(Number(treesInput.value));
        depthVal.textContent = String(Number(depthInput.value));
        setPreset('hour');
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        setTimeout(init, 0);
    }
})();
