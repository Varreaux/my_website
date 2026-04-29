(function () {
    'use strict';

    const COLORS = {
        red: '#ef4444',
        blue: '#2dd4bf',
        perceptron: '#7c5cff',
        adaline: '#0891b2',
        gridLine: '#e2e8f0',
        axis: '#cbd5e1',
        text: '#718096',
    };

    function sign(x) { return x >= 0 ? 1 : -1; }

    function shuffleIndices(n) {
        const a = [];
        for (let i = 0; i < n; i++) a.push(i);
        for (let i = n - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [a[i], a[j]] = [a[j], a[i]];
        }
        return a;
    }

    function init() {
        const canvas = document.getElementById('pd-canvas');
        const errCanvas = document.getElementById('pd-error');
        if (!canvas || !errCanvas) return;

        const dpr = window.devicePixelRatio || 1;

        function setupCanvas(c, h) {
            const w = c.clientWidth;
            c.style.height = h + 'px';
            c.width = Math.round(w * dpr);
            c.height = Math.round(h * dpr);
            const ctx = c.getContext('2d');
            ctx.scale(dpr, dpr);
            return { ctx, w, h };
        }

        const main = setupCanvas(canvas, Math.min(canvas.clientWidth * 0.7, 460));
        const err = setupCanvas(errCanvas, 160);

        function w2x(x) { return ((x + 1) / 2) * main.w; }
        function w2y(y) { return ((1 - y) / 2) * main.h; }

        const state = {
            points: [],
            currentClass: 1,
            perceptron: null,
            adaline: null,
            lr: 0.05,
            running: false,
            runToken: 0,
        };

        function newModel() {
            return {
                w1: 0.02 * (Math.random() - 0.5),
                w2: 0.02 * (Math.random() - 0.5),
                b: 0,
                history: [],
                epoch: 0,
            };
        }

        function resetModels() {
            state.perceptron = newModel();
            state.adaline = newModel();
            renderAll();
        }

        function clearPoints() {
            state.points = [];
            resetModels();
        }

        function predict(m, x, y) {
            return sign(m.w1 * x + m.w2 * y + m.b);
        }

        function trainOneEpochPerceptron() {
            const m = state.perceptron;
            let errors = 0;
            const order = shuffleIndices(state.points.length);
            for (const i of order) {
                const p = state.points[i];
                const pred = predict(m, p.x, p.y);
                if (pred !== p.label) {
                    m.w1 += state.lr * p.label * p.x;
                    m.w2 += state.lr * p.label * p.y;
                    m.b += state.lr * p.label;
                    errors++;
                }
            }
            m.history.push(errors);
            m.epoch++;
        }

        function trainOneEpochAdaline() {
            const m = state.adaline;
            let sse = 0;
            const order = shuffleIndices(state.points.length);
            for (const i of order) {
                const p = state.points[i];
                const net = m.w1 * p.x + m.w2 * p.y + m.b;
                const errVal = p.label - net;
                m.w1 += state.lr * errVal * p.x;
                m.w2 += state.lr * errVal * p.y;
                m.b += state.lr * errVal;
                sse += errVal * errVal;
            }
            m.history.push(sse / Math.max(1, state.points.length));
            m.epoch++;
        }

        function trainOneEpoch() {
            if (state.points.length < 2) return;
            trainOneEpochPerceptron();
            trainOneEpochAdaline();
        }

        async function trainMany(n) {
            if (state.running) return;
            if (state.points.length < 2) return;
            state.running = true;
            const token = ++state.runToken;
            for (let i = 0; i < n; i++) {
                if (token !== state.runToken) break;
                trainOneEpoch();
                if (i % 2 === 0) {
                    renderAll();
                    await new Promise(res => setTimeout(res, 25));
                }
            }
            renderAll();
            state.running = false;
        }

        function drawBoundary(ctx, m, color, dashed) {
            const eps = 1e-6;
            ctx.strokeStyle = color;
            ctx.lineWidth = 2.5;
            ctx.setLineDash(dashed ? [6, 4] : []);
            ctx.beginPath();
            if (Math.abs(m.w2) > eps) {
                const xmin = -1.5, xmax = 1.5;
                const ymin = -(m.w1 * xmin + m.b) / m.w2;
                const ymax = -(m.w1 * xmax + m.b) / m.w2;
                ctx.moveTo(w2x(xmin), w2y(ymin));
                ctx.lineTo(w2x(xmax), w2y(ymax));
            } else if (Math.abs(m.w1) > eps) {
                const x = -m.b / m.w1;
                ctx.moveTo(w2x(x), 0);
                ctx.lineTo(w2x(x), main.h);
            } else {
                ctx.setLineDash([]);
                return;
            }
            ctx.stroke();
            ctx.setLineDash([]);
        }

        function renderMain() {
            const ctx = main.ctx;
            ctx.clearRect(0, 0, main.w, main.h);

            ctx.strokeStyle = COLORS.gridLine;
            ctx.lineWidth = 1;
            ctx.beginPath();
            for (let i = -1; i <= 1.0001; i += 0.25) {
                const px = w2x(i);
                ctx.moveTo(px, 0); ctx.lineTo(px, main.h);
                const py = w2y(i);
                ctx.moveTo(0, py); ctx.lineTo(main.w, py);
            }
            ctx.stroke();

            ctx.strokeStyle = COLORS.axis;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(0, w2y(0)); ctx.lineTo(main.w, w2y(0));
            ctx.moveTo(w2x(0), 0); ctx.lineTo(w2x(0), main.h);
            ctx.stroke();

            drawBoundary(ctx, state.perceptron, COLORS.perceptron, false);
            drawBoundary(ctx, state.adaline, COLORS.adaline, true);

            for (const p of state.points) {
                ctx.beginPath();
                ctx.arc(w2x(p.x), w2y(p.y), 6, 0, Math.PI * 2);
                ctx.fillStyle = p.label === 1 ? COLORS.red : COLORS.blue;
                ctx.fill();
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 2;
                ctx.stroke();
            }
        }

        function renderError() {
            const ctx = err.ctx;
            ctx.clearRect(0, 0, err.w, err.h);
            const pHist = state.perceptron.history;
            const aHist = state.adaline.history;
            const n = Math.max(pHist.length, aHist.length);
            const padL = 36, padR = 12, padT = 12, padB = 24;
            const cw = err.w - padL - padR;
            const ch = err.h - padT - padB;

            ctx.strokeStyle = COLORS.gridLine;
            ctx.lineWidth = 1;
            ctx.strokeRect(padL, padT, cw, ch);

            ctx.font = '11px Inter, system-ui, sans-serif';
            ctx.fillStyle = COLORS.text;
            ctx.textAlign = 'center';
            ctx.fillText('Epoch', padL + cw / 2, err.h - 6);

            ctx.save();
            ctx.translate(12, padT + ch / 2);
            ctx.rotate(-Math.PI / 2);
            ctx.fillText('Error (normalized)', 0, 0);
            ctx.restore();

            if (n < 2) {
                ctx.fillStyle = COLORS.text;
                ctx.textAlign = 'center';
                ctx.fillText('Train to plot error curves', padL + cw / 2, padT + ch / 2);
                return;
            }

            const pMax = Math.max(1, ...pHist);
            const aMax = Math.max(0.01, ...aHist);

            function plot(hist, max, color, dashed) {
                if (hist.length === 0) return;
                ctx.strokeStyle = color;
                ctx.lineWidth = 2;
                ctx.setLineDash(dashed ? [5, 3] : []);
                ctx.beginPath();
                for (let i = 0; i < hist.length; i++) {
                    const x = padL + (cw * i) / Math.max(1, n - 1);
                    const y = padT + ch - (ch * hist[i]) / max;
                    if (i === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                }
                ctx.stroke();
                ctx.setLineDash([]);
            }
            plot(pHist, pMax, COLORS.perceptron, false);
            plot(aHist, aMax, COLORS.adaline, true);
        }

        function computeAccuracy(m) {
            if (state.points.length === 0) return 0;
            let c = 0;
            for (const p of state.points) {
                if (predict(m, p.x, p.y) === p.label) c++;
            }
            return c / state.points.length;
        }

        function renderStats() {
            const stats = document.getElementById('pd-stats');
            if (!stats) return;
            const pAcc = computeAccuracy(state.perceptron);
            const aAcc = computeAccuracy(state.adaline);
            stats.innerHTML =
                'Epoch: <strong>' + state.perceptron.epoch + '</strong>' +
                '   ·   Perceptron acc: <strong>' + (pAcc * 100).toFixed(0) + '%</strong>' +
                '   ·   Adaline acc: <strong>' + (aAcc * 100).toFixed(0) + '%</strong>' +
                '   ·   Points: <strong>' + state.points.length + '</strong>';
        }

        function renderAll() {
            renderMain();
            renderError();
            renderStats();
        }

        canvas.addEventListener('click', e => {
            if (state.running) return;
            const rect = canvas.getBoundingClientRect();
            const x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
            const y = 1 - ((e.clientY - rect.top) / rect.height) * 2;
            if (Math.abs(x) > 1 || Math.abs(y) > 1) return;
            state.points.push({ x, y, label: state.currentClass });
            renderAll();
        });

        const classBtns = document.querySelectorAll('[data-pd-class]');
        classBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                state.currentClass = Number(btn.dataset.pdClass);
                classBtns.forEach(b => b.classList.toggle('active', b === btn));
            });
        });

        document.getElementById('pd-train1').addEventListener('click', () => {
            if (state.running) return;
            trainOneEpoch();
            renderAll();
        });
        document.getElementById('pd-train50').addEventListener('click', () => trainMany(50));
        document.getElementById('pd-reset-weights').addEventListener('click', () => {
            if (state.running) return;
            resetModels();
        });
        document.getElementById('pd-clear').addEventListener('click', () => {
            if (state.running) return;
            clearPoints();
        });

        const lrInput = document.getElementById('pd-lr');
        const lrVal = document.getElementById('pd-lr-val');
        lrInput.addEventListener('input', () => {
            state.lr = Number(lrInput.value);
            lrVal.textContent = state.lr.toFixed(3);
        });
        state.lr = Number(lrInput.value);
        lrVal.textContent = state.lr.toFixed(3);

        const presets = {
            easy: () => {
                const pts = [];
                for (let i = 0; i < 10; i++) {
                    pts.push({ x: -0.7 + Math.random() * 0.4, y: -0.7 + Math.random() * 0.4, label: 1 });
                    pts.push({ x: 0.3 + Math.random() * 0.4, y: 0.3 + Math.random() * 0.4, label: -1 });
                }
                return pts;
            },
            hard: () => {
                const pts = [];
                for (let i = 0; i < 12; i++) {
                    pts.push({ x: -0.5 + Math.random() * 0.6, y: -0.5 + Math.random() * 0.6, label: 1 });
                    pts.push({ x: -0.2 + Math.random() * 0.6, y: -0.2 + Math.random() * 0.6, label: -1 });
                }
                return pts;
            },
            xor: () => {
                const pts = [];
                for (let i = 0; i < 6; i++) {
                    pts.push({ x: -0.75 + Math.random() * 0.3, y: -0.75 + Math.random() * 0.3, label: 1 });
                    pts.push({ x: 0.45 + Math.random() * 0.3, y: 0.45 + Math.random() * 0.3, label: 1 });
                    pts.push({ x: -0.75 + Math.random() * 0.3, y: 0.45 + Math.random() * 0.3, label: -1 });
                    pts.push({ x: 0.45 + Math.random() * 0.3, y: -0.75 + Math.random() * 0.3, label: -1 });
                }
                return pts;
            },
        };

        document.querySelectorAll('[data-pd-preset]').forEach(btn => {
            btn.addEventListener('click', () => {
                if (state.running) return;
                state.points = presets[btn.dataset.pdPreset]();
                resetModels();
            });
        });

        state.points = presets.easy();
        resetModels();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        setTimeout(init, 0);
    }
})();
