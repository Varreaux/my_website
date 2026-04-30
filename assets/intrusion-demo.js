(function () {
    'use strict';

    const COLORS = {
        majority: '#2dd4bf',
        minority: '#ef4444',
        boundary: '#7c5cff',
        gridLine: '#e2e8f0',
        axis: '#cbd5e1',
        text: '#718096',
    };

    function randNormal(mean, std) {
        let u = 0, v = 0;
        while (u === 0) u = Math.random();
        while (v === 0) v = Math.random();
        return mean + std * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }

    function shuffle(arr) {
        const a = arr.slice();
        for (let i = a.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [a[i], a[j]] = [a[j], a[i]];
        }
        return a;
    }

    function generateData(total, imbalance, separation) {
        const minorCount = Math.max(3, Math.round(total * imbalance));
        const majorCount = total - minorCount;
        const sigma = 0.18;
        const offset = separation / 2;
        const points = [];
        for (let i = 0; i < majorCount; i++) {
            points.push({
                x: offset + randNormal(0, sigma),
                y: offset + randNormal(0, sigma),
                label: 0,
            });
        }
        for (let i = 0; i < minorCount; i++) {
            points.push({
                x: -offset + randNormal(0, sigma),
                y: -offset + randNormal(0, sigma),
                label: 1,
            });
        }
        return points;
    }

    function undersample(points) {
        const minor = points.filter(p => p.label === 1);
        const major = shuffle(points.filter(p => p.label === 0));
        const balanced = minor.concat(major.slice(0, minor.length));
        return shuffle(balanced);
    }

    function trainLR(samples, epochs, lr) {
        if (samples.length === 0) return { w1: 0, w2: 0, b: 0 };
        let w1 = 0.01 * (Math.random() - 0.5);
        let w2 = 0.01 * (Math.random() - 0.5);
        let b = 0;
        const n = samples.length;
        for (let e = 0; e < epochs; e++) {
            let dW1 = 0, dW2 = 0, dB = 0;
            for (const p of samples) {
                const z = w1 * p.x + w2 * p.y + b;
                const pred = 1 / (1 + Math.exp(-z));
                const err = pred - p.label;
                dW1 += err * p.x;
                dW2 += err * p.y;
                dB += err;
            }
            w1 -= lr * dW1 / n;
            w2 -= lr * dW2 / n;
            b -= lr * dB / n;
        }
        return { w1, w2, b };
    }

    function predict(model, x, y) {
        return (model.w1 * x + model.w2 * y + model.b) >= 0 ? 1 : 0;
    }

    function metrics(model, points) {
        let TP = 0, FP = 0, TN = 0, FN = 0;
        for (const p of points) {
            const pred = predict(model, p.x, p.y);
            if (p.label === 1 && pred === 1) TP++;
            else if (p.label === 0 && pred === 1) FP++;
            else if (p.label === 0 && pred === 0) TN++;
            else if (p.label === 1 && pred === 0) FN++;
        }
        const total = TP + FP + TN + FN;
        const acc = total === 0 ? 0 : (TP + TN) / total;
        const prec = (TP + FP) === 0 ? 0 : TP / (TP + FP);
        const rec = (TP + FN) === 0 ? 0 : TP / (TP + FN);
        const f1 = (prec + rec) === 0 ? 0 : 2 * prec * rec / (prec + rec);
        return { TP, FP, TN, FN, acc, prec, rec, f1 };
    }

    function init() {
        const canvas = document.getElementById('id-canvas');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');

        const dpr = window.devicePixelRatio || 1;
        const cssWidth = canvas.clientWidth;
        const cssHeight = Math.min(cssWidth * 0.7, 420);
        canvas.style.height = cssHeight + 'px';
        canvas.width = Math.round(cssWidth * dpr);
        canvas.height = Math.round(cssHeight * dpr);
        ctx.scale(dpr, dpr);

        const w2x = (x) => ((x + 1.2) / 2.4) * cssWidth;
        const w2y = (y) => ((1.2 - y) / 2.4) * cssHeight;

        const imbalanceInput = document.getElementById('id-imbalance');
        const imbalanceVal = document.getElementById('id-imbalance-val');
        const separationInput = document.getElementById('id-separation');
        const separationVal = document.getElementById('id-separation-val');
        const undersampleInput = document.getElementById('id-undersample');
        const resampleBtn = document.getElementById('id-resample');

        const accVal = document.getElementById('id-acc');
        const precVal = document.getElementById('id-prec');
        const recVal = document.getElementById('id-rec');
        const f1Val = document.getElementById('id-f1');
        const tpEl = document.getElementById('id-tp');
        const fpEl = document.getElementById('id-fp');
        const fnEl = document.getElementById('id-fn');
        const tnEl = document.getElementById('id-tn');
        const trainCountEl = document.getElementById('id-train-count');

        const TOTAL = 400;
        const EPOCHS = 250;
        const LR = 0.25;

        let raw = [];
        let model = { w1: 0, w2: 0, b: 0 };

        function rebuild() {
            const imbalance = Number(imbalanceInput.value);
            const separation = Number(separationInput.value);
            raw = generateData(TOTAL, imbalance, separation);
            retrain();
        }

        function retrain() {
            const useUnder = undersampleInput.checked;
            const trainSet = useUnder ? undersample(raw) : raw;
            model = trainLR(trainSet, EPOCHS, LR);
            const m = metrics(model, raw);
            renderAll(m, trainSet.length);
        }

        function renderAll(m, trainCount) {
            renderCanvas();
            renderMetrics(m, trainCount);
        }

        function renderCanvas() {
            ctx.clearRect(0, 0, cssWidth, cssHeight);

            ctx.fillStyle = 'rgba(124, 92, 255, 0.05)';
            const eps = 1e-6;
            if (Math.abs(model.w2) > eps) {
                const xLeft = -1.5, xRight = 1.5;
                const yLeft = -(model.w1 * xLeft + model.b) / model.w2;
                const yRight = -(model.w1 * xRight + model.b) / model.w2;
                ctx.beginPath();
                if (model.w2 > 0) {
                    ctx.moveTo(w2x(xLeft), w2y(yLeft));
                    ctx.lineTo(w2x(xRight), w2y(yRight));
                    ctx.lineTo(w2x(xRight), 0);
                    ctx.lineTo(w2x(xLeft), 0);
                } else {
                    ctx.moveTo(w2x(xLeft), w2y(yLeft));
                    ctx.lineTo(w2x(xRight), w2y(yRight));
                    ctx.lineTo(w2x(xRight), cssHeight);
                    ctx.lineTo(w2x(xLeft), cssHeight);
                }
                ctx.closePath();
                ctx.fill();
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

            ctx.strokeStyle = COLORS.boundary;
            ctx.lineWidth = 2.5;
            ctx.beginPath();
            if (Math.abs(model.w2) > eps) {
                const xLeft = -1.5, xRight = 1.5;
                const yLeft = -(model.w1 * xLeft + model.b) / model.w2;
                const yRight = -(model.w1 * xRight + model.b) / model.w2;
                ctx.moveTo(w2x(xLeft), w2y(yLeft));
                ctx.lineTo(w2x(xRight), w2y(yRight));
            } else if (Math.abs(model.w1) > eps) {
                const x = -model.b / model.w1;
                ctx.moveTo(w2x(x), 0);
                ctx.lineTo(w2x(x), cssHeight);
            }
            ctx.stroke();

            for (const p of raw) {
                ctx.beginPath();
                ctx.arc(w2x(p.x), w2y(p.y), 5, 0, Math.PI * 2);
                ctx.fillStyle = p.label === 1 ? COLORS.minority : COLORS.majority;
                ctx.globalAlpha = 0.85;
                ctx.fill();
                ctx.globalAlpha = 1;
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 1.5;
                ctx.stroke();
            }
        }

        function fmt(v) {
            return (v * 100).toFixed(1) + '%';
        }

        function renderMetrics(m, trainCount) {
            accVal.textContent = fmt(m.acc);
            precVal.textContent = fmt(m.prec);
            recVal.textContent = fmt(m.rec);
            f1Val.textContent = fmt(m.f1);
            tpEl.textContent = m.TP;
            fpEl.textContent = m.FP;
            fnEl.textContent = m.FN;
            tnEl.textContent = m.TN;
            if (trainCountEl) trainCountEl.textContent = trainCount;

            const recColor = m.rec >= 0.8 ? '#0d9488' : (m.rec >= 0.5 ? '#d97706' : '#dc2626');
            const precColor = m.prec >= 0.8 ? '#0d9488' : (m.prec >= 0.5 ? '#d97706' : '#dc2626');
            recVal.style.color = recColor;
            precVal.style.color = precColor;
        }

        imbalanceInput.addEventListener('input', () => {
            imbalanceVal.textContent = (Number(imbalanceInput.value) * 100).toFixed(0) + '%';
            rebuild();
        });
        separationInput.addEventListener('input', () => {
            separationVal.textContent = Number(separationInput.value).toFixed(2);
            rebuild();
        });
        undersampleInput.addEventListener('change', retrain);
        resampleBtn.addEventListener('click', rebuild);

        imbalanceVal.textContent = (Number(imbalanceInput.value) * 100).toFixed(0) + '%';
        separationVal.textContent = Number(separationInput.value).toFixed(2);

        rebuild();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        setTimeout(init, 0);
    }
})();
