(function () {
    'use strict';

    const COLORS = {
        red: '#ef4444',
        blue: '#2dd4bf',
        boundary: '#1e293b',
        gridLine: '#e2e8f0',
        text: '#718096',
    };

    function randNormal(mean, std) {
        let u = 0, v = 0;
        while (u === 0) u = Math.random();
        while (v === 0) v = Math.random();
        return mean + std * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }

    function makeKernel(type, params) {
        const gamma = params.gamma;
        const degree = params.degree;
        const coef0 = params.coef0 !== undefined ? params.coef0 : 1;
        if (type === 'linear') {
            return (a, b) => a.x * b.x + a.y * b.y;
        }
        if (type === 'rbf') {
            return (a, b) => {
                const dx = a.x - b.x, dy = a.y - b.y;
                return Math.exp(-gamma * (dx * dx + dy * dy));
            };
        }
        if (type === 'poly') {
            return (a, b) => Math.pow(gamma * (a.x * b.x + a.y * b.y) + coef0, degree);
        }
        if (type === 'sigmoid') {
            return (a, b) => Math.tanh(gamma * (a.x * b.x + a.y * b.y) + coef0);
        }
        throw new Error('unknown kernel: ' + type);
    }

    function buildKernelMatrix(samples, K) {
        const n = samples.length;
        const M = new Float64Array(n * n);
        for (let i = 0; i < n; i++) {
            for (let j = i; j < n; j++) {
                const v = K(samples[i], samples[j]);
                M[i * n + j] = v;
                M[j * n + i] = v;
            }
        }
        return M;
    }

    function trainSMO(samples, K, C, tol, maxPasses) {
        const n = samples.length;
        if (n === 0) return { alpha: [], b: 0 };
        const Kmat = buildKernelMatrix(samples, K);
        const alpha = new Float64Array(n);
        let b = 0;
        const y = samples.map(s => s.label);

        function f(i) {
            let sum = 0;
            for (let j = 0; j < n; j++) {
                if (alpha[j] > 0) sum += alpha[j] * y[j] * Kmat[j * n + i];
            }
            return sum + b;
        }

        let passes = 0;
        let totalIter = 0;
        const maxTotalIter = 200;

        while (passes < maxPasses && totalIter < maxTotalIter) {
            let numChanged = 0;
            for (let i = 0; i < n; i++) {
                const Ei = f(i) - y[i];
                if ((y[i] * Ei < -tol && alpha[i] < C) || (y[i] * Ei > tol && alpha[i] > 0)) {
                    let j = i;
                    while (j === i) j = Math.floor(Math.random() * n);
                    const Ej = f(j) - y[j];

                    const alphaIold = alpha[i];
                    const alphaJold = alpha[j];

                    let L, H;
                    if (y[i] !== y[j]) {
                        L = Math.max(0, alpha[j] - alpha[i]);
                        H = Math.min(C, C + alpha[j] - alpha[i]);
                    } else {
                        L = Math.max(0, alpha[i] + alpha[j] - C);
                        H = Math.min(C, alpha[i] + alpha[j]);
                    }
                    if (L === H) continue;

                    const eta = 2 * Kmat[i * n + j] - Kmat[i * n + i] - Kmat[j * n + j];
                    if (eta >= 0) continue;

                    let aJ = alpha[j] - y[j] * (Ei - Ej) / eta;
                    if (aJ > H) aJ = H;
                    if (aJ < L) aJ = L;
                    if (Math.abs(aJ - alphaJold) < 1e-5) continue;
                    alpha[j] = aJ;

                    alpha[i] = alpha[i] + y[i] * y[j] * (alphaJold - alpha[j]);

                    const b1 = b - Ei
                        - y[i] * (alpha[i] - alphaIold) * Kmat[i * n + i]
                        - y[j] * (alpha[j] - alphaJold) * Kmat[i * n + j];
                    const b2 = b - Ej
                        - y[i] * (alpha[i] - alphaIold) * Kmat[i * n + j]
                        - y[j] * (alpha[j] - alphaJold) * Kmat[j * n + j];

                    if (alpha[i] > 0 && alpha[i] < C) b = b1;
                    else if (alpha[j] > 0 && alpha[j] < C) b = b2;
                    else b = (b1 + b2) / 2;

                    numChanged++;
                }
            }
            totalIter++;
            if (numChanged === 0) passes++;
            else passes = 0;
        }

        return { alpha: Array.from(alpha), b };
    }

    const SV_THRESHOLD = 1e-5;

    function decisionFn(model, samples, K) {
        return function (x) {
            let sum = 0;
            for (let i = 0; i < samples.length; i++) {
                if (model.alpha[i] > SV_THRESHOLD) {
                    sum += model.alpha[i] * samples[i].label * K(samples[i], x);
                }
            }
            return sum + model.b;
        };
    }

    const PRESETS = {
        separable: () => {
            const pts = [];
            for (let i = 0; i < 16; i++) {
                pts.push({ x: -0.55 + randNormal(0, 0.14), y: -0.55 + randNormal(0, 0.14), label: 1 });
                pts.push({ x: 0.55 + randNormal(0, 0.14), y: 0.55 + randNormal(0, 0.14), label: -1 });
            }
            return pts;
        },
        xor: () => {
            const pts = [];
            for (let i = 0; i < 10; i++) {
                pts.push({ x: -0.55 + randNormal(0, 0.13), y: -0.55 + randNormal(0, 0.13), label: 1 });
                pts.push({ x: 0.55 + randNormal(0, 0.13), y: 0.55 + randNormal(0, 0.13), label: 1 });
                pts.push({ x: -0.55 + randNormal(0, 0.13), y: 0.55 + randNormal(0, 0.13), label: -1 });
                pts.push({ x: 0.55 + randNormal(0, 0.13), y: -0.55 + randNormal(0, 0.13), label: -1 });
            }
            return pts;
        },
        circles: () => {
            const pts = [];
            for (let i = 0; i < 24; i++) {
                const t = (i / 24) * 2 * Math.PI;
                pts.push({ x: 0.7 * Math.cos(t) + randNormal(0, 0.04), y: 0.7 * Math.sin(t) + randNormal(0, 0.04), label: -1 });
            }
            for (let i = 0; i < 12; i++) {
                pts.push({ x: randNormal(0, 0.18), y: randNormal(0, 0.18), label: 1 });
            }
            return pts;
        },
    };

    function init() {
        const canvas = document.getElementById('svm-canvas');
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

        const kernelBtns = document.querySelectorAll('[data-svm-kernel]');
        const presetBtns = document.querySelectorAll('[data-svm-preset]');
        const classBtns = document.querySelectorAll('[data-svm-class]');
        const cInput = document.getElementById('svm-c');
        const cVal = document.getElementById('svm-c-val');
        const gammaInput = document.getElementById('svm-gamma');
        const gammaVal = document.getElementById('svm-gamma-val');
        const degreeInput = document.getElementById('svm-degree');
        const degreeVal = document.getElementById('svm-degree-val');
        const clearBtn = document.getElementById('svm-clear');
        const statsEl = document.getElementById('svm-stats');
        const gammaRow = document.getElementById('svm-gamma-row');
        const degreeRow = document.getElementById('svm-degree-row');

        let currentClass = 1;
        let kernel = 'rbf';
        let points = PRESETS.xor();
        let model = null;

        function currentParams() {
            return {
                gamma: Number(gammaInput.value),
                degree: Math.round(Number(degreeInput.value)),
                coef0: 1,
            };
        }

        function setKernel(k) {
            kernel = k;
            kernelBtns.forEach(b => b.classList.toggle('active', b.dataset.svmKernel === k));
            const showGamma = (k === 'rbf' || k === 'poly' || k === 'sigmoid');
            const showDegree = (k === 'poly');
            gammaRow.style.display = showGamma ? '' : 'none';
            degreeRow.style.display = showDegree ? '' : 'none';
        }

        function setClass(c) {
            currentClass = c;
            classBtns.forEach(b => b.classList.toggle('active', Number(b.dataset.svmClass) === c));
        }

        function loadPreset(name) {
            points = PRESETS[name]();
            presetBtns.forEach(b => b.classList.toggle('active', b.dataset.svmPreset === name));
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
            const params = currentParams();
            const K = makeKernel(kernel, params);
            const C = Number(cInput.value);
            model = trainSMO(points, K, C, 1e-3, 4);
            model._kernel = K;
            renderAll();
        }

        function renderAll() {
            renderBackground();
            renderPoints();
            renderStats();
        }

        function renderBackground() {
            ctx.clearRect(0, 0, cssWidth, cssHeight);

            if (model) {
                const decide = decisionFn(model, points, model._kernel);
                const STEP = 6;
                let maxAbs = 0.001;
                const cols = Math.ceil(cssWidth / STEP);
                const rows = Math.ceil(cssHeight / STEP);
                const grid = new Float64Array(cols * rows);
                for (let r = 0; r < rows; r++) {
                    for (let c = 0; c < cols; c++) {
                        const px = c * STEP;
                        const py = r * STEP;
                        const wx = (px / cssWidth) * 2.2 - 1.1;
                        const wy = 1.1 - (py / cssHeight) * 2.2;
                        const v = decide({ x: wx, y: wy });
                        grid[r * cols + c] = v;
                        if (Math.abs(v) > maxAbs) maxAbs = Math.abs(v);
                    }
                }
                for (let r = 0; r < rows; r++) {
                    for (let c = 0; c < cols; c++) {
                        const v = grid[r * cols + c];
                        const t = Math.max(-1, Math.min(1, v / maxAbs));
                        let r0, g0, b0;
                        if (t > 0) {
                            r0 = 239; g0 = 68; b0 = 68;
                        } else {
                            r0 = 45; g0 = 212; b0 = 191;
                        }
                        const alpha = Math.min(0.32, Math.abs(t) * 0.45);
                        ctx.fillStyle = 'rgba(' + r0 + ',' + g0 + ',' + b0 + ',' + alpha + ')';
                        ctx.fillRect(c * STEP, r * STEP, STEP + 1, STEP + 1);
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

        function renderPoints() {
            for (let i = 0; i < points.length; i++) {
                const p = points[i];
                const isSV = model && model.alpha[i] > SV_THRESHOLD;
                ctx.beginPath();
                ctx.arc(w2x(p.x), w2y(p.y), isSV ? 7 : 6, 0, Math.PI * 2);
                ctx.fillStyle = p.label === 1 ? COLORS.red : COLORS.blue;
                ctx.fill();
                ctx.strokeStyle = isSV ? '#1e293b' : '#ffffff';
                ctx.lineWidth = isSV ? 2.5 : 1.5;
                ctx.stroke();
            }
        }

        function renderStats() {
            if (!statsEl) return;
            if (!model) {
                statsEl.textContent = 'Add at least 4 points (both classes) to train.';
                return;
            }
            const svCount = model.alpha.filter(a => a > SV_THRESHOLD).length;
            let correct = 0;
            const decide = decisionFn(model, points, model._kernel);
            for (const p of points) {
                const pred = decide(p) >= 0 ? 1 : -1;
                if (pred === p.label) correct++;
            }
            const acc = points.length > 0 ? (correct / points.length * 100).toFixed(1) : '0.0';
            statsEl.innerHTML =
                'Kernel: <strong>' + kernel + '</strong>' +
                '   ·   Support vectors: <strong>' + svCount + '</strong>' +
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

        kernelBtns.forEach(b => {
            b.addEventListener('click', () => {
                setKernel(b.dataset.svmKernel);
                train();
            });
        });
        presetBtns.forEach(b => {
            b.addEventListener('click', () => loadPreset(b.dataset.svmPreset));
        });
        classBtns.forEach(b => {
            b.addEventListener('click', () => setClass(Number(b.dataset.svmClass)));
        });

        let trainTimeout;
        function debouncedTrain() {
            clearTimeout(trainTimeout);
            trainTimeout = setTimeout(train, 250);
        }

        cInput.addEventListener('input', () => {
            cVal.textContent = Number(cInput.value).toFixed(2);
            debouncedTrain();
        });
        gammaInput.addEventListener('input', () => {
            gammaVal.textContent = Number(gammaInput.value).toFixed(2);
            debouncedTrain();
        });
        degreeInput.addEventListener('input', () => {
            degreeVal.textContent = String(Math.round(Number(degreeInput.value)));
            debouncedTrain();
        });
        clearBtn.addEventListener('click', () => {
            points = [];
            presetBtns.forEach(b => b.classList.remove('active'));
            train();
        });

        cVal.textContent = Number(cInput.value).toFixed(2);
        gammaVal.textContent = Number(gammaInput.value).toFixed(2);
        degreeVal.textContent = String(Math.round(Number(degreeInput.value)));
        setKernel('rbf');
        setClass(1);
        presetBtns.forEach(b => b.classList.toggle('active', b.dataset.svmPreset === 'xor'));
        train();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        setTimeout(init, 0);
    }
})();
