(function () {
    'use strict';

    const COLORS = {
        red: '#ef4444',
        blue: '#2dd4bf',
        split: '#1e293b',
        gridLine: '#e2e8f0',
        highlight: '#f59e0b',
    };

    function randNormal(mean, std) {
        let u = 0, v = 0;
        while (u === 0) u = Math.random();
        while (v === 0) v = Math.random();
        return mean + std * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }

    function impurity(c1, c0, criterion) {
        const total = c1 + c0;
        if (total === 0) return 0;
        const p1 = c1 / total;
        const p0 = c0 / total;
        if (criterion === 'gini') return 1 - p1 * p1 - p0 * p0;
        if (p1 === 0 || p0 === 0) return 0;
        return -(p1 * Math.log2(p1) + p0 * Math.log2(p0));
    }

    function trainTree(points, maxDepth, criterion, minSamples) {
        function build(idxs, depth) {
            let c1 = 0, c0 = 0;
            for (const i of idxs) (points[i].label === 1 ? c1++ : c0++);
            const majority = c1 >= c0 ? 1 : -1;

            if (depth >= maxDepth || c1 === 0 || c0 === 0 || idxs.length < minSamples) {
                return { leaf: true, prediction: majority, count: idxs.length, c1, c0 };
            }

            let bestWeighted = Infinity;
            let bestSplit = null;

            for (let dim = 0; dim < 2; dim++) {
                const sorted = [...idxs].sort((a, b) => (dim === 0 ? points[a].x - points[b].x : points[a].y - points[b].y));
                let leftC1 = 0, leftC0 = 0;
                for (let k = 0; k < sorted.length - 1; k++) {
                    const idx = sorted[k];
                    if (points[idx].label === 1) leftC1++; else leftC0++;
                    const v1 = dim === 0 ? points[sorted[k]].x : points[sorted[k]].y;
                    const v2 = dim === 0 ? points[sorted[k + 1]].x : points[sorted[k + 1]].y;
                    if (v1 === v2) continue;
                    const threshold = (v1 + v2) / 2;
                    const rightC1 = c1 - leftC1;
                    const rightC0 = c0 - leftC0;
                    const leftN = leftC1 + leftC0;
                    const rightN = rightC1 + rightC0;
                    const weighted = (leftN * impurity(leftC1, leftC0, criterion) + rightN * impurity(rightC1, rightC0, criterion)) / idxs.length;
                    if (weighted < bestWeighted) {
                        bestWeighted = weighted;
                        bestSplit = { dim, threshold };
                    }
                }
            }

            if (!bestSplit) {
                return { leaf: true, prediction: majority, count: idxs.length, c1, c0 };
            }

            const left = [], right = [];
            for (const i of idxs) {
                const v = bestSplit.dim === 0 ? points[i].x : points[i].y;
                if (v <= bestSplit.threshold) left.push(i); else right.push(i);
            }

            return {
                leaf: false,
                dim: bestSplit.dim,
                threshold: bestSplit.threshold,
                left: build(left, depth + 1),
                right: build(right, depth + 1),
                count: idxs.length,
                c1, c0,
            };
        }

        if (points.length === 0) return null;
        return build(points.map((_, i) => i), 0);
    }

    function decorateTree(model) {
        let leafCounter = 0;
        let maxDepth = 0;
        function walk(node, depth, xMin, xMax, yMin, yMax) {
            node.rect = { xMin, xMax, yMin, yMax };
            node.depth = depth;
            if (depth > maxDepth) maxDepth = depth;
            if (node.leaf) {
                node.tx = leafCounter++;
                return;
            }
            if (node.dim === 0) {
                walk(node.left, depth + 1, xMin, node.threshold, yMin, yMax);
                walk(node.right, depth + 1, node.threshold, xMax, yMin, yMax);
            } else {
                walk(node.left, depth + 1, xMin, xMax, yMin, node.threshold);
                walk(node.right, depth + 1, xMin, xMax, node.threshold, yMax);
            }
            node.tx = (node.left.tx + node.right.tx) / 2;
        }
        walk(model, 0, -1.1, 1.1, -1.1, 1.1);
        model.leafCount = leafCounter;
        model.treeDepth = maxDepth;
    }

    function predict(node, x, y) {
        while (!node.leaf) {
            const v = node.dim === 0 ? x : y;
            node = v <= node.threshold ? node.left : node.right;
        }
        return node.prediction;
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
        stripes: () => {
            const pts = [];
            for (let i = 0; i < 18; i++) {
                pts.push({ x: randNormal(-0.7, 0.08), y: randNormal(0, 0.55), label: 1 });
                pts.push({ x: randNormal(-0.2, 0.08), y: randNormal(0, 0.55), label: -1 });
                pts.push({ x: randNormal(0.3, 0.08), y: randNormal(0, 0.55), label: 1 });
                pts.push({ x: randNormal(0.8, 0.08), y: randNormal(0, 0.55), label: -1 });
            }
            return pts;
        },
        circles: () => {
            const pts = [];
            for (let i = 0; i < 28; i++) {
                const t = (i / 28) * 2 * Math.PI;
                pts.push({ x: 0.75 * Math.cos(t) + randNormal(0, 0.04), y: 0.75 * Math.sin(t) + randNormal(0, 0.04), label: -1 });
            }
            for (let i = 0; i < 14; i++) {
                pts.push({ x: randNormal(0, 0.18), y: randNormal(0, 0.18), label: 1 });
            }
            return pts;
        },
    };

    function init() {
        const canvas = document.getElementById('dt-canvas');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const treeSvg = document.getElementById('dt-tree-svg');

        const dpr = window.devicePixelRatio || 1;
        const cssWidth = canvas.clientWidth;
        const cssHeight = Math.min(cssWidth * 0.7, 480);
        canvas.style.height = cssHeight + 'px';
        canvas.width = Math.round(cssWidth * dpr);
        canvas.height = Math.round(cssHeight * dpr);
        ctx.scale(dpr, dpr);

        const w2x = (x) => ((x + 1.1) / 2.2) * cssWidth;
        const w2y = (y) => ((1.1 - y) / 2.2) * cssHeight;

        const criterionBtns = document.querySelectorAll('[data-dt-criterion]');
        const presetBtns = document.querySelectorAll('[data-dt-preset]');
        const classBtns = document.querySelectorAll('[data-dt-class]');
        const depthInput = document.getElementById('dt-depth');
        const depthVal = document.getElementById('dt-depth-val');
        const clearBtn = document.getElementById('dt-clear');
        const playBtn = document.getElementById('dt-play');
        const statsEl = document.getElementById('dt-stats');

        let currentClass = 1;
        let criterion = 'gini';
        let points = PRESETS.xor();
        let model = null;
        let highlightedNode = null;
        let internalsBfs = [];
        let currentStep = -1;
        let animationTimer = null;
        const STEP_MS = 650;

        function isRevealed(node) {
            if (currentStep === -1) return true;
            return node.bfsIdx !== undefined && node.bfsIdx < currentStep;
        }

        function collectInternalsBfs(root) {
            const out = [];
            if (!root || root.leaf) return out;
            const queue = [root];
            while (queue.length > 0) {
                const n = queue.shift();
                n.bfsIdx = out.length;
                out.push(n);
                if (!n.left.leaf) queue.push(n.left);
                if (!n.right.leaf) queue.push(n.right);
            }
            return out;
        }

        function stopAnimation() {
            if (animationTimer) {
                clearInterval(animationTimer);
                animationTimer = null;
            }
            currentStep = -1;
            if (playBtn) playBtn.textContent = 'Play';
        }

        function startAnimation() {
            if (animationTimer) {
                stopAnimation();
                renderAll();
                return;
            }
            if (!model || internalsBfs.length === 0) return;
            currentStep = 0;
            playBtn.textContent = 'Stop';
            renderAll();
            animationTimer = setInterval(() => {
                currentStep++;
                if (currentStep > internalsBfs.length) {
                    clearInterval(animationTimer);
                    animationTimer = null;
                    currentStep = -1;
                    playBtn.textContent = 'Play';
                    renderAll();
                    return;
                }
                renderAll();
            }, STEP_MS);
        }

        function setCriterion(c) {
            criterion = c;
            criterionBtns.forEach(b => b.classList.toggle('active', b.dataset.dtCriterion === c));
        }

        function setClass(c) {
            currentClass = c;
            classBtns.forEach(b => b.classList.toggle('active', Number(b.dataset.dtClass) === c));
        }

        function loadPreset(name) {
            points = PRESETS[name]();
            presetBtns.forEach(b => b.classList.toggle('active', b.dataset.dtPreset === name));
            train();
        }

        function train() {
            highlightedNode = null;
            stopAnimation();
            if (points.length < 2) {
                model = null;
                internalsBfs = [];
                renderAll();
                return;
            }
            const labels = new Set(points.map(p => p.label));
            if (labels.size < 2) {
                model = null;
                internalsBfs = [];
                renderAll();
                return;
            }
            const maxDepth = Math.round(Number(depthInput.value));
            model = trainTree(points, maxDepth, criterion, 2);
            decorateTree(model);
            internalsBfs = collectInternalsBfs(model);
            renderAll();
        }

        function renderAll() {
            renderBackground();
            if (model) renderSplits();
            if (model && highlightedNode) renderHighlight();
            renderPoints();
            renderStats();
            renderTree();
        }

        function renderBackground() {
            ctx.clearRect(0, 0, cssWidth, cssHeight);

            if (model) {
                function fillLeaves(node) {
                    if (node.leaf || !isRevealed(node)) {
                        const r = node.rect;
                        const x0 = w2x(r.xMin);
                        const x1 = w2x(r.xMax);
                        const y0 = w2y(r.yMax);
                        const y1 = w2y(r.yMin);
                        const pred = node.leaf ? node.prediction : (node.c1 >= node.c0 ? 1 : -1);
                        ctx.fillStyle = pred === 1
                            ? 'rgba(239, 68, 68, 0.18)'
                            : 'rgba(45, 212, 191, 0.18)';
                        ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
                        return;
                    }
                    fillLeaves(node.left);
                    fillLeaves(node.right);
                }
                fillLeaves(model);
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

            ctx.fillStyle = '#94a3b8';
            ctx.font = '10px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            for (let i = -1; i <= 1.001; i += 0.5) {
                ctx.fillText(i.toFixed(1), w2x(i), w2y(-1) + 4);
            }
            ctx.textAlign = 'right';
            ctx.textBaseline = 'middle';
            for (let i = -0.5; i <= 1.001; i += 0.5) {
                ctx.fillText(i.toFixed(1), w2x(-1) - 4, w2y(i));
            }
        }

        function renderSplits() {
            ctx.strokeStyle = COLORS.split;
            ctx.lineWidth = 1.5;
            function walk(node) {
                if (node.leaf || !isRevealed(node)) return;
                const r = node.rect;
                ctx.beginPath();
                if (node.dim === 0) {
                    const sx = w2x(node.threshold);
                    ctx.moveTo(sx, w2y(r.yMax));
                    ctx.lineTo(sx, w2y(r.yMin));
                } else {
                    const sy = w2y(node.threshold);
                    ctx.moveTo(w2x(r.xMin), sy);
                    ctx.lineTo(w2x(r.xMax), sy);
                }
                ctx.stroke();
                walk(node.left);
                walk(node.right);
            }
            walk(model);
        }

        function renderHighlight() {
            const r = highlightedNode.rect;
            const x = w2x(r.xMin);
            const y = w2y(r.yMax);
            const w = w2x(r.xMax) - w2x(r.xMin);
            const h = w2y(r.yMin) - w2y(r.yMax);
            ctx.save();
            ctx.fillStyle = 'rgba(245, 158, 11, 0.16)';
            ctx.fillRect(x, y, w, h);
            ctx.strokeStyle = COLORS.highlight;
            ctx.lineWidth = 3;
            ctx.strokeRect(x + 1.5, y + 1.5, w - 3, h - 3);
            ctx.restore();
        }

        function renderPoints() {
            for (const p of points) {
                ctx.beginPath();
                ctx.arc(w2x(p.x), w2y(p.y), 6, 0, Math.PI * 2);
                ctx.fillStyle = p.label === 1 ? COLORS.red : COLORS.blue;
                ctx.fill();
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 1.5;
                ctx.stroke();
            }
        }

        function renderStats() {
            if (!statsEl) return;
            if (!model) {
                statsEl.textContent = 'Add points of both classes to train.';
                return;
            }
            let correct = 0;
            for (const p of points) {
                if (predict(model, p.x, p.y) === p.label) correct++;
            }
            const acc = points.length > 0 ? (correct / points.length * 100).toFixed(1) : '0.0';
            statsEl.innerHTML =
                'Criterion: <strong>' + criterion + '</strong>' +
                '   ·   Tree depth used: <strong>' + model.treeDepth + '</strong>' +
                '   ·   Leaves: <strong>' + model.leafCount + '</strong>' +
                '   ·   Training accuracy: <strong>' + acc + '%</strong>' +
                '   ·   Points: <strong>' + points.length + '</strong>';
        }

        function renderTree() {
            if (!treeSvg) return;
            if (!model) {
                treeSvg.innerHTML = '';
                treeSvg.setAttribute('width', '0');
                treeSvg.setAttribute('height', '0');
                return;
            }

            const NODE_W = 78;
            const NODE_H = 30;
            const COL_W = 92;
            const ROW_H = 60;
            const PAD = 14;

            let visLeafCount = 0;
            let visTreeDepth = 0;
            const visibleNodes = [];
            function layoutVisible(node, depth) {
                visibleNodes.push(node);
                if (depth > visTreeDepth) visTreeDepth = depth;
                const stops = node.leaf || !isRevealed(node);
                if (stops) {
                    node.vTx = visLeafCount++;
                    node.vDepth = depth;
                    node.renderAsLeaf = true;
                    return;
                }
                node.renderAsLeaf = false;
                layoutVisible(node.left, depth + 1);
                layoutVisible(node.right, depth + 1);
                node.vTx = (node.left.vTx + node.right.vTx) / 2;
                node.vDepth = depth;
            }
            layoutVisible(model, 0);

            const svgW = Math.max(cssWidth, visLeafCount * COL_W + 2 * PAD);
            const svgH = (visTreeDepth + 1) * ROW_H + 2 * PAD;
            treeSvg.setAttribute('width', svgW);
            treeSvg.setAttribute('height', svgH);
            treeSvg.setAttribute('viewBox', '0 0 ' + svgW + ' ' + svgH);

            const nodeX = n => PAD + (n.vTx + 0.5) * COL_W;
            const nodeY = n => PAD + n.vDepth * ROW_H;

            let html = '';
            for (const n of visibleNodes) {
                if (n.renderAsLeaf) continue;
                const px = nodeX(n);
                const py = nodeY(n) + NODE_H;
                for (const c of [n.left, n.right]) {
                    const cx = nodeX(c);
                    const cy = nodeY(c);
                    html += '<line x1="' + px + '" y1="' + py + '" x2="' + cx + '" y2="' + cy + '" stroke="#cbd5e1" stroke-width="1.5"/>';
                }
            }
            for (let i = 0; i < visibleNodes.length; i++) {
                const n = visibleNodes[i];
                const x = nodeX(n) - NODE_W / 2;
                const y = nodeY(n);
                let fill, stroke, label;
                if (n.renderAsLeaf) {
                    const pred = n.leaf ? n.prediction : (n.c1 >= n.c0 ? 1 : -1);
                    fill = pred === 1 ? 'rgba(239,68,68,0.22)' : 'rgba(45,212,191,0.22)';
                    stroke = pred === 1 ? '#ef4444' : '#0d9488';
                    label = n.c1 + '/' + n.c0;
                } else {
                    fill = '#ffffff';
                    stroke = '#1e293b';
                    const dimLabel = n.dim === 0 ? 'x' : 'y';
                    label = dimLabel + ' ≤ ' + n.threshold.toFixed(2);
                }
                html += '<g class="dt-tree-node" data-idx="' + i + '">';
                html += '<rect x="' + x + '" y="' + y + '" width="' + NODE_W + '" height="' + NODE_H + '" rx="6" ry="6" fill="' + fill + '" stroke="' + stroke + '" stroke-width="1.5"/>';
                html += '<text x="' + nodeX(n) + '" y="' + (y + 19) + '" text-anchor="middle" font-size="11" font-family="Inter, sans-serif" fill="#1e293b">' + label + '</text>';
                html += '</g>';
            }
            treeSvg.innerHTML = html;

            treeSvg.querySelectorAll('.dt-tree-node').forEach(el => {
                const idx = Number(el.dataset.idx);
                el.addEventListener('mouseenter', () => {
                    highlightedNode = visibleNodes[idx];
                    renderBackground();
                    renderSplits();
                    renderHighlight();
                    renderPoints();
                });
                el.addEventListener('mouseleave', () => {
                    highlightedNode = null;
                    renderBackground();
                    renderSplits();
                    renderPoints();
                });
            });
        }

        canvas.addEventListener('click', e => {
            const rect = canvas.getBoundingClientRect();
            const x = ((e.clientX - rect.left) / rect.width) * 2.2 - 1.1;
            const y = 1.1 - ((e.clientY - rect.top) / rect.height) * 2.2;
            if (Math.abs(x) > 1.1 || Math.abs(y) > 1.1) return;
            points.push({ x, y, label: currentClass });
            train();
        });

        criterionBtns.forEach(b => {
            b.addEventListener('click', () => {
                setCriterion(b.dataset.dtCriterion);
                train();
            });
        });
        presetBtns.forEach(b => {
            b.addEventListener('click', () => loadPreset(b.dataset.dtPreset));
        });
        classBtns.forEach(b => {
            b.addEventListener('click', () => setClass(Number(b.dataset.dtClass)));
        });
        depthInput.addEventListener('input', () => {
            depthVal.textContent = String(Math.round(Number(depthInput.value)));
            train();
        });
        clearBtn.addEventListener('click', () => {
            points = [];
            presetBtns.forEach(b => b.classList.remove('active'));
            train();
        });
        if (playBtn) {
            playBtn.addEventListener('click', () => {
                if (animationTimer) {
                    stopAnimation();
                    renderAll();
                } else {
                    startAnimation();
                }
            });
        }

        depthVal.textContent = String(Math.round(Number(depthInput.value)));
        setCriterion('gini');
        setClass(1);
        presetBtns.forEach(b => b.classList.toggle('active', b.dataset.dtPreset === 'xor'));
        train();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        setTimeout(init, 0);
    }
})();
