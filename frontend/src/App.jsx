import React, { useState, useCallback, useEffect } from 'react';
import useGeneration from './hooks/useGeneration.js';
import DrawingCanvas from './components/DrawingCanvas.jsx';

export default function App() {
  const [prompt, setPrompt] = useState('');
  const [brushSize, setBrushSize] = useState(4);
  const [tool, setTool] = useState('brush'); // brush | eraser

  // Generation settings
  const [cfgScale, setCfgScale] = useState(7.5);
  const [adapterScale, setAdapterScale] = useState(0.9);
  const [steps, setSteps] = useState(12);
  const [resolution, setResolution] = useState(512);
  const [seedLocked, setSeedLocked] = useState(false);
  const [seed, setSeed] = useState(42);

  const {
    canvasRef,
    backendStatus,
    genState,
    lastElapsed,
    statusMessage,
    outputImage,
    markDirty,
    updateParams,
    requestHighQuality,
    clearOutput,
  } = useGeneration();

  // Sync params to hook
  useEffect(() => {
    updateParams({ prompt, negativePrompt: '', steps, cfgScale, adapterScale, resolution, seedLocked, seed });
    markDirty();
  }, [prompt, steps, cfgScale, adapterScale, resolution, seedLocked, seed, updateParams, markDirty]);

  const handleClear = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    clearOutput();
  }, [canvasRef, clearOutput]);

  const isReady = backendStatus === 'ready';
  const isGenerating = genState === 'generating';

  const statusColor = {
    connecting: 'var(--status-connecting)',
    loading: 'var(--status-loading)',
    ready: 'var(--status-ready)',
    error: 'var(--status-error)',
  }[backendStatus] || 'var(--text-faint)';

  return (
    <div className="app">

      {/* ── Header ─────────────────────────────────────────── */}
      <header className="header">
        <div className="header__left">
          <span className="header__logo">Canvas</span>
          <span className="header__badge">SDXL T2I-Adapter</span>
        </div>
        <div className="header__right">
          <span className="header__live">
            <span className="header__dot" style={{ background: statusColor }} />
            {isReady ? 'Live' : backendStatus === 'loading' ? 'Loading' : backendStatus === 'error' ? 'Error' : 'Offline'}
          </span>
          {lastElapsed != null && !isGenerating && (
            <span className="header__fps">{lastElapsed}s</span>
          )}
          {isGenerating && (
            <span className="header__fps header__fps--active">running</span>
          )}
        </div>
      </header>

      {/* ── Workspace ──────────────────────────────────────── */}
      <div className="app__workspace">

        {/* Tool rail */}
        <nav className="rail">
          <button
            className={`rail__btn ${tool === 'brush' ? 'rail__btn--active' : ''}`}
            onClick={() => setTool('brush')}
            title="Brush"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="m9.06 11.9 8.07-8.06a2.85 2.85 0 1 1 4.03 4.03l-8.06 8.08" />
              <path d="M7.07 14.94c-1.66 0-3 1.35-3 3.02 0 1.33-2.5 1.52-2 2.02 1.08 1.1 2.49 2.02 4 2.02 2.2 0 4-1.8 4-4.04a3.01 3.01 0 0 0-3-3.02z" />
            </svg>
          </button>
          <button
            className={`rail__btn ${tool === 'eraser' ? 'rail__btn--active' : ''}`}
            onClick={() => setTool('eraser')}
            title="Eraser"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="m7 21-4.3-4.3c-1-1-1-2.5 0-3.4l9.6-9.6c1-1 2.5-1 3.4 0l5.6 5.6c1 1 1 2.5 0 3.4L13 21" />
              <path d="M22 21H7" />
              <path d="m5 11 9 9" />
            </svg>
          </button>

          <div className="rail__sep" />

          <button className="rail__btn" onClick={handleClear} title="Clear canvas">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M3 6h18" /><path d="M8 6V4h8v2" /><path d="m19 6-.7 14a2 2 0 0 1-2 1.8H7.7a2 2 0 0 1-2-1.8L5 6" />
            </svg>
          </button>

          <div className="rail__sep" />

          <div className="rail__slider-group">
            <span className="rail__label">Size</span>
            <input
              type="range"
              min="1"
              max="24"
              value={brushSize}
              onChange={(e) => setBrushSize(Number(e.target.value))}
            />
            <span className="rail__slider-value">{brushSize}</span>
          </div>
        </nav>

        {/* Dual panes */}
        <div className="panes">

          {/* ── Input pane ─────────────────────────────────── */}
          <div className="pane">
            <div className="pane__header">
              <span className="pane__title">Input</span>
            </div>
            <div className="pane__prompt">
              <textarea
                placeholder="Describe the target image..."
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                rows={1}
              />
            </div>
            <div className="pane__frame">
              <div className="pane__viewport pane__viewport--canvas">
                <DrawingCanvas
                  ref={canvasRef}
                  brushSize={brushSize}
                  tool={tool}
                  onChange={markDirty}
                />
              </div>
            </div>
            <div className="pane__footer">
              <span className="pane__footer-text">
                {isReady ? 'Draw to generate' : statusMessage}
              </span>
            </div>
          </div>

          {/* ── Output pane ────────────────────────────────── */}
          <div className="pane">
            <div className="pane__header">
              <span className="pane__title">Output</span>
              <div className="pane__meta">
                {isGenerating && <span className="gen-badge">Generating</span>}
                {!isGenerating && lastElapsed != null && (
                  <span className="elapsed-badge">{lastElapsed}s</span>
                )}
              </div>
            </div>
            <div className="pane__frame">
              <div className="pane__viewport pane__viewport--output">
                {!isReady ? (
                  <div className="output-placeholder">
                    <div className="output-spinner" />
                    <span className="output-placeholder__text">{statusMessage}</span>
                  </div>
                ) : outputImage ? (
                  <>
                    <img className="output-image" src={outputImage} alt="Generated" />
                    {isGenerating && (
                      <div className="output-overlay">
                        <div className="output-spinner" />
                      </div>
                    )}
                  </>
                ) : (
                  <div className="output-placeholder">
                    <span className="output-placeholder__text">
                      Sketch and prompt to see output here
                    </span>
                  </div>
                )}
              </div>
            </div>
            <div className="pane__footer">
              <span className="pane__footer-text">
                {isGenerating ? 'Processing...' : outputImage ? 'Ready' : ''}
              </span>
              <button
                className="pane__hq-btn"
                onClick={requestHighQuality}
                disabled={!isReady || isGenerating}
              >
                HQ Render
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* ── Bottom settings strip ──────────────────────────── */}
      <div className="settings-strip">
        <div className="settings-strip__item">
          <span className="settings-strip__label">Steps</span>
          <input
            type="range" className="settings-strip__slider"
            min="4" max="50" step="1" value={steps}
            onChange={(e) => setSteps(Number(e.target.value))}
          />
          <span className="settings-strip__value">{steps}</span>
        </div>

        <div className="settings-strip__sep" />

        <div className="settings-strip__item">
          <span className="settings-strip__label">CFG</span>
          <input
            type="range" className="settings-strip__slider"
            min="1" max="20" step="0.5" value={cfgScale}
            onChange={(e) => setCfgScale(Number(e.target.value))}
          />
          <span className="settings-strip__value">{cfgScale.toFixed(1)}</span>
        </div>

        <div className="settings-strip__sep" />

        <div className="settings-strip__item">
          <span className="settings-strip__label">Adherence</span>
          <input
            type="range" className="settings-strip__slider"
            min="0.1" max="1.5" step="0.05" value={adapterScale}
            onChange={(e) => setAdapterScale(Number(e.target.value))}
          />
          <span className="settings-strip__value">{adapterScale.toFixed(2)}</span>
        </div>

        <div className="settings-strip__sep" />

        <div className="settings-strip__item">
          <span className="settings-strip__label">Resolution</span>
          <input
            type="range" className="settings-strip__slider"
            min="256" max="1024" step="128" value={resolution}
            onChange={(e) => setResolution(Number(e.target.value))}
          />
          <span className="settings-strip__value">{resolution}px</span>
        </div>

        <div className="settings-strip__sep" />

        <div className="settings-strip__item">
          <span className="settings-strip__label">Seed</span>
          <button
            className={`settings-strip__seed-btn ${seedLocked ? 'settings-strip__seed-btn--active' : ''}`}
            onClick={() => setSeedLocked(!seedLocked)}
          >
            {seedLocked ? 'Fixed' : 'Random'}
          </button>
          {seedLocked && (
            <>
              <input
                type="number"
                className="settings-strip__seed-input"
                value={seed}
                min={0}
                max={999999999}
                onChange={(e) => setSeed(Number(e.target.value))}
              />
              <button
                className="settings-strip__shuffle-btn"
                onClick={() => setSeed(Math.floor(Math.random() * 999999999))}
              >
                RNG
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
