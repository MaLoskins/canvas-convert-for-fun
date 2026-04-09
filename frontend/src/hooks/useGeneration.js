import { useState, useRef, useCallback, useEffect } from 'react';

const WS_URL = 'ws://localhost:8188/ws/generate';
const STATUS_POLL_MS = 3000;

/**
 * Core generation hook — manages the WebSocket connection, the
 * dirty/ready state machine, and exposes everything the UI needs.
 */
export default function useGeneration() {
  const [backendStatus, setBackendStatus] = useState('connecting');
  const [genState, setGenState] = useState('idle');
  const [lastElapsed, setLastElapsed] = useState(null);
  const [statusMessage, setStatusMessage] = useState('Connecting to backend...');
  const [outputImage, setOutputImage] = useState(null);

  const wsRef = useRef(null);
  const reconnectTimer = useRef(null);
  const canvasRef = useRef(null);

  // Dirty / ready flags for the request-reply loop
  const dirtyRef = useRef(false);
  const backendReadyRef = useRef(false);

  // Latest parameter refs (avoids re-creating callbacks on every keystroke)
  const paramsRef = useRef({
    prompt: '',
    negativePrompt: '',
    steps: 12,
    cfgScale: 7.5,
    adapterScale: 0.9,
    resolution: 512,
    seedLocked: false,
    seed: 42,
  });

  const updateParams = useCallback((partial) => {
    Object.assign(paramsRef.current, partial);
  }, []);

  // ── Send latest canvas + params if both dirty & ready ──────
  const trySend = useCallback(() => {
    if (!dirtyRef.current || !backendReadyRef.current) return;
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const canvas = canvasRef.current;
    if (!canvas) return;

    const b64 = canvas.toDataURL('image/png').split(',')[1];
    const p = paramsRef.current;

    dirtyRef.current = false;
    backendReadyRef.current = false;

    ws.send(JSON.stringify({
      type: 'generate',
      prompt: p.prompt,
      negative_prompt: p.negativePrompt,
      image: b64,
      steps: p.steps,
      guidance: p.cfgScale,
      adapter_scale: p.adapterScale,
      seed: p.seedLocked ? p.seed : null,
      size: p.resolution,
    }));
  }, []);

  const markDirty = useCallback(() => {
    dirtyRef.current = true;
    trySend();
  }, [trySend]);

  // ── WebSocket connection ───────────────────────────────────
  const connectWs = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState <= WebSocket.OPEN) return;

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setBackendStatus('loading');
      setStatusMessage('Connected — loading model...');
      ws.send(JSON.stringify({ type: 'status' }));
    };

    ws.onmessage = (evt) => {
      const data = JSON.parse(evt.data);

      switch (data.type) {
        case 'status':
          setBackendStatus(data.status);
          if (data.status === 'loading') setStatusMessage(data.message || 'Loading model...');
          else if (data.status === 'error') setStatusMessage(data.message || 'Backend error');
          else if (data.status === 'ready') setStatusMessage('');
          break;

        case 'ready_for_next':
          backendReadyRef.current = true;
          setGenState('idle');
          trySend();
          break;

        case 'generating':
          setGenState('generating');
          break;

        case 'result':
          setOutputImage('data:image/jpeg;base64,' + data.image);
          setLastElapsed(data.elapsed);
          break;

        case 'skipped':
        case 'pong':
          break;

        case 'error':
          console.error('Generation error:', data.message);
          break;
      }
    };

    ws.onclose = () => {
      setBackendStatus('connecting');
      setStatusMessage('Disconnected — reconnecting...');
      backendReadyRef.current = false;
      reconnectTimer.current = setTimeout(connectWs, 2000);
    };

    ws.onerror = () => ws.close();
  }, [trySend]);

  useEffect(() => {
    connectWs();

    const poll = setInterval(() => {
      const ws = wsRef.current;
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'status' }));
      }
    }, STATUS_POLL_MS);

    return () => {
      clearInterval(poll);
      clearTimeout(reconnectTimer.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, [connectWs]);

  // ── High-quality one-shot render ───────────────────────────
  const requestHighQuality = useCallback(() => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const canvas = canvasRef.current;
    if (!canvas) return;

    const b64 = canvas.toDataURL('image/png').split(',')[1];
    const p = paramsRef.current;

    backendReadyRef.current = false;
    dirtyRef.current = false;
    setGenState('generating');

    ws.send(JSON.stringify({
      type: 'generate',
      prompt: p.prompt,
      negative_prompt: p.negativePrompt,
      image: b64,
      steps: 30,
      guidance: p.cfgScale,
      adapter_scale: p.adapterScale,
      seed: p.seedLocked ? p.seed : null,
      size: 1024,
    }));
  }, []);

  // ── Clear output ───────────────────────────────────────────
  const clearOutput = useCallback(() => setOutputImage(null), []);

  return {
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
  };
}
