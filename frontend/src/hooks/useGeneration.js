import { useState, useRef, useCallback, useEffect } from 'react';

const WS_URL = 'ws://localhost:8188/ws/generate';
const STATUS_POLL_MS = 3000;

export default function useGeneration() {
  const [backendStatus, setBackendStatus] = useState('connecting');
  const [genState, setGenState] = useState('idle');
  const [lastElapsed, setLastElapsed] = useState(null);
  const [statusMessage, setStatusMessage] = useState('Connecting to backend...');
  const [outputImage, setOutputImage] = useState(null);
  const [activeModel, setActiveModel] = useState(null);
  const [modelList, setModelList] = useState([]);

  const wsRef = useRef(null);
  const reconnectTimer = useRef(null);
  const canvasRef = useRef(null);

  const dirtyRef = useRef(false);
  const backendReadyRef = useRef(false);

  const paramsRef = useRef({
    prompt: '',
    negativePrompt: '',
    steps: 12,
    cfgScale: 7.5,
    adapterScale: 0.9,
    resolution: 512,
    seedLocked: false,
    seed: 42,
    unionMode: null,
  });

  const updateParams = useCallback((partial) => {
    Object.assign(paramsRef.current, partial);
  }, []);

  // ── Send if dirty + ready ──────────────────────────────────
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

    const msg = {
      type: 'generate',
      prompt: p.prompt,
      negative_prompt: p.negativePrompt,
      image: b64,
      steps: p.steps,
      guidance: p.cfgScale,
      adapter_scale: p.adapterScale,
      seed: p.seedLocked ? p.seed : null,
      size: p.resolution,
    };
    if (p.unionMode) {
      msg.union_mode = p.unionMode;
    }
    ws.send(JSON.stringify(msg));
  }, []);

  const markDirty = useCallback(() => {
    dirtyRef.current = true;
    trySend();
  }, [trySend]);

  // ── WebSocket ──────────────────────────────────────────────
  const connectWs = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState <= WebSocket.OPEN) return;

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setBackendStatus('loading');
      setStatusMessage('Connected — checking status...');
      ws.send(JSON.stringify({ type: 'status' }));
      ws.send(JSON.stringify({ type: 'list_models' }));
    };

    ws.onmessage = (evt) => {
      const data = JSON.parse(evt.data);

      switch (data.type) {
        case 'status':
          setBackendStatus(data.status);
          setActiveModel(data.model || null);
          if (data.status === 'loading') setStatusMessage(data.message || 'Loading...');
          else if (data.status === 'error') setStatusMessage(data.message || 'Error');
          else if (data.status === 'ready') setStatusMessage('');
          else if (data.status === 'idle') setStatusMessage(data.message || 'No model loaded');
          break;

        case 'model_list':
          setModelList(data.models);
          setActiveModel(data.active);
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
          console.error('Backend error:', data.message);
          setGenState('idle');
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

  // ── Switch model ───────────────────────────────────────────
  const switchModel = useCallback((key) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    backendReadyRef.current = false;
    setBackendStatus('loading');
    setStatusMessage('Switching model...');
    setGenState('idle');
    ws.send(JSON.stringify({ type: 'switch_model', model: key }));
  }, []);

  // ── HQ render ──────────────────────────────────────────────
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

    const msg = {
      type: 'generate',
      prompt: p.prompt,
      negative_prompt: p.negativePrompt,
      image: b64,
      steps: 30,
      guidance: p.cfgScale,
      adapter_scale: p.adapterScale,
      seed: p.seedLocked ? p.seed : null,
      size: 1024,
    };
    if (p.unionMode) {
      msg.union_mode = p.unionMode;
    }
    ws.send(JSON.stringify(msg));
  }, []);

  const clearOutput = useCallback(() => setOutputImage(null), []);

  return {
    canvasRef,
    backendStatus,
    genState,
    lastElapsed,
    statusMessage,
    outputImage,
    activeModel,
    modelList,
    markDirty,
    updateParams,
    switchModel,
    requestHighQuality,
    clearOutput,
  };
}
