import React, { useRef, useEffect, useImperativeHandle, forwardRef, useCallback } from 'react';
import './DrawingCanvas.css';

const DrawingCanvas = forwardRef(({ brushSize = 4, tool = 'brush', onChange }, ref) => {
  const canvasEl = useRef(null);
  const isDrawing = useRef(false);
  const lastPos = useRef(null);

  useImperativeHandle(ref, () => canvasEl.current);

  useEffect(() => {
    const canvas = canvasEl.current;
    const container = canvas.parentElement;

    const resize = () => {
      const rect = container.getBoundingClientRect();
      canvas.width = rect.width;
      canvas.height = rect.height;

      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    };

    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  const getPos = useCallback((e) => {
    const canvas = canvasEl.current;
    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;

    return {
      x: (clientX - rect.left) * (canvas.width / rect.width),
      y: (clientY - rect.top) * (canvas.height / rect.height),
    };
  }, []);

  const strokeColor = tool === 'eraser' ? '#ffffff' : '#111111';

  const drawLine = useCallback((from, to) => {
    const ctx = canvasEl.current.getContext('2d');
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = tool === 'eraser' ? brushSize * 3 : brushSize;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x, to.y);
    ctx.stroke();
  }, [brushSize, tool, strokeColor]);

  const handleStart = useCallback((e) => {
    e.preventDefault();
    isDrawing.current = true;
    const pos = getPos(e);
    lastPos.current = pos;

    const ctx = canvasEl.current.getContext('2d');
    ctx.fillStyle = strokeColor;
    const r = tool === 'eraser' ? (brushSize * 3) / 2 : brushSize / 2;
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, r, 0, Math.PI * 2);
    ctx.fill();
  }, [getPos, brushSize, tool, strokeColor]);

  const handleMove = useCallback((e) => {
    if (!isDrawing.current) return;
    e.preventDefault();
    const pos = getPos(e);
    if (lastPos.current) drawLine(lastPos.current, pos);
    lastPos.current = pos;
  }, [getPos, drawLine]);

  const handleEnd = useCallback(() => {
    if (!isDrawing.current) return;
    isDrawing.current = false;
    lastPos.current = null;
    if (onChange) onChange();
  }, [onChange]);

  return (
    <canvas
      ref={canvasEl}
      className="drawing-canvas"
      onMouseDown={handleStart}
      onMouseMove={handleMove}
      onMouseUp={handleEnd}
      onMouseLeave={handleEnd}
      onTouchStart={handleStart}
      onTouchMove={handleMove}
      onTouchEnd={handleEnd}
    />
  );
});

DrawingCanvas.displayName = 'DrawingCanvas';

export default DrawingCanvas;
