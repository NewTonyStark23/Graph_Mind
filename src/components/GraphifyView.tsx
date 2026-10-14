import { useEffect, useRef, useState } from 'react';
import { BrainMapData, BrainNode, BrainConnection } from '../types';

interface Props {
  brainData: BrainMapData;
  modelName: string;
}

export default function GraphifyView({ brainData, modelName }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animFrameRef = useRef<number>(0);
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const [isAnimating, setIsAnimating] = useState(true);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;

    let frame = 0;
    const animate = () => {
      frame++;
      ctx.clearRect(0, 0, width, height);

      const bgGrad = ctx.createRadialGradient(width / 2, height / 2, 0, width / 2, height / 2, width * 0.7);
      bgGrad.addColorStop(0, 'rgba(30, 41, 59, 1)');
      bgGrad.addColorStop(0.5, 'rgba(15, 23, 42, 1)');
      bgGrad.addColorStop(1, 'rgba(2, 6, 23, 1)');
      ctx.fillStyle = bgGrad;
      ctx.fillRect(0, 0, width, height);

      ctx.strokeStyle = 'rgba(148, 163, 184, 0.05)';
      ctx.lineWidth = 0.5;
      for (let x = 0; x < width; x += 40) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke();
      }
      for (let y = 0; y < height; y += 40) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
      }

      const scaleX = width / 1000;
      const scaleY = height / 1000;

      brainData.connections.forEach((conn: BrainConnection) => {
        const fromNode = brainData.nodes.find((n: BrainNode) => n.id === conn.from);
        const toNode = brainData.nodes.find((n: BrainNode) => n.id === conn.to);
        if (!fromNode || !toNode) return;

        const x1 = fromNode.x * scaleX;
        const y1 = fromNode.y * scaleY;
        const x2 = toNode.x * scaleX;
        const y2 = toNode.y * scaleY;
        const pulse = Math.sin(frame * 0.03 + conn.strength * 5) * 0.3 + 0.7;

        let color: string;
        switch (conn.type) {
          case 'attention': color = `rgba(99, 102, 241, ${conn.strength * pulse * 0.6})`; break;
          case 'residual': color = `rgba(34, 197, 94, ${conn.strength * pulse * 0.5})`; break;
          case 'expert-gate': color = `rgba(234, 179, 8, ${conn.strength * pulse * 0.4})`; break;
          default: color = `rgba(59, 130, 246, ${conn.strength * pulse * 0.5})`;
        }

        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = conn.strength * 2 * pulse;
        const midX = (x1 + x2) / 2 + (Math.sin(frame * 0.01 + conn.strength * 3) * 10);
        const midY = (y1 + y2) / 2;
        ctx.moveTo(x1, y1);
        ctx.quadraticCurveTo(midX, midY, x2, y2);
        ctx.stroke();

        if (conn.strength > 0.5) {
          const t = ((frame * 0.02 * conn.strength) % 1);
          const px = x1 + (x2 - x1) * t;
          const py = y1 + (y2 - y1) * t;
          ctx.beginPath();
          ctx.arc(px, py, 2, 0, Math.PI * 2);
          ctx.fillStyle = color.replace(/[\d.]+\)$/, '0.9)');
          ctx.fill();
        }
      });

      brainData.nodes.forEach((node: BrainNode) => {
        const x = node.x * scaleX;
        const y = node.y * scaleY;
        const baseSize = node.size * Math.min(scaleX, scaleY) * 0.8;
        const pulse = Math.sin(frame * 0.04 + node.weight * 10) * 0.2 + 0.8;
        const size = baseSize * pulse;
        const isHovered = hoveredNode === node.id;

        const glowSize = size * (isHovered ? 3 : 2);
        let glowColor: string;
        switch (node.type) {
          case 'attention': glowColor = 'rgba(99, 102, 241, 0.25)'; break;
          case 'ffn': glowColor = 'rgba(59, 130, 246, 0.25)'; break;
          case 'moe-expert': glowColor = 'rgba(234, 179, 8, 0.25)'; break;
          case 'router': glowColor = 'rgba(168, 85, 247, 0.25)'; break;
          case 'embedding': glowColor = 'rgba(34, 197, 94, 0.25)'; break;
          case 'output': glowColor = 'rgba(239, 68, 68, 0.25)'; break;
          case 'mamba': glowColor = 'rgba(139, 92, 246, 0.25)'; break;
          default: glowColor = 'rgba(148, 163, 184, 0.15)';
        }

        const glow = ctx.createRadialGradient(x, y, 0, x, y, glowSize);
        glow.addColorStop(0, glowColor);
        glow.addColorStop(1, 'transparent');
        ctx.fillStyle = glow;
        ctx.fillRect(x - glowSize, y - glowSize, glowSize * 2, glowSize * 2);

        ctx.beginPath();
        ctx.arc(x, y, size, 0, Math.PI * 2);

        let nodeColor: string;
        switch (node.type) {
          case 'attention': nodeColor = `rgba(99, 102, 241, ${node.activity * pulse})`; break;
          case 'ffn': nodeColor = `rgba(59, 130, 246, ${node.activity * pulse})`; break;
          case 'moe-expert': nodeColor = `rgba(234, 179, 8, ${node.activity * pulse})`; break;
          case 'router': nodeColor = `rgba(168, 85, 247, ${node.activity * pulse})`; break;
          case 'embedding': nodeColor = `rgba(34, 197, 94, ${node.activity * pulse})`; break;
          case 'output': nodeColor = `rgba(239, 68, 68, ${node.activity * pulse})`; break;
          case 'mamba': nodeColor = `rgba(139, 92, 246, ${node.activity * pulse})`; break;
          default: nodeColor = `rgba(148, 163, 184, ${node.activity * pulse * 0.6})`;
        }

        ctx.fillStyle = nodeColor;
        ctx.fill();
        ctx.strokeStyle = isHovered ? 'rgba(255,255,255,0.8)' : 'rgba(255,255,255,0.2)';
        ctx.lineWidth = isHovered ? 2 : 1;
        ctx.stroke();

        if (size > 8 || isHovered) {
          ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
          ctx.font = `${Math.max(8, size * 0.5)}px ui-monospace, monospace`;
          ctx.textAlign = 'center';
          ctx.fillText(node.label, x, y + size + 12);
        }
      });

      ctx.fillStyle = 'rgba(226, 232, 240, 0.95)';
      ctx.font = 'bold 13px ui-monospace, monospace';
      ctx.textAlign = 'left';
      ctx.fillText(`GRAPHIFY · ${modelName}`, 15, 25);
      ctx.fillStyle = 'rgba(148, 163, 184, 0.6)';
      ctx.font = '10px ui-monospace, monospace';
      ctx.fillText(`Nodes: ${brainData.nodes.length} · Connections: ${brainData.connections.length} · Tensors: ${brainData.tensorMap.length}`, 15, 42);

      if (isAnimating) {
        animFrameRef.current = requestAnimationFrame(animate);
      }
    };

    animate();
    return () => { cancelAnimationFrame(animFrameRef.current); };
  }, [brainData, modelName, hoveredNode, isAnimating]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const scaleX = rect.width / 1000;
    const scaleY = rect.height / 1000;

    let found: string | null = null;
    for (const node of brainData.nodes) {
      const x = node.x * scaleX;
      const y = node.y * scaleY;
      const dist = Math.sqrt((mx - x) ** 2 + (my - y) ** 2);
      if (dist < node.size * Math.min(scaleX, scaleY)) {
        found = node.id;
        break;
      }
    }
    setHoveredNode(found);
  };

  const hoveredNodeData = hoveredNode ? brainData.nodes.find((n: BrainNode) => n.id === hoveredNode) : null;

  return (
    <div className="relative w-full h-full">
      <canvas
        ref={canvasRef}
        className="w-full h-full rounded-lg cursor-crosshair"
        style={{ minHeight: '400px' }}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setHoveredNode(null)}
      />
      <div className="absolute bottom-3 right-3 flex gap-2">
        <button
          onClick={() => setIsAnimating(!isAnimating)}
          className="px-3 py-1 bg-slate-800/90 border border-slate-600 rounded text-xs text-slate-300 hover:bg-slate-700 transition-colors backdrop-blur-sm"
        >
          {isAnimating ? '⏸ Pause' : '▶ Play'}
        </button>
      </div>
      {hoveredNodeData && (
        <div className="absolute top-3 right-3 bg-slate-900/95 border border-slate-600 rounded-lg p-3 text-xs max-w-[260px] backdrop-blur-sm shadow-xl">
          <div className="space-y-1.5">
            <div className="font-bold text-blue-400 text-sm">{hoveredNodeData.label}</div>
            <div className="text-slate-400">Type: <span className="text-slate-200">{hoveredNodeData.type}</span></div>
            <div className="text-slate-400">Weight: <span className="text-slate-200">{(hoveredNodeData.weight * 100).toFixed(1)}%</span></div>
            <div className="text-slate-400">Activity: <span className="text-green-400">{(hoveredNodeData.activity * 100).toFixed(1)}%</span></div>
            {hoveredNodeData.tensorName && (
              <div className="border-t border-slate-700 pt-1.5 mt-1.5">
                <div className="text-indigo-400 font-semibold">Tensor (extracted)</div>
                <div className="text-slate-400 mt-1">Name: <span className="text-slate-200 font-mono text-[10px] break-all">{hoveredNodeData.tensorName}</span></div>
                {hoveredNodeData.shape && <div className="text-slate-400">Shape: <span className="text-blue-300 font-mono">{hoveredNodeData.shape}</span></div>}
                {hoveredNodeData.dtype && <div className="text-slate-400">DType: <span className="text-amber-300 font-mono">{hoveredNodeData.dtype}</span></div>}
              </div>
            )}
          </div>
        </div>
      )}
      <div className="absolute bottom-3 left-3 bg-slate-900/90 border border-slate-700 rounded-lg p-2 text-[10px] backdrop-blur-sm">
        <div className="flex flex-wrap gap-x-3 gap-y-1">
          <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-indigo-500"></span>Attention</span>
          <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-blue-500"></span>FFN</span>
          <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-yellow-500"></span>MoE Expert</span>
          <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-purple-500"></span>Router</span>
          <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-green-500"></span>Embedding</span>
          <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-red-500"></span>Output</span>
        </div>
      </div>
    </div>
  );
}
