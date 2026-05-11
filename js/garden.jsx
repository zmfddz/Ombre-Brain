// 花园视图 —— 区域聚合 + 密度切换

const { useState: useStateG, useMemo: useMemoG } = React;

function GardenView({ buckets, onOpen, onOpenCluster }) {
  const [hovered, setHovered] = useStateG(null);
  const [tooltipPos, setTooltipPos] = useStateG({ x: 0, y: 0 });
  const [density, setDensity] = useStateG("smart"); // smart / all / top

  // 显示策略
  const display = useMemoG(() => {
    if (density === "top") {
      // 取 weight top 12
      const sorted = [...buckets].sort((a, b) => (b.weight || 0) - (a.weight || 0)).slice(0, 12);
      return { mode: "scatter", items: sorted, clusters: [] };
    }
    if (density === "all") {
      return { mode: "scatter", items: buckets, clusters: [] };
    }
    // smart: 5x5 聚合
    const grid = {}; // key=ix,iy
    for (const b of buckets) {
      const v = Math.max(0, Math.min(1, b.valence ?? 0.5));
      const a = Math.max(0, Math.min(1, b.arousal ?? 0.5));
      const ix = Math.min(4, Math.floor(v * 5));
      const iy = Math.min(4, Math.floor(a * 5));
      const key = `${ix},${iy}`;
      if (!grid[key]) grid[key] = { ix, iy, items: [] };
      grid[key].items.push(b);
    }
    const clusters = Object.values(grid).map((g) => {
      g.items.sort((a, b) => (b.weight || 0) - (a.weight || 0));
      g.lead = g.items[0];
      // 平均位置（避免代表植物总落在格子中心）
      g.avgV = g.items.reduce((s, x) => s + (x.valence ?? 0.5), 0) / g.items.length;
      g.avgA = g.items.reduce((s, x) => s + (x.arousal ?? 0.5), 0) / g.items.length;
      return g;
    });
    return { mode: "cluster", items: [], clusters };
  }, [buckets, density]);

  // scatter mode 防重叠
  const placedScatter = useMemoG(() => {
    if (display.mode !== "scatter") return [];
    const out = [];
    for (const b of display.items) {
      const v = Math.max(0, Math.min(1, b.valence ?? 0.5));
      const a = Math.max(0, Math.min(1, b.arousal ?? 0.5));
      let x = 6 + v * 88;
      let y = 90 - a * 80;
      let tries = 0;
      while (tries < 8 && out.some((r) => Math.hypot(r.x - x, r.y - y) < 6)) {
        x += (Math.random() - 0.5) * 5;
        y += (Math.random() - 0.5) * 4;
        tries++;
      }
      const hearts = impToHearts(b.importance);
      const size = 32 + hearts * 7;
      out.push({ b, x, y, size });
    }
    return out;
  }, [display]);

  return (
    <div className="garden-pad">
      {/* 控制栏 */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "14px", flexWrap: "wrap", gap: "12px" }}>
        <div style={{ fontFamily: "var(--kai)", fontSize: "0.88rem", color: "var(--ink-soft)" }}>
          一共 <b style={{ fontFamily: "var(--serif)", color: "var(--moss-deep)" }}>{buckets.length}</b> 株植物长在花园里
          {density === "smart" && display.clusters.length < buckets.length && (
            <span style={{ color: "var(--ink-faint)" }}> · 按情感区域聚合显示</span>
          )}
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <span style={{ fontFamily: "var(--mono)", fontSize: "0.65rem", color: "var(--ink-faint)", letterSpacing: "0.1em", textTransform: "uppercase" }}>density</span>
          {[
            { id: "smart", label: "聚合" },
            { id: "all",   label: "全部" },
            { id: "top",   label: "Top 12" },
          ].map((d) => (
            <Tag key={d.id} active={density === d.id} onClick={() => setDensity(d.id)} tone="moss" size="sm">{d.label}</Tag>
          ))}
        </div>
      </div>

      {/* 花园主区 */}
      <div className="garden-stage" style={{
        position: "relative", width: "100%", height: "640px",
        background: "linear-gradient(180deg, oklch(0.97 0.018 90) 0%, oklch(0.945 0.025 95) 100%)",
        border: "1px solid var(--line)", borderRadius: "4px", overflow: "hidden",
      }}>
        {/* 网格 */}
        <svg style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }} preserveAspectRatio="none" viewBox="0 0 100 100">
          <defs>
            <pattern id="grid-pat" width="20" height="20" patternUnits="userSpaceOnUse">
              <path d="M 20 0 L 0 0 0 20" fill="none" stroke="oklch(0.85 0.015 90)" strokeWidth="0.15" />
            </pattern>
          </defs>
          <rect width="100" height="100" fill="url(#grid-pat)" />
          <line x1="50" y1="0" x2="50" y2="100" stroke="oklch(0.78 0.02 90)" strokeWidth="0.25" strokeDasharray="0.8 1.2" />
          <line x1="0" y1="50" x2="100" y2="50" stroke="oklch(0.78 0.02 90)" strokeWidth="0.25" strokeDasharray="0.8 1.2" />
        </svg>

        <AxisLabel pos={{ top: 14, left: 24 }}     title="焦灼"  sub="低 · 激" align="left" />
        <AxisLabel pos={{ top: 14, right: 24 }}    title="雀跃"  sub="高 · 激" align="right" />
        <AxisLabel pos={{ bottom: 14, left: 24 }}  title="低落"  sub="低 · 静" align="left" />
        <AxisLabel pos={{ bottom: 14, right: 24 }} title="安宁"  sub="高 · 静" align="right" />
        <div style={axisCenterStyle("top")}>· arousal ↑ ·</div>
        <div style={axisCenterStyle("bottom")}>· arousal ↓ ·</div>
        <div style={axisSideStyle("left")}>valence ↓</div>
        <div style={axisSideStyle("right")}>valence ↑</div>

        {/* scatter mode */}
        {display.mode === "scatter" && placedScatter.map(({ b, x, y, size }, i) => (
          <PlantButton
            key={b.id}
            bucket={b}
            x={x} y={y} size={size}
            delay={i * 30}
            hovered={hovered === b.id}
            setHovered={setHovered}
            setTooltipPos={setTooltipPos}
            onClick={() => onOpen(b.id)}
          />
        ))}

        {/* cluster mode */}
        {display.mode === "cluster" && display.clusters.map((c, i) => {
          const x = 6 + c.avgV * 88;
          const y = 90 - c.avgA * 80;
          const lead = c.lead;
          const hearts = impToHearts(lead.importance);
          const size = 36 + hearts * 6;
          return (
            <ClusterButton
              key={`${c.ix}-${c.iy}`}
              cluster={c} x={x} y={y} size={size}
              delay={i * 30}
              hovered={hovered === `cl-${c.ix}-${c.iy}`}
              setHovered={setHovered}
              setTooltipPos={setTooltipPos}
              onClick={() => {
                if (c.items.length === 1) onOpen(lead.id);
                else onOpenCluster(c);
              }}
            />
          );
        })}

        {/* tooltip */}
        {hovered && hovered.startsWith("cl-") && (() => {
          const c = display.clusters.find((x) => `cl-${x.ix}-${x.iy}` === hovered);
          if (!c) return null;
          return (
            <div style={tooltipStyle(tooltipPos)}>
              <div style={{ fontFamily: "var(--mono)", fontSize: "0.62rem", color: "var(--ink-faint)", letterSpacing: "0.1em", marginBottom: "6px", textTransform: "uppercase" }}>
                这片角落 · {c.items.length} 株
              </div>
              <div style={{ fontFamily: "var(--serif)", fontWeight: 500, fontSize: "1.05rem", color: "var(--moss-deep)", marginBottom: "6px" }}>
                {c.lead.name}
              </div>
              {c.items.length > 1 && (
                <div style={{ fontFamily: "var(--kai)", fontSize: "0.78rem", color: "var(--ink-soft)", lineHeight: 1.6 }}>
                  和另外 {c.items.length - 1} 株：
                  <span style={{ color: "var(--ink-faint)" }}> {c.items.slice(1, 4).map(x => x.name).join(" · ")}{c.items.length > 4 ? "…" : ""}</span>
                </div>
              )}
              <div style={tooltipFootStyle}>点击查看这一区域</div>
            </div>
          );
        })()}
        {hovered && !hovered.startsWith("cl-") && (() => {
          const item = placedScatter.find((p) => p.b.id === hovered);
          if (!item) return null;
          const b = item.b;
          const m = moodOf(b.valence);
          return (
            <div style={tooltipStyle(tooltipPos)}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "6px", gap: "10px" }}>
                <div style={{ fontFamily: "var(--mono)", fontSize: "0.62rem", color: "var(--ink-faint)", letterSpacing: "0.1em", textTransform: "uppercase" }}>
                  {parseDomains(b.domain).join(" · ") || "未分类"}
                </div>
                <HeartMeter value={impToHearts(b.importance)} size={9} />
              </div>
              <div style={{ fontFamily: "var(--serif)", fontWeight: 500, fontSize: "1.05rem", color: "var(--moss-deep)", marginBottom: "4px" }}>{b.name}</div>
              <div style={{ fontFamily: "var(--kai)", fontSize: "0.78rem", color: "var(--ink-soft)", lineHeight: 1.55, marginBottom: "8px" }}>
                {pickPreview(b) || "（还没有记忆内容）"}
              </div>
              <div style={tooltipFootStyle}>
                {m.label} · {arousalLabel(b.arousal)} · {fmtRelative(b.last_active)}
              </div>
            </div>
          );
        })()}
      </div>

      {/* 图例 */}
      <Legend />
    </div>
  );
}

function PlantButton({ bucket, x, y, size, delay, hovered, setHovered, setTooltipPos, onClick }) {
  const b = bucket;
  return (
    <button
      onClick={onClick}
      onMouseEnter={(e) => {
        setHovered(b.id);
        const rect = e.currentTarget.getBoundingClientRect();
        const parent = e.currentTarget.parentElement.getBoundingClientRect();
        setTooltipPos({ x: rect.left - parent.left + rect.width / 2, y: rect.top - parent.top });
      }}
      onMouseLeave={() => setHovered(null)}
      className="fade-in"
      style={{
        position: "absolute", left: `${x}%`, top: `${y}%`,
        transform: "translate(-50%, -100%)",
        background: "transparent", border: "none", padding: 0, cursor: "pointer",
        animationDelay: `${delay}ms`, transition: "filter 0.2s",
        zIndex: hovered ? 10 : 1,
        filter: hovered ? "drop-shadow(0 4px 12px oklch(0.45 0.08 145 / 0.25))" : "none",
      }}
    >
      <PlantIcon
        valence={b.valence} arousal={b.arousal} importance={b.importance}
        resolved={b.resolved} domain={b.domain} size={size}
      />
      <div style={{
        fontFamily: "var(--kai)", fontSize: "0.7rem",
        color: hovered ? "var(--moss-deep)" : "var(--ink-faint)",
        marginTop: "-2px", maxWidth: "110px", textAlign: "center",
        opacity: hovered ? 1 : 0.7, transition: "all 0.18s",
        whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
      }}>
        {b.name && b.name.length > 9 ? b.name.slice(0, 8) + "…" : b.name}
      </div>
    </button>
  );
}

function ClusterButton({ cluster, x, y, size, delay, hovered, setHovered, setTooltipPos, onClick }) {
  const c = cluster;
  const lead = c.lead;
  const isCluster = c.items.length > 1;
  const key = `cl-${c.ix}-${c.iy}`;
  return (
    <button
      onClick={onClick}
      onMouseEnter={(e) => {
        setHovered(key);
        const rect = e.currentTarget.getBoundingClientRect();
        const parent = e.currentTarget.parentElement.getBoundingClientRect();
        setTooltipPos({ x: rect.left - parent.left + rect.width / 2, y: rect.top - parent.top });
      }}
      onMouseLeave={() => setHovered(null)}
      className="fade-in"
      style={{
        position: "absolute", left: `${x}%`, top: `${y}%`,
        transform: "translate(-50%, -100%)",
        background: "transparent", border: "none", padding: 0, cursor: "pointer",
        animationDelay: `${delay}ms`,
        zIndex: hovered ? 10 : 1,
        filter: hovered ? "drop-shadow(0 4px 12px oklch(0.45 0.08 145 / 0.28))" : "none",
        transition: "filter 0.2s",
      }}
    >
      <div style={{ position: "relative" }}>
        <PlantIcon
          valence={lead.valence} arousal={lead.arousal} importance={lead.importance}
          resolved={lead.resolved} domain={lead.domain} size={size}
        />
        {isCluster && (
          <span style={{
            position: "absolute", top: 0, right: 0,
            background: "var(--moss-deep)", color: "var(--paper)",
            fontFamily: "var(--mono)", fontSize: "0.62rem",
            padding: "1px 6px", borderRadius: "999px",
            border: "2px solid var(--paper)",
            letterSpacing: "0.04em",
          }}>+{c.items.length - 1}</span>
        )}
      </div>
      <div style={{
        fontFamily: "var(--kai)", fontSize: "0.7rem",
        color: hovered ? "var(--moss-deep)" : "var(--ink-faint)",
        marginTop: "-2px", maxWidth: "110px", textAlign: "center",
        opacity: hovered ? 1 : 0.7, transition: "all 0.18s",
        whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
      }}>
        {lead.name && lead.name.length > 9 ? lead.name.slice(0, 8) + "…" : lead.name}
      </div>
    </button>
  );
}

function tooltipStyle(pos) {
  return {
    position: "absolute", left: pos.x, top: pos.y - 14,
    transform: "translate(-50%, -100%)",
    background: "var(--paper)", border: "1px solid var(--line)",
    boxShadow: "0 12px 32px oklch(0.3 0.05 150 / 0.14)",
    padding: "12px 16px", maxWidth: "300px",
    borderRadius: "2px", pointerEvents: "none", zIndex: 20,
  };
}
const tooltipFootStyle = {
  fontFamily: "var(--mono)", fontSize: "0.62rem",
  color: "var(--ink-faint)", letterSpacing: "0.06em",
  paddingTop: "8px", borderTop: "1px dashed var(--line)", marginTop: "8px",
};

function AxisLabel({ pos, title, sub, align }) {
  return (
    <div style={{ position: "absolute", ...pos, textAlign: align, pointerEvents: "none" }}>
      <div style={{ fontFamily: "var(--serif)", fontStyle: "italic", fontSize: "1.15rem", color: "var(--moss-deep)", opacity: 0.55 }}>{title}</div>
      <div style={{ fontFamily: "var(--mono)", fontSize: "0.6rem", color: "var(--ink-faint)", letterSpacing: "0.08em", marginTop: "2px" }}>{sub}</div>
    </div>
  );
}
function axisCenterStyle(side) {
  return {
    position: "absolute", left: "50%", transform: "translateX(-50%)", [side]: "8px",
    fontFamily: "var(--mono)", fontSize: "0.6rem", color: "var(--ink-faint)",
    letterSpacing: "0.1em", pointerEvents: "none",
  };
}
function axisSideStyle(side) {
  return {
    position: "absolute", top: "50%", [side]: "8px",
    transform: side === "left" ? "translateY(-50%) rotate(-90deg)" : "translateY(-50%) rotate(90deg)",
    transformOrigin: "center", fontFamily: "var(--mono)", fontSize: "0.6rem",
    color: "var(--ink-faint)", letterSpacing: "0.1em", pointerEvents: "none",
  };
}

function Legend() {
  const moods = [
    { v: 0.9, key: "🥰 超幸福" },
    { v: 0.72, key: "😊 开心" },
    { v: 0.55, key: "☺️ 平静" },
    { v: 0.4, key: "🥺 有点难过" },
    { v: 0.15, key: "😢 难过" },
  ];
  return (
    <div style={{ marginTop: "20px", display: "flex", flexDirection: "column", gap: "10px" }}>
      <div className="legend-block" style={{ marginTop: "8px" }}>
        <div style={{ display: "flex", flexDirection: "column", gap: "4px", gridColumn: "1 / -1" }}>
          <div style={{ fontFamily: "var(--mono)", fontSize: "0.65rem", color: "var(--ink-faint)", letterSpacing: "0.12em", textTransform: "uppercase" }}>花色 · 情绪</div>
          <div style={{ display: "flex", gap: "12px 18px", flexWrap: "wrap" }}>
            {moods.map((m) => {
              const mm = moodOf(m.v);
              return (
                <span key={m.key} style={{ display: "inline-flex", alignItems: "center", gap: "6px", fontFamily: "var(--kai)", fontSize: "0.78rem", color: "var(--ink-soft)" }}>
                  <span style={{ width: 10, height: 10, borderRadius: "50%", background: mm.color, border: "1px solid oklch(0.78 0.04 80)" }} />
                  {m.key.replace(/^\S+ /, "")}
                </span>
              );
            })}
          </div>
        </div>
      </div>
      <div className="legend-block" style={{ marginTop: "4px" }}>
        <div style={{ display: "flex", flexDirection: "column", gap: "4px", gridColumn: "1 / -1" }}>
          <div style={{ fontFamily: "var(--mono)", fontSize: "0.65rem", color: "var(--ink-faint)", letterSpacing: "0.12em", textTransform: "uppercase" }}>形态 · 主题</div>
        </div>
        <span style={legendItem}>山茶 · 恋爱</span>
        <span style={legendItem}>铃兰 · 内心</span>
        <span style={legendItem}>蕨类 · 数字</span>
        <span style={legendItem}>蒲公英 · 人际</span>
        <span style={legendItem}>鸢尾 · 创作</span>
        <span style={legendItem}>三叶草 · 日常</span>
        <span style={legendItem}>薰衣草 · 身心</span>
        <span style={legendItem}>嫩芽 · 成长</span>
        <span style={legendItem}>干花 · 回忆</span>
      </div>
      <div style={{ marginTop: "4px", display: "flex", flexWrap: "wrap", gap: "6px 18px", fontFamily: "var(--kai)", fontSize: "0.78rem", color: "var(--ink-soft)" }}>
        <span><span style={legendKey}>萤火虫</span> 还在生长（未解决）</span>
        <span><span style={legendKey}>植株高低</span> 重要程度（1–5 心）</span>
        <span><span style={legendKey}>摇摆速度</span> arousal · 越高越急</span>
      </div>
    </div>
  );
}
const legendItem = { fontFamily: "var(--kai)", fontSize: "0.78rem", color: "var(--ink-soft)" };
const legendKey = { fontFamily: "var(--mono)", fontSize: "0.62rem", color: "var(--ink-faint)", letterSpacing: "0.12em", textTransform: "uppercase", marginRight: "6px" };

window.GardenView = GardenView;
