// 时间线藤蔓视图 —— 单根中心藤，植物左右交替挂在藤两侧
// 移动端优先：月份做横向 chapter 标题，不再左右双栏

const { useState: useStateT, useMemo: useMemoT } = React;
const { PlantIcon, parseDomains, moodOf, impToHearts, fmtRelative } = window;

const MONTH_CN = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "十一", "十二"];

function TimelineView({ buckets, onOpen }) {
  // 藤蔓按"种下时间"(created)分组+排序 —— 这是花园的生长史，不是最近触碰史
  // last_active 是检索/更新时间，时间线展示不合适
  const months = useMemoT(() => {
    const map = new Map();
    for (const b of buckets) {
      const t = b.created || b.last_active;
      if (!t) continue;
      const d = new Date(t);
      const key = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}`;
      if (!map.has(key)) map.set(key, { key, year: d.getFullYear(), month: d.getMonth() + 1, items: [] });
      map.get(key).items.push(b);
    }
    const arr = Array.from(map.values()).sort((a, b) => b.key.localeCompare(a.key));
    arr.forEach((m) => {
      m.items.sort((a, b) => new Date(b.created || b.last_active) - new Date(a.created || a.last_active));
    });
    return arr;
  }, [buckets]);

  if (!buckets.length) {
    return <div style={{ padding: "120px 0", textAlign: "center", fontFamily: "var(--serif)", fontStyle: "italic", color: "var(--ink-faint)" }}>花园还空着。</div>;
  }

  return (
    <div style={{ padding: "24px 16px 80px", position: "relative" }}>
      <div style={{ maxWidth: "640px", margin: "0 auto", position: "relative" }}>
        {months.map((m, mi) => (
          <MonthSection key={m.key} month={m} index={mi} isLast={mi === months.length - 1} onOpen={onOpen} />
        ))}
        <div style={{
          textAlign: "center", marginTop: "20px",
          fontFamily: "var(--serif)", fontStyle: "italic",
          fontSize: "0.95rem", color: "var(--ink-faint)",
        }}>· 这里是花园开始的地方 ·</div>
      </div>
    </div>
  );
}

function MonthSection({ month, index, isLast, onOpen }) {
  const m = month;
  const items = m.items;
  const N = items.length;
  // 每株 110px 行高，藤随之延伸
  const ROW = 110;
  const stemH = N * ROW;

  return (
    <div className="fade-in" style={{ position: "relative", marginBottom: "8px", animationDelay: `${index * 80}ms` }}>
      {/* 月份章节标题：横排 */}
      <div style={{
        display: "flex", alignItems: "baseline", gap: "12px",
        padding: "8px 0 12px", borderBottom: "1px dashed var(--line)",
        marginBottom: "8px",
      }}>
        <div style={{
          fontFamily: "var(--serif)", fontStyle: "italic", fontWeight: 500,
          fontSize: "1.7rem", color: "var(--moss-deep)", lineHeight: 1,
        }}>
          {MONTH_CN[m.month - 1]}月
        </div>
        <div style={{
          fontFamily: "var(--mono)", fontSize: "0.68rem",
          color: "var(--ink-faint)", letterSpacing: "0.1em",
        }}>{m.year}</div>
        <div style={{ flex: 1 }} />
        <div style={{
          fontFamily: "var(--kai)", fontSize: "0.78rem",
          color: "var(--ink-soft)",
        }}>{N} 株生长</div>
      </div>

      {/* 藤 + 植物：藤在中线，植物左右交替 */}
      <div style={{ position: "relative", height: stemH + "px" }}>
        {/* 中央藤蔓：单条平滑曲线 */}
        <svg
          style={{ position: "absolute", left: 0, top: 0, width: "100%", height: "100%", overflow: "visible", pointerEvents: "none" }}
          preserveAspectRatio="none"
          viewBox={`0 0 100 ${stemH}`}
        >
          <path
            d={smoothVinePath(stemH, N)}
            stroke="oklch(0.55 0.08 145)"
            strokeWidth="0.5"
            fill="none"
            strokeLinecap="round"
            strokeDasharray="0.8 0.5"
            opacity="0.65"
          />
          {/* 藤上叶子点缀 */}
          {Array.from({ length: N + 1 }).map((_, i) => {
            const y = (i / N) * stemH + 8;
            const dx = (i % 2 ? -1 : 1) * 2;
            return (
              <ellipse key={i} cx={50 + dx} cy={y} rx="2" ry="0.9"
                fill="oklch(0.58 0.10 145)" opacity="0.5"
                transform={`rotate(${i % 2 ? -25 : 25} ${50 + dx} ${y})`} />
            );
          })}
        </svg>

        {items.map((b, i) => (
          <PlantNode
            key={b.id}
            bucket={b}
            yPx={i * ROW + ROW / 2}
            sideRight={i % 2 === 0}
            delay={i * 50}
            onOpen={onOpen}
          />
        ))}
      </div>

      {/* 月份之间的藤蔓接续提示 */}
      {!isLast && (
        <div style={{ display: "flex", justifyContent: "center", padding: "4px 0 8px" }}>
          <svg width="20" height="20" viewBox="0 0 20 20">
            <path d="M 10 0 Q 14 6, 10 10 Q 6 14, 10 20" stroke="oklch(0.55 0.08 145)" strokeWidth="1" fill="none" strokeDasharray="1 1" opacity="0.6" />
          </svg>
        </div>
      )}
    </div>
  );
}

// 中央藤蔓：一根连续柔顺的正弦曲线
function smoothVinePath(h, n) {
  // 正弦波采样：10 个中间点，全部连成同一条 cubic spline
  const segs = Math.max(8, n * 3);
  const amp = 5;     // 摆幅
  const freq = Math.max(1, n * 0.5);  // 总周期数
  const pts = [];
  for (let i = 0; i <= segs; i++) {
    const t = i / segs;
    const y = t * h;
    const x = 50 + Math.sin(t * Math.PI * freq) * amp;
    pts.push([x, y]);
  }
  // 用 Catmull-Rom 拼成 cubic Bezier，曲线连续无折点
  let d = `M ${pts[0][0]} ${pts[0][1]}`;
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[i - 1] || pts[i];
    const p1 = pts[i];
    const p2 = pts[i + 1];
    const p3 = pts[i + 2] || p2;
    const c1x = p1[0] + (p2[0] - p0[0]) / 6;
    const c1y = p1[1] + (p2[1] - p0[1]) / 6;
    const c2x = p2[0] - (p3[0] - p1[0]) / 6;
    const c2y = p2[1] - (p3[1] - p1[1]) / 6;
    d += ` C ${c1x.toFixed(2)} ${c1y.toFixed(2)}, ${c2x.toFixed(2)} ${c2y.toFixed(2)}, ${p2[0].toFixed(2)} ${p2[1].toFixed(2)}`;
  }
  return d;
}

function PlantNode({ bucket, yPx, sideRight, delay, onOpen }) {
  const [hover, setHover] = useStateT(false);
  const b = bucket;
  const hearts = impToHearts(b.importance);
  const size = 44 + hearts * 4;
  const domains = parseDomains(b.domain);
  const m = moodOf(b.valence);

  // 植物贴在中线一侧，标签在植物外侧
  return (
    <div
      className="fade-in"
      style={{
        position: "absolute",
        left: 0, right: 0, top: yPx + "px",
        height: 0,
        animationDelay: `${delay}ms`,
      }}
    >
      <button
        onClick={() => onOpen(b.id)}
        onMouseEnter={() => setHover(true)}
        onMouseLeave={() => setHover(false)}
        style={{
          position: "absolute",
          [sideRight ? "left" : "right"]: "50%",
          top: "0",
          transform: sideRight
            ? "translate(2px, -50%)"
            : "translate(-2px, -50%)",
          background: "transparent", border: "none", padding: "4px",
          cursor: "pointer",
          display: "flex", alignItems: "center",
          flexDirection: sideRight ? "row" : "row-reverse",
          gap: "8px",
          maxWidth: "calc(50% - 4px)",
          zIndex: hover ? 10 : 1,
        }}
      >
        {/* 小连接点 */}
        <span style={{
          flexShrink: 0,
          width: "6px", height: "6px", borderRadius: "50%",
          background: m.color,
          border: "1px solid oklch(0.78 0.04 80)",
          alignSelf: "center",
          marginTop: "2px",
        }} />
        <PlantIcon
          valence={b.valence} arousal={b.arousal} importance={b.importance}
          resolved={b.resolved} domain={b.domain} size={size}
        />
        <div style={{
          textAlign: sideRight ? "left" : "right",
          minWidth: 0,
          flexShrink: 1,
        }}>
          <div style={{
            fontFamily: "var(--serif)", fontWeight: 500,
            fontSize: "0.92rem",
            color: hover ? "var(--moss-deep)" : "var(--ink)",
            lineHeight: 1.25,
            wordBreak: "break-word",
          }}>{b.name}</div>
          <div style={{
            fontFamily: "var(--mono)", fontSize: "0.6rem",
            color: "var(--ink-faint)", letterSpacing: "0.06em",
            marginTop: "4px",
            whiteSpace: "nowrap",
            overflow: "hidden", textOverflow: "ellipsis",
          }}>
            {fmtRelative(b.last_active)} · {domains[0] || "—"}
          </div>
        </div>
      </button>
    </div>
  );
}

window.TimelineView = TimelineView;
