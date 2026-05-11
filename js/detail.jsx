// 详情抽屉

const { useState: useStateD, useEffect: useEffectD } = React;

function DetailDrawer({ bucketId, onClose }) {
  const [data, setData] = useStateD(null);
  const [loading, setLoading] = useStateD(true);

  useEffectD(() => {
    if (!bucketId) return;
    setLoading(true);
    setData(null);
    window.api.fetchBucket(bucketId).then(({ bucket }) => {
      setData(bucket);
      setLoading(false);
    });
  }, [bucketId]);

  useEffectD(() => {
    const onKey = (e) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  if (!bucketId) return null;

  return (
    <div onClick={onClose}
      style={{
        position: "fixed", inset: 0,
        background: "oklch(0.2 0.04 150 / 0.32)",
        backdropFilter: "blur(2px)", zIndex: 100,
        animation: "fade-in 0.25s ease-out",
      }}>
      <div onClick={(e) => e.stopPropagation()}
        style={{
          position: "absolute", right: 0, top: 0, bottom: 0,
          width: "min(720px, 92vw)", background: "var(--paper)",
          borderLeft: "1px solid var(--line)",
          boxShadow: "-30px 0 60px oklch(0.2 0.04 150 / 0.15)",
          overflowY: "auto",
          animation: "drawer-slide 0.32s cubic-bezier(0.2, 0.8, 0.2, 1)",
        }}>
        {/* 关闭按钮：固定在抽屉右上角，始终可见 */}
        <button onClick={onClose} aria-label="关闭"
          style={{
            position: "sticky", top: "12px", float: "right",
            marginRight: "12px", marginTop: "12px",
            zIndex: 20,
            background: "var(--paper)",
            border: "1px solid var(--line)",
            borderRadius: "50%", width: 40, height: 40, cursor: "pointer",
            color: "var(--ink-soft)", fontSize: "1.4rem", lineHeight: 1,
            fontFamily: "var(--serif)",
            boxShadow: "0 2px 8px oklch(0.2 0.04 150 / 0.08)",
            display: "flex", alignItems: "center", justifyContent: "center",
            paddingBottom: "3px",
          }}>×</button>
        {loading || !data ? (
          <div style={{ padding: "120px 0", textAlign: "center", fontFamily: "var(--serif)", fontStyle: "italic", color: "var(--ink-faint)" }}>
            正在翻开这一页…
          </div>
        ) : (
          <DetailContent bucket={data} onClose={onClose} />
        )}
      </div>
      <style>{`
        @keyframes drawer-slide {
          from { transform: translateX(40px); opacity: 0.5; }
          to   { transform: translateX(0); opacity: 1; }
        }
      `}</style>
    </div>
  );
}

const TYPE_LABEL = {
  permanent: "永久",
  dynamic: "动态",
  archive: "归档",
};

function DetailContent({ bucket, onClose }) {
  const b = bucket;
  // 后端 _apply_wikilinks 给 content 自动加 [[关键词]] 双链供 Obsidian 用，前端展示要剥掉
  const content = stripWikilinks(b.content || "");
  const domains = parseDomains(b.domain);
  const tags = parseTags(b.tags);
  const hearts = impToHearts(b.importance);
  const mood = moodOf(b.valence);
  const typeLabel = TYPE_LABEL[b.type] || null;

  return (
    <div>
      <div style={{
        padding: "28px 40px 24px",
        borderBottom: "1px solid var(--line)",
        position: "sticky", top: 0, background: "var(--paper)", zIndex: 10,
      }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: "16px" }}>
          <div style={{ flex: 1 }}>
            <div style={{ fontFamily: "var(--mono)", fontSize: "0.65rem", color: "var(--ink-faint)", letterSpacing: "0.14em", textTransform: "uppercase", marginBottom: "6px" }}>
              {domains.join(" · ") || "未分类"}
            </div>
            <h2 style={{
              fontFamily: "var(--serif)", fontStyle: "italic", fontWeight: 500,
              fontSize: "2.1rem", color: "var(--moss-deep)", margin: 0,
              lineHeight: 1.18, letterSpacing: "-0.005em",
            }}>{b.name}</h2>
            <div style={{ marginTop: "10px", display: "flex", alignItems: "center", gap: "10px", flexWrap: "wrap" }}>
              <HeartMeter value={hearts} size={12} />
              <span style={{ fontFamily: "var(--kai)", fontSize: "0.85rem", color: "var(--ink-soft)" }}>
                {mood.label} · {arousalLabel(b.arousal) === "激" ? "起伏" : arousalLabel(b.arousal) === "稳" ? "平稳" : "安静"}
                {!b.resolved && <span style={{ color: "var(--rose-deep)", marginLeft: "8px" }}>· 还在生长</span>}
              </span>
            </div>
          </div>
          <div style={{ display: "flex", gap: "12px", alignItems: "flex-start" }}>
            <PlantIcon
              valence={b.valence} arousal={b.arousal} importance={b.importance}
              resolved={b.resolved} domain={b.domain} size={72}
            />
          </div>
        </div>

        {tags.length > 0 && (
          <div style={{ display: "flex", gap: "6px", marginTop: "14px", flexWrap: "wrap" }}>
            {tags.map((t) => (
              <span key={t} style={{
                fontFamily: "var(--sans)", fontSize: "0.78rem",
                color: "var(--moss-deep)",
                background: "oklch(0.94 0.03 150)",
                padding: "3px 10px", borderRadius: "999px",
                border: "1px solid oklch(0.85 0.05 150)",
              }}>{t}</span>
            ))}
          </div>
        )}
      </div>

      {/* 情感面板 */}
      <div style={{
        padding: "20px 40px",
        background: "oklch(0.985 0.012 88)",
        borderBottom: "1px solid var(--line)",
        display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "16px",
      }}>
        <Meter label="效价" sub={`valence · ${mood.label}`} value={b.valence} color={mood.color} />
        <Meter label="唤醒度" sub={`arousal · ${arousalLabel(b.arousal)}`} value={b.arousal} color="var(--amber)" />
        <Meter label="权重" sub="weight · AI 想起来的频率" value={Math.min(1, (b.weight || 0) / 10)} raw={b.weight} color="var(--moss)" />
      </div>

      {/* 还在长的事 —— todos 用未开的花苞表示，只在有数据时显示 */}
      {Array.isArray(b.todos) && b.todos.length > 0 && (
        <div style={{ padding: "24px 40px 22px", borderBottom: "1px solid var(--line)" }}>
          <div style={{
            fontFamily: "var(--mono)", fontSize: "0.65rem",
            color: "var(--ink-faint)", letterSpacing: "0.14em",
            textTransform: "uppercase", marginBottom: "16px",
            display: "flex", alignItems: "baseline", gap: "10px",
          }}>
            <span>bud · 还没开的</span>
            <span style={{ fontFamily: "var(--serif)", fontStyle: "italic", fontSize: "0.95rem", color: "var(--moss-deep)", textTransform: "none", letterSpacing: 0 }}>
              {b.todos.length} 朵花苞还含着
            </span>
          </div>
          <ul style={{ listStyle: "none", padding: 0, margin: 0, display: "flex", flexDirection: "column", gap: "10px" }}>
            {b.todos.map((todo, i) => (
              <li key={i} style={{
                display: "flex", alignItems: "flex-start", gap: "12px",
                fontFamily: "var(--kai)", fontSize: "0.98rem",
                color: "var(--ink)", lineHeight: 1.7,
              }}>
                <svg width="16" height="16" viewBox="0 0 16 16" style={{ flexShrink: 0, marginTop: "5px" }}>
                  {/* 花苞：椭圆瓣 + 一小段茎 */}
                  <path d="M 8 4 Q 4 6, 5 11 Q 8 13, 11 11 Q 12 6, 8 4 Z"
                    fill={mood.pale} stroke={mood.color} strokeWidth="1.2" opacity="0.95" />
                  <path d="M 8 13 L 8 16" stroke="oklch(0.45 0.08 145)" strokeWidth="1.1" strokeLinecap="round" />
                </svg>
                <span>{todo}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      <div style={{ padding: "32px 40px 60px" }}>
        <div style={{ fontFamily: "var(--mono)", fontSize: "0.65rem", color: "var(--ink-faint)", letterSpacing: "0.14em", textTransform: "uppercase", marginBottom: "20px", display: "flex", justifyContent: "space-between" }}>
          <span>content · 这棵植物记得的</span>
          {typeLabel && (
            <span style={{ color: "var(--ink-soft)", fontSize: "0.6rem" }}>
              {typeLabel}
            </span>
          )}
        </div>

        {content ? (
          <div style={{
            padding: "22px 24px",
            background: "oklch(0.985 0.012 88)",
            border: "1px solid var(--line)",
            borderRadius: "3px",
          }}>
            <p style={{
              fontFamily: "var(--kai)", fontSize: "1.05rem",
              color: "var(--ink)", lineHeight: 1.85, margin: 0,
              textWrap: "pretty", whiteSpace: "pre-wrap",
            }}>{content}</p>
          </div>
        ) : (
          <div style={{
            padding: "32px 20px", textAlign: "center",
            fontFamily: "var(--serif)", fontStyle: "italic",
            color: "var(--ink-faint)", border: "1px dashed var(--line)", borderRadius: "3px",
          }}>
            这棵植物还没展开它的具体记忆。
          </div>
        )}
      </div>

      <div style={{
        padding: "16px 40px",
        background: "oklch(0.96 0.015 88)",
        fontFamily: "var(--mono)", fontSize: "0.66rem",
        color: "var(--ink-faint)", letterSpacing: "0.08em",
        display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: "8px",
      }}>
        <span>种下：{fmtDate(b.created)}</span>
        <span>最后触碰：{fmtRelative(b.last_active)}</span>
      </div>
    </div>
  );
}

function Meter({ label, sub, value = 0, raw, color }) {
  const v = Math.max(0, Math.min(1, value));
  const display = raw != null ? raw.toFixed(1) : (v * 100).toFixed(0);
  return (
    <div style={{
      padding: "14px 16px", background: "var(--paper)",
      border: "1px solid var(--line)", borderRadius: "2px",
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: "10px" }}>
        <div>
          <div style={{ fontFamily: "var(--mono)", fontSize: "0.62rem", color: "var(--ink-faint)", letterSpacing: "0.12em", textTransform: "uppercase" }}>{label}</div>
          <div style={{ fontFamily: "var(--kai)", fontSize: "0.7rem", color: "var(--ink-faint)", marginTop: "2px" }}>{sub}</div>
        </div>
        <div style={{ fontFamily: "var(--serif)", fontSize: "1.4rem", color: "var(--ink)", fontWeight: 500 }}>
          {display}
        </div>
      </div>
      <div style={{ height: "5px", background: "oklch(0.94 0.018 88)", borderRadius: "3px", overflow: "hidden" }}>
        <div style={{
          height: "100%", width: `${v * 100}%`,
          background: color, borderRadius: "3px", transition: "width 0.6s ease",
        }} />
      </div>
    </div>
  );
}

window.DetailDrawer = DetailDrawer;
