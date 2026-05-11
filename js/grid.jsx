// 卡片网格视图 + 区域 cluster 抽屉

const { useState: useStateGr, useMemo: useMemoGr } = React;

const TYPE_LABEL_INLINE = {
  permanent: "永久",
  dynamic: "动态",
  archive: "归档",
};

function GridView({ buckets, onOpen }) {
  const [q, setQ] = useStateGr("");
  const [domain, setDomain] = useStateGr("全部");
  const [resolved, setResolved] = useStateGr("全部");
  const [sort, setSort] = useStateGr("recent");

  const allDomains = useMemoGr(() => {
    const set = new Set();
    buckets.forEach((b) => parseDomains(b.domain).forEach((d) => set.add(d)));
    return ["全部", ...Array.from(set)];
  }, [buckets]);

  const filtered = useMemoGr(() => {
    let arr = buckets.filter((b) => {
      const ds = parseDomains(b.domain);
      if (domain !== "全部" && !ds.includes(domain)) return false;
      if (resolved === "已解决" && !b.resolved) return false;
      if (resolved === "未解决" && b.resolved) return false;
      if (q.trim()) {
        const needle = q.trim().toLowerCase();
        // 列表接口不返回 content（流量考虑），用 summary 替代做搜索
        const hay = [b.name, b.summary || "", ...parseTags(b.tags), ...parseDomains(b.domain)].join(" ").toLowerCase();
        if (!hay.includes(needle)) return false;
      }
      return true;
    });
    if (sort === "recent")    arr.sort((a, b) => new Date(b.last_active || 0) - new Date(a.last_active || 0));
    if (sort === "important") arr.sort((a, b) => (b.importance || 0) - (a.importance || 0));
    if (sort === "valence")   arr.sort((a, b) => (b.valence || 0) - (a.valence || 0));
    if (sort === "weight")    arr.sort((a, b) => (b.weight || 0) - (a.weight || 0));
    return arr;
  }, [buckets, q, domain, resolved, sort]);

  return (
    <div style={{ padding: "24px 56px 80px" }}>
      {/* 控制条 */}
      <div style={{ display: "flex", gap: "12px", alignItems: "center", marginBottom: "16px", flexWrap: "wrap" }}>
        <div style={{ position: "relative", flex: "1 1 320px", maxWidth: "440px" }}>
          <input
            value={q} onChange={(e) => setQ(e.target.value)}
            placeholder="找一个名字、一段话、一个标签…"
            style={{
              width: "100%", padding: "12px 16px 12px 38px",
              border: "1px solid var(--line)", borderRadius: "2px",
              background: "var(--paper)", fontFamily: "var(--kai)",
              fontSize: "0.95rem", color: "var(--ink)", outline: "none",
              transition: "border-color 0.2s",
            }}
            onFocus={(e) => e.target.style.borderColor = "var(--moss)"}
            onBlur={(e) => e.target.style.borderColor = "var(--line)"}
          />
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6"
            style={{ position: "absolute", left: "14px", top: "50%", transform: "translateY(-50%)", color: "var(--ink-faint)" }}>
            <circle cx="11" cy="11" r="7" />
            <path d="m20 20-3.5-3.5" strokeLinecap="round" />
          </svg>
        </div>

        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: "10px" }}>
          <span style={{ fontFamily: "var(--mono)", fontSize: "0.65rem", color: "var(--ink-faint)", letterSpacing: "0.12em", textTransform: "uppercase" }}>sort</span>
          <select value={sort} onChange={(e) => setSort(e.target.value)}
            style={{
              border: "1px solid var(--line)", background: "var(--paper)",
              padding: "6px 10px", borderRadius: "2px", fontFamily: "var(--kai)",
              fontSize: "0.85rem", color: "var(--ink)", outline: "none", cursor: "pointer",
            }}>
            <option value="recent">最近活跃</option>
            <option value="weight">权重</option>
            <option value="important">最要紧</option>
            <option value="valence">最甜的</option>
          </select>
        </div>
      </div>

      <div style={{ display: "flex", gap: "6px", flexWrap: "wrap", marginBottom: "12px" }}>
        {allDomains.map((d) => (
          <Tag key={d} active={domain === d} onClick={() => setDomain(d)} tone="moss">{d}</Tag>
        ))}
        <span style={{ width: "1px", background: "var(--line)", margin: "0 6px" }} />
        {["全部", "已解决", "未解决"].map((r) => (
          <Tag key={r} active={resolved === r} onClick={() => setResolved(r)} tone="rose">{r}</Tag>
        ))}
      </div>

      <div style={{ fontFamily: "var(--kai)", fontSize: "0.85rem", color: "var(--ink-faint)", marginBottom: "16px" }}>
        共 {filtered.length} 株 {q || domain !== "全部" || resolved !== "全部" ? "· 已筛选" : ""}
      </div>

      {filtered.length === 0 ? (
        <div style={{ padding: "80px 0", textAlign: "center", fontFamily: "var(--serif)", fontStyle: "italic", color: "var(--ink-faint)", fontSize: "1.1rem" }}>
          没找到符合条件的记忆。换个词试试？
        </div>
      ) : (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(300px, 1fr))", gap: "20px" }}>
          {filtered.map((b, i) => (
            <BucketCard key={b.id} bucket={b} onOpen={onOpen} delay={i * 25} />
          ))}
        </div>
      )}
    </div>
  );
}

function BucketCard({ bucket, onOpen, delay }) {
  const [hover, setHover] = useStateGr(false);
  const b = bucket;
  const domains = parseDomains(b.domain);
  const tags = parseTags(b.tags);
  const hearts = impToHearts(b.importance);

  return (
    <button
      onClick={() => onOpen(b.id)}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      className="fade-in"
      style={{
        background: "var(--paper)", border: "1px solid var(--line)",
        borderRadius: "3px", padding: "20px 22px",
        textAlign: "left", cursor: "pointer", position: "relative",
        transition: "all 0.22s ease", animationDelay: `${delay}ms`,
        boxShadow: hover ? "0 12px 28px oklch(0.3 0.05 150 / 0.10)" : "0 1px 2px oklch(0.3 0.05 150 / 0.04)",
        transform: hover ? "translateY(-2px)" : "translateY(0)",
        borderColor: hover ? "var(--moss)" : "var(--line)",
        display: "flex", flexDirection: "column", gap: "12px", minHeight: "230px",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: "12px" }}>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{
            fontFamily: "var(--mono)", fontSize: "0.62rem",
            color: "var(--ink-faint)", letterSpacing: "0.12em",
            textTransform: "uppercase", marginBottom: "4px",
          }}>
            {domains.join(" · ") || "未分类"} · {fmtRelative(b.last_active)}
          </div>
          <div style={{
            fontFamily: "var(--serif)", fontWeight: 500, fontSize: "1.22rem",
            color: "var(--moss-deep)", lineHeight: 1.25, letterSpacing: "0.005em",
          }}>{b.name}</div>
        </div>
        <div style={{ flexShrink: 0 }}>
          <PlantIcon
            valence={b.valence} arousal={b.arousal} importance={b.importance}
            resolved={b.resolved} domain={b.domain} size={52}
          />
        </div>
      </div>

      <div style={{
        fontFamily: "var(--kai)", fontSize: "0.88rem",
        color: "var(--ink-soft)", lineHeight: 1.6, flex: 1,
        display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical",
        overflow: "hidden", textWrap: "pretty",
      }}>
        {pickPreview(b) || "（还没有记忆内容）"}
      </div>

      {tags.length > 0 && (
        <div style={{ display: "flex", gap: "5px", flexWrap: "wrap" }}>
          {tags.slice(0, 4).map((t) => (
            <span key={t} style={{
              fontFamily: "var(--sans)", fontSize: "0.7rem",
              color: "var(--ink-soft)", background: "oklch(0.96 0.018 90)",
              padding: "2px 8px", borderRadius: "999px",
              border: "1px solid var(--line)",
            }}>{t}</span>
          ))}
          {tags.length > 4 && (
            <span style={{ fontFamily: "var(--mono)", fontSize: "0.65rem", color: "var(--ink-faint)", padding: "2px 4px" }}>
              +{tags.length - 4}
            </span>
          )}
        </div>
      )}

      <div style={{
        display: "flex", justifyContent: "space-between", alignItems: "center",
        paddingTop: "10px", borderTop: "1px dashed var(--line)",
        fontFamily: "var(--mono)", fontSize: "0.65rem",
        color: "var(--ink-faint)", letterSpacing: "0.06em",
      }}>
        <span>{TYPE_LABEL_INLINE[b.type] || "记忆"}</span>
        <span style={{ display: "flex", gap: "10px", alignItems: "center" }}>
          {b.todos_count > 0 && (
            <span title={`${b.todos_count} 件待办`}
              style={{ color: "oklch(0.55 0.12 70)", display: "inline-flex", alignItems: "center", gap: "4px" }}>
              <svg width="9" height="9" viewBox="0 0 16 16">
                <path d="M 8 4 Q 4 6, 5 11 Q 8 13, 11 11 Q 12 6, 8 4 Z"
                  fill="oklch(0.92 0.07 65)" stroke="oklch(0.6 0.13 70)" strokeWidth="1.4" />
              </svg>
              {b.todos_count}
            </span>
          )}
          {!b.resolved && (
            <span style={{ color: "var(--rose-deep)", display: "inline-flex", alignItems: "center", gap: "4px" }}>
              <span style={{ width: 5, height: 5, borderRadius: "50%", background: "var(--rose)", display: "inline-block" }} />
              生长中
            </span>
          )}
          <HeartMeter value={hearts} size={9} />
        </span>
      </div>
    </button>
  );
}

// 区域聚合的小列表（被花园视图调用）
function ClusterPanel({ cluster, onOpen, onClose }) {
  if (!cluster) return null;
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
          width: "min(560px, 92vw)", background: "var(--paper)",
          borderLeft: "1px solid var(--line)",
          boxShadow: "-30px 0 60px oklch(0.2 0.04 150 / 0.15)",
          overflowY: "auto", animation: "drawer-slide 0.32s cubic-bezier(0.2, 0.8, 0.2, 1)",
        }}>
        {/* 关闭按钮：sticky 固定 */}
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
        <div style={{ padding: "32px 36px 20px", borderBottom: "1px solid var(--line)" }}>
          <div style={{ fontFamily: "var(--mono)", fontSize: "0.65rem", color: "var(--ink-faint)", letterSpacing: "0.14em", textTransform: "uppercase", marginBottom: "6px" }}>
            这片角落 · {cluster.items.length} 株
          </div>
          <h3 style={{ fontFamily: "var(--serif)", fontStyle: "italic", fontWeight: 500, fontSize: "1.6rem", color: "var(--moss-deep)", margin: 0, lineHeight: 1.2 }}>
            {regionTitle(cluster.ix, cluster.iy)}
          </h3>
        </div>
        <div style={{ padding: "20px 36px 60px", display: "flex", flexDirection: "column", gap: "14px" }}>
          {cluster.items.map((b, i) => (
            <button key={b.id} onClick={() => onOpen(b.id)}
              className="fade-in"
              style={{
                display: "grid", gridTemplateColumns: "44px 1fr auto",
                gap: "14px", alignItems: "center",
                background: "var(--paper)",
                border: "1px solid var(--line)", borderRadius: "2px",
                padding: "12px 16px", cursor: "pointer", textAlign: "left",
                animationDelay: `${i * 30}ms`, transition: "all 0.18s",
              }}
              onMouseEnter={(e) => e.currentTarget.style.borderColor = "var(--moss)"}
              onMouseLeave={(e) => e.currentTarget.style.borderColor = "var(--line)"}
            >
              <PlantIcon valence={b.valence} arousal={b.arousal} importance={b.importance}
                resolved={b.resolved} domain={b.domain} size={40} />
              <div style={{ minWidth: 0 }}>
                <div style={{ fontFamily: "var(--serif)", fontWeight: 500, fontSize: "1.05rem", color: "var(--moss-deep)", marginBottom: "2px" }}>
                  {b.name}
                </div>
                <div style={{ fontFamily: "var(--kai)", fontSize: "0.78rem", color: "var(--ink-soft)", lineHeight: 1.5,
                  display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical", overflow: "hidden",
                }}>
                  {pickPreview(b) || "—"}
                </div>
              </div>
              <div style={{ textAlign: "right", flexShrink: 0 }}>
                <HeartMeter value={impToHearts(b.importance)} size={9} />
                <div style={{ fontFamily: "var(--mono)", fontSize: "0.62rem", color: "var(--ink-faint)", marginTop: "4px" }}>
                  {fmtRelative(b.last_active)}
                </div>
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

function regionTitle(ix, iy) {
  // ix: valence 0-4 低→高；iy: arousal 0-4 低→高
  const v = ix <= 1 ? "低" : ix >= 3 ? "高" : "中";
  const a = iy <= 1 ? "静" : iy >= 3 ? "激" : "稳";
  if (v === "高" && a === "激") return "雀跃 · 兴奋鲜活";
  if (v === "高" && a === "静") return "安宁 · 宁静的甜";
  if (v === "高")               return "温和的好心情";
  if (v === "低" && a === "激") return "焦灼 · 起伏不定";
  if (v === "低" && a === "静") return "低沉 · 安静的低落";
  if (v === "低")               return "平淡里的失落";
  return "中间地带";
}

window.GridView = GridView;
window.ClusterPanel = ClusterPanel;
