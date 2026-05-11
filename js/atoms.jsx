// 共享原子组件 + 工具

const { useState, useEffect, useMemo, useRef } = React;

// ───── domain → 植物形态 ─────
// 8 个主域 → 8 种植物，每个主域下的子域共享同一形态
//   日常 → clover    人际 → dandelion   成长 → tendril   身心 → lavender
//   兴趣 → iris      数字 → fern        事务 → wheat     内心 → bell
const DOMAIN_FORM = {
  // 日常 → 三叶草
  "日常": "clover",
  "饮食": "clover", "穿搭": "clover", "出行": "clover", "居家": "clover", "购物": "clover",
  // 人际 → 蒲公英
  "人际": "dandelion",
  "家庭": "dandelion", "恋爱": "camellia", "友谊": "dandelion", "社交": "dandelion",
  // 成长 → 嫩芽卷须
  "成长": "tendril",
  "工作": "tendril", "学习": "tendril", "考试": "tendril", "求职": "tendril",
  // 身心 → 薰衣草
  "身心": "lavender",
  "健康": "lavender", "心理": "lavender", "睡眠": "lavender", "运动": "lavender",
  // 兴趣 → 鸢尾
  "兴趣": "iris",
  "游戏": "iris", "影视": "iris", "音乐": "iris", "阅读": "iris", "创作": "iris", "手工": "iris",
  // 数字 → 蕨类
  "数字": "fern",
  "编程": "fern", "AI": "fern", "硬件": "fern", "网络": "fern",
  // 事务 → 麦穗
  "事务": "wheat",
  "财务": "wheat", "计划": "wheat", "待办": "wheat",
  // 内心 → 铃兰
  "内心": "bell",
  "情绪": "bell", "回忆": "dried", "梦境": "bell", "自省": "bell",
  // legacy
  "未分类": "wild",
};

const DOMAIN_LABEL_FALLBACK = "wild";

// 后端返回数组；保险起见也兼容字符串输入
function parseDomains(domain) {
  if (!domain) return [];
  if (Array.isArray(domain)) return domain.filter(Boolean).map((s) => String(s).trim());
  return String(domain).split(/[·,，\/]/).map((s) => s.trim()).filter(Boolean);
}

function getForm(domain) {
  const list = parseDomains(domain);
  if (!list.length) return DOMAIN_LABEL_FALLBACK;
  // 取第一个能映射到形态的 domain
  for (const d of list) {
    if (DOMAIN_FORM[d]) return DOMAIN_FORM[d];
  }
  return DOMAIN_LABEL_FALLBACK;
}

function parseTags(tags) {
  if (!tags) return [];
  if (Array.isArray(tags)) return tags;
  return String(tags).split(/[,，]/).map((s) => s.trim()).filter(Boolean);
}

// ───── 工具 ─────
function fmtRelative(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  const now = new Date();
  const diff = (now - d) / 1000;
  if (diff < 60) return "刚刚";
  if (diff < 3600) return Math.floor(diff / 60) + " 分钟前";
  if (diff < 86400) return Math.floor(diff / 3600) + " 小时前";
  if (diff < 86400 * 7) return Math.floor(diff / 86400) + " 天前";
  if (diff < 86400 * 30) return Math.floor(diff / 86400 / 7) + " 周前";
  if (diff < 86400 * 365) return Math.floor(diff / 86400 / 30) + " 月前";
  return Math.floor(diff / 86400 / 365) + " 年前";
}

function fmtDate(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  const m = (d.getMonth() + 1).toString().padStart(2, "0");
  const day = d.getDate().toString().padStart(2, "0");
  return `${d.getFullYear()}.${m}.${day}`;
}

// 去掉 Obsidian 双链标记 [[关键词]] -> 关键词
// 后端 _apply_wikilinks 给 content 自动加双链，前端展示要剥掉
function stripWikilinks(s) {
  if (!s) return "";
  return String(s).replace(/\[\[([^\[\]]+?)\]\]/g, "$1");
}

// 卡片/tooltip/聚合面板的预览文本：
//   统一用 content 的前若干字（preview 字段，由后端 list 接口截取）
//   summary 留在 API 里以备将来，但 UI 不再用 —— LLM 提炼跟 content 嚼烂的米饭一样
function pickPreview(b) {
  if (!b) return "";
  return stripWikilinks(b.preview || b.content || "");
}

// importance 1-10 → 1-5 心
function impToHearts(imp) {
  if (imp == null) return 1;
  const v = Math.max(1, Math.min(10, imp));
  return Math.max(1, Math.round(v / 2));
}

// valence → mood 段
function moodOf(v) {
  if (v == null) v = 0.5;
  if (v >= 0.8)  return { key: "bliss",   label: "超幸福",   color: "oklch(0.78 0.13 30)",  pale: "oklch(0.92 0.07 28)" };
  if (v >= 0.65) return { key: "happy",   label: "开心",     color: "oklch(0.82 0.10 50)",  pale: "oklch(0.94 0.06 50)" };
  if (v >= 0.5)  return { key: "calm",    label: "平静",     color: "oklch(0.82 0.06 130)", pale: "oklch(0.93 0.04 130)" };
  if (v >= 0.35) return { key: "tender",  label: "有点难过", color: "oklch(0.72 0.07 290)", pale: "oklch(0.90 0.04 290)" };
  return                { key: "sad",     label: "难过",     color: "oklch(0.62 0.05 250)", pale: "oklch(0.86 0.03 250)" };
}

function arousalLabel(a) {
  if (a == null) return "—";
  if (a >= 0.7) return "激";
  if (a >= 0.45) return "稳";
  return "静";
}

// ───── 植物 SVG ─────
// 每种 form 对应一种几何画法，用 css 变量挂上 mood 颜色
function PlantIcon({ valence = 0.5, arousal = 0.5, importance = 5, resolved = true, domain = "", size = 56, secondaryDomain = null }) {
  const form = getForm(domain);
  const mood = moodOf(valence);
  const hearts = impToHearts(importance);
  const heightFactor = 0.6 + (hearts / 5) * 0.4;   // 1心矮、5心高
  const flowerScale  = 0.7 + (hearts / 5) * 0.4;
  const swayDur = 6 - arousal * 3.5; // 高 arousal 摇得快
  const stem = "oklch(0.45 0.08 145)";
  const leaf = "oklch(0.58 0.12 145)";
  const leafDeep = "oklch(0.46 0.10 148)";

  return (
    <svg width={size} height={size} viewBox="0 0 60 60" style={{ overflow: "visible" }}>
      <defs>
        <radialGradient id={`petal-${mood.key}`} cx="50%" cy="50%">
          <stop offset="0%" stopColor={mood.pale} />
          <stop offset="100%" stopColor={mood.color} />
        </radialGradient>
      </defs>

      {/* 地面 */}
      <ellipse cx="30" cy="56" rx="14" ry="1.2" fill="oklch(0.85 0.02 90)" opacity="0.5" />

      {/* form 分支 */}
      {form === "camellia"  && <Camellia  hf={heightFactor} fs={flowerScale} mood={mood} stem={stem} leaf={leaf} swayDur={swayDur} />}
      {form === "bell"      && <Bell      hf={heightFactor} fs={flowerScale} mood={mood} stem={stem} leaf={leaf} swayDur={swayDur} />}
      {form === "dandelion" && <Dandelion hf={heightFactor} fs={flowerScale} mood={mood} stem={stem} leaf={leaf} swayDur={swayDur} />}
      {form === "clover"    && <Clover    hf={heightFactor} fs={flowerScale} mood={mood} stem={stem} leaf={leaf} swayDur={swayDur} />}
      {form === "tendril"   && <Tendril   hf={heightFactor} fs={flowerScale} mood={mood} stem={stem} leaf={leaf} swayDur={swayDur} />}
      {form === "lavender"  && <Lavender  hf={heightFactor} fs={flowerScale} mood={mood} stem={stem} leaf={leaf} swayDur={swayDur} />}
      {form === "iris"      && <Iris      hf={heightFactor} fs={flowerScale} mood={mood} stem={stem} leaf={leaf} swayDur={swayDur} />}
      {form === "fern"      && <Fern      hf={heightFactor} fs={flowerScale} mood={mood} stem={stem} leaf={leafDeep} swayDur={swayDur} />}
      {form === "wheat"     && <Wheat     hf={heightFactor} fs={flowerScale} mood={mood} stem={stem} leaf={leaf} swayDur={swayDur} />}
      {form === "dried"     && <Dried     hf={heightFactor} fs={flowerScale} mood={mood} stem={stem} leaf={leaf} swayDur={swayDur} />}
      {form === "wild"      && <Wild      hf={heightFactor} fs={flowerScale} mood={mood} stem={stem} leaf={leaf} swayDur={swayDur} />}

      {/* 未解决：萤火虫光点 */}
      {!resolved && (
        <g style={{ animation: "firefly 2.6s ease-in-out infinite" }}>
          <circle cx="44" cy="14" r="1.6" fill="oklch(0.92 0.18 90)" opacity="0.95" />
          <circle cx="44" cy="14" r="3.5" fill="oklch(0.92 0.18 90)" opacity="0.35" />
        </g>
      )}
    </svg>
  );
}

// ────── 各种植物形态 ──────
function StemPath({ d, stem, swayDur, children }) {
  // iOS Safari 不支持 SVG <g> 上的 transformOrigin 内联样式；改用 CSS class + duration 变量
  return (
    <g className="plant-sway-g" style={{ "--sway-dur": `${swayDur}s` }}>
      <path d={d} stroke={stem} strokeWidth="1.4" fill="none" strokeLinecap="round" />
      {children}
    </g>
  );
}

function Camellia({ hf, fs, mood, stem, leaf, swayDur }) {
  const top = 56 - 44 * hf;
  return (
    <StemPath d={`M30 56 Q 28 ${30+(56-top)/2}, 30 ${top}`} stem={stem} swayDur={swayDur}>
      <ellipse cx="22" cy={56 - 18 * hf} rx="6" ry="2.5" fill={leaf} transform={`rotate(-25 22 ${56 - 18 * hf})`} />
      <ellipse cx="38" cy={56 - 28 * hf} rx="5" ry="2.2" fill={leaf} transform={`rotate(30 38 ${56 - 28 * hf})`} />
      <g transform={`translate(30 ${top}) scale(${fs})`}>
        {[0, 60, 120, 180, 240, 300].map((a, i) => (
          <ellipse key={i} cx="0" cy="-4" rx="3.2" ry="5.5" fill={`url(#petal-${mood.key})`} opacity="0.95" transform={`rotate(${a})`} />
        ))}
        {[30, 90, 150, 210, 270, 330].map((a, i) => (
          <ellipse key={i} cx="0" cy="-2.5" rx="2.5" ry="4" fill={mood.pale} opacity="0.95" transform={`rotate(${a})`} />
        ))}
        <circle cx="0" cy="0" r="1.8" fill="oklch(0.85 0.13 80)" />
      </g>
    </StemPath>
  );
}

function Bell({ hf, fs, mood, stem, leaf, swayDur }) {
  const top = 56 - 42 * hf;
  return (
    <StemPath d={`M30 56 Q 30 40, 32 ${top + 6} Q 32 ${top}, 30 ${top}`} stem={stem} swayDur={swayDur}>
      <ellipse cx="22" cy={56 - 14 * hf} rx="7" ry="2.5" fill={leaf} transform={`rotate(-20 22 ${56 - 14 * hf})`} />
      {/* 三个朝下的铃铛 */}
      {[0, 1, 2].map((i) => {
        const yy = top + 4 + i * 7;
        return (
          <g key={i} transform={`translate(${33 + i * 0.5} ${yy}) scale(${fs})`}>
            <path d="M -3 0 Q -4 5, 0 6 Q 4 5, 3 0 Z" fill={`url(#petal-${mood.key})`} opacity="0.92" />
            <path d="M -1 0 L 0 -2 L 1 0" stroke={stem} strokeWidth="0.6" fill="none" />
          </g>
        );
      })}
    </StemPath>
  );
}

function Dandelion({ hf, fs, mood, stem, leaf, swayDur }) {
  const top = 56 - 44 * hf;
  return (
    <StemPath d={`M30 56 Q 30 36, 30 ${top}`} stem={stem} swayDur={swayDur}>
      <path d={`M 22 52 Q 26 50, 26 46`} stroke={leaf} strokeWidth="1.5" fill="none" strokeLinecap="round" />
      <path d={`M 38 50 Q 34 48, 34 44`} stroke={leaf} strokeWidth="1.5" fill="none" strokeLinecap="round" />
      <g transform={`translate(30 ${top}) scale(${fs})`}>
        <circle cx="0" cy="0" r="1.5" fill={mood.color} />
        {Array.from({ length: 12 }).map((_, i) => {
          const a = (i * 30 * Math.PI) / 180;
          const r = 6;
          return <line key={i} x1="0" y1="0" x2={Math.cos(a) * r} y2={Math.sin(a) * r}
            stroke={mood.pale} strokeWidth="0.8" opacity="0.85" />;
        })}
        {Array.from({ length: 12 }).map((_, i) => {
          const a = (i * 30 * Math.PI) / 180;
          const r = 6.5;
          return <circle key={i} cx={Math.cos(a) * r} cy={Math.sin(a) * r} r="0.9" fill={mood.color} opacity="0.95" />;
        })}
      </g>
    </StemPath>
  );
}

function Clover({ hf, fs, mood, stem, leaf, swayDur }) {
  const top = 56 - 28 * hf;
  return (
    <StemPath d={`M30 56 Q 30 46, 30 ${top}`} stem={stem} swayDur={swayDur}>
      {/* 三叶 */}
      <ellipse cx="24" cy={top + 4} rx="4" ry="3" fill={leaf} />
      <ellipse cx="36" cy={top + 4} rx="4" ry="3" fill={leaf} />
      <ellipse cx="30" cy={top - 1} rx="4" ry="3" fill={leaf} />
      {/* 小花 */}
      <g transform={`translate(30 ${top - 6}) scale(${fs})`}>
        {[0, 72, 144, 216, 288].map((a, i) => (
          <circle key={i} cx="0" cy="-3" r="1.8" fill={`url(#petal-${mood.key})`} transform={`rotate(${a})`} />
        ))}
        <circle cx="0" cy="0" r="1.2" fill="oklch(0.9 0.13 80)" />
      </g>
    </StemPath>
  );
}

function Tendril({ hf, fs, mood, stem, leaf, swayDur }) {
  const top = 56 - 38 * hf;
  return (
    <StemPath d={`M30 56 Q 24 44, 32 36 Q 38 28, 30 ${top}`} stem={stem} swayDur={swayDur}>
      {/* 卷须 */}
      <path d={`M 30 ${top} q -4 -2, -2 -6 q 4 -1, 2 -5`} stroke={leaf} strokeWidth="1.2" fill="none" strokeLinecap="round" />
      <ellipse cx="35" cy={56 - 22 * hf} rx="3" ry="1.5" fill={leaf} transform={`rotate(40 35 ${56 - 22 * hf})`} />
      {/* 嫩芽 */}
      <g transform={`translate(28 ${top - 8}) scale(${fs})`}>
        <ellipse cx="0" cy="0" rx="2" ry="4" fill={mood.pale} />
        <ellipse cx="0" cy="-1" rx="1.2" ry="3" fill={mood.color} opacity="0.85" />
      </g>
    </StemPath>
  );
}

function Lavender({ hf, fs, mood, stem, leaf, swayDur }) {
  const top = 56 - 46 * hf;
  return (
    <StemPath d={`M30 56 Q 30 36, 30 ${top}`} stem={stem} swayDur={swayDur}>
      <ellipse cx="24" cy={56 - 12 * hf} rx="5" ry="1.6" fill={leaf} transform={`rotate(-30 24 ${56 - 12 * hf})`} />
      {/* 穗状花序 */}
      <g transform={`translate(30 ${top + 4}) scale(${fs})`}>
        {[0, 1, 2, 3, 4].map((i) => (
          <g key={i} transform={`translate(0 ${-i * 3})`}>
            <ellipse cx={-1.5} cy="0" rx="1.4" ry="1.8" fill={`url(#petal-${mood.key})`} opacity="0.9" />
            <ellipse cx={1.5} cy="-1" rx="1.4" ry="1.8" fill={`url(#petal-${mood.key})`} opacity="0.9" />
          </g>
        ))}
      </g>
    </StemPath>
  );
}

function Iris({ hf, fs, mood, stem, leaf, swayDur }) {
  const top = 56 - 44 * hf;
  return (
    <StemPath d={`M30 56 Q 30 36, 30 ${top}`} stem={stem} swayDur={swayDur}>
      <path d={`M 25 ${56 - 16 * hf} L 28 ${56 - 32 * hf}`} stroke={leaf} strokeWidth="2" strokeLinecap="round" />
      <path d={`M 35 ${56 - 18 * hf} L 32 ${56 - 30 * hf}`} stroke={leaf} strokeWidth="2" strokeLinecap="round" />
      <g transform={`translate(30 ${top}) scale(${fs})`}>
        {/* 三片下垂 + 三片直立 */}
        <path d="M 0 0 Q -7 4, -4 8 Q 0 6, 0 0" fill={`url(#petal-${mood.key})`} opacity="0.95" />
        <path d="M 0 0 Q 7 4, 4 8 Q 0 6, 0 0" fill={`url(#petal-${mood.key})`} opacity="0.95" />
        <path d="M 0 0 Q -3 -7, 0 -8 Q 3 -7, 0 0" fill={mood.pale} opacity="0.95" />
        <ellipse cx="-3" cy="-3" rx="2" ry="4" fill={mood.color} opacity="0.85" transform="rotate(-30 -3 -3)" />
        <ellipse cx="3" cy="-3" rx="2" ry="4" fill={mood.color} opacity="0.85" transform="rotate(30 3 -3)" />
        <circle cx="0" cy="0" r="1.2" fill="oklch(0.92 0.13 80)" />
      </g>
    </StemPath>
  );
}

function Fern({ hf, fs, mood, stem, leaf, swayDur }) {
  const top = 56 - 42 * hf;
  return (
    <StemPath d={`M30 56 Q 30 36, 30 ${top}`} stem={stem} swayDur={swayDur}>
      {/* 蕨叶：左右对称小叶片 */}
      {[0, 1, 2, 3, 4, 5].map((i) => {
        const y = 54 - i * (8 * hf);
        const w = 8 - i * 1.1;
        if (w < 1.5) return null;
        return (
          <g key={i}>
            <ellipse cx={30 - w * 0.7} cy={y} rx={w} ry="1.6" fill={leaf} opacity="0.9" transform={`rotate(${-25} ${30 - w * 0.7} ${y})`} />
            <ellipse cx={30 + w * 0.7} cy={y} rx={w} ry="1.6" fill={leaf} opacity="0.9" transform={`rotate(${25} ${30 + w * 0.7} ${y})`} />
          </g>
        );
      })}
      {/* 顶端孢子点 */}
      <circle cx="30" cy={top} r={1.5 * fs} fill={mood.color} opacity="0.85" />
    </StemPath>
  );
}

function Wheat({ hf, fs, mood, stem, leaf, swayDur }) {
  // 麦穗：顶端有一串成对的小麦粒，下方两片叶
  const top = 56 - 44 * hf;
  const earStart = top + 4;             // 麦粒起始位置
  const earEnd   = top + 22 * hf;       // 麦粒终止位置
  const grainCount = 6;
  const goldA = "oklch(0.78 0.13 80)";  // 浅金
  const goldB = "oklch(0.68 0.15 70)";  // 深金
  return (
    <StemPath d={`M30 56 Q 31 38, 30 ${top}`} stem={stem} swayDur={swayDur}>
      {/* 下方两片细长叶子 */}
      <path d={`M30 44 Q 18 40, 14 30`} stroke={leaf} strokeWidth="1.6" fill="none" strokeLinecap="round" opacity="0.85" />
      <path d={`M30 48 Q 44 44, 48 36`} stroke={leaf} strokeWidth="1.6" fill="none" strokeLinecap="round" opacity="0.85" />

      {/* 麦穗：成对的小麦粒沿茎排列 */}
      {Array.from({ length: grainCount }).map((_, i) => {
        const t = i / (grainCount - 1);
        const y = earEnd - t * (earEnd - earStart);
        const off = 2.6 * fs;
        const len = 4 * fs;
        const wid = 1.6 * fs;
        const isTop = i >= grainCount - 2;
        const c = isTop ? goldA : goldB;
        return (
          <g key={i}>
            {/* 左麦粒 */}
            <ellipse cx={30 - off} cy={y} rx={wid} ry={len} fill={c}
              transform={`rotate(-22 ${30 - off} ${y})`} opacity="0.95" />
            {/* 右麦粒 */}
            <ellipse cx={30 + off} cy={y} rx={wid} ry={len} fill={c}
              transform={`rotate(22 ${30 + off} ${y})`} opacity="0.95" />
            {/* 麦芒 */}
            <line x1={30 - off - 1} y1={y - len} x2={30 - off - 3} y2={y - len - 4}
              stroke={c} strokeWidth="0.5" opacity="0.6" />
            <line x1={30 + off + 1} y1={y - len} x2={30 + off + 3} y2={y - len - 4}
              stroke={c} strokeWidth="0.5" opacity="0.6" />
          </g>
        );
      })}

      {/* 顶端的小尖（让麦穗有头） */}
      <line x1="30" y1={top} x2="30" y2={top - 5 * fs} stroke={goldA} strokeWidth="0.7" opacity="0.7" />
      <circle cx="30" cy={top} r={0.8 * fs} fill={mood.color} opacity="0.7" />
    </StemPath>
  );
}

function Dried({ hf, fs, mood, stem, leaf, swayDur }) {
  const top = 56 - 40 * hf;
  // 干花：色调淡一点
  const dryColor = "oklch(0.78 0.05 50)";
  return (
    <StemPath d={`M30 56 Q 30 36, 30 ${top}`} stem={"oklch(0.55 0.05 60)"} swayDur={swayDur}>
      <path d={`M 22 ${56 - 18 * hf} L 28 ${56 - 28 * hf}`} stroke={dryColor} strokeWidth="1" />
      <path d={`M 38 ${56 - 20 * hf} L 32 ${56 - 30 * hf}`} stroke={dryColor} strokeWidth="1" />
      <g transform={`translate(30 ${top}) scale(${fs})`}>
        {[0, 60, 120, 180, 240, 300].map((a, i) => (
          <ellipse key={i} cx="0" cy="-3" rx="2" ry="4" fill={mood.pale} opacity="0.7" stroke={dryColor} strokeWidth="0.4" transform={`rotate(${a})`} />
        ))}
        <circle cx="0" cy="0" r="1.2" fill={dryColor} />
      </g>
    </StemPath>
  );
}

function Wild({ hf, fs, mood, stem, leaf, swayDur }) {
  const top = 56 - 38 * hf;
  return (
    <StemPath d={`M30 56 Q 28 40, 30 ${top}`} stem={stem} swayDur={swayDur}>
      <ellipse cx="24" cy={56 - 14 * hf} rx="5" ry="2" fill={leaf} transform={`rotate(-20 24 ${56 - 14 * hf})`} />
      <g transform={`translate(30 ${top}) scale(${fs})`}>
        {[0, 90, 180, 270].map((a, i) => (
          <ellipse key={i} cx="0" cy="-3" rx="2.5" ry="4" fill={`url(#petal-${mood.key})`} opacity="0.9" transform={`rotate(${a})`} />
        ))}
        <circle cx="0" cy="0" r="1.4" fill="oklch(0.9 0.13 80)" />
      </g>
    </StemPath>
  );
}

// ───── 心心刻度 ─────
function HeartMeter({ value = 1, max = 5, size = 11 }) {
  return (
    <span style={{ display: "inline-flex", gap: "2px", alignItems: "center" }}>
      {Array.from({ length: max }).map((_, i) => {
        const filled = i < value;
        return (
          <svg key={i} width={size} height={size} viewBox="0 0 16 16">
            <path
              d="M8 14 C 3 10.5, 1 8, 1 5.2 C 1 3, 2.5 1.5, 4.5 1.5 C 6 1.5, 7.2 2.5, 8 4 C 8.8 2.5, 10 1.5, 11.5 1.5 C 13.5 1.5, 15 3, 15 5.2 C 15 8, 13 10.5, 8 14 Z"
              fill={filled ? "oklch(0.68 0.13 22)" : "transparent"}
              stroke={filled ? "oklch(0.55 0.13 22)" : "oklch(0.78 0.04 22)"}
              strokeWidth="1.2"
            />
          </svg>
        );
      })}
    </span>
  );
}

// ───── 标签 chip ─────
function Tag({ children, active, onClick, tone = "moss", size = "md" }) {
  const tones = {
    moss:  { fg: "var(--moss-deep)", bg: "oklch(0.94 0.025 150)", border: "oklch(0.84 0.045 150)" },
    rose:  { fg: "var(--rose-deep)", bg: "oklch(0.95 0.035 22)",  border: "oklch(0.86 0.06 22)" },
    amber: { fg: "oklch(0.42 0.10 70)", bg: "oklch(0.96 0.05 80)", border: "oklch(0.86 0.09 75)" },
    plain: { fg: "var(--ink-soft)", bg: "transparent",            border: "var(--line)" },
  };
  const t = tones[tone] || tones.moss;
  const activeStyle = active ? { background: t.fg, color: "var(--paper)", borderColor: t.fg } : {};
  const padding = size === "sm" ? "2px 8px" : "3px 10px";
  const fs = size === "sm" ? "0.7rem" : "0.75rem";
  return (
    <button
      onClick={onClick}
      style={{
        fontFamily: "var(--sans)", fontSize: fs, fontWeight: 500,
        color: t.fg, background: t.bg, border: `1px solid ${t.border}`,
        borderRadius: "999px", padding, cursor: onClick ? "pointer" : "default",
        letterSpacing: "0.02em", whiteSpace: "nowrap",
        transition: "all 0.18s ease", ...activeStyle,
      }}
    >{children}</button>
  );
}

// ───── 统计带 ─────
function StatBand({ buckets }) {
  const stats = useMemo(() => {
    const avgValence = buckets.length ? buckets.reduce((s, b) => s + (b.valence || 0), 0) / buckets.length : 0;
    const unresolved = buckets.filter((b) => !b.resolved).length;
    const importantUnresolved = buckets.filter((b) => !b.resolved && (b.importance || 0) >= 7).length;
    const permanent = buckets.filter((b) => b.type === "permanent").length;
    const recent = buckets.filter((b) => {
      if (!b.last_active) return false;
      return (new Date() - new Date(b.last_active)) / 86400000 < 7;
    }).length;
    return { avgValence, unresolved, importantUnresolved, recent, permanent, bucketCount: buckets.length };
  }, [buckets]);

  const moodNow = moodOf(stats.avgValence);

  const items = [
    { label: "记忆桶", value: stats.bucketCount, sub: stats.permanent > 0 ? `其中 ${stats.permanent} 株是永久` : "都还在生长" },
    { label: "平均心情", value: moodNow.label, sub: `valence · ${(stats.avgValence * 100).toFixed(0)}` },
    { label: "还在生长", value: stats.unresolved, sub: stats.importantUnresolved > 0 ? `其中 ${stats.importantUnresolved} 项要紧` : "都是小事" },
    { label: "本周触碰", value: stats.recent, sub: stats.recent > 0 ? "还在被记起" : "近期安静" },
  ];

  return (
    <div className="stat-band" style={{
      display: "grid", gridTemplateColumns: "repeat(4, 1fr)",
      borderTop: "1px solid var(--line)", borderBottom: "1px solid var(--line)",
      background: "oklch(0.985 0.012 88)",
    }}>
      {items.map((item, i) => (
        <div key={i} className="stat-cell" style={{
          padding: "20px 28px",
          borderRight: i < items.length - 1 ? "1px solid var(--line)" : "none",
          display: "flex", flexDirection: "column", gap: "4px",
        }}>
          <div style={{ fontFamily: "var(--mono)", fontSize: "0.68rem", color: "var(--ink-faint)", letterSpacing: "0.12em", textTransform: "uppercase" }}>{item.label}</div>
          <div style={{ fontFamily: "var(--serif)", fontSize: "2.1rem", fontWeight: 500, color: "var(--moss-deep)", lineHeight: 1, marginTop: "4px" }}>{item.value}</div>
          <div style={{ fontFamily: "var(--kai)", fontSize: "0.82rem", color: "var(--ink-soft)", marginTop: "2px" }}>{item.sub}</div>
        </div>
      ))}
    </div>
  );
}

// ───── Hero ─────
function Hero({ source }) {
  const today = new Date();
  const dateStr = `${today.getFullYear()} 年 ${today.getMonth() + 1} 月 ${today.getDate()} 日`;
  return (
    <header className="hero-pad" style={{ position: "relative", overflow: "hidden" }}>
      {/* 装饰：右侧的植物角落 */}
      <HeroOrnament />

      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", marginBottom: "8px", gap: "12px", flexWrap: "wrap", position: "relative", zIndex: 2 }}>
        <div style={{ fontFamily: "var(--mono)", fontSize: "0.7rem", color: "var(--ink-faint)", letterSpacing: "0.18em", textTransform: "uppercase" }}>
          memory · garden · for two
        </div>
        <div style={{ fontFamily: "var(--kai)", fontSize: "0.85rem", color: "var(--ink-faint)" }}>
          {dateStr} · {source === "live" ? "已连接" : source === "error" ? "未连接" : "正在加载"}
          <span style={{
            display: "inline-block", width: 6, height: 6, borderRadius: "50%",
            background: source === "live" ? "var(--leaf-bright)" : source === "error" ? "var(--rose)" : "var(--amber)",
            marginLeft: 8, verticalAlign: "middle",
          }} />
        </div>
      </div>
      <h1 className="hero-title" style={{
        fontFamily: "var(--serif)", fontWeight: 500, fontStyle: "italic",
        margin: 0, lineHeight: 1.0,
        color: "var(--moss-deep)", letterSpacing: "-0.01em",
        whiteSpace: "nowrap", position: "relative", zIndex: 2,
      }}>
        记忆花园
      </h1>
      <div className="hero-sub" style={{
        fontFamily: "var(--kai)", fontSize: "1rem",
        color: "var(--ink-faint)", letterSpacing: "0.06em",
        marginTop: "10px", position: "relative", zIndex: 2,
      }}>—— 我们俩共同打理的那块地方</div>
    </header>
  );
}

// 右侧的植物剪影 + 飞舞的小花瓣 —— 安静、不抢标题
function HeroOrnament() {
  const moss   = "oklch(0.45 0.08 145)";
  const mossL  = "oklch(0.62 0.10 145)";
  const leaf   = "oklch(0.70 0.13 130)";
  const rose   = "oklch(0.72 0.13 25)";
  const peach  = "oklch(0.82 0.10 50)";
  const cream  = "oklch(0.92 0.06 80)";
  return (
    <div className="hero-ornament" aria-hidden="true" style={{
      position: "absolute",
      right: "0", top: "0", bottom: "0",
      width: "min(46%, 520px)",
      pointerEvents: "none",
      zIndex: 1,
    }}>
      <svg viewBox="0 0 520 240" preserveAspectRatio="xMaxYMid meet"
        style={{ position: "absolute", right: "0", top: "0", height: "100%", width: "100%" }}>
        <defs>
          <radialGradient id="hero-glow" cx="80%" cy="40%" r="50%">
            <stop offset="0%" stopColor="oklch(0.95 0.06 70 / 0.5)" />
            <stop offset="100%" stopColor="oklch(0.95 0.06 70 / 0)" />
          </radialGradient>
          <linearGradient id="hero-petal" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={cream} />
            <stop offset="100%" stopColor={peach} />
          </linearGradient>
        </defs>

        {/* 暖晕 */}
        <ellipse cx="420" cy="100" rx="180" ry="120" fill="url(#hero-glow)" opacity="0.7" />

        {/* 弧形拱门：一根藤从右下蜿蜒到右上，框出标题 */}
        <path
          d="M 520 230 C 380 220, 360 130, 440 60 S 520 -10, 520 -10"
          stroke={moss} strokeWidth="1.4" fill="none" strokeLinecap="round"
          opacity="0.75"
        />
        {/* 第二根更细的藤，错位平行 */}
        <path
          d="M 510 240 C 410 200, 380 120, 470 50"
          stroke={mossL} strokeWidth="0.9" fill="none" strokeLinecap="round"
          strokeDasharray="2 2" opacity="0.5"
        />

        {/* 藤上的叶子 */}
        {[
          { cx: 460, cy: 200, rx: 12, ry: 4, rot: -25 },
          { cx: 405, cy: 168, rx: 14, ry: 5, rot: 30 },
          { cx: 388, cy: 130, rx: 10, ry: 4, rot: -50 },
          { cx: 410, cy: 95,  rx: 12, ry: 4, rot: 60 },
          { cx: 450, cy: 60,  rx: 10, ry: 3.5, rot: -35 },
          { cx: 488, cy: 30,  rx: 9,  ry: 3, rot: 40 },
        ].map((l, i) => (
          <ellipse key={i} cx={l.cx} cy={l.cy} rx={l.rx} ry={l.ry}
            fill={i % 2 ? leaf : mossL}
            transform={`rotate(${l.rot} ${l.cx} ${l.cy})`}
            opacity="0.85" />
        ))}

        {/* 一朵主花：山茶风 */}
        <g transform="translate(415 145)">
          {[0, 72, 144, 216, 288].map((a, i) => (
            <ellipse key={i} cx="0" cy="-8" rx="6" ry="10"
              fill="url(#hero-petal)"
              transform={`rotate(${a})`}
              opacity="0.95" />
          ))}
          <circle cx="0" cy="0" r="3" fill={rose} />
          <circle cx="0" cy="0" r="1.2" fill="oklch(0.4 0.12 30)" />
        </g>

        {/* 几朵小花苞 */}
        <g transform="translate(478 78)">
          <circle cx="0" cy="0" r="3.5" fill={peach} opacity="0.9" />
          <circle cx="0" cy="0" r="1.5" fill={rose} />
          <line x1="0" y1="2" x2="-2" y2="10" stroke={mossL} strokeWidth="0.6" />
        </g>
        <g transform="translate(440 105)">
          <circle cx="0" cy="0" r="2.5" fill={cream} opacity="0.95" />
          <circle cx="0" cy="0" r="1" fill={peach} />
        </g>

        {/* 飞舞的花瓣（飘散） */}
        {[
          { cx: 350, cy: 80,  rot: 20  },
          { cx: 320, cy: 130, rot: -30 },
          { cx: 295, cy: 60,  rot: 50  },
          { cx: 270, cy: 100, rot: -10 },
        ].map((p, i) => (
          <ellipse key={`p${i}`}
            cx={p.cx} cy={p.cy} rx="3" ry="5"
            fill={cream}
            transform={`rotate(${p.rot} ${p.cx} ${p.cy})`}
            opacity={0.55 - i * 0.08}
            className="petal-float"
            style={{ animationDelay: `${i * 0.6}s` }}
          />
        ))}

        {/* 蕨类小卷须，从顶端伸出 */}
        <path
          d="M 480 25 Q 472 18, 478 12 Q 484 6, 478 2"
          stroke={mossL} strokeWidth="0.7" fill="none" strokeLinecap="round"
          opacity="0.7"
        />
      </svg>
    </div>
  );
}

// ───── 视图切换器 ─────
function ViewSwitcher({ view, setView }) {
  const tabs = [
    { id: "garden",   label: "花园",   sub: "情感坐标" },
    { id: "timeline", label: "藤蔓",   sub: "按月生长" },
    { id: "grid",     label: "卡片",   sub: "全部记忆桶" },
  ];
  return (
    <div className="view-switcher" style={{
      display: "flex", borderBottom: "1px solid var(--line)", background: "var(--paper)",
      overflowX: "auto", WebkitOverflowScrolling: "touch", scrollbarWidth: "none",
    }}>
      {tabs.map((t) => {
        const active = view === t.id;
        return (
          <button key={t.id} onClick={() => setView(t.id)}
            style={{
              background: "transparent", border: "none",
              borderBottom: active ? "2px solid var(--moss-deep)" : "2px solid transparent",
              padding: "14px 16px 14px 0", marginRight: "24px", cursor: "pointer",
              display: "flex", alignItems: "baseline", gap: "8px",
              fontFamily: "var(--serif)", flexShrink: 0,
              color: active ? "var(--moss-deep)" : "var(--ink-faint)",
              transition: "all 0.18s", whiteSpace: "nowrap",
            }}>
            <span style={{ fontSize: "1.3rem", fontWeight: 500, letterSpacing: "0.02em" }}>{t.label}</span>
            <span style={{ fontFamily: "var(--mono)", fontSize: "0.62rem", letterSpacing: "0.1em", textTransform: "uppercase", opacity: active ? 0.7 : 0.5 }}>{t.sub}</span>
          </button>
        );
      })}
    </div>
  );
}

// 全局动画
const swayCSS = `
@keyframes plant-sway { 0% { transform: rotate(-2.5deg); } 100% { transform: rotate(3deg); } }
.plant-sway-g {
  transform-box: fill-box;
  transform-origin: 50% 100%;
  animation: plant-sway var(--sway-dur, 4s) ease-in-out infinite alternate;
  will-change: transform;
}
@keyframes firefly {
  0%, 100% { opacity: 0.4; transform: translate(0,0); }
  50% { opacity: 1; transform: translate(2px, -3px); }
}
@keyframes fade-in { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
.fade-in { animation: fade-in 0.4s ease-out backwards; }

@keyframes petal-float {
  0%   { transform: translate(0,0) rotate(0deg); opacity: 0.55; }
  50%  { transform: translate(-6px, -10px) rotate(20deg); opacity: 0.8; }
  100% { transform: translate(0,0) rotate(0deg); opacity: 0.55; }
}
.petal-float {
  transform-box: fill-box;
  transform-origin: center;
  animation: petal-float 6s ease-in-out infinite;
}

.hero-pad { padding: 44px 56px 28px; }
.hero-title { font-size: 4.2rem; }
.hero-sub { padding-left: 0; }
.view-switcher { padding: 0 56px; }
.view-switcher::-webkit-scrollbar { display: none; }
.stat-band { grid-template-columns: repeat(4, 1fr) !important; }
.legend-block { display: grid; gap: 8px 18px; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); }

@media (max-width: 720px) {
  .hero-pad { padding: 28px 20px 18px; }
  .hero-title { font-size: 2.8rem; letter-spacing: 0; }
  .hero-sub { font-size: 0.88rem !important; margin-top: 6px !important; }
  .hero-ornament { width: 38% !important; opacity: 0.55; }
  .view-switcher { padding: 0 20px; }
  .stat-band { grid-template-columns: repeat(2, 1fr) !important; }
  .stat-cell { padding: 14px 16px !important; }
  .stat-cell:nth-child(2) { border-right: none !important; }
  .stat-cell:nth-child(1), .stat-cell:nth-child(2) { border-bottom: 1px solid var(--line); }
  .legend-block { grid-template-columns: repeat(2, 1fr); }
  .garden-pad { padding: 20px 12px 32px !important; }
  .garden-stage { height: 520px !important; }
}
.garden-pad { padding: 24px 56px 56px; }
`;
{
  const styleEl = document.createElement("style");
  styleEl.textContent = swayCSS;
  document.head.appendChild(styleEl);
}

Object.assign(window, {
  fmtRelative, fmtDate, parseDomains, parseTags, getForm,
  impToHearts, moodOf, arousalLabel, pickPreview, stripWikilinks,
  PlantIcon, HeartMeter, Tag, StatBand, Hero, ViewSwitcher,
});
