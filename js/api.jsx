// API 层 —— 直连 memory.fanfan.party
// 后端字段：
//   id, name, domain[], valence, arousal, importance, tags[], resolved,
//   type ("dynamic"|"permanent"|"archive"), weight, created, last_active
//   单桶详情多一个 content (整段文字)
const API_BASE = (() => {
  const h = window.location.hostname;
  if (h.includes("memory.fanfan.party")) return "";
  return "https://memory.fanfan.party";
})();

async function tryFetch(path) {
  const res = await fetch(API_BASE + path, { credentials: "include" });
  if (!res.ok) throw new Error("status " + res.status);
  return await res.json();
}

async function fetchBuckets() {
  try {
    const real = await tryFetch("/api/buckets");
    if (Array.isArray(real)) return { source: "live", buckets: real };
    if (real && Array.isArray(real.buckets)) return { source: "live", buckets: real.buckets };
    return { source: "error", buckets: [], error: "返回结构不对" };
  } catch (e) {
    return { source: "error", buckets: [], error: e.message || "连接失败" };
  }
}

async function fetchBucket(id) {
  try {
    const real = await tryFetch(`/api/bucket/${id}`);
    if (real && !real.error) return { source: "live", bucket: real };
    return { source: "error", bucket: null, error: real?.error || "返回结构不对" };
  } catch (e) {
    return { source: "error", bucket: null, error: e.message || "连接失败" };
  }
}

async function fetchStats() {
  try {
    const real = await tryFetch("/api/stats");
    if (real && !real.error) return { source: "live", stats: real };
    return { source: "error", stats: null };
  } catch (e) {
    return { source: "error", stats: null };
  }
}

window.api = { fetchBuckets, fetchBucket, fetchStats };
