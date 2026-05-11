# ============================================================
# Module: MCP Server Entry Point (server.py)
# 模块：MCP 服务器主入口
#
# Starts the Ombre Brain MCP service and registers memory
# operation tools for Claude to call.
# 启动 Ombre Brain MCP 服务，注册记忆操作工具供 Claude 调用。
#
# Core responsibilities:
# 核心职责：
#   - Initialize config, bucket manager, dehydrator, decay engine
#     初始化配置、记忆桶管理器、脱水器、衰减引擎
#   - Expose 5 MCP tools:
#     暴露 5 个 MCP 工具：
#       breath — Surface unresolved memories or search by keyword
#                浮现未解决记忆 或 按关键词检索
#       hold   — Store a single memory
#                存储单条记忆
#       grow   — Diary digest, auto-split into multiple buckets
#                日记归档，自动拆分多桶
#       trace  — Modify metadata / resolved / delete
#                修改元数据 / resolved 标记 / 删除
#       pulse  — System status + bucket listing
#                系统状态 + 所有桶列表
#
# Startup:
# 启动方式：
#   Local:  python server.py
#   Remote: OMBRE_TRANSPORT=streamable-http python server.py
#   Docker: docker-compose up
# ============================================================

import os
import sys
import random
import logging
import asyncio
import httpx

# --- Ensure same-directory modules can be imported ---
# --- 确保同目录下的模块能被正确导入 ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP

from bucket_manager import BucketManager
from dehydrator import Dehydrator
from decay_engine import DecayEngine
from utils import load_config, setup_logging

# --- Load config & init logging / 加载配置 & 初始化日志 ---
config = load_config()
setup_logging(config.get("log_level", "INFO"))
logger = logging.getLogger("ombre_brain")

# --- Initialize three core components / 初始化三大核心组件 ---
bucket_mgr = BucketManager(config)                  # Bucket manager / 记忆桶管理器
dehydrator = Dehydrator(config)                      # Dehydrator / 脱水器
decay_engine = DecayEngine(config, bucket_mgr)       # Decay engine / 衰减引擎

# --- Create MCP server instance / 创建 MCP 服务器实例 ---
# host="0.0.0.0" so Docker container's SSE is externally reachable
# stdio mode ignores host (no network)
mcp = FastMCP(
    "Ombre Brain",
    host="0.0.0.0",
    port=8000,
)

import re as _re_inline  # for inlining jsx into the home page

# Cache the inlined HTML so we only build it once per process
_INLINED_HOME_HTML: str | None = None

def _build_inlined_home() -> str:
    """
    Read index.html and inline every <script type="text/babel" src="..."> tag
    so the home page is fully self-contained — zero child requests.

    Why: Cloudflare Access treats sub-resource fetches (Babel's XHR for jsx) as
    API calls and challenges them with 302, even when the parent page already
    passed auth via cookie. Inlining the jsx avoids the second challenge entirely.
    """
    base = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base, "index.html"), "r", encoding="utf-8") as f:
        html = f.read()

    # Matches both <script type="text/babel" src="js/api.jsx"></script> and
    # the bare <script type="text/babel" src="tweaks-panel.jsx"></script> form.
    pattern = _re_inline.compile(
        r'<script\s+type="text/babel"\s+src="([^"]+)"\s*></script>'
    )

    def replace(match):
        src = match.group(1)
        # Only inline same-directory paths (defense against weird srcs)
        if src.startswith("http://") or src.startswith("https://") or src.startswith("//"):
            return match.group(0)
        file_path = os.path.join(base, src)
        # Resolve and bound-check
        real = os.path.realpath(file_path)
        if not real.startswith(os.path.realpath(base) + os.sep):
            return match.group(0)
        if not os.path.isfile(real):
            return match.group(0)
        with open(real, "r", encoding="utf-8") as ff:
            jsx_body = ff.read()
        # Comment marker helps when you View Source / debug
        return (
            f'<script type="text/babel" data-inlined-from="{src}">\n'
            f'{jsx_body}\n'
            f'</script>'
        )

    return pattern.sub(replace, html)


@mcp.custom_route("/", methods=["GET"])
async def serve_home(request):
    """记忆花园首页(memory-garden 前端) —— 内联所有 jsx,规避 CF Access 二次拦截"""
    from starlette.responses import HTMLResponse
    global _INLINED_HOME_HTML
    if _INLINED_HOME_HTML is None:
        _INLINED_HOME_HTML = _build_inlined_home()
    return HTMLResponse(_INLINED_HOME_HTML)


@mcp.custom_route("/legacy", methods=["GET"])
async def serve_legacy_home(request):
    """老版 landing 页(我们的小家),保留访问入口便于回退"""
    from starlette.responses import HTMLResponse
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index_legacy.html")
    if not os.path.isfile(html_path):
        return HTMLResponse("Legacy page not found", status_code=404)
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


# ── Static asset routes for memory-garden frontend ──
# memory-garden 前端的静态资源路由(jsx / 顶层 jsx 文件)
# Babel-in-browser 会从这些路径 fetch 源码再做 JSX → JS 转译
_STATIC_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _safe_static_file(rel_path: str, allowed_dirs: tuple) -> str | None:
    """
    Resolve a static file path safely, preventing directory traversal.
    Returns absolute path or None if disallowed.
    """
    base = os.path.realpath(_STATIC_BASE_DIR)
    target = os.path.realpath(os.path.join(base, rel_path))
    if not target.startswith(base + os.sep) and target != base:
        return None
    rel = os.path.relpath(target, base).replace(os.sep, "/")
    if not any(rel == d or rel.startswith(d + "/") for d in allowed_dirs):
        return None
    if not os.path.isfile(target):
        return None
    return target


@mcp.custom_route("/js/{filename}", methods=["GET"])
async def serve_jsx_module(request):
    """记忆花园的 jsx 模块(api / atoms / garden / grid / timeline / detail / app)"""
    from starlette.responses import FileResponse, JSONResponse
    filename = request.path_params["filename"]
    target = _safe_static_file(f"js/{filename}", allowed_dirs=("js",))
    if not target:
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(target, media_type="application/javascript")


@mcp.custom_route("/tweaks-panel.jsx", methods=["GET"])
async def serve_tweaks_panel(request):
    """顶层 tweaks-panel.jsx"""
    from starlette.responses import FileResponse, JSONResponse
    target = _safe_static_file("tweaks-panel.jsx", allowed_dirs=("tweaks-panel.jsx",))
    if not target:
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(target, media_type="application/javascript")

# =============================================================
# /health endpoint: lightweight keepalive
# 轻量保活接口
# For Cloudflare Tunnel or reverse proxy to ping, preventing idle timeout
# 供 Cloudflare Tunnel 或反代定期 ping，防止空闲超时断连
# =============================================================
@mcp.custom_route("/health", methods=["GET"])
async def health_check(request):
    from starlette.responses import JSONResponse
    try:
        stats = await bucket_mgr.get_stats()
        return JSONResponse({
            "status": "ok",
            "buckets": stats["permanent_count"] + stats["dynamic_count"],
            "decay_engine": "running" if decay_engine.is_running else "stopped",
        })
    except Exception as e:
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)


# =============================================================
# REST API endpoints for frontend (fanfan.party)
# 前端 API 接口
# =============================================================
import re as _re_wl
_WIKILINK_RE = _re_wl.compile(r"\[\[([^\[\]]+?)\]\]")
def _strip_wikilinks(s: str) -> str:
    """[[关键词]] -> 关键词. Used for previews and any user-facing text."""
    return _WIKILINK_RE.sub(r"\1", s or "")


@mcp.custom_route("/api/buckets", methods=["GET"])
async def api_list_buckets(request):
    """列出所有记忆桶的摘要信息"""
    from starlette.responses import JSONResponse
    try:
        include_archive = request.query_params.get("archive", "false").lower() == "true"
        buckets = await bucket_mgr.list_all(include_archive=include_archive)
        result = []
        for b in buckets:
            meta = b.get("metadata", {})
            score = decay_engine.calculate_score(meta)
            # --- Preview text: prefer LLM summary; fallback to content head ---
            # --- 卡片预览：优先 LLM 脱水的 summary，没有就回退到 content 截取 ---
            # 老桶没有 summary 字段，这条 fallback 保证它们也能在列表显示一段预览
            content_stripped = _strip_wikilinks(b.get("content", ""))
            preview = content_stripped[:120] + ("…" if len(content_stripped) > 120 else "")
            result.append({
                "id": b["id"],
                "name": meta.get("name", "未命名"),
                "domain": meta.get("domain", []),
                "valence": meta.get("valence", 0.5),
                "arousal": meta.get("arousal", 0.3),
                "importance": meta.get("importance", 5),
                "tags": meta.get("tags", []),
                "resolved": meta.get("resolved", False),
                "type": meta.get("type", "dynamic"),
                "weight": round(score, 2),
                "created": meta.get("created", ""),
                "last_active": meta.get("last_active", ""),
                # --- Dehydration outputs (list view uses summary for preview) ---
                # --- 脱水产物（列表用 summary 做预览） ---
                "summary": _strip_wikilinks(meta.get("summary", "")),
                "preview": preview,  # 兜底预览：content 截取 + 去 wikilinks
                "todos_count": len(meta.get("todos", []) or []),
            })
        result.sort(key=lambda x: x["weight"], reverse=True)
        return JSONResponse(result, headers={"Access-Control-Allow-Origin": "*"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/bucket/{bucket_id}", methods=["GET"])
async def api_get_bucket(request):
    """获取单条记忆桶的完整内容"""
    from starlette.responses import JSONResponse
    bucket_id = request.path_params["bucket_id"]
    try:
        bucket = await bucket_mgr.get(bucket_id)
        if not bucket:
            return JSONResponse({"error": "未找到"}, status_code=404)
        meta = bucket.get("metadata", {})
        score = decay_engine.calculate_score(meta)
        return JSONResponse({
            "id": bucket["id"],
            "content": bucket.get("content", ""),
            "name": meta.get("name", "未命名"),
            "domain": meta.get("domain", []),
            "valence": meta.get("valence", 0.5),
            "arousal": meta.get("arousal", 0.3),
            "importance": meta.get("importance", 5),
            "tags": meta.get("tags", []),
            "resolved": meta.get("resolved", False),
            "type": meta.get("type", "dynamic"),
            "weight": round(score, 2),
            "created": meta.get("created", ""),
            "last_active": meta.get("last_active", ""),
            # --- Dehydration outputs ---
            # --- 脱水产物 ---
            # core_facts / keywords / emotion_state 留在 metadata 里以备将来，但 API 不返回
            # core_facts 跟 content 是嚼烂的米饭，前端用 content 即可
            "summary": _strip_wikilinks(meta.get("summary", "")),
            "todos": meta.get("todos", []) or [],
        }, headers={"Access-Control-Allow-Origin": "*"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/stats", methods=["GET"])
async def api_stats(request):
    """系统状态"""
    from starlette.responses import JSONResponse
    try:
        stats = await bucket_mgr.get_stats()
        return JSONResponse({
            "permanent_count": stats["permanent_count"],
            "dynamic_count": stats["dynamic_count"],
            "archive_count": stats["archive_count"],
            "total_size_kb": stats["total_size_kb"],
            "decay_running": decay_engine.is_running,
        }, headers={"Access-Control-Allow-Origin": "*"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# =============================================================
# Internal helper: merge-or-create
# 内部辅助：检查是否可合并，可以则合并，否则新建
# Shared by hold and grow to avoid duplicate logic
# hold 和 grow 共用，避免重复逻辑
# =============================================================
async def _merge_or_create(
    content: str,
    tags: list,
    importance: int,
    domain: list,
    valence: float,
    arousal: float,
    name: str = "",
    # --- Dehydration extraction (optional; pre-computed by caller) ---
    # --- 脱水产物（调用方预先算好传入） ---
    extracted: dict = None,
) -> tuple[str, bool]:
    """
    Check if a similar bucket exists for merging; merge if so, create if not.
    Returns (bucket_id_or_name, is_merged).
    检查是否有相似桶可合并，有则合并，无则新建。
    返回 (桶ID或名称, 是否合并)。
    """
    extracted = extracted or {}
    try:
        existing = await bucket_mgr.search(content, limit=1)
    except Exception as e:
        logger.warning(f"Search for merge failed, creating new / 合并搜索失败，新建: {e}")
        existing = []

    if existing and existing[0].get("score", 0) > config.get("merge_threshold", 75):
        bucket = existing[0]
        try:
            merged = await dehydrator.merge(bucket["content"], content)
            # --- Re-extract structured fields from the merged content ---
            # --- 合并后重新提取脱水产物（todos 等可能因新内容变化） ---
            try:
                merged_extracted = await dehydrator.extract_structured(merged)
            except Exception as e:
                logger.warning(f"Re-extract after merge failed / 合并后重提脱水失败: {e}")
                merged_extracted = {}
            update_kwargs = {
                "content": merged,
                "tags": list(set(bucket["metadata"].get("tags", []) + tags)),
                "importance": max(bucket["metadata"].get("importance", 5), importance),
                "domain": list(set(bucket["metadata"].get("domain", []) + domain)),
                "valence": valence,
                "arousal": arousal,
            }
            # Only overwrite dehydration fields if new extraction produced non-empty
            # 仅在重新提取有产物时覆盖原有脱水字段
            if merged_extracted.get("summary"):
                update_kwargs["summary"] = merged_extracted["summary"]
            if merged_extracted.get("core_facts"):
                update_kwargs["core_facts"] = merged_extracted["core_facts"]
            if merged_extracted.get("todos"):
                # Union with existing todos to avoid losing in-flight items
                # 与已有 todos 取并集，避免丢失进行中事项
                old_todos = bucket["metadata"].get("todos", []) or []
                seen = set()
                union = []
                for t in old_todos + merged_extracted["todos"]:
                    if t and t not in seen:
                        seen.add(t)
                        union.append(t)
                update_kwargs["todos"] = union
            if merged_extracted.get("keywords"):
                update_kwargs["keywords"] = merged_extracted["keywords"]
            if merged_extracted.get("emotion_state"):
                update_kwargs["emotion_state"] = merged_extracted["emotion_state"]

            await bucket_mgr.update(bucket["id"], **update_kwargs)
            return bucket["metadata"].get("name", bucket["id"]), True
        except Exception as e:
            logger.warning(f"Merge failed, creating new / 合并失败，新建: {e}")

    bucket_id = await bucket_mgr.create(
        content=content,
        tags=tags,
        importance=importance,
        domain=domain,
        valence=valence,
        arousal=arousal,
        name=name or None,
        summary=extracted.get("summary", ""),
        core_facts=extracted.get("core_facts", []),
        todos=extracted.get("todos", []),
        keywords=extracted.get("keywords", []),
        emotion_state=extracted.get("emotion_state", ""),
    )
    return bucket_id, False


# =============================================================
# Tool 1: breath — Breathe
# 工具 1：breath — 呼吸
#
# No args: surface highest-weight unresolved memories (active push)
# 无参数：浮现权重最高的未解决记忆
# With args: search by keyword + emotion coordinates
# 有参数：按关键词+情感坐标检索记忆
# =============================================================
@mcp.tool()
async def breath(
    query: str = "",
    max_results: int = 3,
    domain: str = "",
    valence: float = -1,
    arousal: float = -1,
) -> str:
    """检索记忆或浮现未解决记忆。query 为空时自动推送权重最高的未解决桶（每次对话开头用这个）；有 query 时按关键词检索（用关键词而非整句话，更准）。domain 逗号分隔可缩小范围，valence/arousal 传 0~1 启用情感共鸣，-1 忽略。max_results 默认3，需要更多可调大。"""
    await decay_engine.ensure_started()

    # --- No args: surfacing mode (weight pool active push) ---
    # --- 无参数：浮现模式（权重池主动推送）---
    if not query.strip():
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
        except Exception as e:
            logger.error(f"Failed to list buckets for surfacing / 浮现列桶失败: {e}")
            return "记忆系统暂时无法访问。"

        unresolved = [
            b for b in all_buckets
            if not b["metadata"].get("resolved", False)
            and b["metadata"].get("type") != "permanent"
        ]
        if not unresolved:
            return "权重池平静，没有需要处理的记忆。"

        scored = sorted(
            unresolved,
            key=lambda b: decay_engine.calculate_score(b["metadata"]),
            reverse=True,
        )
        top = scored[:2]
        results = []
        for b in top:
            try:
                summary = await dehydrator.dehydrate(b["content"], b["metadata"])
                await bucket_mgr.touch(b["id"])
                score = decay_engine.calculate_score(b["metadata"])
                results.append(f"[权重:{score:.2f}] {summary}")
            except Exception as e:
                logger.warning(f"Failed to dehydrate surfaced bucket / 浮现脱水失败: {e}")
                continue
        if not results:
            return "权重池平静，没有需要处理的记忆。"
        return "=== 浮现记忆 ===\n" + "\n---\n".join(results)

    # --- With args: search mode / 有参数：检索模式 ---
    domain_filter = [d.strip() for d in domain.split(",") if d.strip()] or None
    q_valence = valence if 0 <= valence <= 1 else None
    q_arousal = arousal if 0 <= arousal <= 1 else None

    try:
        matches = await bucket_mgr.search(
            query,
            limit=max_results,
            domain_filter=domain_filter,
            query_valence=q_valence,
            query_arousal=q_arousal,
        )
    except Exception as e:
        logger.error(f"Search failed / 检索失败: {e}")
        return "检索过程出错，请稍后重试。"

    results = []
    for bucket in matches:
        try:
            summary = await dehydrator.dehydrate(bucket["content"], bucket["metadata"])
            await bucket_mgr.touch(bucket["id"])
            results.append(summary)
        except Exception as e:
            logger.warning(f"Failed to dehydrate search result / 检索结果脱水失败: {e}")
            continue

    # --- Random surfacing: when search returns < 3, 40% chance to float old memories ---
    # --- 随机浮现：检索结果不足 3 条时，40% 概率从低权重旧桶里漂上来 ---
    if len(matches) < 3 and random.random() < 0.4:
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
            matched_ids = {b["id"] for b in matches}
            low_weight = [
                b for b in all_buckets
                if b["id"] not in matched_ids
                and decay_engine.calculate_score(b["metadata"]) < 2.0
            ]
            if low_weight:
                drifted = random.sample(low_weight, min(random.randint(1, 3), len(low_weight)))
                drift_results = []
                for b in drifted:
                    summary = await dehydrator.dehydrate(b["content"], b["metadata"])
                    drift_results.append(f"[surface_type: random]\n{summary}")
                results.append("--- 忽然想起来 ---\n" + "\n---\n".join(drift_results))
        except Exception as e:
            logger.warning(f"Random surfacing failed / 随机浮现失败: {e}")

    if not results:
        return "未找到相关记忆。"

    return "\n---\n".join(results)


# =============================================================
# Tool 2: hold — Hold on to this
# 工具 2：hold — 握住，留下来
# =============================================================
@mcp.tool()
async def hold(
    content: str,
    tags: str = "",
    importance: int = 5,
) -> str:
    """存储单条记忆。自动打标+合并相似桶。tags 逗号分隔，importance 1-10（里程碑8-10，日常3-5）。一句话的事用 hold，一大段用 grow。"""
    await decay_engine.ensure_started()

    # --- Input validation / 输入校验 ---
    if not content or not content.strip():
        return "内容为空，无法存储。"

    importance = max(1, min(10, importance))
    extra_tags = [t.strip() for t in tags.split(",") if t.strip()]

    # --- Step 1: auto-tagging + structured extraction in parallel ---
    # --- 自动打标 + 结构化脱水提取（并行调用 LLM，少等一轮）---
    try:
        analysis, extracted = await asyncio.gather(
            dehydrator.analyze(content),
            dehydrator.extract_structured(content),
        )
    except Exception as e:
        logger.warning(f"Auto-tagging or extraction failed, using defaults / 打标或脱水失败: {e}")
        analysis = {
            "domain": ["未分类"], "valence": 0.5, "arousal": 0.3,
            "tags": [], "suggested_name": "",
        }
        extracted = {
            "summary": "", "core_facts": [], "todos": [],
            "keywords": [], "emotion_state": "",
        }

    domain = analysis["domain"]
    valence = analysis["valence"]
    arousal = analysis["arousal"]
    auto_tags = analysis["tags"]
    suggested_name = analysis.get("suggested_name", "")

    all_tags = list(dict.fromkeys(auto_tags + extra_tags))

    # --- Step 2: merge or create / 合并或新建 ---
    result_name, is_merged = await _merge_or_create(
        content=content,
        tags=all_tags,
        importance=importance,
        domain=domain,
        valence=valence,
        arousal=arousal,
        name=suggested_name,
        extracted=extracted,
    )

    if is_merged:
        return (
            f"已合并到现有记忆桶: {result_name}\n"
            f"主题域: {', '.join(domain)} | 情感: V{valence:.1f}/A{arousal:.1f}"
        )
    return (
        f"已创建新记忆桶: {result_name}\n"
        f"主题域: {', '.join(domain)} | 情感: V{valence:.1f}/A{arousal:.1f} | 标签: {', '.join(all_tags)}"
    )


# =============================================================
# Tool 3: grow — Grow, fragments become memories
# 工具 3：grow — 生长，一天的碎片长成记忆
# =============================================================
@mcp.tool()
async def grow(content: str) -> str:
    """日记归档。自动拆分长内容为多个记忆桶。"""
    await decay_engine.ensure_started()

    if not content or not content.strip():
        return "内容为空，无法整理。"

    # --- Step 1: let API split and organize / 让 API 拆分整理 ---
    try:
        items = await dehydrator.digest(content)
    except Exception as e:
        logger.error(f"Diary digest failed / 日记整理失败: {e}")
        return f"日记整理失败: {e}"

    if not items:
        return "内容为空或整理失败。"

    results = []
    created = 0
    merged = 0

    # --- Step 2: parallel structured extraction for all items ---
    # --- 并行做所有 item 的脱水提取（一次过，避免逐条串行等 LLM）---
    try:
        extracted_list = await asyncio.gather(
            *[dehydrator.extract_structured(item["content"]) for item in items],
            return_exceptions=True,
        )
    except Exception as e:
        logger.warning(f"Batch extraction failed / 批量脱水失败: {e}")
        extracted_list = [None] * len(items)

    # --- Step 3: merge or create each item (with per-item error handling) ---
    # --- 逐条合并或新建（单条失败不影响其他）---
    for item, item_extracted in zip(items, extracted_list):
        if isinstance(item_extracted, Exception) or item_extracted is None:
            item_extracted = {
                "summary": "", "core_facts": [], "todos": [],
                "keywords": [], "emotion_state": "",
            }
        try:
            result_name, is_merged = await _merge_or_create(
                content=item["content"],
                tags=item.get("tags", []),
                importance=item.get("importance", 5),
                domain=item.get("domain", ["未分类"]),
                valence=item.get("valence", 0.5),
                arousal=item.get("arousal", 0.3),
                name=item.get("name", ""),
                extracted=item_extracted,
            )

            if is_merged:
                results.append(f"  📎 合并 → {result_name}")
                merged += 1
            else:
                domains_str = ",".join(item.get("domain", []))
                results.append(
                    f"  📝 新建 [{item.get('name', result_name)}] "
                    f"主题:{domains_str} V{item.get('valence', 0.5):.1f}/A{item.get('arousal', 0.3):.1f}"
                )
                created += 1
        except Exception as e:
            logger.warning(
                f"Failed to process diary item / 日记条目处理失败: "
                f"{item.get('name', '?')}: {e}"
            )
            results.append(f"  ⚠️ 失败: {item.get('name', '未知条目')}")

    summary = f"=== 日记整理完成 ===\n拆分为 {len(items)} 条 | 新建 {created} 桶 | 合并 {merged} 桶\n"
    return summary + "\n".join(results)


# =============================================================
# Tool 4: trace — Trace, redraw the outline of a memory
# 工具 4：trace — 描摹，重新勾勒记忆的轮廓
# Also handles deletion (delete=True)
# 同时承接删除功能
# =============================================================
@mcp.tool()
async def trace(
    bucket_id: str,
    name: str = "",
    domain: str = "",
    valence: float = -1,
    arousal: float = -1,
    importance: int = -1,
    tags: str = "",
    resolved: int = -1,
    delete: bool = False,
) -> str:
    """修改记忆元数据。bucket_id 通过 pulse 查看。resolved=1 标记已解决（桶权重骤降沉底），resolved=0 重新激活，delete=True 删除桶。其余字段只传需改的，-1 或空串表示不改。"""

    if not bucket_id or not bucket_id.strip():
        return "请提供有效的 bucket_id。"

    # --- Delete mode / 删除模式 ---
    if delete:
        success = await bucket_mgr.delete(bucket_id)
        return f"已遗忘记忆桶: {bucket_id}" if success else f"未找到记忆桶: {bucket_id}"

    bucket = await bucket_mgr.get(bucket_id)
    if not bucket:
        return f"未找到记忆桶: {bucket_id}"

    # --- Collect only fields actually passed / 只收集用户实际传入的字段 ---
    updates = {}
    if name:
        updates["name"] = name
    if domain:
        updates["domain"] = [d.strip() for d in domain.split(",") if d.strip()]
    if 0 <= valence <= 1:
        updates["valence"] = valence
    if 0 <= arousal <= 1:
        updates["arousal"] = arousal
    if 1 <= importance <= 10:
        updates["importance"] = importance
    if tags:
        updates["tags"] = [t.strip() for t in tags.split(",") if t.strip()]
    if resolved in (0, 1):
        updates["resolved"] = bool(resolved)

    if not updates:
        return "没有任何字段需要修改。"

    success = await bucket_mgr.update(bucket_id, **updates)
    if not success:
        return f"修改失败: {bucket_id}"

    changed = ", ".join(f"{k}={v}" for k, v in updates.items())
    # Explicit hint about resolved state change semantics
    # 特别提示 resolved 状态变化的语义
    if "resolved" in updates:
        if updates["resolved"]:
            changed += " → 已沉底，只在关键词触发时重新浮现"
        else:
            changed += " → 已重新激活，将参与浮现排序"
    return f"已修改记忆桶 {bucket_id}: {changed}"


# =============================================================
# Tool 5: pulse — Heartbeat, system status + memory listing
# 工具 5：pulse — 脉搏，系统状态 + 记忆列表
# =============================================================
@mcp.tool()
async def pulse(include_archive: bool = False) -> str:
    """系统状态和所有记忆桶摘要（含 bucket_id，供 trace 使用）。include_archive=True 时包含归档桶。"""
    try:
        stats = await bucket_mgr.get_stats()
    except Exception as e:
        return f"获取系统状态失败: {e}"

    status = (
        f"=== Ombre Brain 记忆系统 ===\n"
        f"固化记忆桶: {stats['permanent_count']} 个\n"
        f"动态记忆桶: {stats['dynamic_count']} 个\n"
        f"归档记忆桶: {stats['archive_count']} 个\n"
        f"总存储大小: {stats['total_size_kb']:.1f} KB\n"
        f"衰减引擎: {'运行中' if decay_engine.is_running else '已停止'}\n"
    )

    # --- List all bucket summaries / 列出所有桶摘要 ---
    try:
        buckets = await bucket_mgr.list_all(include_archive=include_archive)
    except Exception as e:
        return status + f"\n列出记忆桶失败: {e}"

    if not buckets:
        return status + "\n记忆库为空。"

    lines = []
    for b in buckets:
        meta = b.get("metadata", {})
        if meta.get("type") == "permanent":
            icon = "📦"
        elif meta.get("type") == "archived":
            icon = "🗄️"
        elif meta.get("resolved", False):
            icon = "✅"
        else:
            icon = "💭"
        try:
            score = decay_engine.calculate_score(meta)
        except Exception:
            score = 0.0
        domains = ",".join(meta.get("domain", []))
        val = meta.get("valence", 0.5)
        aro = meta.get("arousal", 0.3)
        resolved_tag = " [已解决]" if meta.get("resolved", False) else ""
        lines.append(
            f"{icon} [{meta.get('name', b['id'])}]{resolved_tag} "
            f"id:{b['id']} "
            f"主题:{domains} "
            f"情感:V{val:.1f}/A{aro:.1f} "
            f"重要:{meta.get('importance', '?')} "
            f"权重:{score:.2f} "
            f"标签:{','.join(meta.get('tags', []))}"
        )

    return status + "\n=== 记忆列表 ===\n" + "\n".join(lines)


# --- Entry point / 启动入口 ---
if __name__ == "__main__":
    transport = config.get("transport", "stdio")
    logger.info(f"Ombre Brain starting | transport: {transport}")

    # --- Application-level keepalive: remote mode only, ping /health every 60s ---
    # --- 应用层保活：仅远程模式下启动，每 60 秒 ping 一次 /health ---
    # Prevents Cloudflare Tunnel from dropping idle connections
    if transport in ("sse", "streamable-http"):
        async def _keepalive_loop():
            await asyncio.sleep(10)  # Wait for server to fully start
            async with httpx.AsyncClient() as client:
                while True:
                    try:
                        await client.get("http://localhost:8000/health", timeout=5)
                        logger.debug("Keepalive ping OK / 保活 ping 成功")
                    except Exception as e:
                        logger.warning(f"Keepalive ping failed / 保活 ping 失败: {e}")
                    await asyncio.sleep(60)

        import threading

        def _start_keepalive():
            loop = asyncio.new_event_loop()
            loop.run_until_complete(_keepalive_loop())

        t = threading.Thread(target=_start_keepalive, daemon=True)
        t.start()

    mcp.run(transport=transport)
