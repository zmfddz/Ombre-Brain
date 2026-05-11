# 测试新加的脱水产物解析逻辑（不需要真 API）
import sys, os, asyncio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

from utils import load_config
from dehydrator import Dehydrator
from bucket_manager import BucketManager

config = load_config()
d = Dehydrator(config)

# 1. _parse_structured: 标准 JSON
raw1 = '''{"core_facts": ["事实A", "事实B"], "emotion_state": "复杂", "todos": ["买菜", "回信"], "keywords": ["周末", "雨"], "summary": "下雨天在家"}'''
r1 = d._parse_structured(raw1)
assert r1["summary"] == "下雨天在家", f"summary 错: {r1['summary']}"
assert r1["core_facts"] == ["事实A", "事实B"], f"core_facts 错: {r1['core_facts']}"
assert r1["todos"] == ["买菜", "回信"], f"todos 错: {r1['todos']}"
assert r1["keywords"] == ["周末", "雨"], f"keywords 错: {r1['keywords']}"
assert r1["emotion_state"] == "复杂"
print("[OK] 标准 JSON 解析")

# 2. markdown 代码块包裹
raw2 = '```json\n{"summary": "test", "todos": ["a"]}\n```'
r2 = d._parse_structured(raw2)
assert r2["summary"] == "test"
assert r2["todos"] == ["a"]
print("[OK] markdown 代码块包裹")

# 3. 非法 JSON → 默认值
raw3 = "this is not json"
r3 = d._parse_structured(raw3)
assert r3 == {"summary": "", "core_facts": [], "todos": [], "keywords": [], "emotion_state": ""}
print("[OK] 非法 JSON 走默认")

# 4. 部分字段缺失
raw4 = '{"todos": ["只有 todo"]}'
r4 = d._parse_structured(raw4)
assert r4["todos"] == ["只有 todo"]
assert r4["summary"] == ""
assert r4["core_facts"] == []
print("[OK] 部分字段缺失")

# 5. 数组里有空值/None
raw5 = '{"todos": ["有效", "", null, "另一个"]}'
r5 = d._parse_structured(raw5)
assert r5["todos"] == ["有效", "另一个"], f"过滤空值失败: {r5['todos']}"
print("[OK] 过滤空值")

# 6. 超长字段截断
long_summary = "x" * 500
raw6 = f'{{"summary": "{long_summary}"}}'
r6 = d._parse_structured(raw6)
assert len(r6["summary"]) == 200, f"summary 应该被截到 200，实际 {len(r6['summary'])}"
print("[OK] 长字段截断")

# 7. extract_structured 没 API 时安全返回
async def test_no_api():
    d_noapi = Dehydrator({"dehydration": {"api_key": ""}})
    r = await d_noapi.extract_structured("hello")
    assert r == {"summary": "", "core_facts": [], "todos": [], "keywords": [], "emotion_state": ""}
    print("[OK] 无 API 时安全降级")
asyncio.run(test_no_api())

# 8. bucket_manager 写入并读回新字段
async def test_create_with_extract():
    bm = BucketManager(config)
    bid = await bm.create(
        content="测试内容",
        tags=["t1"],
        domain=["日常"],
        summary="一句话总结",
        core_facts=["事实1", "事实2"],
        todos=["待办1"],
        keywords=["关键词1"],
        emotion_state="平静",
    )
    bucket = await bm.get(bid)
    meta = bucket["metadata"]
    assert meta["summary"] == "一句话总结"
    assert meta["core_facts"] == ["事实1", "事实2"]
    assert meta["todos"] == ["待办1"]
    assert meta["keywords"] == ["关键词1"]
    assert meta["emotion_state"] == "平静"
    print("[OK] bucket create 写入新字段")

    # update 改 todos
    await bm.update(bid, todos=["更新的待办"])
    bucket2 = await bm.get(bid)
    assert bucket2["metadata"]["todos"] == ["更新的待办"]
    print("[OK] bucket update 改新字段")

    # 老字段还在
    assert bucket2["metadata"]["tags"] == ["t1"]
    print("[OK] 老字段不丢")

    # 清理
    await bm.delete(bid)

asyncio.run(test_create_with_extract())

print("\n全部通过 ✨")
