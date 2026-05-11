// 主 App

const { useState: useStateA, useEffect: useEffectA } = React;

const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "defaultView": "garden",
  "showStats": true,
  "warmth": 50
}/*EDITMODE-END*/;

function App() {
  const [tweaks, setTweak] = window.useTweaks ? window.useTweaks(TWEAK_DEFAULTS) : [TWEAK_DEFAULTS, () => {}];
  const [buckets, setBuckets] = useStateA([]);
  const [source, setSource] = useStateA("loading");
  const [errMsg, setErrMsg] = useStateA("");
  const [view, setView] = useStateA(tweaks.defaultView || "garden");
  const [openId, setOpenId] = useStateA(null);
  const [openCluster, setOpenCluster] = useStateA(null);

  const loadBuckets = React.useCallback(() => {
    setSource("loading");
    setErrMsg("");
    window.api.fetchBuckets().then(({ source, buckets, error }) => {
      setBuckets(buckets || []);
      setSource(source);
      if (error) setErrMsg(error);
    });
  }, []);

  useEffectA(() => { loadBuckets(); }, [loadBuckets]);

  useEffectA(() => {
    const w = (tweaks.warmth ?? 50) / 100;
    const lightness = 0.97 + w * 0.005;
    const chroma = 0.008 + w * 0.012;
    document.documentElement.style.setProperty("--paper", `oklch(${lightness} ${chroma} 88)`);
  }, [tweaks.warmth]);

  return (
    <div>
      <Hero source={source} />
      {tweaks.showStats !== false && <StatBand buckets={buckets} />}
      <ViewSwitcher view={view} setView={setView} />

      {source === "loading" ? (
        <div style={{ padding: "120px 0", textAlign: "center", fontFamily: "var(--serif)", fontStyle: "italic", color: "var(--ink-faint)" }}>
          正在唤醒花园…
        </div>
      ) : source === "error" ? (
        <div style={{ padding: "100px 24px", textAlign: "center", maxWidth: "440px", margin: "0 auto" }}>
          <div style={{ fontFamily: "var(--serif)", fontStyle: "italic", fontSize: "1.4rem", color: "var(--moss-deep)", marginBottom: "12px" }}>
            还连不上花园
          </div>
          <div style={{ fontFamily: "var(--kai)", fontSize: "0.92rem", color: "var(--ink-soft)", lineHeight: 1.7, marginBottom: "20px" }}>
            没能从 memory.fanfan.party 拿到记忆桶。
            <br />
            可能是后端没启动、登录失效、或者跨域被拦了。
          </div>
          {errMsg && (
            <div style={{ fontFamily: "var(--mono)", fontSize: "0.7rem", color: "var(--ink-faint)", marginBottom: "20px", letterSpacing: "0.04em" }}>
              {errMsg}
            </div>
          )}
          <button onClick={loadBuckets} style={{
            fontFamily: "var(--serif)", fontStyle: "italic", fontSize: "0.95rem",
            padding: "8px 20px", background: "var(--moss-deep)", color: "var(--paper)",
            border: "none", borderRadius: "999px", cursor: "pointer", letterSpacing: "0.04em",
          }}>再试一次</button>
        </div>
      ) : view === "garden" ? (
        <GardenView buckets={buckets} onOpen={setOpenId} onOpenCluster={setOpenCluster} />
      ) : view === "timeline" ? (
        <TimelineView buckets={buckets} onOpen={setOpenId} />
      ) : (
        <GridView buckets={buckets} onOpen={setOpenId} />
      )}

      {openCluster && !openId && (
        <ClusterPanel
          cluster={openCluster}
          onOpen={(id) => { setOpenId(id); setOpenCluster(null); }}
          onClose={() => setOpenCluster(null)}
        />
      )}
      {openId && <DetailDrawer bucketId={openId} onClose={() => setOpenId(null)} />}

      {window.TweaksPanel && (
        <window.TweaksPanel title="花园偏好">
          <window.TweakSection label="默认视图" />
          <window.TweakRadio
            label="开始于"
            value={tweaks.defaultView}
            onChange={(v) => { setTweak("defaultView", v); setView(v); }}
            options={["garden", "timeline", "grid"]}
          />
          <window.TweakSection label="界面" />
          <window.TweakToggle
            label="显示统计带"
            value={tweaks.showStats}
            onChange={(v) => setTweak("showStats", v)}
          />
          <window.TweakSlider
            label="纸张暖度"
            value={tweaks.warmth}
            onChange={(v) => setTweak("warmth", v)}
            min={0} max={100} step={5} unit="%"
          />
        </window.TweaksPanel>
      )}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
