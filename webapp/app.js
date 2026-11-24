// Campus Solar Explorer - UI. Model logic lives in model.js.
(async function () {
  "use strict";
  const LAND = {
    "roof": { label: "Rooftops", color: getCss("--roof") },
    "garage top": { label: "Garage-top canopies", color: getCss("--garage") },
    "parking": { label: "Parking canopies", color: getCss("--parking") },
    "open land": { label: "Open land", color: getCss("--open") },
  };
  const STRATEGY_HINT = {
    land: "Rooftops first, then garage tops and parking, open land last; each site uses its densest design (east-west roofs, dense ground arrays).",
    cost: "Cheapest kWh first on any land: usually trackers on open land, then 10° roof rows.",
    optimal: "Cheapest kWh once open land carries a price (land value), and never exporting more than the grid cap to ERCOT.",
  };
  const PART_LABEL = { base: "base", scale: "size (economies of scale)", height: "height / crane", reroof: "re-roof R&R",
                       garage: "garage structure", interconnection: "interconnection", siteprep: "site prep & fencing" };
  const COST_LABEL = { roof: "Roof rows $/Wdc", roof_east_west: "Roof east-west $/Wdc", garage_canopy: "Garage canopy $/Wdc",
                       carport: "Parking canopy $/Wdc", ground_fixed: "Ground fixed $/Wdc", ground_tracker: "Tracker $/Wdc" };

  function getCss(v) { return getComputedStyle(document.documentElement).getPropertyValue(v).trim(); }
  const $ = id => document.getElementById(id);
  const fmt = (v, d = 1) => (v == null || !isFinite(v)) ? "–" : v.toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });

  // ---------------------------------------------------------------- state (URL hash overrides for sharing / tests)
  const C = await (await fetch("data/common.json")).json();
  const hash = Object.fromEntries(new URLSearchParams(location.hash.slice(1)));
  const S = {
    vintage: hash.vintage || "2025", scenario: hash.scenario || "central",
    strategy: hash.strategy || "land", targetPct: +(hash.target || 10),
    include: { "roof": true, "garage top": true, "parking": true, "open land": true },
    polo: true, allowEastWest: true, openDesigns: { fixed: true, tracker: true },
    usdPerW: { ...C.costs.usd_per_wdc }, costModel: JSON.parse(JSON.stringify(C.costs.model)), landValuePerAcre: C.land_value_usd_per_acre, maxExport: C.max_export_frac,
  };
  const SITES = {};
  async function sites() {
    if (!SITES[S.vintage]) SITES[S.vintage] = await (await fetch(`data/sites_${S.vintage}.json`)).json();
    return SITES[S.vintage];
  }

  // ---------------------------------------------------------------- map
  const map = L.map("map", { crs: L.CRS.Simple, minZoom: -3, maxZoom: 4, zoomSnap: 0.25, preferCanvas: true, attributionControl: false });
  const B = [[0, 0], [C.tile.size, C.tile.size]];
  L.imageOverlay("data/basemap.jpg", B).addTo(map);
  map.fitBounds(B);
  L.control.scale({ metric: true, imperial: false, maxWidth: 160 }).addTo(map);
  const renderer = L.canvas({ padding: 0.5 });
  let layers = {};       // site id -> polygon layer

  $("legend").innerHTML = Object.entries(LAND).map(([k, v]) =>
    `<div><span class="sw" style="background:${v.color}"></span>${v.label}</div>`).join("") +
    `<div><span class="sw out" style="border-color:#8a8984;border-style:dashed"></span>candidate, not selected</div>` +
    `<div style="color:#8a8984">partly built sites are lighter</div>`;

  function drawSites(list) {
    Object.values(layers).forEach(l => map.removeLayer(l));
    layers = {};
    for (const s of list) {
      const latlngs = s.geom.map(poly => poly.map(ring => ring.map(([x, y]) => [y, x])));
      const lyr = L.polygon(latlngs, { renderer, weight: 1, color: LAND[s.land].color, fillOpacity: 0.05 });
      lyr.on("click", e => L.popup().setLatLng(e.latlng).setContent(popup(s)).openOn(map));
      lyr.addTo(map);
      layers[s.id] = lyr;
    }
  }

  function popup(site) {
    const pick = lastSel && lastSel.chosen.find(c => c.site.id === site.id);
    const rows = site.options.map(o => {
      const m = Model.metrics(o, S, C, site);
      const on = pick && pick.opt === o;
      return `<tr${on ? ' style="font-weight:700"' : ""}><td>${o.design}</td><td>${fmt(o.kwdc, 0)} kWdc</td>` +
             `<td>${fmt(o.kwh / 1e3, 0)} MWh/yr</td><td>$${fmt(m.usdPerW, 2)}/W</td><td>${fmt(100 * m.lcoe, 1)} ¢/kWh</td></tr>`;
    }).join("");
    const i = site.info || {};
    const extra = [
      i.footprint_m2 ? `footprint ${fmt(i.footprint_m2, 0)} m²` : "", i.height_m ? `height ${fmt(i.height_m, 0)} m` : "",
      i.year ? `built ${i.year}` : "", i.shade_loss_pct != null ? `shade loss ${fmt(i.shade_loss_pct, 1)} %` : "",
      i.canopy_frac != null ? `tree cover ${fmt(100 * i.canopy_frac, 0)} %` : "", i.slope_deg != null ? `slope ${fmt(i.slope_deg, 1)}°` : "",
      i.canopy || "", i.source && !String(i.source).startsWith("LiDAR") ? i.source : "",
    ].filter(Boolean).join(" · ");
    const usable = site.options[0].area_m2;
    const shown = pick ? pick : { opt: site.options[0], m: Model.metrics(site.options[0], S, C, site) };
    const nz = Object.entries(shown.m.parts).filter(([k, v]) => Math.abs(v) > 1)
      .map(([k, v]) => `${PART_LABEL[k]} ${v < 0 ? "−" : ""}$${fmt(Math.abs(v) / 1e3, 0)}k`).join(" · ");
    return `<b>${site.name}</b> <span style="color:#52514e">${site.id} · ${LAND[site.land].label}</span><br>` +
      `usable ${fmt(usable, 0)} m²${extra ? " · " + extra : ""}<br>` +
      (pick ? `<span style="color:${LAND[site.land].color}">● selected${pick.frac < 0.999 ? ` (${fmt(100 * pick.frac, 0)} % of site)` : ""}</span>` : "not selected") +
      `<table style="margin-top:4px">${rows}</table>` +
      `<div style="color:#52514e;margin-top:3px">Capex (${shown.opt.design}): ${nz}</div>`;
  }

  // ---------------------------------------------------------------- controls
  for (const b of $("strategy").querySelectorAll("button")) {
    b.classList.toggle("on", b.dataset.v === S.strategy);
    b.onclick = () => { S.strategy = b.dataset.v; $("strategy").querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b)); update(true); };
  }
  $("include").innerHTML = Object.entries(LAND).map(([k, v]) =>
    `<label><input type="checkbox" data-k="${k}" checked><span class="sw" style="background:${v.color}"></span>${v.label}</label>`).join("") +
    `<label title="Polo Fields tract (in use as fields; Bonfire Memorial cut out)"><input type="checkbox" id="polo" checked>Polo Fields</label>`;
  $("include").querySelectorAll("input[data-k]").forEach(c => c.onchange = () => { S.include[c.dataset.k] = c.checked; update(true); });
  $("polo").onchange = e => { S.polo = e.target.checked; update(true); };

  const A = [];
  for (const [k, lab] of Object.entries(COST_LABEL)) A.push(`<span>${lab}</span><input type="number" step="0.05" min="0" data-cost="${k}" value="${S.usdPerW[k]}">`);
  const CM = S.costModel;
  const P = [  // [label, getter path, step]
    ["Size exponent (−20 %/10× ≈ 0.10)", ["scale", "exponent"], 0.01],
    ["Height adder $/W per m above 12 m", ["height", "usd_per_w_per_m"], 0.001],
    ["Re-roof R&R $/W (bldg ≥ 20 yr)", ["reroof", "usd_per_w"], 0.05],
    ["Garage structural $/W", ["garage", "usd_per_w"], 0.05],
    ["Interconnection, small site $ (< 250 kW)", ["interconnection", "lv_fixed_usd"], 10000],
    ["Interconnection, MV fixed $", ["interconnection", "mv_fixed_usd"], 10000],
    ["MV ductbank $/m (× dist. to substation)", ["interconnection", "mv_usd_per_m"], 50],
    ["Site prep & fencing $/m² (open land)", ["siteprep", "usd_per_m2"], 0.5],
  ];
  A.push(`<span style="grid-column:1/-1;margin-top:4px;font-weight:600">Cost model v2 (placeholders)</span>`);
  P.forEach(([lab, path, step], i) => A.push(`<span>${lab}</span><input type="number" step="${step}" min="0" data-cm="${i}" value="${CM[path[0]][path[1]]}">`));
  A.push(`<span style="grid-column:1/-1;margin-top:4px;font-weight:600">Strategy settings</span>`);
  A.push(`<span>Open-land value ($/acre, Optimal)</span><input type="number" step="25000" min="0" id="landValue" value="${S.landValuePerAcre}">`);
  A.push(`<span>Grid cap: max export (%, Optimal)</span><input type="number" step="1" min="0" max="100" id="maxExport" value="${100 * S.maxExport}">`);
  A.push(`<span>Allow east-west roof racks</span><input type="checkbox" id="ew" checked>`);
  A.push(`<span>Open land: fixed tilt / tracker</span><span><input type="checkbox" id="fx" checked> <input type="checkbox" id="tr" checked></span>`);
  $("assump").innerHTML = A.join("");
  $("assump").querySelectorAll("input[data-cost]").forEach(i => i.onchange = () => { S.usdPerW[i.dataset.cost] = +i.value; update(true); });
  $("assump").querySelectorAll("input[data-cm]").forEach(inp => inp.onchange = () => {
    const [a, b] = P[+inp.dataset.cm][1]; CM[a][b] = +inp.value; update(true); });
  $("landValue").onchange = e => { S.landValuePerAcre = +e.target.value; update(true); };
  $("maxExport").onchange = e => { S.maxExport = +e.target.value / 100; update(true); };
  $("ew").onchange = e => { S.allowEastWest = e.target.checked; update(true); };
  $("fx").onchange = e => { S.openDesigns.fixed = e.target.checked; update(true); };
  $("tr").onchange = e => { S.openDesigns.tracker = e.target.checked; update(true); };
  $("vintage").value = S.vintage; $("scenario").value = S.scenario;
  $("vintage").onchange = async e => { S.vintage = e.target.value; drawSites(await sites()); update(true); };
  $("scenario").onchange = e => { S.scenario = e.target.value; update(true); };
  $("target").oninput = e => { S.targetPct = +e.target.value; schedule(); };
  $("barMetric").onchange = () => renderCharts();
  $("season").onchange = () => renderCharts();
  $("csv").onclick = downloadCsv;

  // ---------------------------------------------------------------- charts
  Chart.defaults.font.family = "system-ui, -apple-system, Segoe UI, Roboto, sans-serif";
  Chart.defaults.font.size = 11; Chart.defaults.color = getCss("--ink2");
  const grid = { color: getCss("--line") };
  const bars = new Chart($("bars"), {
    type: "bar", data: { labels: [], datasets: [{ data: [], backgroundColor: [], borderRadius: 4, borderSkipped: false, barThickness: 16 }] },
    options: { indexAxis: "y", maintainAspectRatio: false, animation: false, plugins: { legend: { display: false },
      tooltip: { callbacks: { label: c => ` ${fmt(c.parsed.x, 2)} ${$("barMetric").selectedOptions[0].text.match(/\((.*)\)/)?.[1] || ""}` } } },
      scales: { x: { grid, beginAtZero: true }, y: { grid: { display: false } } } },
  });
  const capexChart = new Chart($("capexParts"), {
    type: "bar", data: { labels: [], datasets: [{ data: [], backgroundColor: [], borderRadius: 4, borderSkipped: false, barThickness: 14 }] },
    options: { indexAxis: "y", maintainAspectRatio: false, animation: false, plugins: { legend: { display: false },
      tooltip: { callbacks: { label: c => ` $${fmt(c.parsed.x, 2)} M` } } },
      scales: { x: { grid, title: { display: true, text: "$M" } }, y: { grid: { display: false } } } },
  });
  const day = new Chart($("day"), {
    type: "line", data: { labels: [...Array(24).keys()].map(h => `${h}:00`), datasets: [] },
    options: { maintainAspectRatio: false, animation: false, interaction: { mode: "index", intersect: false },
      plugins: { legend: { position: "bottom", labels: { boxWidth: 10, boxHeight: 10 } },
        tooltip: { callbacks: { label: c => ` ${c.dataset.label}: ${fmt(c.parsed.y, 1)} MW` } } },
      scales: { x: { grid, ticks: { maxTicksLimit: 8 } }, y: { grid, stacked: true, beginAtZero: true, title: { display: true, text: "MW" } } } },
  });

  // ---------------------------------------------------------------- compute + render
  let lastSel = null, lastSum = null, pending = false;
  function schedule() { if (!pending) { pending = true; requestAnimationFrame(() => { pending = false; update(false); }); } }

  async function update(settingsChanged) {
    const list = await sites();
    if (settingsChanged) {
      const mx = Model.maxPct(list, S, C);
      const slider = $("target");
      slider.max = Math.max(1, Math.floor(mx * 2) / 2);
      if (S.targetPct > +slider.max) S.targetPct = +slider.max;
      slider.value = S.targetPct;
      const mxv = +slider.max;
      const marks = [[5, "5 %"], [10, "10 % target"], [mxv, `max ${fmt(mx, 1)} %`]].filter(([v]) => v <= mxv);
      let lastPos = -1;
      $("ticks").innerHTML = marks.map(([v, t], i) => {
        const pos = 100 * v / mxv, isMax = i === marks.length - 1;
        const show = isMax || (pos - lastPos > 14 && 100 - pos > 22);
        if (show) lastPos = pos;
        const style = isMax ? `left:${pos}%;transform:translateX(-100%);text-align:right` : `left:${pos}%`;
        return `<span style="${style}" data-v="${v}" title="${t}">${show ? t : ""}</span>`;
      }).join("");
      $("ticks").querySelectorAll("span").forEach(sp => sp.onclick = () => { S.targetPct = +sp.dataset.v; slider.value = S.targetPct; update(false); });
      $("strategyHint").textContent = STRATEGY_HINT[S.strategy];
    }
    const sel = Model.select(list, S, C);
    const sum = Model.summarize(sel, S, C);
    lastSel = sel; lastSum = sum;
    $("pctLabel").textContent = `${fmt(S.targetPct, 1)} %`;
    const gwhT = S.targetPct / 100 * sel.campus;
    $("targetHint").textContent = sum.gwh + 0.05 < gwhT
      ? `Reached ${fmt(sum.gwh, 1)} of ${fmt(gwhT, 1)} GWh/yr - ${sel.gridCapped ? "grid cap reached" : "no more eligible sites"}.`
      : `${fmt(gwhT, 1)} GWh/yr of ${fmt(sel.campus, 0)} GWh campus use (FY2026, ${S.scenario}).`;
    styleMap(list, sel);
    renderTiles(sum);
    renderCharts();
    renderTable(sel);
    document.body.dataset.result = JSON.stringify({ strategy: S.strategy, target: S.targetPct, vintage: S.vintage,
      sites: sum.T.sites, MWdc: +(sum.T.kwdc / 1e3).toFixed(1), GWh: +sum.gwh.toFixed(1), newLandHa: +(sum.T.newLand / 1e4).toFixed(1),
      capexM: +(sum.T.capex / 1e6).toFixed(1), lcoe: +(100 * sum.lcoe).toFixed(2), exportGWh: +sum.exportGwh.toFixed(1),
      peakMW: +sum.peak.toFixed(1), maxPct: +$("target").max });
  }

  function styleMap(list, sel) {
    const chosen = new Map(sel.chosen.map(c => [c.site.id, c]));
    for (const s of list) {
      const l = layers[s.id]; if (!l) continue;
      const eligible = S.include[s.land] && (S.polo || !s.polo);
      const c = chosen.get(s.id), col = LAND[s.land].color;
      if (c) l.setStyle({ color: "#ffffff", weight: 0.7, fillColor: col, fillOpacity: 0.45 + 0.45 * c.frac, opacity: 0.95, dashArray: null });
      else if (eligible) l.setStyle({ color: "#ffffff", weight: 1, fillOpacity: 0, opacity: 0.75, dashArray: "2 2" });
      else l.setStyle({ color: "#ffffff", weight: 0.5, fillOpacity: 0, opacity: 0.15, dashArray: "1 3" });
    }
  }

  function renderTiles(m) {
    const t = (k, v, s) => `<div class="tile"><div class="v">${v}</div><div class="k">${k}</div>${s ? `<div class="s">${s}</div>` : ""}</div>`;
    $("tiles").innerHTML = [
      t("Energy, GWh/yr", fmt(m.gwh, 1), `${fmt(m.pct, 1)} % of campus use`),
      t("Capacity, MWdc", fmt(m.T.kwdc / 1e3, 1), `${m.T.sites} sites · peak ${fmt(m.peak, 0)} MWac`),
      t("New land, ha", fmt(m.T.newLand / 1e4, 1), `built area ${fmt(m.T.area / 1e4, 1)} ha`),
      t("Capex, $M", fmt(m.T.capex / 1e6, 0), "placeholder $/W"),
      t("LCOE, ¢/kWh", fmt(100 * m.lcoe, 1), `${fmt(100 * m.lcoeItc, 1)} with 30 % ITC`),
      t("Simple payback, yr", fmt(m.payback10, 0), `at 10 ¢/kWh · ${fmt(m.payback56, 0)} at 5.6 ¢`),
    ].join("");
  }

  function renderCharts() {
    if (!lastSum) return;
    const metric = $("barMetric").value;
    const keys = Object.keys(LAND);
    const val = { kwh: b => b.kwh / 1e6, kwdc: b => b.kwdc / 1e3, area: b => b.area / 1e4, capex: b => b.capex / 1e6, sites: b => b.sites };
    bars.data.labels = keys.map(k => LAND[k].label);
    bars.data.datasets[0].data = keys.map(k => lastSum.by[k] ? val[metric](lastSum.by[k]) : 0);
    bars.data.datasets[0].backgroundColor = keys.map(k => LAND[k].color);
    bars.update();

    const parts = Object.entries(lastSum.parts).filter(([k, v]) => k === "base" || Math.abs(v) > 1e3);
    capexChart.data.labels = parts.map(([k]) => PART_LABEL[k]);
    capexChart.data.datasets[0].data = parts.map(([, v]) => v / 1e6);
    capexChart.data.datasets[0].backgroundColor = parts.map(([k, v]) => k === "base" ? getCss("--accent") : v < 0 ? "#8a8984" : "#b4646c");
    capexChart.update();
    $("capexHint").textContent = `Total $${fmt(lastSum.T.capex / 1e6, 1)} M = $${fmt(lastSum.T.capex / Math.max(lastSum.T.kwdc * 1000, 1), 2)}/Wdc on average; ` +
      `O&M $${fmt(lastSum.T.om / 1e6, 2)} M/yr.`;

    const months = { summer: [5, 6, 7], winter: [11, 0, 1], year: [...Array(12).keys()] }[$("season").value];
    const d = Model.typicalDay(lastSel, C, months);
    const ds = keys.filter(k => d.lands[k]).map(k => ({ label: LAND[k].label, data: d.lands[k], borderColor: LAND[k].color,
      backgroundColor: LAND[k].color + "cc", fill: true, pointRadius: 0, borderWidth: 1, stack: "pv" }));
    ds.push({ label: "Campus load − CHP", data: d.load, borderColor: getCss("--ink"), borderWidth: 2, borderDash: [5, 3],
      pointRadius: 0, fill: false, stack: "load" });
    day.data.datasets = ds; day.update();
    $("gridHint").textContent = `Over the year: ${fmt(lastSum.usedPct, 0)} % of PV used on campus, ${fmt(lastSum.exportGwh, 1)} GWh exported to ERCOT; ` +
      `PV peak ${fmt(lastSum.peak, 0)} MWac vs average residual load ${fmt(lastSel.load.reduce((a, b) => a + b, 0) / 8760, 0)} MW.`;
  }

  function renderTable(sel) {
    const rows = [...sel.chosen].sort((a, b) => b.opt.kwh * b.frac - a.opt.kwh * a.frac);
    $("table").querySelector("tbody").innerHTML = rows.slice(0, 300).map(c =>
      `<tr data-id="${c.site.id}"><td title="${c.site.id}">${c.site.name.slice(0, 20)}</td><td>${LAND[c.site.land].label.split(" ")[0]}</td>` +
      `<td>${c.opt.design}</td><td>${fmt(c.opt.kwdc * c.frac, 0)}</td><td>${fmt(c.opt.kwh * c.frac / 1e3, 0)}</td><td>${fmt(c.m.usdPerW, 2)}</td><td>${fmt(100 * c.m.lcoe, 1)}</td></tr>`).join("");
    $("table").querySelectorAll("tbody tr").forEach(tr => tr.onclick = () => {
      const l = layers[tr.dataset.id]; if (!l) return;
      map.fitBounds(l.getBounds(), { maxZoom: 2, padding: [60, 60] });
      const s = sel.chosen.find(c => c.site.id === tr.dataset.id).site;
      L.popup().setLatLng(l.getBounds().getCenter()).setContent(popup(s)).openOn(map);
    });
  }

  function downloadCsv() {
    const head = ["site_id", "name", "land_type", "design", "fraction_built", "kwdc", "kwh_per_yr", "area_m2", "new_land_m2", "capex_usd", "usd_per_wdc", "lcoe_usd_per_kwh"];
    const lines = [head.join(",")].concat(lastSel.chosen.map(c => [c.site.id, `"${c.site.name.replace(/"/g, "'")}"`, c.site.land, c.opt.design,
      c.frac.toFixed(3), (c.opt.kwdc * c.frac).toFixed(1), Math.round(c.opt.kwh * c.frac), (c.opt.area_m2 * c.frac).toFixed(0),
      (c.opt.new_land_m2 * c.frac).toFixed(0), Math.round(c.m.capex * c.frac), c.m.usdPerW.toFixed(3), c.m.lcoe.toFixed(4)].join(",")));
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/csv" }));
    a.download = `solar_${S.strategy}_${S.targetPct}pct_${S.vintage}.csv`; a.click();
  }

  drawSites(await sites());
  update(true);
})();
