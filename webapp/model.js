// Portfolio model - mirrors scripts/11_portfolios.py (selection, costs, hourly grid balance).
// Pure functions: usable from the page (window.Model) and from tests.
(function (root) {
  "use strict";
  const LAND_ORDER = { "roof": 0, "garage top": 1, "parking": 2, "open land": 3 };
  const M2_PER_ACRE = 4046.86;
  const HOURS = 8760;

  // capital cost components - mirror of scripts/costs.py (keep in sync)
  function capexBreakdown(site, opt, s) {
    const M = s.costModel, w = opt.kwdc * 1000, d = site.cost || {};
    const base = w * s.usdPerW[opt.cost_key];
    let f = Math.pow(Math.max(opt.kwdc, 1) / M.scale.ref_kwdc[opt.cost_key], -M.scale.exponent);
    f = Math.min(Math.max(f, M.scale.min_factor), M.scale.max_factor);
    const out = { base, scale: base * (f - 1), height: 0, reroof: 0, garage: 0, interconnection: 0, siteprep: 0 };
    const L = site.land, ok = v => v !== null && v !== undefined && isFinite(v);
    if ((L === "roof" || L === "garage top") && ok(d.height_m))
      out.height = w * M.height.usd_per_w_per_m * Math.max(d.height_m - M.height.free_m, 0);
    if (L === "roof" && ok(d.age_yr) && d.age_yr >= M.reroof.age_yr) out.reroof = w * M.reroof.usd_per_w;
    if (L === "garage top") out.garage = w * M.garage.usd_per_w;
    if (L === "parking" || L === "open land") {
      const I = M.interconnection, dist = ok(d.dist_m) ? d.dist_m : I.default_dist_m;
      out.interconnection = opt.kwdc < I.lv_max_kwdc ? I.lv_fixed_usd : I.mv_fixed_usd + I.mv_usd_per_m * dist;
    }
    if (L === "open land") out.siteprep = M.siteprep.usd_per_m2 * opt.new_land_m2;
    return out;
  }

  // per-option economics for the current settings
  function metrics(opt, s, C, site) {
    const om = C.costs.om[opt.cost_key] * opt.kwdc;
    const parts = capexBreakdown(site, opt, s);
    const capex = Object.values(parts).reduce((a, b) => a + b, 0);
    const f = C.costs.finance, kwhL = opt.kwh * f.levelized_energy_factor;
    const lcoe = (capex * f.crf + om) / kwhL;
    const landAnnual = (s.landValuePerAcre / M2_PER_ACRE) * f.crf;     // $ per m2 per year
    const landPen = opt.new_land_m2 * landAnnual / kwhL;               // $ per kWh
    return { capex, parts, om, lcoe, landPen, usdPerW: capex / (opt.kwdc * 1000), kwhPerM2: opt.kwh / Math.max(opt.area_m2, 1) };
  }

  function candidates(sites, s, C) {
    const out = [];
    for (const site of sites) {
      if (!s.include[site.land]) continue;
      if (site.polo && !s.polo) continue;
      let best = null;
      for (const o of site.options) {
        if (o.design.startsWith("east-west") && !s.allowEastWest) continue;
        if (site.land === "open land" && !s.openDesigns[o.design.startsWith("tracker") ? "tracker" : "fixed"]) continue;
        const m = metrics(o, s, C, site);
        const key = s.strategy === "land" ? -m.kwhPerM2 : s.strategy === "cost" ? m.lcoe : m.lcoe + m.landPen;
        if (!best || key < best.key) best = { site, opt: o, m, key };
      }
      if (best) out.push(best);
    }
    if (s.strategy === "land")
      out.sort((a, b) => LAND_ORDER[a.site.land] - LAND_ORDER[b.site.land] || b.opt.kwh - a.opt.kwh);
    else out.sort((a, b) => a.key - b.key);
    return out;
  }

  function addProfile(prof, shape, kwh, scale) {      // MW per hour
    const k = kwh / scale / 1000;
    for (let h = 0; h < HOURS; h++) prof[h] += shape[h] * k;
  }
  function exportFrac(prof, load) {
    let e = 0, t = 0;
    for (let h = 0; h < HOURS; h++) { t += prof[h]; if (prof[h] > load[h]) e += prof[h] - load[h]; }
    return t > 0 ? e / t : 0;
  }

  // fill to the target; optimal skips sites that would push export above the cap
  function select(sites, s, C) {
    const campus = C.campus_gwh[s.scenario];
    const target = s.targetPct / 100 * campus * 1e6;
    const load = C.residual_load_mw[s.scenario];
    const cands = candidates(sites, s, C);
    const prof = new Float64Array(HOURS);
    const chosen = [];
    let tot = 0, gridCapped = false;
    for (const c of cands) {
      if (tot >= target - 1) break;
      let frac = Math.min(1, (target - tot) / c.opt.kwh);
      if (s.strategy === "optimal") {
        const trial = Float64Array.from(prof);
        addProfile(trial, C.shapes[c.opt.shape], c.opt.kwh * frac, C.shape_scale);
        if (exportFrac(trial, load) > s.maxExport) { gridCapped = true; continue; }
      }
      addProfile(prof, C.shapes[c.opt.shape], c.opt.kwh * frac, C.shape_scale);
      chosen.push({ ...c, frac });
      tot += c.opt.kwh * frac;
    }
    return { chosen, prof, load, target, reached: tot, gridCapped, campus };
  }

  function summarize(sel, s, C) {
    const f = C.costs.finance;
    const by = {};
    const T = { kwdc: 0, kwh: 0, area: 0, newLand: 0, capex: 0, om: 0, sites: 0 };
    const parts = { base: 0, scale: 0, height: 0, reroof: 0, garage: 0, interconnection: 0, siteprep: 0 };
    for (const c of sel.chosen) {
      const k = c.frac, L = c.site.land;
      by[L] = by[L] || { kwdc: 0, kwh: 0, area: 0, newLand: 0, capex: 0, sites: 0 };
      const add = { kwdc: c.opt.kwdc * k, kwh: c.opt.kwh * k, area: c.opt.area_m2 * k,
                    newLand: c.opt.new_land_m2 * k, capex: c.m.capex * k };
      for (const [n, v] of Object.entries(add)) { by[L][n] += v; T[n] += v; }
      by[L].sites += 1; T.sites += 1; T.om += c.m.om * k;
      for (const [n, v] of Object.entries(c.m.parts)) parts[n] += v * k;
    }
    const kwhL = T.kwh * f.levelized_energy_factor;
    const lcoe = T.kwh ? (T.capex * f.crf + T.om) / kwhL : 0;
    const lcoeItc = T.kwh ? (T.capex * (1 - f.itc_sensitivity) * f.crf + T.om) / kwhL : 0;
    let used = 0, peak = 0;
    for (let h = 0; h < HOURS; h++) { used += Math.min(sel.prof[h], sel.load[h]); peak = Math.max(peak, sel.prof[h]); }
    const gwh = T.kwh / 1e6, usedGwh = used / 1e3;
    const price = f.electricity_price_usd_per_kwh;
    const payback = p => (T.kwh * p > T.om ? T.capex / (T.kwh * p - T.om) : Infinity);
    return { by, T, parts, gwh, pct: 100 * gwh / sel.campus, lcoe, lcoeItc, peak, usedGwh,
             exportGwh: Math.max(gwh - usedGwh, 0), usedPct: gwh ? 100 * usedGwh / gwh : 0,
             payback56: payback(price.base_purchase_fy23), payback10: payback(price.ues_rate_fy27) };
  }

  // largest % reachable with the current settings (all candidates; optimal respects the grid cap)
  function maxPct(sites, s, C) {
    const big = { ...s, targetPct: 1000 };
    const sel = select(sites, big, C);
    return 100 * sel.reached / 1e6 / sel.campus;
  }

  // average day (24 h) for a month range, by land type and the load
  function typicalDay(sel, C, months) {
    const startOfMonth = [0, 744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016, 8760];
    const hours = [];
    for (const m of months) for (let h = startOfMonth[m]; h < startOfMonth[m + 1]; h++) hours.push(h);
    const nd = hours.length / 24;
    const lands = {}, load = new Array(24).fill(0);
    for (const c of sel.chosen) {
      const L = c.site.land, sh = C.shapes[c.opt.shape], k = c.opt.kwh * c.frac / C.shape_scale / 1000;
      lands[L] = lands[L] || new Array(24).fill(0);
      for (const h of hours) lands[L][h % 24] += sh[h] * k / nd;
    }
    for (const h of hours) load[h % 24] += sel.load[h] / nd;
    return { lands, load };
  }

  root.Model = { select, summarize, maxPct, typicalDay, metrics, capexBreakdown, LAND_ORDER };
})(typeof window !== "undefined" ? window : globalThis);
