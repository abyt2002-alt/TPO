const normalizeSizeKey = (value) => String(value || '').toUpperCase().replace(/\s+/g, '').trim()

const slabSortKey = (value) => {
  const text = String(value || '').toLowerCase()
  const match = text.match(/(\d+)/)
  return match ? Number(match[1]) : Number.MAX_SAFE_INTEGER
}

export const normalizePlannerPeriodsFromData = (data) => {
  const explicit = Array.isArray(data?.periods) ? data.periods.map((p) => String(p)) : []
  if (explicit.length) return explicit
  const rawMonthly = data?.monthly_results
  if (Array.isArray(rawMonthly)) {
    const derived = rawMonthly
      .map((row) => String(row?.period || '').trim())
      .filter(Boolean)
    if (derived.length) return derived
  }
  if (rawMonthly && typeof rawMonthly === 'object') {
    const keys = Object.keys(rawMonthly).map((k) => String(k))
    if (keys.length) return keys
  }
  return []
}

const getSeriesValueAt = (seriesValue, monthIdx, periodKey) => {
  if (Array.isArray(seriesValue)) return Number(seriesValue?.[monthIdx] || 0)
  if (seriesValue && typeof seriesValue === 'object') return Number(seriesValue?.[periodKey] || 0)
  return Number(seriesValue || 0)
}

const buildWeightMapForSize = (sizeResult, slabKeys) => {
  const keys = (Array.isArray(slabKeys) ? slabKeys : []).map((s) => String(s))
  if (!keys.length) return {}
  const raw = sizeResult?.modeled_weights && typeof sizeResult.modeled_weights === 'object'
    ? sizeResult.modeled_weights
    : {}
  const out = {}
  let sum = 0
  keys.forEach((k) => {
    const w = Number(raw?.[k] || 0)
    const safe = Number.isFinite(w) && w > 0 ? w : 0
    out[k] = safe
    sum += safe
  })
  if (sum <= 0) {
    const eq = 1 / keys.length
    keys.forEach((k) => { out[k] = eq })
    return out
  }
  keys.forEach((k) => { out[k] = out[k] / sum })
  return out
}

const computeOtherWeightedDiscount = (slabKey, discountMap, weightMap) => {
  const target = String(slabKey)
  let num = 0
  let den = 0
  Object.entries(weightMap || {}).forEach(([k, wRaw]) => {
    const kStr = String(k)
    if (kStr === target) return
    const w = Number(wRaw || 0)
    const d = Number(discountMap?.[kStr] || 0)
    if (!Number.isFinite(w) || w <= 0) return
    if (!Number.isFinite(d)) return
    num += w * d
    den += w
  })
  if (den <= 0) return 0
  return num / den
}

export const computeCrossSizePlannerData = ({ data, periods, scenarioDiscountsByPeriod }) => {
  if (!data?.success || !periods?.length) return data

  const defaultsMatrix = data?.defaults_matrix || {}
  const baselineSlabMatrix = data?.baseline_slab_matrix || {}
  const baseSummary = data?.summary_3m || {}
  const sizeResults = Array.isArray(data?.size_results) ? data.size_results : []
  const sizeResultByKey = {}
  sizeResults.forEach((row) => {
    const k = normalizeSizeKey(row?.size)
    if (k) sizeResultByKey[k] = row
  })

  const ref12Qty = Number(baseSummary?.['12-ML']?.reference_qty || 0)
  const ref18Qty = Number(baseSummary?.['18-ML']?.reference_qty || 0)
  const e12From18 = Number(data?.cross_elasticity_12_from_18 || 0)
  const e18From12 = Number(data?.cross_elasticity_18_from_12 || 0)

  const monthlyResults = periods.map((periodKey, monthIdx) => {
    const sizes = {}
    ;['12-ML', '18-ML'].forEach((sizeKey) => {
      const sizeDefaults = defaultsMatrix?.[sizeKey] || {}
      const slabKeys = Object.keys(sizeDefaults).sort((a, b) => slabSortKey(a) - slabSortKey(b))
      if (!slabKeys.length) return
      const sizeResult = sizeResultByKey?.[normalizeSizeKey(sizeKey)] || {}
      const slabRowsModel = Array.isArray(sizeResult?.slabs) ? sizeResult.slabs : []
      const slabModelByKey = {}
      slabRowsModel.forEach((r) => {
        const k = String(r?.slab || '')
        if (k) slabModelByKey[k] = r
      })
      const weightMap = buildWeightMapForSize(sizeResult, slabKeys)
      const scenarioMapThisMonth = {}
      slabKeys.forEach((slabKey) => {
        const fallback = Number(sizeDefaults?.[slabKey]?.[monthIdx] || 0)
        scenarioMapThisMonth[slabKey] = Number(
          scenarioDiscountsByPeriod?.[periodKey]?.[sizeKey]?.[slabKey] ?? fallback
        )
      })

      const slabs = slabKeys.map((slabKey) => {
        const model = slabModelByKey?.[slabKey] || {}
        const defaultDiscount = Number(sizeDefaults?.[slabKey]?.[monthIdx] || model?.default_discount_pct || 0)
        const scenarioDiscount = Number(scenarioMapThisMonth?.[slabKey] ?? defaultDiscount)
        const lagUsed = monthIdx > 0
          ? Number(
            scenarioDiscountsByPeriod?.[periods?.[monthIdx - 1]]?.[sizeKey]?.[slabKey]
            ?? sizeDefaults?.[slabKey]?.[monthIdx - 1]
            ?? model?.default_discount_pct
            ?? defaultDiscount
          )
          : Number(model?.default_discount_pct ?? defaultDiscount)
        const otherWeighted = computeOtherWeightedDiscount(slabKey, scenarioMapThisMonth, weightMap)
        const coefBase = Number(model?.coef_base_discount_pct || 0)
        const coefLag = Number(model?.coef_lag1_base_discount_pct || 0)
        const coefOther = Number(model?.coef_other_slabs_weighted_base_discount_pct || 0)
        const basePrice = Number(model?.base_price || 0)
        const clpPrice = Number(model?.clp_price || basePrice)
        const cogsPerUnit = Number(model?.cogs_per_unit || 0)
        const nonDiscountBaseline = Number(baselineSlabMatrix?.[sizeKey]?.[slabKey]?.[monthIdx] || 0)
        const discountComponentScenario = (coefBase * scenarioDiscount) + (coefLag * lagUsed) + (coefOther * otherWeighted)
        const preCrossQty = Math.max(nonDiscountBaseline + discountComponentScenario, 0)
        return {
          slab: slabKey,
          default_discount_pct: defaultDiscount,
          scenario_discount_pct: scenarioDiscount,
          default_lag_used_pct: defaultDiscount,
          lag_used_pct: lagUsed,
          other_weighted_default_pct: otherWeighted,
          other_weighted_scenario_pct: otherWeighted,
          discount_component_default_qty: discountComponentScenario,
          discount_component_scenario_qty: discountComponentScenario,
          non_discount_baseline_qty: nonDiscountBaseline,
          baseline_qty: nonDiscountBaseline,
          default_world_qty: preCrossQty,
          pre_cross_qty: preCrossQty,
          final_qty: preCrossQty,
          base_price: basePrice,
          clp_price: clpPrice,
          cogs_per_unit: cogsPerUnit,
          baseline_revenue: 0,
          scenario_revenue: 0,
          baseline_revenue_gross: 0,
          scenario_revenue_gross: 0,
          baseline_revenue_net: 0,
          scenario_revenue_net: 0,
          baseline_profit: 0,
          scenario_profit: 0,
          baseline_investment: 0,
          scenario_investment: 0,
          scenario_investment_positive: 0,
        }
      })

      const baselineTotal = slabs.reduce((s, x) => s + Number(x?.non_discount_baseline_qty || 0), 0)
      const preTotal = slabs.reduce((s, x) => s + Number(x?.pre_cross_qty || 0), 0)
      sizes[sizeKey] = {
        size: sizeKey,
        baseline_total_qty: baselineTotal,
        baseline_total_qty_default_world: preTotal,
        pre_cross_total_qty: preTotal,
        final_total_qty: preTotal,
        slabs,
      }
    })
    return {
      period: periodKey,
      sizes,
      impact: {
        prev12_qty: Number(sizes?.['12-ML']?.baseline_total_qty || 0),
        prev18_qty: Number(sizes?.['18-ML']?.baseline_total_qty || 0),
        pre12_qty: Number(sizes?.['12-ML']?.pre_cross_total_qty || 0),
        pre18_qty: Number(sizes?.['18-ML']?.pre_cross_total_qty || 0),
        final12_qty: Number(sizes?.['12-ML']?.final_total_qty || 0),
        final18_qty: Number(sizes?.['18-ML']?.final_total_qty || 0),
        own12_pct: 0,
        own18_pct: 0,
        overall12_pct: 0,
        overall18_pct: 0,
      },
    }
  })

  const pre12_3m = monthlyResults.reduce((s, row) => s + Number(row?.sizes?.['12-ML']?.pre_cross_total_qty || 0), 0)
  const pre18_3m = monthlyResults.reduce((s, row) => s + Number(row?.sizes?.['18-ML']?.pre_cross_total_qty || 0), 0)
  const own12 = ref12Qty > 0 ? ((pre12_3m - ref12Qty) / ref12Qty) * 100 : 0
  const own18 = ref18Qty > 0 ? ((pre18_3m - ref18Qty) / ref18Qty) * 100 : 0
  const adjusted12Pct = own12 + (e12From18 * own18)
  const adjusted18Pct = own18 + (e18From12 * own12)
  const final12_3m = ref12Qty > 0 ? Math.max(ref12Qty * (1 + adjusted12Pct / 100), 0) : Math.max(pre12_3m, 0)
  const final18_3m = ref18Qty > 0 ? Math.max(ref18Qty * (1 + adjusted18Pct / 100), 0) : Math.max(pre18_3m, 0)

  ;['12-ML', '18-ML'].forEach((sizeKey) => {
    const target = sizeKey === '12-ML' ? final12_3m : final18_3m
    const cells = []
    let sumPre = 0
    let sumBase = 0
    monthlyResults.forEach((row) => {
      const slabs = row?.sizes?.[sizeKey]?.slabs || []
      slabs.forEach((slab) => {
        const pre = Math.max(Number(slab?.pre_cross_qty || 0), 0)
        const base = Math.max(Number(slab?.non_discount_baseline_qty || 0), 0)
        sumPre += pre
        sumBase += base
        cells.push({ slab, pre, base })
      })
    })
    if (!cells.length) return
    let shares
    if (sumPre > 0) shares = cells.map((c) => c.pre / sumPre)
    else if (sumBase > 0) shares = cells.map((c) => c.base / sumBase)
    else shares = cells.map(() => 1 / cells.length)
    cells.forEach((c, idx) => {
      c.slab.final_qty = Math.max(target * shares[idx], 0)
    })
  })

  const summary = {
    '12-ML': {
      baseline_qty: 0,
      scenario_qty_additive: 0,
      final_qty: 0,
      baseline_revenue: 0,
      scenario_revenue: 0,
      baseline_revenue_gross: 0,
      scenario_revenue_gross: 0,
      baseline_revenue_net: 0,
      scenario_revenue_net: 0,
      baseline_profit: 0,
      scenario_profit: 0,
      baseline_investment: 0,
      scenario_investment: 0,
      investment_change_positive: 0,
    },
    '18-ML': {
      baseline_qty: 0,
      scenario_qty_additive: 0,
      final_qty: 0,
      baseline_revenue: 0,
      scenario_revenue: 0,
      baseline_revenue_gross: 0,
      scenario_revenue_gross: 0,
      baseline_revenue_net: 0,
      scenario_revenue_net: 0,
      baseline_profit: 0,
      scenario_profit: 0,
      baseline_investment: 0,
      scenario_investment: 0,
      investment_change_positive: 0,
    },
  }

  monthlyResults.forEach((row) => {
    ;['12-ML', '18-ML'].forEach((sizeKey) => {
      const block = row?.sizes?.[sizeKey]
      if (!block) return
      let baselineRevenueTotal = 0
      let scenarioRevenueTotal = 0
      let baselineRevenueGrossTotal = 0
      let scenarioRevenueGrossTotal = 0
      let baselineRevenueNetTotal = 0
      let scenarioRevenueNetTotal = 0
      let baselineProfitTotal = 0
      let scenarioProfitTotal = 0
      let baselineInvestmentTotal = 0
      let scenarioInvestmentTotal = 0
      let scenarioInvestmentPositiveTotal = 0
      const slabs = block?.slabs || []
      slabs.forEach((slab) => {
        const baseQty = Number(slab?.non_discount_baseline_qty || 0)
        const finalQty = Number(slab?.final_qty || 0)
        const dspPrice = Number(slab?.base_price || 0)
        const clpPrice = Number(slab?.clp_price || dspPrice)
        const defaultDiscount = Number(slab?.default_discount_pct || 0)
        const scenarioDiscount = Number(slab?.scenario_discount_pct || 0)
        const cogs = Number(slab?.cogs_per_unit || 0)
        const baselineRevenueGross = baseQty * dspPrice
        const scenarioRevenueGross = finalQty * dspPrice
        const baselineRevenueNet = baseQty * clpPrice
        const scenarioRevenueNet = finalQty * clpPrice
        const baselineRevenue = baselineRevenueGross
        const scenarioRevenue = scenarioRevenueGross
        const baselineProfit = baselineRevenueNet - (baseQty * cogs)
        const scenarioProfit = scenarioRevenueNet - (finalQty * cogs)
        const baselineInvestment = baseQty * dspPrice * (defaultDiscount / 100)
        const scenarioInvestment = finalQty * dspPrice * (scenarioDiscount / 100)
        const scenarioInvestmentPositive = finalQty * dspPrice * (Math.max(0, scenarioDiscount - defaultDiscount) / 100)
        slab.baseline_revenue = baselineRevenue
        slab.scenario_revenue = scenarioRevenue
        slab.baseline_revenue_gross = baselineRevenueGross
        slab.scenario_revenue_gross = scenarioRevenueGross
        slab.baseline_revenue_net = baselineRevenueNet
        slab.scenario_revenue_net = scenarioRevenueNet
        slab.baseline_profit = baselineProfit
        slab.scenario_profit = scenarioProfit
        slab.baseline_investment = baselineInvestment
        slab.scenario_investment = scenarioInvestment
        slab.scenario_investment_positive = scenarioInvestmentPositive
        baselineRevenueTotal += baselineRevenue
        scenarioRevenueTotal += scenarioRevenue
        baselineRevenueGrossTotal += baselineRevenueGross
        scenarioRevenueGrossTotal += scenarioRevenueGross
        baselineRevenueNetTotal += baselineRevenueNet
        scenarioRevenueNetTotal += scenarioRevenueNet
        baselineProfitTotal += baselineProfit
        scenarioProfitTotal += scenarioProfit
        baselineInvestmentTotal += baselineInvestment
        scenarioInvestmentTotal += scenarioInvestment
        scenarioInvestmentPositiveTotal += scenarioInvestmentPositive
      })
      block.final_total_qty = slabs.reduce((s, slab) => s + Number(slab?.final_qty || 0), 0)
      block.baseline_revenue_total = baselineRevenueTotal
      block.scenario_revenue_total = scenarioRevenueTotal
      block.baseline_revenue_gross_total = baselineRevenueGrossTotal
      block.scenario_revenue_gross_total = scenarioRevenueGrossTotal
      block.baseline_revenue_net_total = baselineRevenueNetTotal
      block.scenario_revenue_net_total = scenarioRevenueNetTotal
      block.baseline_profit_total = baselineProfitTotal
      block.scenario_profit_total = scenarioProfitTotal
      block.baseline_investment_total = baselineInvestmentTotal
      block.scenario_investment_total = scenarioInvestmentTotal
      block.scenario_investment_positive_total = scenarioInvestmentPositiveTotal

      summary[sizeKey].baseline_qty += Number(block?.baseline_total_qty || 0)
      summary[sizeKey].scenario_qty_additive += Number(block?.pre_cross_total_qty || 0)
      summary[sizeKey].final_qty += Number(block?.final_total_qty || 0)
      summary[sizeKey].baseline_revenue += baselineRevenueTotal
      summary[sizeKey].scenario_revenue += scenarioRevenueTotal
      summary[sizeKey].baseline_revenue_gross += baselineRevenueGrossTotal
      summary[sizeKey].scenario_revenue_gross += scenarioRevenueGrossTotal
      summary[sizeKey].baseline_revenue_net += baselineRevenueNetTotal
      summary[sizeKey].scenario_revenue_net += scenarioRevenueNetTotal
      summary[sizeKey].baseline_profit += baselineProfitTotal
      summary[sizeKey].scenario_profit += scenarioProfitTotal
      summary[sizeKey].baseline_investment += baselineInvestmentTotal
      summary[sizeKey].scenario_investment += scenarioInvestmentTotal
      summary[sizeKey].investment_change_positive += scenarioInvestmentPositiveTotal
    })
  })

  const finalizeSizeSummary = (sizeKey) => {
    const s = summary[sizeKey]
    const refQty = Number(baseSummary?.[sizeKey]?.reference_qty || 0)
    const refRevGross = Number((baseSummary?.[sizeKey]?.reference_revenue_gross ?? baseSummary?.[sizeKey]?.reference_revenue) || 0)
    const refRevNet = Number(baseSummary?.[sizeKey]?.reference_revenue_net ?? 0)
    const refProfit = Number(baseSummary?.[sizeKey]?.reference_profit || 0)
    const refInvestment = Number(baseSummary?.[sizeKey]?.reference_investment || 0)
    const refAvail = Number(baseSummary?.[sizeKey]?.reference_available || 0)
    return {
      baseline_qty: s.baseline_qty,
      scenario_qty_additive: s.scenario_qty_additive,
      discount_component_qty: s.scenario_qty_additive - s.baseline_qty,
      final_qty: s.final_qty,
      volume_delta_pct: s.baseline_qty > 0 ? ((s.final_qty - s.baseline_qty) / s.baseline_qty) * 100 : 0,
      volume_delta_additive_pct: s.baseline_qty > 0 ? ((s.scenario_qty_additive - s.baseline_qty) / s.baseline_qty) * 100 : 0,
      baseline_revenue: s.baseline_revenue,
      scenario_revenue: s.scenario_revenue,
      baseline_revenue_gross: s.baseline_revenue_gross,
      scenario_revenue_gross: s.scenario_revenue_gross,
      baseline_revenue_net: s.baseline_revenue_net,
      scenario_revenue_net: s.scenario_revenue_net,
      revenue_delta_pct: s.baseline_revenue > 0 ? ((s.scenario_revenue - s.baseline_revenue) / s.baseline_revenue) * 100 : 0,
      revenue_gross_delta_pct: s.baseline_revenue_gross > 0 ? ((s.scenario_revenue_gross - s.baseline_revenue_gross) / s.baseline_revenue_gross) * 100 : 0,
      revenue_net_delta_pct: s.baseline_revenue_net > 0 ? ((s.scenario_revenue_net - s.baseline_revenue_net) / s.baseline_revenue_net) * 100 : 0,
      baseline_profit: s.baseline_profit,
      scenario_profit: s.scenario_profit,
      profit_delta_pct: Math.abs(s.baseline_profit) > 1e-9 ? ((s.scenario_profit - s.baseline_profit) / Math.abs(s.baseline_profit)) * 100 : 0,
      baseline_investment: s.baseline_investment,
      scenario_investment: s.scenario_investment,
      investment_change_positive: s.investment_change_positive,
      investment_delta_pct: s.baseline_investment > 0 ? ((s.scenario_investment - s.baseline_investment) / s.baseline_investment) * 100 : 0,
      reference_qty: refQty,
      reference_revenue_gross: refRevGross,
      reference_revenue_net: refRevNet,
      reference_revenue: refRevGross,
      reference_profit: refProfit,
      reference_investment: refInvestment,
      vs_reference_volume_pct: refQty > 0 ? ((s.final_qty - refQty) / refQty) * 100 : 0,
      vs_reference_revenue_pct: refRevGross > 0 ? ((s.scenario_revenue - refRevGross) / refRevGross) * 100 : 0,
      vs_reference_revenue_gross_pct: refRevGross > 0 ? ((s.scenario_revenue_gross - refRevGross) / refRevGross) * 100 : 0,
      vs_reference_revenue_net_pct: refRevNet > 0 ? ((s.scenario_revenue_net - refRevNet) / refRevNet) * 100 : 0,
      vs_reference_profit_pct: Math.abs(refProfit) > 1e-9 ? ((s.scenario_profit - refProfit) / Math.abs(refProfit)) * 100 : 0,
      vs_reference_investment_pct: refInvestment > 0 ? ((s.scenario_investment - refInvestment) / refInvestment) * 100 : 0,
      investment_change_positive_vs_reference_pct: refInvestment > 0 ? ((s.investment_change_positive - refInvestment) / refInvestment) * 100 : 0,
      reference_available: refAvail,
    }
  }

  const s12 = finalizeSizeSummary('12-ML')
  const s18 = finalizeSizeSummary('18-ML')
  const totalBaselineQty = s12.baseline_qty + s18.baseline_qty
  const totalScenarioAddQty = s12.scenario_qty_additive + s18.scenario_qty_additive
  const totalFinalQty = s12.final_qty + s18.final_qty
  const totalBaselineRevenue = s12.baseline_revenue + s18.baseline_revenue
  const totalScenarioRevenue = s12.scenario_revenue + s18.scenario_revenue
  const totalBaselineRevenueGross = s12.baseline_revenue_gross + s18.baseline_revenue_gross
  const totalScenarioRevenueGross = s12.scenario_revenue_gross + s18.scenario_revenue_gross
  const totalBaselineRevenueNet = s12.baseline_revenue_net + s18.baseline_revenue_net
  const totalScenarioRevenueNet = s12.scenario_revenue_net + s18.scenario_revenue_net
  const totalBaselineProfit = s12.baseline_profit + s18.baseline_profit
  const totalScenarioProfit = s12.scenario_profit + s18.scenario_profit
  const totalBaselineInvestment = s12.baseline_investment + s18.baseline_investment
  const totalScenarioInvestment = s12.scenario_investment + s18.scenario_investment
  const totalInvestmentChangePositive = s12.investment_change_positive + s18.investment_change_positive
  const refTotalQty = Number(baseSummary?.TOTAL?.reference_qty || 0)
  const refTotalRevGross = Number((baseSummary?.TOTAL?.reference_revenue_gross ?? baseSummary?.TOTAL?.reference_revenue) || 0)
  const refTotalRevNet = Number(baseSummary?.TOTAL?.reference_revenue_net || 0)
  const refTotalProfit = Number(baseSummary?.TOTAL?.reference_profit || 0)
  const refTotalInvestment = Number(baseSummary?.TOTAL?.reference_investment || 0)
  const refTotalAvail = Number(baseSummary?.TOTAL?.reference_available || 0)
  const baselineVolumeMl = (s12.baseline_qty * 12) + (s18.baseline_qty * 18)
  const scenarioVolumeMlAdd = (s12.scenario_qty_additive * 12) + (s18.scenario_qty_additive * 18)
  const finalVolumeMl = (s12.final_qty * 12) + (s18.final_qty * 18)
  const refVolumeMl = (Number(baseSummary?.['12-ML']?.reference_qty || 0) * 12) + (Number(baseSummary?.['18-ML']?.reference_qty || 0) * 18)

  const summary3m = {
    '12-ML': s12,
    '18-ML': s18,
    TOTAL: {
      baseline_qty: totalBaselineQty,
      scenario_qty_additive: totalScenarioAddQty,
      discount_component_qty: totalScenarioAddQty - totalBaselineQty,
      final_qty: totalFinalQty,
      volume_delta_pct: totalBaselineQty > 0 ? ((totalFinalQty - totalBaselineQty) / totalBaselineQty) * 100 : 0,
      volume_delta_additive_pct: totalBaselineQty > 0 ? ((totalScenarioAddQty - totalBaselineQty) / totalBaselineQty) * 100 : 0,
      baseline_revenue: totalBaselineRevenue,
      scenario_revenue: totalScenarioRevenue,
      baseline_revenue_gross: totalBaselineRevenueGross,
      scenario_revenue_gross: totalScenarioRevenueGross,
      baseline_revenue_net: totalBaselineRevenueNet,
      scenario_revenue_net: totalScenarioRevenueNet,
      revenue_delta_pct: totalBaselineRevenue > 0 ? ((totalScenarioRevenue - totalBaselineRevenue) / totalBaselineRevenue) * 100 : 0,
      revenue_gross_delta_pct: totalBaselineRevenueGross > 0 ? ((totalScenarioRevenueGross - totalBaselineRevenueGross) / totalBaselineRevenueGross) * 100 : 0,
      revenue_net_delta_pct: totalBaselineRevenueNet > 0 ? ((totalScenarioRevenueNet - totalBaselineRevenueNet) / totalBaselineRevenueNet) * 100 : 0,
      baseline_profit: totalBaselineProfit,
      scenario_profit: totalScenarioProfit,
      profit_delta_pct: Math.abs(totalBaselineProfit) > 1e-9 ? ((totalScenarioProfit - totalBaselineProfit) / Math.abs(totalBaselineProfit)) * 100 : 0,
      baseline_investment: totalBaselineInvestment,
      scenario_investment: totalScenarioInvestment,
      investment_change_positive: totalInvestmentChangePositive,
      investment_delta_pct: totalBaselineInvestment > 0 ? ((totalScenarioInvestment - totalBaselineInvestment) / totalBaselineInvestment) * 100 : 0,
      reference_qty: refTotalQty,
      reference_revenue_gross: refTotalRevGross,
      reference_revenue_net: refTotalRevNet,
      reference_revenue: refTotalRevGross,
      reference_profit: refTotalProfit,
      reference_investment: refTotalInvestment,
      vs_reference_volume_pct: refTotalQty > 0 ? ((totalFinalQty - refTotalQty) / refTotalQty) * 100 : 0,
      vs_reference_revenue_pct: refTotalRevGross > 0 ? ((totalScenarioRevenue - refTotalRevGross) / refTotalRevGross) * 100 : 0,
      vs_reference_revenue_gross_pct: refTotalRevGross > 0 ? ((totalScenarioRevenueGross - refTotalRevGross) / refTotalRevGross) * 100 : 0,
      vs_reference_revenue_net_pct: refTotalRevNet > 0 ? ((totalScenarioRevenueNet - refTotalRevNet) / refTotalRevNet) * 100 : 0,
      vs_reference_profit_pct: Math.abs(refTotalProfit) > 1e-9 ? ((totalScenarioProfit - refTotalProfit) / Math.abs(refTotalProfit)) * 100 : 0,
      vs_reference_investment_pct: refTotalInvestment > 0 ? ((totalScenarioInvestment - refTotalInvestment) / refTotalInvestment) * 100 : 0,
      investment_change_positive_vs_reference_pct: refTotalInvestment > 0 ? ((totalInvestmentChangePositive - refTotalInvestment) / refTotalInvestment) * 100 : 0,
      reference_available: refTotalAvail,
      baseline_volume_ml: baselineVolumeMl,
      scenario_volume_ml_additive: scenarioVolumeMlAdd,
      final_volume_ml: finalVolumeMl,
      reference_volume_ml: refVolumeMl,
      volume_ml_delta_pct: baselineVolumeMl > 0 ? ((finalVolumeMl - baselineVolumeMl) / baselineVolumeMl) * 100 : 0,
      volume_ml_delta_additive_pct: baselineVolumeMl > 0 ? ((scenarioVolumeMlAdd - baselineVolumeMl) / baselineVolumeMl) * 100 : 0,
      vs_reference_volume_ml_pct: refVolumeMl > 0 ? ((finalVolumeMl - refVolumeMl) / refVolumeMl) * 100 : 0,
    },
  }

  return {
    ...data,
    monthly_results: monthlyResults,
    summary_3m: summary3m,
  }
}

