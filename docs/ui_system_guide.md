# UI System Guide (HRI App Style)

## Goal
Use this guide to recreate the same UI quality and structure used in this app:
- Left navigation
- Top step tabs
- Main content cards
- Right settings panel
- Clean KPI rows and chart containers

This is a practical style system, not a generic design note.

---

## 1) App Shell Structure

Use a 3-column shell on desktop:
- Left nav: fixed, narrow
- Center canvas: main work area
- Right panel: settings/filters

Recommended proportions:
- Left nav: `240px`
- Right panel: `320px`
- Center: remaining width (`1fr`)

Page layout:
1. App header/title row
2. Step tabs row (Step 1..Step 6)
3. Main card sections (KPI + charts + tables)

Mobile/tablet:
- Left nav becomes drawer
- Right panel becomes drawer
- Center remains primary

---

## 2) Visual Tokens

Use CSS variables (or Tailwind theme extension):

```css
:root {
  --bg-page: #f3f4f6;
  --bg-card: #ffffff;
  --border: #dbe1ea;
  --text-main: #2f3440;
  --text-sub: #5f6876;
  --text-muted: #8a93a3;

  --brand: #4a8fe7;
  --success: #2db777;
  --danger: #e25b5b;
  --warning: #f4b544;

  --chip-blue: #eaf3ff;
  --chip-green: #ebf9f2;
  --chip-red: #feecec;
  --chip-amber: #fff6e8;
}
```

Typography:
- Family: Inter (or similar clean sans)
- Title: 30/700
- Section title: 20/700
- Card metric: 40/700
- Body: 14/500
- Helper: 12/500

Spacing:
- Outer page padding: `16px`
- Card radius: `10px`
- Card border: `1px solid var(--border)`
- Internal card padding: `14px`
- Vertical gap between blocks: `12px`

---

## 3) Component Specs

### 3.1 Left Sidebar
- Flat background `#f5f6f8`
- Navigation item:
  - inactive: subtle text
  - active: thin blue border + white fill
- Keep icon + label aligned in one row

### 3.2 Step Tabs
- 1 row, equal-height tabs
- Active tab:
  - blue border
  - medium bold label
- Keep tab text short (2 lines max)

### 3.3 KPI Cards
- Show metric label, value, and delta
- Delta color:
  - green for positive
  - red for negative
  - muted for near zero
- Avoid long helper text inside card body

### 3.4 Filter Inputs
- Compact controls (`36px` to `40px`)
- Multi-select:
  - searchable list
  - select all / clear
  - selected count summary
- Keep filter labels above controls

### 3.5 Charts
- Always wrap charts in a card
- Keep one intent per chart:
  - trend
  - comparison
  - decomposition
- Use only 3-4 stable colors:
  - blue / green / orange / red
- Put legends inside chart area bottom when possible

### 3.6 Tables
- Sticky header
- Dense rows
- Right-align numeric columns
- Format large values in compact units for visual scan (`M`, `K`)

---

## 4) Data Formatting Rules

Numbers:
- Volume: compact for cards (`2.34M`), full in drill tables
- Revenue: compact in cards, comma style in tables
- Percent: always signed (`+2.45%`, `-1.20%`)
- CTS / GM: 2 decimals

Date labels:
- Use short month format (`Jan 2026`)
- Keep consistent across tabs/charts/tables

---

## 5) Page Template (Recommended)

For each functional step page:
1. Section header + 1-line helper
2. KPI strip (3 cards, sometimes 5 if needed)
3. Primary interaction area (filters/settings)
4. Primary chart
5. Supporting table/drilldown

Do not:
- Place long text blocks above charts
- Mix too many controls and charts in one card
- Use multiple competing accent colors

---

## 6) Interaction Principles

- Keep compute buttons explicit (`Apply Changes`, `Run`, `Reset`)
- Do not auto-run heavy recomputation on every keystroke
- Show local section loading states (not full screen)
- Preserve user edits when switching tabs if possible
- Use modal for deep scenario details only

---

## 7) Professional Quality Checklist

Before demo/release:
1. No overflow text in cards/tables/charts
2. All metrics have consistent units
3. Sidebar + stepper + right panel alignment is stable
4. Colors are consistent across all pages
5. Filters are compact and searchable
6. Every chart has clear title and axis labels
7. Loading/error states are visible and non-blocking
8. Mobile width does not cause horizontal clipping

---

## 8) Quick Starter Skeleton (React JSX)

```jsx
<div className="app-shell">
  <aside className="left-nav">{/* nav */}</aside>
  <main className="center-canvas">
    <section className="step-tabs">{/* step buttons */}</section>
    <section className="kpi-row">{/* cards */}</section>
    <section className="content-card">{/* chart/table */}</section>
  </main>
  <aside className="right-settings">{/* filters/settings */}</aside>
</div>
```

---

## 9) Naming Conventions (for consistency)

- `Panel`: right filter/settings containers
- `Card`: metric/chart/table wrappers
- `Row`: horizontal metric rows
- `Section`: page-level block
- `Delta`: % change line
- `Reference`: comparison baseline mode (`Y-o-Y`, `Q-o-Q`)

Keep naming stable across components so future teams can ship changes faster.

