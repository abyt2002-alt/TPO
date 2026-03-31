# Frontend Skeleton Blueprint (Price Promo Optimization App)

## Purpose
This file defines a reusable frontend skeleton so the new app keeps the same clean structure:
- fixed top header
- left navigation rail
- central working canvas
- right settings/filter panel
- consistent colors, typography, spacing, and component behavior

Use this as the baseline before building feature pages.

---

## 1) Page Layout Structure

## Global shell
- **Header (top, fixed height 64px)**
  - App logo/icon + app title
  - Mobile controls (menu + settings toggle)
- **Body (full height below header)**
  - **Left Sidebar** (desktop)
    - width: `256px`
    - vertical navigation only
  - **Main Content**
    - fluid width
    - scrollable page area
    - page sections as cards
  - **Right Sidebar** (desktop optional, settings/filters)
    - width: `320px`
    - sticky section header
    - scrollable content

## Mobile behavior
- Left nav collapses to overlay menu.
- Right settings opens as slide-in drawer.
- Main content remains primary focus.

---

## 2) Visual System

## Typography
- Font: `Inter`
- Base text color: `#333333`
- Secondary text: `#666666`
- Small helper text: `#999999`

## Color tokens
- Primary: `#458EE2`
- Success/Green: `#41C185`
- Accent/Amber: `#FFBD59`
- Warning soft: `#FFCF87`
- Danger: `#E85D5D`
- Canvas background: `#F5F5F5`
- Card surface: `#FFFFFF`

## Surfaces and borders
- Card background: white
- Border: light gray (`#E5E7EB` style)
- Radius:
  - cards: `12px`
  - small controls: `8px`
- Shadows:
  - subtle only (`shadow-sm`)

---

## 3) Spacing and Grid Rules

- Outer page padding: `16px` mobile, `24px` tablet, `32px` desktop
- Section gap: `16px`
- Card internal padding: `16px` (small), `20px` (primary)
- Form control heights:
  - input/select: `40px`
  - primary button: `40px`

## KPI rows
- Use 3-column layout on desktop for primary KPI cards.
- Collapse to 1-column on mobile.

---

## 4) Component Blueprint

## A) Header
- Left: brand icon + title (`text-xl`, bold)
- Right: mobile-only action buttons

## B) Left Navigation
- Vertical stack of nav items:
  - icon + label
  - active state: white background + primary border
  - hover state: light border highlight

## C) Main Content
- Top-level sections in card blocks:
  - Page title + short description
  - KPI strip
  - charts/tables
  - action area

## D) Right Settings Panel
- Sticky heading: `Settings`
- Inputs grouped by section cards
- Actions at bottom:
  - primary action button
  - reset/secondary button

---

## 5) Interaction Rules

- Avoid full-page blocking loaders.
- Use section-level loading states only.
- Keep user inputs editable while background fetch runs where possible.
- Show short clear error banners inside the active section only.
- Keep destructive actions explicit (confirm when needed).

---

## 6) Recommended Folder Skeleton

```text
frontend/src/
  components/
    layout/
      AppShell.jsx
      LeftNav.jsx
      RightPanel.jsx
      TopHeader.jsx
    common/
      KPIBox.jsx
      SectionCard.jsx
      EmptyState.jsx
      LoadingState.jsx
  pages/
    Dashboard.jsx
    Step1.jsx
    Step2.jsx
    Step3.jsx
    Step4.jsx
    Step5.jsx
  services/
    api.js
  styles/
    tokens.css
```

---

## 7) Starter JSX Skeleton

```jsx
export default function AppShell({ children, rightSidebar }) {
  return (
    <div className="h-screen flex flex-col bg-canvas">
      <header className="h-16 bg-white border-b border-gray-200 px-6 flex items-center">
        <h1 className="text-xl font-bold text-body">Price Promo Optimization</h1>
      </header>

      <div className="flex-1 flex min-h-0">
        <aside className="hidden lg:block w-64 bg-white border-r border-gray-200 p-4">
          {/* Left nav */}
        </aside>

        <main className="flex-1 overflow-y-auto p-6">
          {children}
        </main>

        {rightSidebar && (
          <aside className="hidden lg:block w-80 bg-white border-l border-gray-200">
            <div className="p-4 border-b border-gray-200 font-semibold">Settings</div>
            <div className="p-4 overflow-y-auto">{rightSidebar}</div>
          </aside>
        )}
      </div>
    </div>
  );
}
```

---

## 8) Quality Checklist

Before shipping any page:
1. Left nav, center content, right settings alignment is clean.
2. No text overflow in KPI cards.
3. Mobile layout works without horizontal scroll.
4. Filter controls are compact and consistent.
5. Chart colors match token set (blue/green/amber/red only).
6. Loading and error states are visible but non-disruptive.
7. Page remains usable while data refresh runs.

