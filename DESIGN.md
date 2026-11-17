---
name: Federated Computing Lab
description: ETH / Birkhäuser Scientific Monograph & Technical Atlas for Multi-Objective Reinforcement Learning in Federated Computing
colors:
  vermilion-calibration: "#C82D14"
  vermilion-bright: "#D9381E"
  cobalt-trajectory: "#1D4ED8"
  viridian-measured: "#0F766E"
  ochre-warning: "#B45309"
  vellum-ground: "#F5F2EB"
  cream-plate: "#FAF8F3"
  stone-recessed: "#EFECE4"
  stone-well: "#E4DFD3"
  drafting-rule: "#D5CFC2"
  obsidian-ink: "#121316"
  graphite-secondary: "#3A3832"
  slate-muted: "#57534A"
typography:
  display:
    fontFamily: "'Schibsted Grotesk', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
    fontSize: "2.5rem"
    fontWeight: 700
    lineHeight: 1.1
    letterSpacing: "-0.03em"
  headline:
    fontFamily: "'Schibsted Grotesk', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
    fontSize: "1.375rem"
    fontWeight: 700
    lineHeight: 1.2
    letterSpacing: "-0.025em"
  title:
    fontFamily: "'Schibsted Grotesk', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
    fontSize: "1.03125rem"
    fontWeight: 700
    lineHeight: 1.3
    letterSpacing: "-0.02em"
  body:
    fontFamily: "'Schibsted Grotesk', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
    fontSize: "0.875rem"
    fontWeight: 400
    lineHeight: 1.55
  label:
    fontFamily: "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
    fontSize: "0.75rem"
    fontWeight: 600
    lineHeight: 1.2
    letterSpacing: "0.04em"
rounded:
  xs: "2px"
  sm: "3px"
  md: "4px"
spacing:
  xs: "6px"
  sm: "10px"
  md: "16px"
  lg: "24px"
  xl: "48px"
components:
  button-primary:
    backgroundColor: "{colors.vermilion-calibration}"
    textColor: "{colors.cream-plate}"
    rounded: "{rounded.md}"
    padding: "11px 20px"
  button-secondary:
    backgroundColor: "{colors.stone-recessed}"
    textColor: "{colors.obsidian-ink}"
    rounded: "{rounded.md}"
    padding: "11px 18px"
  plate-container:
    backgroundColor: "{colors.cream-plate}"
    textColor: "{colors.obsidian-ink}"
    rounded: "{rounded.md}"
    padding: "20px"
---

# Design System: Federated Computing Lab

## Overview

**Creative North Star: "The ETH / Birkhäuser Scientific Monograph & Technical Atlas"**

Federated Computing Lab rejects the category default of pitch-black SaaS dashboards and glowing neon cards in favor of the tactile, high-contrast authority of a Swiss polytechnic engineering folio and scientific atlas. Every screen behaves like a calibrated figure plate printed on warm technical vellum (`#F5F2EB`) and cream drafting stock (`#FAF8F3`), structured by exposed `1px` obsidian hairlines (`#121316`) and subtle stone rules (`#D5CFC2`).

Because the product's core scientific contribution is anchoring a multi-node Federated Learning simulator to a single physical wall-meter reading (`PC1`: `0.065 kWh` over `190 min` = `20.53 W`) while modeling cluster-wide latency, active/idle power, and carbon intensity across `PC2`–`PC4`, the visual system treats provenance as structural architecture. Physical measurements, calibrated synthetic scenarios, and non-dominated Pareto frontiers each carry distinct typographic and chromatic stamps so an evaluator never confuses a modeled scenario with a physical meter observation.

**Key Characteristics:**
- Exposed `1px` architectural hairline grids (`#121316` outer plate frames, `#D5CFC2` internal subdivisions) instead of soft floating cards.
- Strict typographic tri-voice separation: `Schibsted Grotesk` for structural headings and controls, `STIX Two Text` for governing physical equations, and `JetBrains Mono` strictly for tabular numbers, seeds, parameters, and stdout logs.
- Simultaneous mechanism-and-proof layouts: synchronous active compute bars sit inline with hatched idle straggler-wait tails.
- High-contrast quantitative legibility across both live browser sessions (`demo_server.py`) and standalone offline single-file archives (`dashboard.html`).

## Colors

The palette pairs warm archival paper neutrals with four high-contrast scientific plotting inks chosen for maximum multi-series differentiation on light vellum grounds.

### Primary
- **Vermilion Calibration Ink** (`#C82D14` / `#D9381E`): Marks the primary `Run Simulation` actuator, the `Q-Learning RL` dynamic subset policy, active uplink paths in the hub-and-spoke topology, and non-dominated Pareto frontier curves.

### Secondary
- **Cobalt Trajectory Blue** (`#1D4ED8`): Represents the full-participation baseline (`FedAvg · All 4`) and dominated sweep points in multi-objective scalarization plots.
- **Viridian Measured Anchor** (`#0F766E`): Identifies physical wall-meter provenance (`MEASURED` badges on `PC1`), carbon-reduction delta stamps, and the `Lowest Energy 2` policy.

### Tertiary
- **Ochre Precision Amber** (`#B45309`): Encodes the `Fastest 2` greedy latency heuristic, live engine running states, and verification stamps.

### Neutral
- **Technical Vellum Ground** (`#F5F2EB`): Primary page background across all routes.
- **Cream Figure Plate** (`#FAF8F3`): Elevated surface for figure plates, command decks, and SVG chart canvases.
- **Recessed Stone Well** (`#EFECE4` / `#E4DFD3`): Used for table headers, segmented control tracks, formula callouts, and unselected standby node rows.
- **Drafting Hairline Rule** (`#D5CFC2`): Internal table and ledger cell dividers.
- **Obsidian Structural Ink** (`#121316`): Primary typography, outer plate borders (`1px solid`), and coordinator node fills.
- **Graphite Secondary Ink** (`#3A3832`): Supporting prose and table cell values (contrast > 8.5:1).
- **Slate Caption Ink** (`#57534A`): Axis labels, units, and parameter field captions (contrast > 6.2:1).

### Named Rules
**The Provenance Polarity Rule.** Every hardware or telemetry table must visually separate `PC1` (`Viridian Measured Anchor` stamp) from `PC2`–`PC4` (`Recessed Stone` scenario stamp) so physical measurements and modeled ratios are distinguishable at a glance.

**The White-Free Series Rule.** Never plot a data series in white or low-contrast grey on vellum or cream chart stages; all four policies (`#D9381E`, `#1D4ED8`, `#B45309`, `#0F766E`) maintain high-contrast ink visibility on both interactive SVGs and Matplotlib PNG exports.

## Typography

**Display Font:** `Schibsted Grotesk` (with `-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif` fallback)
**Body Font:** `Schibsted Grotesk` (with system sans-serif fallback)
**Equation Serif Font:** `STIX Two Text` (with `Georgia, serif` fallback)
**Label/Mono Font:** `JetBrains Mono` (with `ui-monospace, SFMono-Regular, Menlo, Consolas, monospace` fallback)

**Character:** Swiss neo-grotesk structural clarity paired with authentic mathematical serif notation and tabular monospace telemetry.

### Hierarchy
- **Display** (`700`, `clamp(1.75rem, 2.7vw, 2.5rem)`, `1.1` line-height, `-0.03em` tracking): Monograph cover title (`<h1>`).
- **Headline** (`700`, `1.375rem` / `22px`, `1.2` line-height, `-0.025em` tracking): Plate section headers (`<h2>`).
- **Title** (`700`, `1.03125rem` / `16.5px`, `1.3` line-height, `-0.02em` tracking): Panel and figure plate titles.
- **Body** (`400`, `0.875rem` / `14px`, `1.55` line-height, `65–72ch` max measure): Analytical explanations, mechanism summaries, and table descriptions.
- **Label / Telemetry** (`600`–`700`, `0.6875rem`–`0.78125rem` / `11px`–`12.5px`, `0.04em` tracking, `tabular-nums`): Plate index tags (`PLATE 01`), axis ticks, numeric inputs, and provenance stamps.

### Named Rules
**The Heading Leads First Rule.** Every `<h1>`, `<h2>`, and `<h3>` leads its container directly; never place a decorative kicker or eyebrow above a heading. Plate numbers and figure tags sit beside the heading in an inline baseline flex row (`head-inline`) after the heading element.

**The Strict Monospace Mandate Rule.** `JetBrains Mono` is reserved exclusively for numerical values, units, seeds, file paths, equations, and stdout logs—never as a decorative voice for ordinary prose.

## Layout

The application uses two coordinated spatial grammars:
- **Monograph Cover (`/`):** Full-width top masthead over a `1440px` centered container (`28px` horizontal padding). The opening viewport is a `1.02fr : 1.18fr` split architectural plate joining the research thesis and provenance ledger on the left with the interactive **Synchronous Round Timing & Idle Power Dissector** on the right.
- **Console Workspace (`/overview`, `/pipeline`, `/replay`, `/specs`):** A `1480px` max-width two-column asymmetric grid (`276px` sticky left navigation spine + `minmax(0, 1fr)` main empirical column with `24px` gap).
- **Responsive Breakpoints:**
  - At `≤ 1120px`, split 2-column plates (`mono-hero-grid`, `math-and-roster-grid`, `stage-grid`, `topo-grid`, `spec-two-col`) stack into a single vertical flow and 4-column grids reflow to `2×2`.
  - At `≤ 900px`, `.app-frame` collapses into a single column with the navigation spine at the top in a `2×2` tab grid.
  - At `≤ 600px`, all multi-column strips collapse to `1fr` for zero-horizontal-scroll mobile inspection.

## Elevation & Depth

This system is **flat and structural by default**, relying on tonal paper layering (`#F5F2EB` ground → `#FAF8F3` figure plate → `#EFECE4` recessed well) and crisp `1px` solid ink borders (`#121316` outer frames, `#D5CFC2` inner hairlines) rather than drop shadows or glass blur.

### Shadow Vocabulary
- **Floating Tooltip Elevation** (`box-shadow: 0 10px 26px rgba(18, 19, 22, 0.14)`): Applied exclusively to the cursor-tracking `.tooltip` overlay on SVG charts so it separates cleanly from underlying gridlines.

### Named Rules
**The Hairline-Over-Halo Rule.** Never use zero-offset colored glows, glassmorphism (`backdrop-filter`), or neobrutalist hard offset shadows (`4px 4px 0`). Boundaries are drawn with `1px solid` drafting lines.

## Shapes

- **Corner Radius Strategy:** Tight, technical corners (`2px` for badges and progress tracks, `3px` for emblems and nav items, `4px` for plates, buttons, and inputs).
- **Border Strategy:** `1px solid #121316` (`--line-strong`) defines primary plates, buttons, and ledger strips; `1px solid #D5CFC2` (`--line`) subdivides internal rows and cells.
- **Hatched Pattern Encoding:** Idle waiting tails in synchronous round bars use a `-45deg` repeating linear stripe (`#D6CEBF` / `#EAE4D7`) to visually distinguish wasted standby time from solid vermilion active compute.

## Components

### Buttons
- **Shape:** Technical `4px` radius with `1px solid #121316` structural border.
- **Primary (`.btn-run`, `.btn-vermilion`):** Vermilion Calibration Ink (`#C82D14`) background, `#FFFFFF` text, `600` weight, `7px 16px` (console) or `11px 20px` (cover) padding; darkens to `#A9230D` on hover.
- **Secondary (`.btn-tune`, `.btn-architectural`, `.btn-pill`):** Vellum (`#F5F2EB`) or Recessed Stone (`#EFECE4`) background, `#121316` ink, `1px solid #121316` border; shifts to `#EFECE4` / `#E4DFD3` on hover.

### Chips & Parameter Fields (`.chip-field`)
- **Style:** Inline label + tabular monospace numeric input (`44px` width, right-aligned) inside a `#F5F2EB` container with `1px solid #D5CFC2` border (`4px` radius).
- **State:** Shifts border to `#C82D14` and background to `#FAF8F3` on `:focus-within`.

### Plates & Ledger Strips (`.panel`, `.editorial-strip`)
- **Corner Style:** `4px` radius.
- **Background:** Cream Figure Plate (`#FAF8F3`), with lead summary cells tinted in `rgba(217, 56, 30, 0.075)`.
- **Border:** `1px solid #121316` outer perimeter with `1px solid #121316` or `#D5CFC2` internal cell dividers.
- **Internal Padding:** `20px` for panels; `16px 20px` for summary ledger cells.

### Navigation Spine (`.nav-rail`, `.nav-step`)
- **Style:** Sticky `276px` left folio spine containing the geometric hub-and-spoke emblem, 4 ordered plate tabs (`01`–`04`), a 3-row provenance anchor summary box, and run export controls.
- **Active State:** Spring-animated layout indicator (`layoutId="sidebar-active"`) with `1px solid #C82D14` border, soft vermilion fill, and `#C82D14` step index badge.

### Signature Component: Synchronous Round Timing & Idle Power Dissector (`.gantt-ledger`)
- **Purpose:** Visualizes why synchronous `FedAvg` wastes energy by decomposing each client node's round duration (`PC1`–`PC4` + `30 W Coordinator`) into active compute + network uplink (solid vermilion bar) vs. idle straggler-wait tail (diagonal hatched bar), paired with exact per-node `Wh` accounting.

## Do's and Don'ts

### Do:
- **Do** place `<h1>`, `<h2>`, and `<h3>` as the first element of their header group, putting `PLATE` or `FIG.` badges beside them in a `.head-inline` flex row.
- **Do** apply `.num` (`font-variant-numeric: tabular-nums`) and `white-space: nowrap` to all numeric table cells, parameter inputs, and KPI readouts.
- **Do** preserve the explicit provenance distinction between `PC1` (`MEASURED WALL METER`) and `PC2`–`PC4` (`MODELED SCENARIO`) across every table and chart.
- **Do** keep `run_project.py`'s `make_figures()` synchronized with the monograph palette (`#FAF8F3` figure ground, `#D9381E`, `#1D4ED8`, `#B45309`, `#0F766E`) and embed PNG provenance metadata.

### Don't:
- **Don't** place kickers or eyebrows above headings anywhere in the interface.
- **Don't** use generic rounded icon cards, nested card containers, or colored side-border stripes (`border-left` > 1px).
- **Don't** use training-data default display fonts or apply monospace to body prose.
- **Don't** claim or imply that `PC2`–`PC4` hardware profiles or simulated round CO₂ emissions are physical meter measurements.
