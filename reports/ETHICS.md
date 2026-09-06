# TerraShift — Ethics & Responsible Use

## Purpose
TerraShift publishes open-data-derived, aggregate land-cover change
statistics and maps for three publicly documented Indian change hotspots.
It is a demonstration of transparent environmental monitoring; it is NOT
an enforcement, surveillance, or adjudication tool.

## Data rights & privacy
- Inputs are open satellite products (Copernicus Sentinel-2, ESA
  WorldCover). No personal data is processed at any stage; the 90 m grid
  cannot resolve individuals, homes, or vehicles.
- No ground-truth collection was performed; no field visits, no
  interviews, no third-party private data.

## Honest reporting
- Accuracy metrics come from spatial-block cross-validation, the strict
  protocol that avoids inflated scores from spatial autocorrelation.
  Reported numbers include per-class recall, macro-F1 and Cohen's κ.
- Products are minimum-6-ha patches of disjoint class transitions; small
  or ambiguous change is deliberately EXCLUDED rather than guessed.
- Limitations (90 m resolution, label-year mismatch, optical-only gaps)
  are documented in README and REPORT.

## Potential misuse & mitigations
- **Blaming land owners:** change patches indicate land-cover
  transition, not legality. Converting forest to farmland may be
  authorised; built-up growth may be planned development. All products
  are labelled "change detection", never "illegal activity".
- **Policy overreach:** numbers are model estimates with quantified
  error, not census facts. Any policy use requires ground verification.
- **Stigmatised regions:** AOIs were chosen for documented, widely
  reported change dynamics (urbanisation, deforestation alerts, coalfield
  subsidence), not to single out communities.

## Environmental footprint
- Windowed COG reads (not full scenes) keep downloads small; per-scene
  windows are cached and reused across runs.
- Static precomputed artifacts served serverlessly: no idle compute.

## Attribution
Sentinel-2: Copernicus/ESA. WorldCover: ESA WorldCover consortium
(CC BY 4.0). Basemap: Esri World Imagery. TerraShift is not affiliated
with these providers.
