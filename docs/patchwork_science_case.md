# Patchwork science case: mean molecular weight across the sub-Neptunes

Status 2026-10-22. Patchwork fit v1.6, 15 targets fitted (13 used; TOI-125 b
and TOI-270 c held because the GP absorbs transit depth), 6 still fitting
(GJ 3090 b, TOI-776 b, TOI-776 c, TOI-836 b, HD 15337 c, K2-18 b). Figures:
`scripts/patchwork/plot_sing_library.py`; published values with citations:
`scripts/patchwork/literature_subneptunes.json`,
`scripts/patchwork/literature_g395h.json`.

## The question

A sub-Neptune's transmission spectrum is set first by the scale height of its
atmosphere, H = k T / (μ m_u g). The mean molecular weight μ is therefore the
quantity JWST transmission spectra constrain most directly. It separates the
hypotheses about what these planets are: a primordial H₂/He envelope over rock
(μ ≈ 2.3), a metal-enriched or "miscible" envelope (μ ≈ 4–10), or a water- or
CO₂-dominated secondary atmosphere (μ ≈ 18–44).

## What the literature says (2024–2026)

- **High μ is common, but not universal.** GJ 1214 b ([M/H] ≈ 3.5, μ ≳ 15;
  Ohno+2025, arXiv:2410.10186), GJ 9827 d (H₂O > 31 %, μ > 7;
  Piaulet-Ghorayeb+2024, arXiv:2410.03527), GJ 357 b (μ > 8 at 3σ; Adams
  Redai+2025, arXiv:2507.07165), L 98-59 d (μ ≈ 10–32; Banerjee+2024,
  arXiv:2408.15707), TOI-270 d (μ = 5.5 ± 1.2; Benneke+2024, contested by
  Holmberg & Madhusudhan 2024), GJ 3090 b (> 100× solar; Ahrer+2025,
  arXiv:2504.20428). Against these, TOI-421 b (Davenport+2025,
  arXiv:2501.01498), K2-18 b (Madhusudhan+2023; Schmidt+2025; Fernández-
  Rodríguez+2026), TOI-732 c = LTT 3780 c (Rigby+2025, arXiv:2512.15844) and
  TOI-1231 b (Sarkar+2026, arXiv:2607.00973) are H₂-rich, the last three with
  CH₄.
- **No mass–metallicity trend yet.** Carbon abundances show no rise below
  ~10 M⊕; whether there is a "carbon plateau or cliff" is open, and the
  sample is too small to say (review, arXiv:2509.19247; PNAS
  10.1073/pnas.2416194122).
- **Temperature matters.** H₂O abundance rises with T_eq, from a
  non-detection in K2-18 b to > 30 % in GJ 9827 d; a water cold trap with
  T_crit ≳ 350 K is proposed (arXiv:2509.19247). Aerosol muting is strongest
  at 500–800 K, with clearer skies below ~500 K (Brande+2024, ApJL 961, L23,
  arXiv:2310.07714). Above ~850 K there is no CH₄ to photolyse into haze
  (Davenport+2025) — though HD 86226 c at 1311 K is flat (arXiv:2507.13439).
- **Formation diagnostics.** μ together with H₂O/CH₄ separates water-rich from
  water-poor building blocks (arXiv:2606.20464): K2-18 b and TOI-270 d look
  water-rich, TOI-421 b and GJ 3470 b water-poor.
- **Youth.** The ~20 Myr progenitor V1298 Tau b has a metal-poor H₂-rich
  atmosphere (Barat+2025), an order of magnitude below mature sub-Neptunes.
- **Many limits are cloud-dependent.** Most COMPASS-style constraints
  (TOI-836 c, TOI-776 c, HD 15337 b/c, TOI-125 b/c) are lower limits on
  metallicity that hold only if the opaque level is deep. Clouds and high μ
  mute features identically over G395H alone.

## What Patchwork adds

1. **One reduction for every planet.** Every target goes through identical
   exoTEDRF stages, one CRDS context, one fit (v1.6: joint NRS1+NRS2 white
   light, GP red noise, Stage 5.5 spot handling, Stage 6.5 contamination
   limits). Population trends are then not reduction-to-reduction scatter.
   Validation: v1.6 white-light depths match every published G395H depth to
   about 1 % (TOI-836.01 1.005/1.004, HD 15337 b 0.989/1.007, TOI-1231 b
   1.009/0.998, LTT 3780 c 1.001/0.970).
2. **First spectra.** Six Patchwork targets have no published JWST
   transmission spectrum: TOI-4336 A b (309 K), TOI-1468 c (338 K),
   TOI-270 c (490 K), HIP 9618 b (696 K), TOI-451 d (714 K), TOI-451 c
   (864 K).
3. **The temperate regime, densely sampled.** Five targets at 309–415 K
   (TOI-4336 A b, TOI-1231 b, TOI-1468 c, LTT 3780 c, L 98-59 d), plus K2-18 b
   (still fitting), straddle the proposed T_crit ≈ 350 K cold trap and the
   < 500 K "clear" regime — the place the literature says μ and clarity
   should change.
4. **Young planets.** TOI-451 (Psc–Eri stream, 120 Myr; Newton+2021) gives
   two young sub-Neptunes, c and d, to set against V1298 Tau b and the
   mature, metal-rich population. Their masses are poorly known
   (σ(g) = 76 % and 46 %), which limits any scale-height statement today.

## First result: CH₄ appears only in the coolest planets

`patchwork_band_indices`: the 3.3 µm CH₄ and 4.3 µm CO₂ band contrasts, each
band minus neighbouring windows on the same detector (so no NRS1–NRS2 offset
enters), in reference H₂/He scale heights (μ = 2.3).

| Planet | T_eq (K) | CH₄ 3.3 µm (H) | CO₂ 4.3 µm (H) | Published |
|---|---|---|---|---|
| TOI-4336 A b | 309 | **+0.91 ± 0.24 (3.8σ)** | +0.83 ± 0.51 | none |
| TOI-1231 b | 330 | +1.09 ± 0.58 (1.9σ) | +0.49 ± 0.64 | CH₄ strong, CO₂ moderate |
| TOI-1468 c | 338 | +0.95 ± 0.53 (1.8σ) | +1.26 ± 0.67 | none |
| LTT 3780 c | 362 | **+3.41 ± 0.67 (5.1σ)** | −0.80 ± 1.10 | CH₄ ~1 % in H₂ |
| L 98-59 d | 415 | +0.21 ± 0.23 | +0.15 ± 0.65 | high μ, tentative S |
| GJ 357 b | 518 | −0.11 ± 0.35 | −0.02 ± 0.59 | μ > 8 |
| GJ 1214 b | 567 | +0.21 ± 0.12 | −0.32 ± 0.20 | [M/H] ≈ 3.5, hazy |
| TOI-836.01 | 654 | −0.53 ± 0.51 | +1.96 ± 0.75 (2.6σ) | featureless |
| GJ 9827 d | 674 | −0.12 ± 0.29 | −0.85 ± 0.71 | steam world |
| HIP 9618 b | 696 | +0.34 ± 0.30 | −0.01 ± 0.33 | none |
| TOI-451 d | 714 | −0.54 ± 0.41 | +0.05 ± 0.71 | none |
| TOI-451 c | 864 | −0.19 ± 0.71 | −0.83 ± 1.05 | none |
| HD 15337 b | 1002 | −0.57 ± 1.77 | −0.43 ± 1.54 | flat, Z > 50× |

- The index recovers both published CH₄ detections in the sample (LTT 3780 c,
  TOI-1231 b), which is the check that it measures CH₄.
- **TOI-4336 A b shows a 3.8σ CH₄ band with no published spectrum** — a new
  candidate detection on a 309 K, 3.3 M⊕ planet. It needs the same check the
  published planets passed (an independent reduction, and a fit with a
  forward model rather than an index).
- Every planet at ≥ 415 K is consistent with no CH₄. The switch-off between
  362 and 415 K sits where the cold-trap and clear-sky regimes are proposed.
  With five cool planets this is suggestive, not established; K2-18 b
  (267 K) is the next point.
- TOI-836.01's 2.6σ CO₂ contrast contradicts the published featureless
  spectrum at low significance; it is the first target to re-examine.

## From band contrasts to μ — what is still missing

A band contrast in reference scale heights falls as 1/μ only at fixed
chemistry and cloud deck, so the index is not yet a μ measurement. The step
that turns Patchwork into a population μ study:

1. **A uniform forward-model fit per planet.** Chemical-equilibrium TauREx
   models (taurex-fastchem is installed) on a grid of metallicity (1–1000×
   solar, which spans μ 2.3–16) × opaque-pressure (clouds) × C/O, fitted to
   each combined spectrum with a free NRS1–NRS2 offset. Output: a posterior
   on μ and on the cloud pressure, the same way for every target. Needs the
   ExoMol R15000 cross-sections (`download_linelists.py`, several GB) — best
   on Fir.
2. **Break the cloud–μ degeneracy where possible.** Ten targets have NIRISS
   SOSS overlap. Shorter wavelengths separate a deep cloud deck from a heavy
   atmosphere, as for GJ 9827 d.
3. **Propagate the scale-height normalisation.** σ(g) runs from 5 % to 76 %
   (TOI-451 c), which scales any μ quoted in scale-height units.
4. **Resolve the held targets.** TOI-125 b's v1.6 depth is 28–34 % below the
   published value because the GP absorbs the transit. An out-of-transit-
   trained GP (proposed v1.7) fixes that, and TOI-270 c also needs a rule
   against step regressors inside the transit.

## Caveats on the current numbers

- Band contrasts use archive g and a uniform T_eq (T_eff √(R★/2a)).
- Spectra are as measured; an NRS1–NRS2 offset (−111 to +27 ppm here) is real
  for most targets. The indices avoid it by design; the stacked figure does
  not correct it.
- GP-inflated channel errors make χ²_ν < 1 for several spectra, so the indices
  are probably conservative.
