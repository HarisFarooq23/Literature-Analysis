# Data Dictionary — llm_extract clean dataset (domain: saf_hefa)

Feedstock quality, catalyst structure, hydroprocessing conditions, deoxygenation/upgrading pathway, product distribution, and finished-SAF property data from the HEFA literature on waste cooking oil conversion to jet-range SPK. multi_experiment=true: one record per catalyst/condition combination, not one per paper.


| Column | Description | Unit |
|---|---|---|
| `paper_id` | Stable blake3 hash, primary key, joins back to scientific-pdf-pipeline output | - |
| `doi` / `title` / `first_author` / `journal` / `publication_year` | Bibliographic provenance | - |
| `agreement_status` | `full_agreement` \| `resolved_after_escalation` \| `unresolved_conflict` | categorical |
| `primary_confidence` / `cross_check_confidence` | Each model's self-reported confidence | 0-1 |
| `n_unresolved_fields` | Count of fields no model majority could resolve for this paper | count |
| `experiment_id` | Key: `{paper_id}_{experiment_idx}`. Stable within one dataset build (repeated `build-dataset` runs against the same extracted_records.jsonl); NOT guaranteed stable across a full re-extraction, since experiment order comes from cross-model matching, not fixed paper structure. For an ID surviving re-extraction, join on `(paper_id, evidence_location)` instead. | - |
| `experiment_idx` | Position of this experiment within its paper (0-indexed) | - |
| `n_experiments` / `n_claims` | Total experiments/claims extracted from this row's paper | count |
| `experiment_match_status` | Cross-model agreement for THIS experiment specifically | categorical |
| `match_method` | `identity_key` \\| `position` -- how this experiment was paired across models; `position` warrants extra scrutiny | categorical |
| `n_unresolved_experiment_fields` | Fields within this experiment with no resolved majority value | count |
| `evidence_location` / `evidence_text` | Where in the paper this experiment's data came from, and the verbatim supporting text | - |
| `ml_core_features_complete` | True iff every one of `active_metal`, `support_type`, `metal_loading_wt_pct`, `reaction_temperature_C`, `reaction_pressure_bar`, `oil_conversion_pct` is non-null for this row -- filter on this for a ready-to-model shallow-ML subset | boolean |
| `feedstock_type` | Lipid feedstock used in the paper | categorical |
| `ffa_content_pct` | Free fatty acid content of the feedstock | wt% |
| `moisture_content_pct` | Moisture content of the feedstock | wt% |
| `acid_value_mgKOHg` | Acid value of the feedstock | mg KOH/g |
| `iodine_value` | Iodine value (degree of unsaturation) | g I2/100g |
| `dominant_fatty_acid_chain` | Dominant fatty-acid carbon-chain range reported for the feedstock (free text, e.g. "C16-C18"). Null if not reported.
 | - |
| `feedstock_sulfur_ppm` | Sulfur content of the feedstock (catalyst-poison relevant) | ppm |
| `active_metal` | Active metal(s)/alloy exactly as the authors wrote it, ISOLATED from support and loading (e.g. "NiMo" or "Pt", not "10% NiMo/Al2O3" or "10 wt% Pt"). Support goes in support_material, loading goes in metal_loading_wt_pct -- do not fold either into this field. Do not expand trade names or invent oxidation states.
 | - |
| `support_material` | Support material exactly as written, e.g. "gamma-Al2O3", "HZSM-5", "SAPO-11", "activated_carbon". Canonical form added post-extraction.
 | - |
| `support_type` | Structural class of the support, classified from support_material. Set from the support's known family: HZSM-5/H-Beta/SAPO-n/MFI-type frameworks -> zeolite; SBA-n/MCM-n/KIT-n -> mesoporous_silica; Al2O3/TiO2/SiO2/CeO2/ZrO2/MgO single-component oxides -> metal_oxide; two-or-more-oxide combinations (e.g. CeO2-ZrO2, Al2O3-SiO2) -> mixed_metal_oxide; activated_carbon/CNT/graphene/graphitic carbon -> activated_carbon; UiO-n/ZIF-n/MIL-n framework names -> MOF; layered-double-hydroxide-derived supports -> hydrotalcite_derived. Use not_specified only if support_material itself is null; use other if support_material is given but doesn't fit any class above.
 | categorical |
| `promoter` | Promoter element/oxide if reported (null if none reported) | - |
| `promoter_loading_wt_pct` | Promoter loading relative to support or total catalyst mass, exactly as basis is reported. Null if no promoter is reported, even if promoter identity itself is also null.
 | wt% |
| `metal_loading_wt_pct` | Active metal loading relative to support | wt% |
| `synthesis_method` | Catalyst preparation route, from the paper's Experimental/Synthesis/ Preparation section. incipient_wetness_impregnation is by far the most common in HEFA catalyst papers -- do not default to it if the method isn't actually described; use not_specified instead.
 | categorical |
| `calcination_temperature_C` | Calcination temperature after synthesis, before reduction/reaction | °C |
| `calcination_time_h` | Calcination hold time | h |
| `calcination_atmosphere` | Atmosphere during calcination | categorical |
| `reduction_temperature_C` | Pre-reaction reduction/activation temperature (typically under H2 or H2/inert flow), reported separately from the reaction temperature itself. Do not confuse with reaction_temperature_C below.
 | °C |
| `reduction_time_h` | Pre-reaction reduction/activation hold time | h |
| `bet_surface_area_m2g` | BET surface area of the fresh catalyst | m²/g |
| `average_pore_diameter_nm` | Average pore diameter (BJH/BET) | nm |
| `metal_dispersion_pct` | Metal dispersion from H2/CO chemisorption | % |
| `metal_particle_size_nm` | Average metal particle/crystallite size | nm |
| `particle_size_method` | Technique the metal_particle_size_nm value came from. TEM and XRD-Scherrer routinely disagree for the same catalyst (different averaging, different sensitivity to small particles) -- recording which method produced the value is necessary to compare particle sizes across papers correctly. Set both only if the paper reports values from both methods and metal_particle_size_nm above is itself an average of the two as stated by the authors.
 | categorical |
| `acidity_type` | Dominant acid site type (from NH3-TPD, pyridine-FTIR, etc.) | categorical |
| `nh3_uptake_mmol_g` | Total acid site density from NH3-TPD | mmol/g |
| `reducibility_temp_C` | Peak reduction temperature from H2-TPR (proxy for metal-support interaction strength). If the paper reports a SINGLE reduction peak, use this field. If it distinguishes two peaks (a common pattern for NiMo/Al2O3-type catalysts: a lower-temperature peak for easily reducible, weakly-bound surface/octahedral species and a higher- temperature peak for bulk/tetrahedral species strongly interacting with the support), set tpr_peak_low_C and tpr_peak_high_C below instead, and set this field to whichever peak the authors describe as dominant (usually the higher-intensity one) if they single one out -- otherwise leave this null rather than picking arbitrarily.
 | °C |
| `tpr_peak_low_C` | Lower-temperature H2-TPR reduction peak, when the paper reports two (or more, in which case record the lowest) distinct peaks rather than a single reduction temperature. Typically assigned to weakly- bound, easily reducible surface metal-oxide species. Null if the paper reports only one undifferentiated reduction temperature (use reducibility_temp_C for that case instead).
 | °C |
| `tpr_peak_high_C` | Higher-temperature H2-TPR reduction peak, when the paper reports two (or more, in which case record the highest) distinct peaks. Typically assigned to bulk metal-oxide species or species in strong interaction with the support (e.g. nickel/cobalt aluminate-like phases). Null if the paper reports only one undifferentiated reduction temperature.
 | °C |
| `co2_uptake_mmol_g` | Total basic site density from CO2-TPD, when reported. Basicity (alongside acidity from NH3-TPD, see nh3_uptake_mmol_g) affects triglyceride/FFA adsorption and is reported for a meaningful minority of HEFA catalyst characterization studies -- null if CO2-TPD is not performed or not reported, do not infer from acidity data.
 | mmol/g |
| `catalyst_type` | Structural classification of the catalyst material itself (distinct from support_type, which classifies the SUPPORT). supported_metal: metal/metal-oxide active phase dispersed on a separate support (the overwhelming majority of HEFA hydroprocessing catalysts). bulk_metal: unsupported metal or metal-oxide catalyst with no distinct support phase. bifunctional_catalyst: paper explicitly frames the catalyst as combining a hydrogenation function (metal) and an acid function (e.g. zeolite/SAPO) for isomerization/cracking in the same material, not merely "metal on an acidic support" as an incidental description -- only use this value if the paper itself uses bifunctional framing, not as a default for any zeolite- supported metal.
 | categorical |
| `reaction_temperature_C` | Hydroprocessing temperature | °C |
| `reaction_pressure_bar` | H2 pressure | bar |
| `h2_to_oil_ratio_NL_L` | H2-to-oil feed ratio | NL H2 / L oil |
| `lhsv_h` | Liquid hourly space velocity | h⁻¹ |
| `whsv_h` | Weight hourly space velocity | h⁻¹ |
| `reactor_type` | Reactor configuration | categorical |
| `deoxygenation_pathway` | Dominant deoxygenation route. HDO=hydrodeoxygenation (-H2O, chain retained). DCO=decarbonylation (-CO, chain shortened). DCO2= decarboxylation (-CO2). mixed_HDO_DC=both. Infer from gas-phase byproducts (H2O/CO/CO2) or chain-length shift if not stated.
 | categorical |
| `downstream_processing` | Upgrading step after deoxygenation. hydroisomerization=skeletal branching. hydrocracking=chain scission. Set none only if straight-chain n-paraffin product with no further processing is explicitly described.
 | categorical |
| `deoxygenation_degree_pct` | Fraction of feedstock oxygen removed | % |
| `oil_conversion_pct` | Feedstock (triglyceride/FFA) conversion. NOT required: some papers report only yield or selectivity for a given experiment without an explicit conversion number (e.g. a screening-table row marked "n.d." or reporting only product yield) -- treating this as mandatory would fail validation for the entire experiment list over one missing value in one row, losing an otherwise fully extractable paper.
 | % |
| `iso_to_normal_paraffin_ratio` | Iso-to-normal paraffin ratio in the liquid product | ratio |
| `c8_c16_jet_range_selectivity_pct` | Selectivity to C8-C16 jet-boiling-range hydrocarbons | % |
| `residual_oxygenate_content_pct` | Residual oxygen-containing compounds in liquid product | % |
| `jet_fraction_yield_pct` | Yield of distilled jet-range cut relative to feedstock mass | wt% |
| `tof_h` | Turnover frequency if reported. Record the numeric value only; report the basis (total_metal / accessible_metal / BET / mass) in evidence_text of the EvidenceSpan for this experiment.
 | h⁻¹ |
| `activation_energy_kJ_mol` | Apparent activation energy if reported (Arrhenius analysis) | kJ/mol |
| `kinetic_model` | The kinetic rate model the paper actually fits to its data, if any quantitative kinetic analysis (rate constant, reaction order, or Arrhenius parameters) is reported at all. pseudo_first_order: rate expressed as first-order in a single reactant concentration (by far the most common simplification in HEFA/hydroprocessing kinetics). power_law: rate = k * [reactant]^n with an explicitly fitted non-unity order n. LHHW (Langmuir-Hinshelwood-Hougen-Watson): explicit adsorption-term denominator in the rate expression -- only use this if the paper's rate equation itself has that form, not merely because adsorption is discussed qualitatively. Set not_specified if no quantitative kinetic model is fitted at all (most papers only report conversion/selectivity with no rate law -- that is the common case, not an extraction failure).
 | categorical |
| `rate_constant` | Numeric value of the fitted rate constant k, exactly as reported (whatever basis/units the paper uses -- record the actual unit string in rate_constant_unit_reported, since the basis varies widely: min^-1, h^-1, mol/(g_cat*h), L/(mol*h), etc. and is NOT standardized across this literature the way TOF's h^-1 basis is). Null if kinetic_model is not_specified.
 | varies -- see rate_constant_unit_reported |
| `rate_constant_unit_reported` | The rate constant's unit EXACTLY as printed in the paper (e.g. "h^-1", "L mol^-1 h^-1", "mol g_cat^-1 h^-1 bar^-1"). Do not normalize or convert -- record verbatim so unit harmonisation can be done deliberately downstream rather than guessed at extraction time.
 | - |
| `reaction_order` | Fitted reaction order n if kinetic_model is power_law (or an apparent/pseudo order reported for pseudo_first_order / pseudo_second_order, which is 1.0 / 2.0 by definition of the model name -- still record it if the paper states it explicitly). Null if no order is reported.
 | - |
| `pre_exponential_factor` | Arrhenius pre-exponential factor A, exactly as the paper reports the NUMBER (do not compute or convert). Null if no Arrhenius analysis is reported alongside activation_energy_kJ_mol.
 | same basis as rate_constant |
| `pre_exponential_factor_is_log10` | true if the value in pre_exponential_factor is log10(A) as printed (papers commonly report "ln A" or "log A" instead of A itself, since A is often 10-20 orders of magnitude and unwieldy to print directly -- e.g. "ln A = 13.2" or "log10(A) = 5.7"). false if the paper prints A directly (e.g. "A = 2.3 x 10^13"). This distinction is NOT cosmetic: treating a log-scale value as the raw factor (or vice versa) corrupts any downstream Arrhenius-based feature by many orders of magnitude. If the paper uses natural log (ln A) rather than log10, still set this true and note the ln-vs-log10 distinction in this experiment's evidence_text -- the boolean's job is only to flag "this number is on a log scale", not which log base.
 | boolean |
| `coke_amount_wt_pct` | Coke deposited on catalyst after reaction (from TGA) | wt% |
| `structure_property_summary` | A single 1-3 sentence synthesis, IN YOUR OWN WORDS, of how THIS experiment's catalyst structure relates to its performance outcome -- e.g. "NiMo/SAPO-11 with 15 wt% loading and predominantly Bronsted acid sites (0.42 mmol/g NH3 uptake) gave 92% jet-range selectivity at 350C, consistent with strong hydroisomerization activity from the bifunctional acid/metal sites." Ground this ONLY in values already captured elsewhere in this SAME experiment record (active_ metal, support_type, characterization fields, reaction conditions, performance fields) -- do not introduce a number, mechanism, or claim that isn't already one of this experiment's own extracted fields or directly stated in the source text for this experiment. Paraphrase; do not quote a sentence from the paper verbatim. This field exists so each experiment record is independently useful as a retrievable text passage (for RAG / semantic search) without needing the full paper's context, and as a same-record sanity check that connects the numeric fields into their intended causal story.
 | - |
| `freezing_point_C` | Freezing point of the finished SPK/jet fraction | °C |
| `density_15C_kg_m3` | Density at 15°C | kg/m³ |
| `kinematic_viscosity_neg20C_mm2_s` | Kinematic viscosity at -20°C | mm²/s |
| `flash_point_C` | Flash point | °C |
| `net_heat_of_combustion_MJ_kg` | Net heat of combustion / specific energy | MJ/kg |
| `astm_reference_standard` | SAF standard the paper explicitly benchmarks against. Only set if the paper cites the standard by name — do not infer from "SAF" or "jet fuel".
 | categorical |
| `meets_freezing_point_spec` | Freezing point <= -40°C (ASTM D7566 Annex A2) (nullable: null = not reported, not 'fails spec') | boolean |
| `meets_density_spec` | Density at 15°C <= 772 kg/m³ (ASTM D7566 Annex A2 SBC limit) (nullable: null = not reported, not 'fails spec') | boolean |
| `meets_viscosity_spec` | Kinematic viscosity at -20°C <= 8.0 mm²/s (ASTM D7566 Annex A2) (nullable: null = not reported, not 'fails spec') | boolean |
| `meets_flash_point_spec` | Flash point >= 38°C (ASTM D7566/D1655 minimum) (nullable: null = not reported, not 'fails spec') | boolean |
| `meets_combustion_spec` | Net heat of combustion >= 42.8 MJ/kg (ASTM D7566/D1655 minimum) (nullable: null = not reported, not 'fails spec') | boolean |
| `high_jet_range_selectivity` | C8-C16 selectivity >= 70% (nullable: null = not reported, not 'fails spec') | boolean |
| `high_conversion` | Feedstock conversion >= 95% (nullable: null = not reported, not 'fails spec') | boolean |

**Nullability note:** every extracted field and binary target uses null/NA to mean 'not reported in the source paper.' Treat null and False as distinct categories in any downstream model -- collapsing them reintroduces the exact bias the literature already has (most papers under-report properties they did not optimize for).