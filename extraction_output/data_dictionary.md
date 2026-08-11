# Data Dictionary — llm_extract clean dataset (domain: biodiesel)

Catalyst, process condition, and finished-fuel-property data from the waste-cooking-oil biodiesel (catalytic transesterification) literature.


| Column | Description | Unit |
|---|---|---|
| `paper_id` | Stable blake3 hash, primary key, joins back to scientific-pdf-pipeline output | - |
| `doi` / `title` / `first_author` / `journal` / `publication_year` | Bibliographic provenance | - |
| `agreement_status` | `full_agreement` \| `resolved_after_escalation` \| `unresolved_conflict` | categorical |
| `primary_confidence` / `cross_check_confidence` | Each model's self-reported confidence | 0-1 |
| `n_unresolved_fields` | Count of fields no model majority could resolve for this paper | count |
| `catalyst_type` | Catalyst family | categorical |
| `catalyst_loading_wt_pct` | Catalyst loading relative to oil | wt% |
| `methanol_to_oil_molar_ratio` | Methanol-to-oil molar ratio | molar ratio |
| `reaction_temperature_C` | Reaction temperature | °C |
| `reaction_time_min` | Reaction time | min |
| `wco_ffa_content_pct` | Free fatty acid content of the waste cooking oil feedstock | % |
| `fame_yield_pct` | FAME (biodiesel) yield | % |
| `cetane_number` | Cetane number | dimensionless |
| `kinematic_viscosity_40C_cSt` | Kinematic viscosity at 40°C | cSt (mm²/s) |
| `cfpp_or_pour_point_C` | Cold filter plugging point, or pour point if CFPP not reported | °C |
| `oxidative_stability_h` | Oxidative (Rancimat) stability | h |
| `density_kg_m3` | Density | kg/m³ |
| `flash_point_C` | Flash point | °C |
| `reference_standard` | Fuel standard referenced by the paper | categorical |
| `meets_cetane_spec` | Cetane number >= 51 (ASTM D6751 minimum) (nullable: null = not reported, not 'fails spec') | boolean |
| `meets_viscosity_spec` | Kinematic viscosity at 40°C within the ASTM D6751/EN 14214 diesel-like band (nullable: null = not reported, not 'fails spec') | boolean |
| `meets_oxidative_spec` | Oxidative stability >= 6 h (EN 14214 minimum) (nullable: null = not reported, not 'fails spec') | boolean |
| `meets_cfpp_target` | CFPP/pour point <= 0°C (adjust to your target climate) (nullable: null = not reported, not 'fails spec') | boolean |

**Nullability note:** every extracted field and binary target uses null/NA to mean 'not reported in the source paper.' Treat null and False as distinct categories in any downstream model -- collapsing them reintroduces the exact bias the literature already has (most papers under-report properties they did not optimize for).