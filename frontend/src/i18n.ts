import { createContext, useContext } from 'react'

// Interface text in English and Hindi. Rainfall categories follow IMD's Hindi terminology; numbers, units,
// metric abbreviations (RMSE, ETS, ...) and place names stay as in the source data.
// The Hindi text needs review by a native speaker before operational use.

export type Lang = 'en' | 'hi'
export const LANGS: Lang[] = ['en', 'hi']

const STRINGS = {
  title: { en: 'Regime-Aware Rainfall Forecast Post-Processing', hi: 'रेजीम-आधारित वर्षा पूर्वानुमान पश्च-प्रसंस्करण' },
  subtitle: {
    en: 'SIH 26080 · NCMRWF / MoES · corrects NWP rainfall forecasts; does not forecast weather from scratch',
    hi: 'SIH 26080 · NCMRWF / MoES · NWP वर्षा पूर्वानुमान में सुधार करता है; स्वयं मौसम का पूर्वानुमान नहीं बनाता',
  },
  langToggle: { en: 'हिन्दी', hi: 'English' },
  langToggleLabel: { en: 'Switch to Hindi', hi: 'अंग्रेज़ी में देखें' },
  backendDown: { en: 'Backend unavailable', hi: 'सर्वर उपलब्ध नहीं' },
  modeReal: { en: 'Real data', hi: 'वास्तविक डेटा' },
  modeDemo: { en: 'Offline demonstration · real data', hi: 'ऑफ़लाइन प्रदर्शन · वास्तविक डेटा' },
  modeDemoTitle: {
    en: 'Runs offline from saved forecast products; every value comes from real data',
    hi: 'सहेजे गए पूर्वानुमान उत्पादों से ऑफ़लाइन चलता है; हर मान वास्तविक डेटा से है',
  },
  controlsLabel: { en: 'Forecast controls', hi: 'पूर्वानुमान नियंत्रण' },
  issued: { en: 'Forecast issued (00 UTC)', hi: 'पूर्वानुमान जारी (00 UTC)' },
  lead: { en: 'Lead', hi: 'अग्रिम अवधि' },
  leadGroup: { en: 'Lead day', hi: 'अग्रिम दिन' },
  day: { en: 'Day {n}', hi: 'दिन {n}' },
  mapLayer: { en: 'Map layer', hi: 'मानचित्र परत' },
  findDistrict: { en: 'Find district', hi: 'ज़िला खोजें' },
  findPlaceholder: { en: 'e.g. Mumbai', hi: 'जैसे Mumbai' },
  findHint: { en: 'Type a district name and choose a suggestion', hi: 'ज़िले का नाम लिखें और सुझाव चुनें' },
  export: { en: 'Export', hi: 'निर्यात' },
  caseStudy: { en: 'Case study', hi: 'केस स्टडी' },
  chooseCase: { en: 'Choose an event…', hi: 'एक घटना चुनें…' },
  period_validation: { en: 'validation', hi: 'सत्यापन' },
  period_test: { en: 'test', hi: 'परीक्षण' },
  period_operational_test: { en: 'live-mode test', hi: 'लाइव-मोड परीक्षण' },
  caseNote: {
    en: 'IMD observed up to {mm} mm in {district} on this day (verification only).',
    hi: 'इस दिन {district} में IMD ने {mm} mm तक वर्षा प्रेक्षित की (केवल सत्यापन)।',
  },
  downloadCsv: { en: 'District table (CSV)', hi: 'ज़िला तालिका (CSV)' },
  printPdf: { en: 'Print / save PDF', hi: 'प्रिंट / PDF सहेजें' },
  mapLabel: { en: 'Map', hi: 'मानचित्र' },
  loadingMap: { en: 'Loading map…', hi: 'मानचित्र लोड हो रहा है…' },
  mapUnavailable: { en: 'Map boundaries unavailable.', hi: 'मानचित्र सीमाएँ उपलब्ध नहीं।' },
  validFor: { en: 'Valid {date} (24 h ending 08:30 IST). ', hi: 'मान्य {date} (08:30 IST पर समाप्त 24 घंटे)। ' },
  mapCaption: {
    en: 'District values: area-weighted mean of 0.25° cells (probabilities: highest cell). Boundaries: Survey of India index maps via DataMeet (Census 2011 districts).',
    hi: 'ज़िला मान: 0.25° कोशिकाओं का क्षेत्र-भारित औसत (संभावना: सबसे ऊँची कोशिका)। सीमाएँ: DataMeet के माध्यम से भारतीय सर्वेक्षण विभाग के सूचकांक मानचित्र (जनगणना 2011 ज़िले)।',
  },
  noProducts: { en: 'No forecast products have been generated yet.', hi: 'अभी तक कोई पूर्वानुमान उत्पाद नहीं बना है।' },
  noData: { en: 'no data', hi: 'डेटा नहीं' },

  layer_raw_mm: { en: 'Raw NWP', hi: 'मूल NWP' },
  layer_corrected_mm: { en: 'Regime-aware', hi: 'रेजीम-आधारित' },
  layer_qm_mm: { en: 'Quantile-mapped', hi: 'क्वांटाइल-मैप्ड' },
  layer_unet_mm: { en: 'U-Net (deep learning)', hi: 'U-Net (डीप लर्निंग)' },
  note_unet_mm: { en: 'whole-map deep-learning corrector; wetter than observed on average', hi: 'पूरे मानचित्र पर डीप-लर्निंग सुधारक; औसतन प्रेक्षण से अधिक गीला' },
  layer_observed_mm: { en: 'Observed (IMD)', hi: 'प्रेक्षित (IMD)' },
  layer_p_heavy_max: { en: 'P(≥64.5 mm)', hi: 'P(≥64.5 mm)' },
  layer_p_very_heavy_max: { en: 'P(≥115.6 mm)', hi: 'P(≥115.6 mm)' },
  note_qm_mm: { en: 'preserves heavy-rain frequency', hi: 'भारी वर्षा की आवृत्ति बनाए रखता है' },
  note_observed_mm: { en: 'observation, for verification only', hi: 'प्रेक्षण, केवल सत्यापन के लिए' },
  rain0: { en: '< 2.5 mm (dry / very light)', hi: '< 2.5 mm (शुष्क / बहुत हल्की)' },
  rain1: { en: '2.5–15.5 light', hi: '2.5–15.5 हल्की' },
  rain2: { en: '15.6–64.4 moderate', hi: '15.6–64.4 मध्यम' },
  rain3: { en: '64.5–115.5 heavy', hi: '64.5–115.5 भारी' },
  rain4: { en: '115.6–204.4 very heavy', hi: '115.6–204.4 बहुत भारी' },
  rain5: { en: '≥ 204.5 extremely heavy', hi: '≥ 204.5 अत्यधिक भारी' },
  legend: { en: 'Legend', hi: 'संकेत' },

  sourceLabel: { en: 'Forecast source', hi: 'पूर्वानुमान स्रोत' },
  live: { en: 'Live forecast', hi: 'लाइव पूर्वानुमान' },
  archive: { en: 'Archived forecast', hi: 'संग्रहीत पूर्वानुमान' },
  forecastLive: { en: 'NOAA GEFSv12 operational, control member', hi: 'NOAA GEFSv12 परिचालन, नियंत्रण सदस्य' },
  forecastArchive: { en: 'NOAA GEFSv12 reforecast, control member', hi: 'NOAA GEFSv12 पुनः-पूर्वानुमान, नियंत्रण सदस्य' },
  verificationPending: {
    en: 'Verification pending: IMD has not published observations for these days',
    hi: 'सत्यापन लंबित: IMD ने इन दिनों के प्रेक्षण अभी प्रकाशित नहीं किए हैं',
  },
  verificationAttached: {
    en: 'Verification: IMD observed rainfall attached',
    hi: 'सत्यापन: IMD प्रेक्षित वर्षा संलग्न',
  },
  outOfSeason: {
    en: 'Valid outside June–September. The models were trained on monsoon-season days only, so these values are outside their tested range.',
    hi: 'जून–सितंबर से बाहर मान्य। मॉडल केवल मानसून ऋतु के दिनों पर प्रशिक्षित हैं, इसलिए ये मान उनकी परखी गई सीमा से बाहर हैं।',
  },

  regimeTitle: { en: 'Weather regime · valid {date}', hi: 'मौसम रेजीम · मान्य {date}' },
  regimeBars: { en: 'Regime probabilities', hi: 'रेजीम संभावनाएँ' },
  regime_NORMAL: { en: 'Normal', hi: 'सामान्य' },
  regime_ACTIVE: { en: 'Active monsoon', hi: 'सक्रिय मानसून' },
  regime_BREAK: { en: 'Break monsoon', hi: 'मानसून विराम' },
  regime_MONSOON_DEPRESSION: { en: 'Monsoon depression', hi: 'मानसून अवदाब' },
  classifierLive: { en: 'a LightGBM classifier on forecast fields only', hi: 'केवल पूर्वानुमान क्षेत्रों पर LightGBM वर्गीकारक' },
  classifierArchive: {
    en: 'a LightGBM classifier on forecast fields and rainfall observed before the forecast was issued',
    hi: 'पूर्वानुमान क्षेत्रों और पूर्वानुमान जारी होने से पहले प्रेक्षित वर्षा पर LightGBM वर्गीकारक',
  },
  regimeNote: {
    en: 'Predicted by {model} (validation macro F1 {f1}{rule}). Treat as guidance, not certainty.',
    hi: '{model} द्वारा अनुमानित (सत्यापन macro F1 {f1}{rule})। इसे निश्चितता नहीं, मार्गदर्शन मानें।',
  },
  ruleScore: { en: '; a simple rule scores {v}', hi: '; एक सरल नियम का अंक {v}' },
  notEvaluated: { en: 'not evaluated', hi: 'मूल्यांकित नहीं' },

  districtTitle: { en: 'District forecast', hi: 'ज़िला पूर्वानुमान' },
  selectDistrict: { en: 'Select a district on the map.', hi: 'मानचित्र पर एक ज़िला चुनें।' },
  noCoverage: {
    en: 'No IMD 0.25° land cells overlap this district; no values are shown.',
    hi: 'इस ज़िले से कोई IMD 0.25° स्थल कोशिका नहीं मिलती; कोई मान नहीं दिखाया गया।',
  },
  lowCoverage: {
    en: 'Only {pct}% of this district is covered by the 0.25° grid; treat values with caution.',
    hi: 'इस ज़िले का केवल {pct}% भाग 0.25° ग्रिड में आता है; मानों को सावधानी से लें।',
  },
  areaMean: { en: 'Area-mean rainfall, valid {date} (day {n})', hi: 'क्षेत्र-औसत वर्षा, मान्य {date} (दिन {n})' },
  rowRaw: { en: 'Raw NWP (GEFS)', hi: 'मूल NWP (GEFS)' },
  rowCorrected: { en: 'Regime-aware correction', hi: 'रेजीम-आधारित सुधार' },
  rowQm: { en: 'Quantile-mapped', hi: 'क्वांटाइल-मैप्ड' },
  rowUnet: { en: 'U-Net (deep learning)', hi: 'U-Net (डीप लर्निंग)' },
  rowHeavy: { en: 'P(heavy ≥ 64.5 mm), highest cell', hi: 'P(भारी ≥ 64.5 mm), सबसे ऊँची कोशिका' },
  rowVeryHeavy: { en: 'P(very heavy ≥ 115.6 mm), highest cell', hi: 'P(बहुत भारी ≥ 115.6 mm), सबसे ऊँची कोशिका' },
  rowObserved: { en: 'Observed (IMD) — verification', hi: 'प्रेक्षित (IMD) — सत्यापन' },
  na: { en: 'n/a', hi: 'उपलब्ध नहीं' },

  explainTitle: { en: 'Why this correction', hi: 'यह सुधार क्यों' },
  correction: { en: 'Correction', hi: 'सुधार' },
  heavyRain: { en: 'Heavy rain', hi: 'भारी वर्षा' },
  correctionModel: {
    en: 'C1: LightGBM Tweedie with predicted regime probabilities and local regime as inputs',
    hi: 'C1: LightGBM Tweedie, अनुमानित रेजीम संभावनाओं और स्थानीय रेजीम को इनपुट के रूप में लेकर',
  },
  heavyModel: {
    en: 'LightGBM binary with regime inputs; probabilities used as fitted (isotonic calibration did not improve validation scores)',
    hi: 'रेजीम इनपुट के साथ LightGBM द्वि-वर्गीय; संभावनाएँ जैसी प्रशिक्षित हुईं वैसी ही (आइसोटोनिक अंशांकन से सत्यापन अंक नहीं सुधरे)',
  },
  validationRmse: { en: 'Validation RMSE', hi: 'सत्यापन RMSE' },
  rmseLine: {
    en: 'raw {raw} mm → corrected {corrected} mm (reforecast validation years, this lead)',
    hi: 'मूल {raw} mm → सुधारित {corrected} mm (पुनः-पूर्वानुमान सत्यापन वर्ष, यही अग्रिम दिन)',
  },
  contributions: {
    en: 'Largest contributions to the correction on this day (mean |SHAP|, share):',
    hi: 'इस दिन सुधार में सबसे बड़े योगदान (औसत |SHAP|, हिस्सा):',
  },
  contribBars: { en: 'Feature contributions', hi: 'इनपुट योगदान' },
  f_nwp_precip_mm: { en: 'Forecast rain at the cell', hi: 'कोशिका पर पूर्वानुमानित वर्षा' },
  f_nwp_log1p: { en: 'Forecast rain (log)', hi: 'पूर्वानुमानित वर्षा (लॉग)' },
  f_nwp_mean_3x3: { en: 'Forecast rain, 3×3 neighbourhood mean', hi: 'पूर्वानुमानित वर्षा, 3×3 पड़ोस औसत' },
  f_nwp_max_3x3: { en: 'Forecast rain, 3×3 neighbourhood max', hi: 'पूर्वानुमानित वर्षा, 3×3 पड़ोस अधिकतम' },
  f_nwp_mean_7x7: { en: 'Forecast rain, 7×7 neighbourhood mean', hi: 'पूर्वानुमानित वर्षा, 7×7 पड़ोस औसत' },
  f_nwp_max_7x7: { en: 'Forecast rain, 7×7 neighbourhood max', hi: 'पूर्वानुमानित वर्षा, 7×7 पड़ोस अधिकतम' },
  f_obs_climatology_mm: { en: 'Observed climatology (training years)', hi: 'प्रेक्षित जलवायु-औसत (प्रशिक्षण वर्ष)' },
  f_doy_sin: { en: 'Season (day of year)', hi: 'ऋतु (वर्ष का दिन)' },
  f_doy_cos: { en: 'Season (day of year)', hi: 'ऋतु (वर्ष का दिन)' },
  f_lat: { en: 'Latitude', hi: 'अक्षांश' },
  f_lon: { en: 'Longitude', hi: 'देशांतर' },
  f_lead_day: { en: 'Forecast lead time', hi: 'पूर्वानुमान अग्रिम अवधि' },
  f_elevation_m: { en: 'Terrain elevation', hi: 'भू-ऊँचाई' },
  f_coast_km: { en: 'Distance to coast', hi: 'तट से दूरी' },
  f_p_NORMAL: { en: 'Regime probability: normal', hi: 'रेजीम संभावना: सामान्य' },
  f_p_ACTIVE: { en: 'Regime probability: active', hi: 'रेजीम संभावना: सक्रिय' },
  f_p_BREAK: { en: 'Regime probability: break', hi: 'रेजीम संभावना: विराम' },
  f_p_MONSOON_DEPRESSION: { en: 'Regime probability: depression', hi: 'रेजीम संभावना: अवदाब' },
  f_local_INLAND: { en: 'Local regime: inland', hi: 'स्थानीय रेजीम: अंतर्देशीय' },
  f_local_COASTAL: { en: 'Local regime: coastal', hi: 'स्थानीय रेजीम: तटीय' },
  f_local_OROGRAPHIC: { en: 'Local regime: orographic', hi: 'स्थानीय रेजीम: पर्वतीय' },

  verifTitle: { en: 'Model comparison (validation 2016–2017)', hi: 'मॉडल तुलना (सत्यापन 2016–2017)' },
  verifLead: { en: 'Verification lead day', hi: 'सत्यापन अग्रिम दिन' },
  metric: { en: 'Metric', hi: 'मापक' },
  verifNote: { en: 'Bold = best in row. No confidence intervals on this table.', hi: 'मोटा = पंक्ति में सर्वश्रेष्ठ। इस तालिका में विश्वास अंतराल नहीं हैं।' },
  model_A_raw_nwp: { en: 'Raw NWP', hi: 'मूल NWP' },
  model_B1_quantile_mapping: { en: 'Quantile mapping', hi: 'क्वांटाइल मैपिंग' },
  model_B2_global_lgbm: { en: 'Global LightGBM', hi: 'वैश्विक LightGBM' },
  model_C1_regime_features: { en: 'Regime-aware C1', hi: 'रेजीम-आधारित C1' },
  model_C2_regime_split: { en: 'Regime-split C2', hi: 'रेजीम-विभाजित C2' },

  bulletinTitle: { en: 'District heavy-rain bulletin', hi: 'ज़िला भारी वर्षा बुलेटिन' },
  bulletinIssued: { en: 'Forecast issued {init} 00 UTC · day {n}, valid {date} (24 h ending 08:30 IST)', hi: 'पूर्वानुमान जारी {init} 00 UTC · दिन {n}, मान्य {date} (08:30 IST पर समाप्त 24 घंटे)' },
  bulletinTop: { en: 'Districts ranked by chance of heavy rain (≥ 64.5 mm) somewhere in the district', hi: 'ज़िले में कहीं भारी वर्षा (≥ 64.5 mm) की संभावना के क्रम में ज़िले' },
  district: { en: 'District', hi: 'ज़िला' },
  state: { en: 'State', hi: 'राज्य' },
  bulletinFooter: {
    en: 'Post-processed NWP guidance, not an official IMD warning. Probabilities are model estimates; see the validation scores on the dashboard. Rainfall categories follow IMD thresholds.',
    hi: 'पश्च-प्रसंस्कृत NWP मार्गदर्शन, IMD की आधिकारिक चेतावनी नहीं। संभावनाएँ मॉडल अनुमान हैं; डैशबोर्ड पर सत्यापन अंक देखें। वर्षा श्रेणियाँ IMD सीमाओं के अनुसार हैं।',
  },
} as const

export type StringKey = keyof typeof STRINGS
export const STRING_KEYS = Object.keys(STRINGS) as StringKey[]

export function translate(lang: Lang, key: StringKey, vars: Record<string, string | number> = {}): string {
  return STRINGS[key][lang].replace(/\{(\w+)\}/g, (m, name: string) => (name in vars ? String(vars[name]) : m))
}

export function hasKey(key: string): key is StringKey {
  return key in STRINGS
}

export const LangContext = createContext<Lang>('en')

export function useT() {
  const lang = useContext(LangContext)
  return (key: StringKey, vars?: Record<string, string | number>) => translate(lang, key, vars)
}
