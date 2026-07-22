"""
=============================================================================
 Panel de Modelado Epidemiológico COVID-19 — Versión General
 Trabajo de Fin de Máster (TFM) — ML + XAI + Simulación de Políticas SEIR
 V3: Selección dinámica de país, soporte multi-algoritmo, tema clínico claro
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import io
import tempfile, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import shap
import streamlit as st

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="COVID-19 · Panel de Modelado Epidemiológico",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# PALETA DE COLORES — Tema clínico claro
# ─────────────────────────────────────────────────────────────────────────────
C = {
    "real":      "#D62839",
    "pred":      "#0077B6",
    "strict":    "#0A9E6E",
    "loose":     "#E07A10",
    "accent":    "#0096C7",
    "accent2":   "#48CAE4",
    "neutral2":  "#5A8FA8",
    "purple":    "#7B2FBE",
    "gold":      "#C9800A",
    "bg":        "#F4F7FA",
    "white":     "#FFFFFF",
    "border":    "#C8DCE8",
    "border2":   "#A0BFD0",
    "text":      "#0D2035",
    "muted":     "#4A7A95",
    "dim":       "#8AABB8",
    "shadow":    "rgba(13,32,53,0.08)",
}

# ─────────────────────────────────────────────────────────────────────────────
# INTERNACIONALIZACIÓN (ES / EN)
# ─────────────────────────────────────────────────────────────────────────────
if "lang" not in st.session_state:
    st.session_state.lang = "es"

T = {
    "es": {
        "brand_title": "Panel COVID-19",
        "brand_tagline": "TFM · ML + XAI + SEIR",
        "lang_label": "🌐 Idioma",
        "dataset_label": "📂 Conjunto de datos",
        "dataset_help": "Sube el archivo Europe_countries_covid19.csv",
        "test_country_label": "🌍 País de test",
        "test_country_note": "El país seleccionado se excluye del entrenamiento. Todos los demás entrenan el modelo.",
        "train_n_countries": "Entrenamiento ({n} países):",
        "study_period_label": "📅 Periodo de estudio",
        "period_pre": "Pre-vacunación",
        "period_full": "Periodo completo COVID-19",
        "period_badge_pre": "🟢 Pre-vacunación",
        "period_badge_full": "🟡 Periodo completo",
        "model_features_label": "⚙️ Variables del modelo",
        "vacc_warning": "⚠️ <b>people_fully_vaccinated_per_hundred</b> = 0 en modo pre-vacunación.",
        "min_feature_warning": "⚠️ Selecciona al menos 1 variable.",
        "algorithm_label": "🤖 Algoritmo",
        "hyperparams_label": "🔧 Hiperparámetros",
        "max_depth_unlimited": "max_depth (0=ilimitado)",
        "summary_model": "Modelo:",
        "summary_test": "Test:",
        "summary_train": "Entrenamiento:",
        "summary_features": "Variables:",
        "summary_period": "Periodo:",
        "summary_period_pre_short": "Pre-vacc.",
        "summary_period_full_short": "Completo",
        "hero_title": "🦠 COVID-19 · <span>Panel Epidemiológico</span>",
        "hero_subtitle": "Machine Learning + IA Explicable + Simulación de Políticas SEIR",
        "hero_meta": "TFM · ML · SHAP · SEIR",
        "gate_title": "Sube tu conjunto de datos para empezar",
        "gate_text": "Sube <b style='color:{text_color};'>Europe_countries_covid19.csv</b> desde la barra lateral. Después elige el <b>país de test</b>, el <b>algoritmo</b> y los <b>hiperparámetros</b> — el modelo se entrena automáticamente con el resto de países.",
        "gate_chip_shap": "Explicabilidad SHAP",
        "gate_chip_seir": "Simulación SEIR",
        "gate_chip_policy": "Escenarios de Política",
        "err_prepare_data": "❌ Error al preparar los datos: {e}",
        "warn_not_enough_data": "⚠️ No hay datos suficientes tras el filtrado. Prueba a cambiar el periodo, el país o las variables.",
        "err_train_model": "❌ Error al entrenar el modelo: {e}",
        "active_config": "{icon} <b>{algo_icon} {algo_name}</b> entrenado con <b>{n_train}</b> países · Evaluado en <b style='color:{pred_color};'>{test_country}</b> · <b>{n_test}</b> días de test · <b>{n_rows}</b> filas de entrenamiento · <b>{n_feats}</b> variables · <b>{period_tag}</b>",
        "sec1_title": "Rendimiento del Modelo",
        "sec1_subtitle": "{algo_icon} {algo_name} entrenado con {n_train} países · Evaluado en zero-shot sobre {test_country}",
        "metric_r2": "R²",
        "metric_r2_sub": "Varianza explicada",
        "metric_mse_sub": "Error cuadrático medio",
        "metric_mae_sub": "Error absoluto medio",
        "metric_pearson": "Pearson ρ",
        "metric_pearson_sub": "Predicho vs Real",
        "leg_beta_obs": "β observado",
        "leg_beta_pred": "β predicho",
        "growth_threshold": "Umbral β = 1/7 ≈ {gamma:.3f}",
        "title_obs_vs_pred": "β observado vs predicho · {country}",
        "axis_date": "Fecha",
        "axis_beta": "β",
        "axis_real_beta": "β real",
        "axis_pred_beta": "β predicho",
        "colorbar_time": "Tiempo",
        "tick_start": "Inicio", "tick_mid": "Mitad", "tick_end": "Fin",
        "leg_predictions": "Predicciones",
        "leg_perfect_fit": "Ajuste perfecto",
        "title_real_vs_pred": "β real vs predicho · {country}",
        "training_strategy": "<b>Estrategia de entrenamiento:</b> Generalización zero-shot — el modelo se entrena con <b>{n_train}</b> países ({sample}{ellipsis}) y se evalúa en <b>{test_country}</b> sin haberlo visto durante el entrenamiento.",
        "sec2_title": "Importancia de Variables y Explicabilidad",
        "sec2_subtitle": "SHAP (modelos de árboles) o importancia incorporada (otros)",
        "shap_explainer": "<b>SHAP (SHapley Additive exPlanations)</b> asigna a cada variable un valor de contribución. SHAP positivo → aumenta β; SHAP negativo → reduce β.",
        "tab_beeswarm": "🐝 Beeswarm",
        "tab_mean_shap": "📊 Media |SHAP|",
        "beeswarm_caption": "Cada punto = un día. Color = valor de la variable (rojo=alto, azul=bajo). Eje X = impacto SHAP en β.",
        "title_global_importance": "Importancia global de variables (media |SHAP|) · {country}",
        "axis_mean_shap": "media(|valor SHAP|)",
        "sec3_title": "Distribución SHAP por Variable",
        "sec3_subtitle": "Distribución individual del impacto SHAP para cada variable — los niveles categóricos se muestran por separado",
        "sec3_info": "Cada mini-gráfico muestra cómo los valores de <b>una variable</b> afectan a β mediante SHAP. <b>Eje X</b> = valor SHAP (impacto en β) · <b>Color</b> = valor de la variable (azul = bajo, rojo = alto). Las variables categóricas (niveles enteros 0–3 o 0–4) se muestran con filas distintas por nivel.",
        "axis_policy_level": "Nivel de política",
        "axis_shap_value": "Valor SHAP",
        "colorbar_feat_value": "Valor de la variable\n(bajo → alto)",
        "tick_low": "Bajo", "tick_mid2": "Medio", "tick_high": "Alto",
        "sec4_title": "SHAP Dependence Plot — Vista Detallada",
        "sec4_subtitle": "Valor de la variable vs. impacto SHAP · coloreado por la variable con mayor interacción",
        "sec4_info": "<b>Eje X</b> = valor de la variable seleccionada &nbsp;·&nbsp; <b>Eje Y</b> = su impacto SHAP en β &nbsp;·&nbsp; <b>Color</b> = valor de la variable con mayor interacción (detección automática o manual). Las variables categóricas usan <b>diagramas de puntos con jitter</b> con etiquetas enteras.",
        "primary_feature_label": "📌 Variable principal",
        "color_by_label": "🎨 Colorear por (variable de interacción)",
        "auto_best_match": "🔁 Auto (mejor coincidencia SHAP)",
        "dot_size_label": "Tamaño de punto",
        "colored_by": "&nbsp;coloreado por&nbsp;",
        "auto_detected": "Detección automática",
        "manual": "Manual",
        "axis_shap_value_for": "Valor SHAP de {feat}",
        "metric_mean_shap": "Media |SHAP|",
        "metric_global_importance": "Importancia global",
        "metric_interaction": "Interacción |r|",
        "metric_with": "con {feat}",
        "feat_type_label": "<b>Tipo de variable:</b><br>{val}<br><br><b>Dirección:</b><br>{dir}<br><br><b>Variable de color:</b><br>{color_feat}<br><small style='color:{dim};'>Revela efectos de interacción sobre β</small>",
        "feat_type_categorical": "Categórica (niveles enteros)",
        "feat_type_continuous": "Continua",
        "dir_higher_increases": "📈 Mayor → aumenta β",
        "dir_higher_reduces": "📉 Mayor → reduce β",
        "sec5_title": "Explicabilidad Local — SHAP Waterfall · {country}",
        "sec5_subtitle": "Selecciona una fecha para ver por qué el modelo predijo ese valor de β",
        "sec5_info": "<b>SHAP local</b> explica una predicción individual. Cada barra muestra cómo una variable empuja β hacia arriba (rojo) o hacia abajo (azul).",
        "select_date": "🗓️ Selecciona una fecha",
        "no_data_for_date": "No hay datos para esta fecha.",
        "metric_date": "Fecha",
        "metric_selected_day": "Día seleccionado",
        "metric_real_beta": "β real",
        "metric_predicted_beta": "β predicho",
        "metric_error": "Error: {err:.4f}",
        "growing": "🔴 Creciendo",
        "declining": "🟢 Decreciendo",
        "values_this_day": "Valores en este día:",
        "shap_caption_scenario": "Explicación SHAP de tu escenario",
        "sec6_title": "Simulador de Políticas · {country}",
        "sec6_subtitle": "Ajusta las palancas de política para predecir β en un escenario personalizado",
        "sec6_info": "Usa los deslizadores para configurar un escenario de política hipotético para <b>{country}</b>. Los valores tienen un <b>retardo (lag) de 14 días</b>. Algoritmo: {algo_icon} <b>{algo_name}</b> · {n_feats} variables activas.",
        "metric_predicted_beta_scenario": "β predicho",
        "metric_custom_scenario": "Tu escenario personalizado",
        "metric_epidemic_status": "Estado de la epidemia",
        "metric_vs_historical": "vs. Media histórica",
        "metric_avg_beta": "β medio = {beta:.4f}",
        "sec7_title": "Comparación de Escenarios SEIR · {country}",
        "sec7_subtitle": "Dos escenarios de simulación — comparación de enfoques de β y análisis de impacto de políticas",
        "sec7_info": "<b>Escenario 1</b> compara tres formas de impulsar el modelo SEIR a lo largo de todo el periodo epidémico: β constante (enfoque clásico), β predicho por ML y β estimado directamente.<br><b>Escenario 2</b> aísla una ventana de política estable para evaluar el impacto de niveles de restricción altos y bajos en condiciones controladas, y muestra qué políticas estaban activas.",
        "comp_active_infectious": "I(t) — Infecciosos activos",
        "comp_new_infections": "Nuevas infecciones por día",
        "comp_select_label": "📊 Compartimento SEIR a visualizar (aplica a ambos escenarios)",
        "comp_desc_I": "Personas actualmente infecciosas — la carga epidémica en cada momento. Los picos marcan el ápice de la epidemia.",
        "comp_desc_new": "Flujo diario E→I (incidencia) — los picos identifican las olas epidémicas.",
        "scenario1_banner_title": "📐 Escenario 1 — comparación de enfoques de β",
        "scenario1_banner_sub": "Desde el primer mes de la pandemia · β constante (clásico) · β predicho (ML) · β estimado",
        "scenario1_info": "<b>1. SEIR clásico — β constante:</b> Usa un único β fijo igual a la media en el periodo seleccionado. Representa el supuesto tradicional de una tasa de transmisión constante.<br><br><b>2. SEIR dinámico — β predicho:</b> β varía diariamente según las predicciones del modelo de ML a partir de las variables NPI observadas.<br><br><b>3. SEIR dinámico — β estimado:</b> β(t) tomado directamente de la estimación epidemiológica — verdad de campo epidemiológica.",
        "s1_start_label": "📅 Escenario 1 — inicio",
        "s1_start_help": "Por defecto: primer caso registrado para este país en el dataset.",
        "s1_end_label": "📅 Escenario 1 — fin",
        "s1_not_enough_data": "No hay datos suficientes en la ventana del Escenario 1.",
        "s1_const_beta_label": "β constante (ajustable)",
        "s1_mean_beta": "β medio (periodo)",
        "leg_beta_const": "β constante = {val:.4f}",
        "leg_beta_pred_ml": "β predicho (modelo ML)",
        "leg_beta_estimated": "β estimado",
        "title_s1_beta_series": "Escenario 1 — series β · {country}",
        "title_s1_seir": "Escenario 1 · {comp} — {country}",
        "leg_classical_seir": "🔘 SEIR clásico (β = {val:.4f})",
        "leg_dynamic_pred": "🔵 SEIR dinámico — β predicho",
        "leg_dynamic_estim": "🔴 SEIR dinámico — β estimado",
        "scen_classical": "Clásico (β const.)",
        "scen_pred_ml": "β predicho (ML)",
        "scen_estimated": "β estimado",
        "peak_day": "día pico {day} · final {final:,}",
        "s1_interpretation": "📌 <b>Escenario 1 — Interpretación ({comp}):</b> El SEIR clásico de β constante diverge de la trayectoria estimada en <b>{diff_const}</b> en el pico, lo que ilustra la limitación de asumir una tasa de transmisión fija. El β predicho por ML se ajusta más (<b>{diff_pred}</b> frente al estimado en el pico).",
        "scenario2_banner_title": "🏛️ Escenario 2 — impacto de restricciones de política",
        "scenario2_banner_sub": "Ventana de política · Caso base · Restricciones altas · Restricciones bajas",
        "scenario2_info": "<b>1. Caso base:</b> β estimado o predicho sin modificaciones — seleccionable más abajo.<br><b>2. Restricciones altas:</b> el ML predice β bajo valores estrictos de NPI — totalmente ajustable.<br><b>3. Restricciones bajas:</b> el ML predice β bajo valores relajados de NPI — totalmente ajustable.",
        "s2_start_label": "📅 Escenario 2 — inicio",
        "s2_end_label": "📅 Escenario 2 — fin",
        "s2_not_enough_data": "No hay datos suficientes en la ventana del Escenario 2 — prueba a ampliar el rango de fechas.",
        "active_policies": "📋 Valores de política activos en la ventana seleccionada (media)",
        "base_case_source": "📌 Fuente de β para el caso base",
        "base_estimated": "β estimado",
        "base_predicted_ml": "β predicho (modelo ML)",
        "expand_high_restr": "🟢 Restricciones altas — editar valores NPI",
        "expand_low_restr": "🟡 Restricciones bajas — editar valores NPI",
        "leg_beta_base": "β base ({label})",
        "leg_beta_high_restr": "β restricciones altas",
        "leg_beta_low_restr": "β restricciones bajas",
        "title_s2_beta_series": "Escenario 2 — series β · {country} ({start} → {end})",
        "leg_base_case": "{icon} Caso base ({label})",
        "leg_high_restr": "🟢 Restricciones altas",
        "leg_low_restr": "🟡 Restricciones bajas",
        "title_s2_seir": "Escenario 2 · {comp} — {country}",
        "scen_base": "Base ({label})",
        "scen_high_restr": "Restricciones altas",
        "scen_low_restr": "Restricciones bajas",
        "s2_interpretation": "📌 <b>Escenario 2 — Interpretación ({comp}):</b> En esta ventana de política ({start} → {end}), usando <b>{label}</b> como base (pico {base_peak:,}), las restricciones altas desplazan el pico en <b>{diff_s}</b> ({strict_peak:,}) y las restricciones bajas en <b>{diff_l}</b> ({loose_peak:,}).",
        "footer_text": "🦠 Panel Epidemiológico COVID-19 &nbsp;·&nbsp; Trabajo de Fin de Máster (TFM) &nbsp;·&nbsp; ML + XAI + SEIR",
    },
    "en": {
        "brand_title": "COVID-19 Dashboard",
        "brand_tagline": "TFM · ML + XAI + SEIR",
        "lang_label": "🌐 Language",
        "dataset_label": "📂 Dataset",
        "dataset_help": "Upload Europe_countries_covid19.csv",
        "test_country_label": "🌍 Test Country",
        "test_country_note": "Selected country is removed from training. All others train the model.",
        "train_n_countries": "Train ({n} countries):",
        "study_period_label": "📅 Study Period",
        "period_pre": "Pre-vaccination",
        "period_full": "Full COVID-19 period",
        "period_badge_pre": "🟢 Pre-vaccination",
        "period_badge_full": "🟡 Full period",
        "model_features_label": "⚙️ Model Features",
        "vacc_warning": "⚠️ <b>people_fully_vaccinated_per_hundred</b> = 0 in pre-vaccination mode.",
        "min_feature_warning": "⚠️ Select at least 1 feature.",
        "algorithm_label": "🤖 Algorithm",
        "hyperparams_label": "🔧 Hyperparameters",
        "max_depth_unlimited": "max_depth (0=unlimited)",
        "summary_model": "Model:",
        "summary_test": "Test:",
        "summary_train": "Train:",
        "summary_features": "Features:",
        "summary_period": "Period:",
        "summary_period_pre_short": "Pre-vacc.",
        "summary_period_full_short": "Full",
        "hero_title": "🦠 COVID-19 · <span>Epidemiological Dashboard</span>",
        "hero_subtitle": "Machine Learning + Explainable AI + SEIR Policy Simulation",
        "hero_meta": "TFM · ML · SHAP · SEIR",
        "gate_title": "Upload your dataset to begin",
        "gate_text": "Upload <b style='color:{text_color};'>Europe_countries_covid19.csv</b> via the sidebar. Then choose your <b>test country</b>, <b>algorithm</b>, and <b>hyperparameters</b> — the model trains on all remaining countries automatically.",
        "gate_chip_shap": "SHAP Explainability",
        "gate_chip_seir": "SEIR Simulation",
        "gate_chip_policy": "Policy Scenarios",
        "err_prepare_data": "❌ Error preparing data: {e}",
        "warn_not_enough_data": "⚠️ Not enough data after filtering. Try changing period, country, or features.",
        "err_train_model": "❌ Model training failed: {e}",
        "active_config": "{icon} <b>{algo_icon} {algo_name}</b> trained on <b>{n_train}</b> countries · Testing on <b style='color:{pred_color};'>{test_country}</b> · <b>{n_test}</b> test days · <b>{n_rows}</b> train rows · <b>{n_feats}</b> features · <b>{period_tag}</b>",
        "sec1_title": "Model Performance",
        "sec1_subtitle": "{algo_icon} {algo_name} trained on {n_train} countries · Tested zero-shot on {test_country}",
        "metric_r2": "R²",
        "metric_r2_sub": "Explained variance",
        "metric_mse_sub": "Mean Sq. Error",
        "metric_mae_sub": "Mean Abs. Error",
        "metric_pearson": "Pearson ρ",
        "metric_pearson_sub": "Predicted vs Real",
        "leg_beta_obs": "Observed β",
        "leg_beta_pred": "Predicted β",
        "growth_threshold": "Threshold β = 1/7 ≈ {gamma:.3f}",
        "title_obs_vs_pred": "Observed vs Predicted β · {country}",
        "axis_date": "Date",
        "axis_beta": "β",
        "axis_real_beta": "Real β",
        "axis_pred_beta": "Predicted β",
        "colorbar_time": "Time",
        "tick_start": "Start", "tick_mid": "Mid", "tick_end": "End",
        "leg_predictions": "Predictions",
        "leg_perfect_fit": "Perfect fit",
        "title_real_vs_pred": "β Real vs Predicted · {country}",
        "training_strategy": "<b>Training strategy:</b> Zero-shot generalization — model is trained on <b>{n_train}</b> countries ({sample}{ellipsis}) and tested on <b>{test_country}</b> without seeing it during training.",
        "sec2_title": "Feature Importance & Explainability",
        "sec2_subtitle": "SHAP (tree models) or built-in importance (others)",
        "shap_explainer": "<b>SHAP (SHapley Additive exPlanations)</b> assigns each feature a contribution value. Positive SHAP → increases β; Negative SHAP → reduces β.",
        "tab_beeswarm": "🐝 Beeswarm",
        "tab_mean_shap": "📊 Mean |SHAP|",
        "beeswarm_caption": "Each dot = one day. Colour = feature value (red=high, blue=low). X-axis = SHAP impact on β.",
        "title_global_importance": "Global Feature Importance (mean |SHAP|) · {country}",
        "axis_mean_shap": "mean(|SHAP value|)",
        "sec3_title": "Per-Feature SHAP Distribution",
        "sec3_subtitle": "Individual SHAP impact distribution for each variable — categorical levels shown separately",
        "sec3_info": "Each mini-plot shows how the values of <b>one feature</b> affect β via SHAP. <b>X-axis</b> = SHAP value (impact on β) · <b>Color</b> = feature value (blue = low, red = high). Categorical variables (integer levels 0–3 or 0–4) are shown with distinct rows per level.",
        "axis_policy_level": "Policy level",
        "axis_shap_value": "SHAP value",
        "colorbar_feat_value": "Feature value\n(low → high)",
        "tick_low": "Low", "tick_mid2": "Mid", "tick_high": "High",
        "sec4_title": "SHAP Dependence Plot — Detailed View",
        "sec4_subtitle": "Feature value vs. SHAP impact · colored by the most interacting variable",
        "sec4_info": "<b>X-axis</b> = value of the selected feature &nbsp;·&nbsp; <b>Y-axis</b> = its SHAP impact on β &nbsp;·&nbsp; <b>Color</b> = value of the feature that interacts most (auto-detected or manual). Categorical variables use <b>jittered strip plots</b> with integer tick labels.",
        "primary_feature_label": "📌 Primary feature",
        "color_by_label": "🎨 Color by (interaction variable)",
        "auto_best_match": "🔁 Auto (SHAP best match)",
        "dot_size_label": "Dot size",
        "colored_by": "&nbsp;colored by&nbsp;",
        "auto_detected": "Auto-detected",
        "manual": "Manual",
        "axis_shap_value_for": "SHAP value for {feat}",
        "metric_mean_shap": "Mean |SHAP|",
        "metric_global_importance": "Global importance",
        "metric_interaction": "Interaction |r|",
        "metric_with": "with {feat}",
        "feat_type_label": "<b>Feature type:</b><br>{val}<br><br><b>Direction:</b><br>{dir}<br><br><b>Color variable:</b><br>{color_feat}<br><small style='color:{dim};'>Reveals interaction effects on β</small>",
        "feat_type_categorical": "Categorical (integer levels)",
        "feat_type_continuous": "Continuous",
        "dir_higher_increases": "📈 Higher → increases β",
        "dir_higher_reduces": "📉 Higher → reduces β",
        "sec5_title": "Local Explainability — SHAP Waterfall · {country}",
        "sec5_subtitle": "Select a date to see why the model predicted that β",
        "sec5_info": "<b>Local SHAP</b> explains a single prediction. Each bar shows how a feature pushes β up (red) or down (blue).",
        "select_date": "🗓️ Select date",
        "no_data_for_date": "No data for this date.",
        "metric_date": "Date",
        "metric_selected_day": "Selected day",
        "metric_real_beta": "Real β",
        "metric_predicted_beta": "Predicted β",
        "metric_error": "Error: {err:.4f}",
        "growing": "🔴 Growing",
        "declining": "🟢 Declining",
        "values_this_day": "Values on this day:",
        "shap_caption_scenario": "SHAP explanation for your scenario",
        "sec6_title": "Policy Simulator · {country}",
        "sec6_subtitle": "Adjust policy levers to predict β for a custom scenario",
        "sec6_info": "Use the sliders to configure a hypothetical policy scenario for <b>{country}</b>. Values are <b>lagged 14 days</b>. Algorithm: {algo_icon} <b>{algo_name}</b> · {n_feats} features active.",
        "metric_predicted_beta_scenario": "Predicted β",
        "metric_custom_scenario": "Your custom scenario",
        "metric_epidemic_status": "Epidemic Status",
        "metric_vs_historical": "vs. Historical avg",
        "metric_avg_beta": "Avg β = {beta:.4f}",
        "sec7_title": "SEIR Scenario Comparison · {country}",
        "sec7_subtitle": "Two simulation scenarios — β approach comparison and policy impact analysis",
        "sec7_info": "<b>Scenario 1</b> compares three ways of driving the SEIR model across the full epidemic period: constant β (classical approach), ML-predicted β, and directly estimated β.<br><b>Scenario 2</b> isolates a stable policy window to evaluate the impact of high and low restriction levels under controlled conditions, and shows which policies were active.",
        "comp_active_infectious": "I(t) — Active infectious",
        "comp_new_infections": "New infections per day",
        "comp_select_label": "📊 SEIR compartment to visualize (applies to both scenarios)",
        "comp_desc_I": "People currently infectious — the epidemic burden at each moment. Peaks mark the epidemic apex.",
        "comp_desc_new": "Daily flow E→I (incidence) — peaks identify epidemic waves.",
        "scenario1_banner_title": "📐 Scenario 1 — β approach comparison",
        "scenario1_banner_sub": "From the first month of the pandemic · Constant β (classical) · Predicted β (ML) · Estimated β",
        "scenario1_info": "<b>1. Classical SEIR — constant β:</b> Uses a single fixed β equal to the mean over the selected period. Represents the traditional assumption of a constant transmission rate.<br><br><b>2. Dynamic SEIR — predicted β:</b> β varies daily according to ML model predictions from observed NPI features.<br><br><b>3. Dynamic SEIR — estimated β:</b> β(t) taken directly from epidemiological estimation — epidemiological ground truth.",
        "s1_start_label": "📅 Scenario 1 — start",
        "s1_start_help": "Default: first case recorded for this country in the dataset.",
        "s1_end_label": "📅 Scenario 1 — end",
        "s1_not_enough_data": "Not enough data in Scenario 1 window.",
        "s1_const_beta_label": "Constant β (adjustable)",
        "s1_mean_beta": "Mean β (period)",
        "leg_beta_const": "β constant = {val:.4f}",
        "leg_beta_pred_ml": "β predicted (ML model)",
        "leg_beta_estimated": "β estimated",
        "title_s1_beta_series": "Scenario 1 — β series · {country}",
        "title_s1_seir": "Scenario 1 · {comp} — {country}",
        "leg_classical_seir": "🔘 Classical SEIR (β = {val:.4f})",
        "leg_dynamic_pred": "🔵 Dynamic SEIR — predicted β",
        "leg_dynamic_estim": "🔴 Dynamic SEIR — estimated β",
        "scen_classical": "Classical (const. β)",
        "scen_pred_ml": "Predicted β (ML)",
        "scen_estimated": "Estimated β",
        "peak_day": "peak day {day} · final {final:,}",
        "s1_interpretation": "📌 <b>Scenario 1 — Interpretation ({comp}):</b> The classical constant-β SEIR diverges from the estimated trajectory by <b>{diff_const}</b> at peak, illustrating the limitation of assuming a fixed transmission rate. The ML-predicted β tracks more closely (<b>{diff_pred}</b> vs estimated at peak).",
        "scenario2_banner_title": "🏛️ Scenario 2 — Policy restriction impact",
        "scenario2_banner_sub": "Policy window · Base case · High restrictions · Low restrictions",
        "scenario2_info": "<b>1. Base case:</b> estimated or predicted β without modifications — selectable below.<br><b>2. High restrictions:</b> ML predicts β under strict NPI values — fully adjustable.<br><b>3. Low restrictions:</b> ML predicts β under relaxed NPI values — fully adjustable.",
        "s2_start_label": "📅 Scenario 2 — start",
        "s2_end_label": "📅 Scenario 2 — end",
        "s2_not_enough_data": "Not enough data in Scenario 2 window — try widening the date range.",
        "active_policies": "📋 Active policy values in selected window (mean)",
        "base_case_source": "📌 Base case β source",
        "base_estimated": "Estimated β",
        "base_predicted_ml": "Predicted β (ML model)",
        "expand_high_restr": "🟢 High restrictions — edit NPI values",
        "expand_low_restr": "🟡 Low restrictions — edit NPI values",
        "leg_beta_base": "β base ({label})",
        "leg_beta_high_restr": "β high restrictions",
        "leg_beta_low_restr": "β low restrictions",
        "title_s2_beta_series": "Scenario 2 — β series · {country} ({start} → {end})",
        "leg_base_case": "{icon} Base case ({label})",
        "leg_high_restr": "🟢 High restrictions",
        "leg_low_restr": "🟡 Low restrictions",
        "title_s2_seir": "Scenario 2 · {comp} — {country}",
        "scen_base": "Base ({label})",
        "scen_high_restr": "High restrictions",
        "scen_low_restr": "Low restrictions",
        "s2_interpretation": "📌 <b>Scenario 2 — Interpretation ({comp}):</b> In this policy window ({start} → {end}), using <b>{label}</b> as baseline (peak {base_peak:,}), high restrictions shift the peak by <b>{diff_s}</b> ({strict_peak:,}) and low restrictions by <b>{diff_l}</b> ({loose_peak:,}).",
        "footer_text": "🦠 COVID-19 Epidemiological Dashboard &nbsp;·&nbsp; Master's Thesis (TFM) &nbsp;·&nbsp; ML + XAI + SEIR",
    },
}

def t(key, **kwargs):
    """Devuelve el texto traducido para la clave dada, en el idioma activo."""
    txt = T[st.session_state.lang].get(key, T["es"].get(key, key))
    if kwargs:
        try:
            return txt.format(**kwargs)
        except (KeyError, IndexError):
            return txt
    return txt

def feat_label(feat_key):
    """Devuelve la etiqueta traducida de una variable NPI."""
    meta = ALL_FEATURES_META[feat_key]
    return meta["label_es"] if st.session_state.lang == "es" else meta["label_en"]

def algo_desc(algo_key):
    """Devuelve la descripción traducida de un algoritmo."""
    meta = ALGO_CATALOGUE[algo_key]
    return meta["desc_es"] if st.session_state.lang == "es" else meta["desc_en"]

# ─────────────────────────────────────────────────────────────────────────────
# CATÁLOGO DE VARIABLES
# ─────────────────────────────────────────────────────────────────────────────
ALL_FEATURES_META = {
    "Facial_Coverings_lag14":                  {"label_es": "Uso obligatorio de mascarillas",        "label_en": "Mandatory face masks",      "range": (0, 4, 1),       "icon": "😷"},
    "stringency_index_lag14":                  {"label_es": "Índice de severidad",                   "label_en": "Stringency index",          "range": (0.0,100.0,1.0),"icon": "📋"},
    "Workplace_closing_lag14":                 {"label_es": "Cierre de centros de trabajo",          "label_en": "Workplace closing",         "range": (0, 3, 1),       "icon": "🏢"},
    "people_fully_vaccinated_per_hundred":     {"label_es": "Vacunación completa (%)",                "label_en": "Fully vaccinated (%)",      "range": (0.0,100.0,0.5),"icon": "💉"},
    "Restrictions_on_gatherings_lag14":        {"label_es": "Restricciones a reuniones",              "label_en": "Gathering restrictions",    "range": (0, 4, 1),       "icon": "👥"},
    "School_closing_lag14":                    {"label_es": "Cierre de centros escolares",            "label_en": "School closures",           "range": (0, 3, 1),       "icon": "🏫"},
    "mobility_transit_lag14":                  {"label_es": "Movilidad en transporte (%)",            "label_en": "Transit mobility (%)",      "range": (-80.0,20.0,1.0),"icon": "🚇"},
    "Stay_at_home_lag14":                      {"label_es": "Confinamiento domiciliario",  "label_en": "Stay-at-home orders",       "range": (0, 3, 1),       "icon": "🏠"},
}

DEFAULT_FEATURES = [
    "Facial_Coverings_lag14",
    "Restrictions_on_gatherings_lag14",
    "School_closing_lag14",
    "mobility_transit_lag14",
    "Stay_at_home_lag14",
]

# ─────────────────────────────────────────────────────────────────────────────
# CATÁLOGO DE ALGORITMOS
# ─────────────────────────────────────────────────────────────────────────────
ALGO_CATALOGUE = {
    "XGBoost": {
        "icon": "⚡",
        "desc_es": "Árboles potenciados por gradiente — rápido, preciso, gestiona valores faltantes de forma nativa.",
        "desc_en": "Gradient boosted trees — fast, accurate, handles missing values natively.",
        "supports_shap": True,
        "shap_type": "tree",
    },
    "Random Forest": {
        "icon": "🌲",
        "desc_es": "Conjunto de árboles de decisión — robusto, baja varianza, entrenamiento paralelo.",
        "desc_en": "Ensemble of decision trees — robust, low variance, parallel training.",
        "supports_shap": True,
        "shap_type": "tree",
    },
    "Gradient Boosting": {
        "icon": "📈",
        "desc_es": "Potenciación secuencial con fuerte regularización — alta precisión.",
        "desc_en": "Sequential boosting with strong regularization — high accuracy.",
        "supports_shap": True,
        "shap_type": "tree",
    },
    "Extra Trees": {
        "icon": "🌳",
        "desc_es": "Árboles extremadamente aleatorizados — entrenamiento rápido, robusto frente a valores atípicos.",
        "desc_en": "Extremely randomized trees — fast training, robust to outliers.",
        "supports_shap": True,
        "shap_type": "tree",
    },
}



# ─────────────────────────────────────────────────────────────────────────────
# CSS — Tema clínico claro profesional
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'DM Sans', 'Helvetica Neue', sans-serif;
    color: {C['text']};
}}

.main {{
    background-color: {C['bg']};
    background-image:
        radial-gradient(ellipse at 10% 20%, rgba(0,150,199,0.06) 0%, transparent 45%),
        radial-gradient(ellipse at 90% 70%, rgba(214,40,57,0.04) 0%, transparent 45%),
        radial-gradient(ellipse at 50% 50%, rgba(10,158,110,0.03) 0%, transparent 60%);
}}
.main::before {{
    content: '';
    position: fixed; top:0; left:0; width:100%; height:100%;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='900' height='900' opacity='0.03'%3E%3Ccircle cx='150' cy='150' r='45' fill='none' stroke='%230077B6' stroke-width='2'/%3E%3Cline x1='150' y1='105' x2='150' y2='68' stroke='%230077B6' stroke-width='1.2'/%3E%3Cline x1='195' y1='150' x2='232' y2='150' stroke='%230077B6' stroke-width='1.2'/%3E%3Cline x1='105' y1='150' x2='68' y2='150' stroke='%230077B6' stroke-width='1.2'/%3E%3Cline x1='150' y1='195' x2='150' y2='232' stroke='%230077B6' stroke-width='1.2'/%3E%3Ccircle cx='150' cy='150' r='10' fill='%230077B6'/%3E%3Ccircle cx='550' cy='380' r='35' fill='none' stroke='%23D62839' stroke-width='1.5'/%3E%3Cline x1='550' y1='345' x2='550' y2='315' stroke='%23D62839' stroke-width='1.1'/%3E%3Cline x1='585' y1='380' x2='615' y2='380' stroke='%23D62839' stroke-width='1.1'/%3E%3Ccircle cx='550' cy='380' r='8' fill='%23D62839'/%3E%3Ccircle cx='780' cy='680' r='40' fill='none' stroke='%230A9E6E' stroke-width='1.5'/%3E%3Cline x1='780' y1='640' x2='780' y2='605' stroke='%230A9E6E' stroke-width='1.1'/%3E%3Cline x1='820' y1='680' x2='855' y2='680' stroke='%230A9E6E' stroke-width='1.1'/%3E%3Ccircle cx='780' cy='680' r='9' fill='%230A9E6E'/%3E%3Cline x1='150' y1='150' x2='550' y2='380' stroke='%230077B6' stroke-width='0.7' opacity='0.5'/%3E%3Cline x1='550' y1='380' x2='780' y2='680' stroke='%23D62839' stroke-width='0.7' opacity='0.5'/%3E%3C/svg%3E");
    background-size: 700px 700px;
    pointer-events: none; z-index: 0;
}}
.block-container {{ padding-top:1.2rem; padding-bottom:2rem; position:relative; z-index:1; }}

/* Barra lateral */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #DFF0F9 0%, #EBF4FA 50%, #EDF5FB 100%);
    border-right: 1px solid {C['border']};
}}
[data-testid="stSidebar"] * {{ color: {C['text']} !important; }}

.sidebar-brand {{
    background: linear-gradient(135deg, rgba(0,119,182,0.1), rgba(10,158,110,0.08));
    border: 1px solid {C['border2']}; border-radius: 12px;
    padding: 1rem; margin-bottom: 1rem; text-align: center;
    box-shadow: 0 2px 8px {C['shadow']};
}}
.sidebar-brand h2 {{ color: {C['text']} !important; font-size:1.1rem; margin:0.3rem 0 0 0; font-weight:700; }}
.sidebar-brand .tagline {{ color: {C['muted']} !important; font-size:0.7rem; font-family:'DM Mono',monospace !important; }}

.sidebar-label {{
    color: {C['accent']}; font-size:0.75rem; font-weight:600;
    text-transform:uppercase; letter-spacing:0.09em; margin-bottom:0.3rem; margin-top:0.2rem;
}}

/* Cabeceras de sección */
.section-header {{
    background: linear-gradient(90deg, rgba(0,119,182,0.07), rgba(0,119,182,0.01));
    border-left: 3px solid {C['pred']}; border-radius: 0 8px 8px 0;
    padding: 0.65rem 1.3rem; margin: 1.8rem 0 0.9rem 0;
    box-shadow: 0 1px 4px {C['shadow']};
}}
.section-header h2 {{ color:{C['text']}; margin:0; font-size:1.15rem; font-weight:600; letter-spacing:-0.01em; }}
.section-header p  {{ color:{C['muted']}; margin:0.2rem 0 0 0; font-size:0.81rem; }}

/* Tarjetas de métricas */
.metric-card {{
    background: {C['white']}; border: 1px solid {C['border']}; border-top: 3px solid {C['pred']};
    border-radius: 10px; padding: 1rem 1.1rem; text-align: center;
    margin-bottom: 0.5rem; box-shadow: 0 2px 10px {C['shadow']};
    transition: box-shadow 0.2s, transform 0.15s;
}}
.metric-card:hover {{ box-shadow: 0 4px 18px rgba(0,119,182,0.13); transform: translateY(-1px); }}
.metric-card .label {{ color:{C['muted']}; font-size:0.69rem; text-transform:uppercase; letter-spacing:0.1em; font-weight:600; }}
.metric-card .value {{ color:{C['text']}; font-size:1.8rem; font-weight:700; font-family:'DM Mono',monospace; margin:0.2rem 0 0.1rem 0; line-height:1.1; }}
.metric-card .sub   {{ color:{C['dim']}; font-size:0.7rem; }}

/* Cajas de info/aviso/éxito */
.info-box {{
    background: linear-gradient(135deg, rgba(0,119,182,0.05), rgba(72,202,228,0.04));
    border: 1px solid {C['border']}; border-left: 3px solid {C['accent']};
    border-radius: 8px; padding: 0.85rem 1.1rem; margin:0.7rem 0;
    color:{C['text']}; font-size:0.86rem; line-height:1.6;
    box-shadow: 0 1px 4px {C['shadow']};
}}
.warn-box {{
    background: rgba(214,40,57,0.04); border: 1px solid rgba(214,40,57,0.22);
    border-left: 3px solid {C['real']}; border-radius:8px;
    padding:0.85rem 1.1rem; margin:0.7rem 0; color:#6B0D1A; font-size:0.86rem;
}}
.success-box {{
    background: rgba(10,158,110,0.05); border: 1px solid rgba(10,158,110,0.22);
    border-left: 3px solid {C['strict']}; border-radius:8px;
    padding:0.85rem 1.1rem; margin:0.7rem 0; color:#083D2A; font-size:0.86rem;
}}

/* Tarjeta de algoritmo */
.algo-card {{
    background: {C['white']}; border: 1px solid {C['border']};
    border-radius: 10px; padding: 0.8rem 1rem; margin-bottom:0.4rem;
    box-shadow: 0 1px 5px {C['shadow']};
}}
.algo-card .algo-name {{ font-size:0.9rem; font-weight:700; color:{C['pred']}; }}
.algo-card .algo-desc {{ font-size:0.78rem; color:{C['muted']}; margin-top:0.2rem; }}

/* Insignia de país */
.country-badge {{
    display:inline-flex; align-items:center; gap:0.4rem;
    background:linear-gradient(135deg,rgba(0,119,182,0.1),rgba(10,158,110,0.08));
    border:1px solid {C['border2']}; border-radius:20px;
    padding:0.3rem 0.9rem; font-size:0.82rem; font-weight:600;
    color:{C['pred']}; font-family:'DM Mono',monospace;
}}

/* Insignia de periodo */
.period-badge {{
    display:inline-block; padding:0.22rem 0.75rem; border-radius:20px;
    font-size:0.71rem; font-weight:600; letter-spacing:0.05em;
    text-transform:uppercase; font-family:'DM Mono',monospace;
}}
.period-pre  {{ background:rgba(10,158,110,0.1); color:{C['strict']}; border:1px solid rgba(10,158,110,0.3); }}
.period-full {{ background:rgba(224,122,16,0.1);  color:{C['loose']};  border:1px solid rgba(224,122,16,0.3); }}

/* Chip de variable */
.feat-chip {{
    display:inline-block; background:rgba(0,119,182,0.08);
    border:1px solid rgba(0,119,182,0.2); border-radius:16px;
    padding:0.18rem 0.65rem; font-size:0.71rem; color:{C['pred']};
    margin:0.12rem; font-family:'DM Mono',monospace;
}}

/* Separador */
.styled-divider {{
    border:none; height:1px;
    background:linear-gradient(90deg,transparent,{C['border2']},transparent);
    margin:1.8rem 0;
}}

/* Cabecera principal */
.hero-area {{ text-align:center; padding:1.4rem 0 0.4rem 0; }}
.hero-area h1 {{
    font-size:2.3rem; font-weight:800; color:{C['text']};
    margin:0; letter-spacing:-0.03em;
}}
.hero-area h1 span {{
    background:linear-gradient(90deg,{C['pred']},{C['strict']});
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
}}
.hero-area .subtitle {{ font-size:0.98rem; color:{C['muted']}; margin-top:0.4rem; }}
.hero-area .meta {{ font-size:0.75rem; color:{C['dim']}; margin-top:0.25rem; font-family:'DM Mono',monospace; }}

/* Píldoras de países de entrenamiento */
.train-pill {{
    display:inline-block; background:rgba(0,150,199,0.08);
    border:1px solid rgba(0,150,199,0.22); border-radius:12px;
    padding:0.15rem 0.55rem; font-size:0.7rem; color:{C['accent']};
    margin:0.1rem; font-family:'DM Mono',monospace;
}}

/* Plotly */
.js-plotly-plot {{ border-radius:10px; border:1px solid {C['border']}; box-shadow:0 2px 8px {C['shadow']}; }}

/* Ajustes de Streamlit */
.stMultiSelect [data-baseweb="tag"] {{
    background-color:rgba(0,119,182,0.12) !important;
    border-color:rgba(0,119,182,0.3) !important; color:{C['pred']} !important;
}}
.stTabs [data-baseweb="tab-list"] {{ background-color:{C['bg']}; border-bottom:2px solid {C['border']}; }}
.stTabs [aria-selected="true"] {{ border-bottom:2px solid {C['pred']} !important; color:{C['pred']} !important; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TEMA DE PLOTLY
# ─────────────────────────────────────────────────────────────────────────────
PLT = dict(
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#F8FAFC",
    font=dict(color=C["text"], family="DM Sans, Helvetica Neue, sans-serif"),
    xaxis=dict(gridcolor="rgba(180,200,215,0.4)", linecolor=C["border2"], zerolinecolor=C["border2"],
               tickfont=dict(color=C["muted"]), showgrid=True, gridwidth=0.5),
    yaxis=dict(gridcolor="rgba(180,200,215,0.4)", linecolor=C["border2"], zerolinecolor=C["border2"],
               tickfont=dict(color=C["muted"]), showgrid=True, gridwidth=0.5),
    legend=dict(bgcolor="rgba(255,255,255,0.95)", bordercolor=C["border"], borderwidth=1, font=dict(color=C["text"])),
    margin=dict(l=60, r=30, t=50, b=50),
)

# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ─────────────────────────────────────────────────────────────────────────────

def section_header(icon, title, subtitle=""):
    st.markdown(f'<div class="section-header"><h2>{icon} {title}</h2>{"<p>"+subtitle+"</p>" if subtitle else ""}</div>', unsafe_allow_html=True)

def metric_card(label, value, sub=""):
    st.markdown(f'<div class="metric-card"><div class="label">{label}</div><div class="value">{value}</div><div class="sub">{sub}</div></div>', unsafe_allow_html=True)

def info_box(text):    st.markdown(f'<div class="info-box">{text}</div>',    unsafe_allow_html=True)
def warn_box(text):    st.markdown(f'<div class="warn-box">{text}</div>',    unsafe_allow_html=True)
def success_box(text): st.markdown(f'<div class="success-box">{text}</div>', unsafe_allow_html=True)
def hr():              st.markdown('<hr class="styled-divider">',            unsafe_allow_html=True)

def apply_theme(fig):
    fig.update_layout(**PLT)
    fig.update_xaxes(
        gridcolor="rgba(180,200,215,0.4)", linecolor=C["border2"],
        zerolinecolor=C["border2"], tickfont=dict(color=C["muted"]),
        showgrid=True, gridwidth=0.5, showline=True,
        mirror=False,
    )
    fig.update_yaxes(
        gridcolor="rgba(180,200,215,0.4)", linecolor=C["border2"],
        zerolinecolor=C["border2"], tickfont=dict(color=C["muted"]),
        showgrid=True, gridwidth=0.5, showline=True,
        mirror=False,
    )
    return fig

def shap_fig_to_bytes():
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor="#FFFFFF", edgecolor="none")
    buf.seek(0); plt.close("all")
    return buf

def pills(items, cls="train-pill"):
    return " ".join(f'<span class="{cls}">{i}</span>' for i in items)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUCTOR DE MODELOS
# ─────────────────────────────────────────────────────────────────────────────

def build_model(algo_name: str, params: dict):
    """Devuelve un estimador basado en árboles — todos soportan SHAP TreeExplainer."""
    if algo_name == "XGBoost":
        return XGBRegressor(
            n_estimators=params.get("n_estimators", 300),
            learning_rate=params.get("learning_rate", 0.05),
            max_depth=params.get("max_depth", 3),
            subsample=params.get("subsample", 0.8),
            random_state=42,
        )
    elif algo_name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", None) or None,
            min_samples_leaf=params.get("min_samples_leaf", 2),
            random_state=42, n_jobs=-1,
        )
    elif algo_name == "Gradient Boosting":
        return GradientBoostingRegressor(
            n_estimators=params.get("n_estimators", 200),
            learning_rate=params.get("learning_rate", 0.05),
            max_depth=params.get("max_depth", 3),
            subsample=params.get("subsample", 0.8),
            random_state=42,
        )
    elif algo_name == "Extra Trees":
        return ExtraTreesRegressor(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", None) or None,
            min_samples_leaf=params.get("min_samples_leaf", 2),
            random_state=42, n_jobs=-1,
        )
    raise ValueError(f"Algoritmo desconocido: {algo_name}")

def supports_tree_shap(algo_name):
    return True  # los 4 algoritmos soportan SHAP TreeExplainer

# ─────────────────────────────────────────────────────────────────────────────
# CLAVE DE CACHÉ — usa SHA-256 de los bytes del archivo para que dos archivos distintos
# nunca coincidan y el mismo archivo siempre use la caché, sin importar la ruta temporal.
# ─────────────────────────────────────────────────────────────────────────────
import hashlib

def file_hash(uploaded_file) -> str:
    """Devuelve un hash hexadecimal estable del contenido del archivo subido."""
    uploaded_file.seek(0)
    digest = hashlib.sha256(uploaded_file.read()).hexdigest()
    uploaded_file.seek(0)
    return digest

# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE DATOS — indexada por hash de contenido, no por ruta temporal
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="📦 Cargando datos…")
def load_csv_cached(file_hash: str, csv_path: str) -> pd.DataFrame:
    """file_hash es la clave de caché; csv_path solo se usa para leer el archivo."""
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner="🔄 Preparando partición entrenamiento/test…")
def prepare_splits(file_hash: str, csv_path: str, test_country: str,
                   features: tuple, period: str):
    df = load_csv_cached(file_hash, csv_path)

    if period == "Pre-vaccination":
        df = df[df["people_fully_vaccinated_per_hundred"] == 0.0]

    available = sorted(df["country"].dropna().unique().tolist())
    train_countries = [c for c in available if c != test_country]

    df_all   = df[df["country"].isin(available)].copy()
    train_df = df_all[df_all["country"].isin(train_countries)].copy()
    test_df  = df_all[df_all["country"] == test_country].copy()

    test_df["I_real"] = test_df["new_cases_smoothed"].rolling(7).sum()
    test_df["E_real"] = test_df["new_cases_smoothed"].rolling(5).sum()

    required = list(features) + ["beta"]
    train_df = train_df.dropna(subset=required).reset_index(drop=True)
    test_df  = test_df.dropna(subset=required).reset_index(drop=True)

    return train_df, test_df, train_countries


@st.cache_resource(show_spinner="🧠 Entrenando modelo…")
def train_model_cached(file_hash, csv_path, test_country, features,
                       period, algo_name, params_str):
    import json
    params = json.loads(params_str)
    train_df, _, _ = prepare_splits(file_hash, csv_path, test_country, features, period)
    model = build_model(algo_name, params)
    model.fit(train_df[list(features)], train_df["beta"])
    return model


@st.cache_resource(show_spinner="🔍 Calculando valores SHAP…")
def compute_shap_cached(_model, X_test_vals, feature_names, algo_name):
    X_df = pd.DataFrame(X_test_vals, columns=feature_names)
    explainer   = shap.TreeExplainer(_model)
    shap_values = explainer.shap_values(X_df)
    return explainer, shap_values

# ─────────────────────────────────────────────────────────────────────────────
# SEIR
# ─────────────────────────────────────────────────────────────────────────────

def simulate_seir(beta_series, population, E0, I0):
    """Ejecuta el modelo SEIR — devuelve un diccionario con las trayectorias S, E, I, R, new_infected."""
    sigma, gamma = 1/5, 1/7
    S, E, I, R = population - E0 - I0, E0, I0, 0.0
    traj = {"S": [], "E": [], "I": [], "R": [], "new_infected": []}
    for beta in beta_series:
        ne = beta * S * I / population
        ni = sigma * E
        nr = gamma * I
        S -= ne; E += ne - ni; I += ni - nr; R += nr
        traj["S"].append(S)
        traj["E"].append(E)
        traj["I"].append(I)
        traj["R"].append(R)
        traj["new_infected"].append(ni)
    return traj


def safe_peak(vals):
    """Devuelve el valor pico de una lista/array, gestionando NaN y vacíos de forma segura."""
    arr = np.array(vals, dtype=float)
    arr = arr[~np.isnan(arr)]
    return int(arr.max()) if len(arr) > 0 else 0

def safe_final(vals):
    """Devuelve el último valor no-NaN."""
    arr = np.array(vals, dtype=float)
    arr = arr[~np.isnan(arr)]
    return int(arr[-1]) if len(arr) > 0 else 0

def safe_peak_day(vals):
    """Devuelve el día del pico (indexado desde 1), ignorando NaN."""
    arr = np.array(vals, dtype=float)
    if len(arr) == 0 or np.all(np.isnan(arr)):
        return 1
    return int(np.nanargmax(arr)) + 1

# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════  BARRA LATERAL  ══════════════════════
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # ── 0. SELECTOR DE IDIOMA / LANGUAGE SELECTOR ──
    lang_choice = st.radio(
        t("lang_label"),
        options=["es", "en"],
        index=0 if st.session_state.lang == "es" else 1,
        format_func=lambda l: "Español" if l == "es" else "English",
        horizontal=True,
        key="lang_radio",
    )
    if lang_choice != st.session_state.lang:
        st.session_state.lang = lang_choice
        st.rerun()

    st.markdown(f"""
    <div class="sidebar-brand">
        <div style='font-size:2rem;'>🦠</div>
        <h2>{t("brand_title")}</h2>
        <div class="tagline">{t("brand_tagline")}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── 1. DATASET ──
    st.markdown(f'<p class="sidebar-label">{t("dataset_label")}</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("CSV", type=["csv"], label_visibility="collapsed",
                                 help=t("dataset_help"))

    st.markdown("---")

    # We need the CSV loaded before we can offer country/algo choices
    if uploaded is not None:
        # ── Compute content hash and detect file change ──
        csv_hash = file_hash(uploaded)

        # If a different file was uploaded, wipe all cached data & models
        prev_hash = st.session_state.get("_csv_hash", None)
        if prev_hash != csv_hash:
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state["_csv_hash"] = csv_hash

        # Write temp file only once per hash (reuse path stored in session)
        if st.session_state.get("_tmp_path_hash") != csv_hash:
            uploaded.seek(0)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(uploaded.read())
                st.session_state["_tmp_path"]      = tmp.name
                st.session_state["_tmp_path_hash"] = csv_hash

        TMP_PATH = st.session_state["_tmp_path"]

        raw_df = load_csv_cached(csv_hash, TMP_PATH)
        all_countries = sorted(raw_df["country"].dropna().unique().tolist())

        # ── 2. TEST COUNTRY ──
        st.markdown(f'<p class="sidebar-label">{t("test_country_label")}</p>', unsafe_allow_html=True)
        st.markdown(f"<small style='color:{C['dim']};font-size:0.71rem;'>{t('test_country_note')}</small>", unsafe_allow_html=True)
        test_country = st.selectbox(
            "Test country",
            options=all_countries,
            index=all_countries.index("Spain") if "Spain" in all_countries else 0,
            label_visibility="collapsed",
        )
        train_countries_preview = [c for c in all_countries if c != test_country]
        st.markdown(f"<div style='margin:0.4rem 0 0.2rem 0;'><span style='font-size:0.72rem;color:{C['muted']};'>{t('train_n_countries', n=len(train_countries_preview))}</span></div>", unsafe_allow_html=True)
        st.markdown(pills(train_countries_preview[:8]) + (" …" if len(train_countries_preview) > 8 else ""), unsafe_allow_html=True)

        st.markdown("---")

        # ── 3. STUDY PERIOD ──
        st.markdown(f'<p class="sidebar-label">{t("study_period_label")}</p>', unsafe_allow_html=True)
        period = st.radio("Period", ["Pre-vaccination", "Full COVID-19 period"],
                          index=0, label_visibility="collapsed",
                          format_func=lambda p: t("period_pre") if p == "Pre-vaccination" else t("period_full"))
        if period == "Pre-vaccination":
            st.markdown(f'<span class="period-badge period-pre">{t("period_badge_pre")}</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="period-badge period-full">{t("period_badge_full")}</span>', unsafe_allow_html=True)

        st.markdown("---")

        # ── 4. FEATURES ──
        st.markdown(f'<p class="sidebar-label">{t("model_features_label")}</p>', unsafe_allow_html=True)
        if period == "Pre-vaccination":
            st.markdown(f"<div class='warn-box' style='padding:0.45rem 0.65rem;font-size:0.73rem;margin:0.3rem 0;'>{t('vacc_warning')}</div>", unsafe_allow_html=True)

        feat_options = list(ALL_FEATURES_META.keys())
        feat_fmt = lambda f: f"{ALL_FEATURES_META[f]['icon']} {feat_label(f)}"
        selected_feats = st.multiselect(
            "Features", options=feat_options, default=DEFAULT_FEATURES,
            format_func=feat_fmt, label_visibility="collapsed",
        )
        if len(selected_feats) < 1:
            st.warning(t("min_feature_warning"))
            selected_feats = DEFAULT_FEATURES

        FEATURES      = selected_feats
        FEAT_LABELS   = {f: feat_label(f) for f in FEATURES}
        FEAT_RANGES   = {f: ALL_FEATURES_META[f]["range"]  for f in FEATURES}

        st.markdown("---")

        # ── 5. ALGORITHM ──
        st.markdown(f'<p class="sidebar-label">{t("algorithm_label")}</p>', unsafe_allow_html=True)
        algo_name = st.selectbox(
            "Algorithm",
            options=list(ALGO_CATALOGUE.keys()),
            index=list(ALGO_CATALOGUE.keys()).index("Extra Trees"),
            format_func=lambda a: f"{ALGO_CATALOGUE[a]['icon']} {a}",
            label_visibility="collapsed",
        )
        algo_meta = ALGO_CATALOGUE[algo_name]
        st.markdown(f"<div class='algo-card'><div class='algo-name'>{algo_meta['icon']} {algo_name}</div><div class='algo-desc'>{algo_desc(algo_name)}</div></div>", unsafe_allow_html=True)

        # ── 6. HYPERPARAMETERS ──
        st.markdown(f'<p class="sidebar-label">{t("hyperparams_label")}</p>', unsafe_allow_html=True)
        hp = {}

        if algo_name in ("XGBoost", "Gradient Boosting"):
            hp["n_estimators"]  = st.slider("n_estimators",  50, 600, 300, 50)
            hp["learning_rate"] = st.select_slider("learning_rate", [0.005,0.01,0.02,0.05,0.1,0.2], value=0.05)
            hp["max_depth"]     = st.slider("max_depth", 1, 8, 3)
            hp["subsample"]     = st.slider("subsample", 0.4, 1.0, 0.8, 0.05)
        elif algo_name in ("Random Forest", "Extra Trees"):
            hp["n_estimators"]     = st.slider("n_estimators", 50, 500, 200, 50)
            hp["max_depth"]        = st.slider(t("max_depth_unlimited"), 0, 20, 0)
            hp["min_samples_leaf"] = st.slider("min_samples_leaf", 1, 10, 2)

        st.markdown("---")
        st.markdown(f"""
        <div style='font-size:0.7rem; color:{C['dim']}; font-family:DM Mono,monospace;'>
        <b style='color:{C['muted']};'>{t("summary_model")}</b> {algo_name}<br>
        <b style='color:{C['muted']};'>{t("summary_test")}</b> {test_country}<br>
        <b style='color:{C['muted']};'>{t("summary_train")}</b> {len(train_countries_preview)} {"países" if st.session_state.lang=="es" else "countries"}<br>
        <b style='color:{C['muted']};'>{t("summary_features")}</b> {len(FEATURES)}<br>
        <b style='color:{C['muted']};'>{t("summary_period")}</b> {t("summary_period_pre_short") if period=="Pre-vaccination" else t("summary_period_full_short")}
        </div>
        """, unsafe_allow_html=True)

    else:
        # No CSV yet — placeholder defaults
        period        = "Pre-vaccination"
        test_country  = "Spain"
        FEATURES      = DEFAULT_FEATURES
        FEAT_LABELS   = {f: feat_label(f) for f in FEATURES}
        FEAT_RANGES   = {f: ALL_FEATURES_META[f]["range"]  for f in FEATURES}
        algo_name     = "Extra Trees"
        algo_meta     = ALGO_CATALOGUE[algo_name]
        hp            = {}
        TMP_PATH      = None
        csv_hash      = None
        train_countries_preview = []

# ─────────────────────────────────────────────────────────────────────────────
# TÍTULO PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
period_color = C["strict"] if period == "Pre-vaccination" else C["loose"]
period_tag   = t("period_pre") if period == "Pre-vaccination" else t("period_full")

st.markdown(f"""
<div class="hero-area">
    <h1>{t("hero_title")}</h1>
    <p class="subtitle">{t("hero_subtitle")}</p>
    <p class="meta">      
        &nbsp;&nbsp;{t("hero_meta")}
    </p>
</div>
""", unsafe_allow_html=True)
hr()

# ─────────────────────────────────────────────────────────────────────────────
# PUERTA DE ENTRADA
# ─────────────────────────────────────────────────────────────────────────────
if uploaded is None:
    st.markdown(f"""
    <div style='text-align:center;padding:3.5rem 1rem;'>
        <div style='font-size:4rem;margin-bottom:1rem;'>📂</div>
        <h2 style='color:{C["pred"]};font-weight:700;'>{t("gate_title")}</h2>
        <p style='color:{C["muted"]};max-width:540px;margin:1rem auto;line-height:1.7;'>
            {t("gate_text", text_color=C["text"])}
        </p>
        <div style='margin-top:1.5rem;display:flex;justify-content:center;gap:0.7rem;flex-wrap:wrap;'>
            <span class='feat-chip'>{t("gate_chip_shap")}</span>
            <span class='feat-chip'>{t("gate_chip_seir")}</span>
            <span class='feat-chip'>{t("gate_chip_policy")}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# CARGA, PARTICIÓN Y ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────
import json

try:
    train_df, test_df, train_countries = prepare_splits(
        csv_hash, TMP_PATH, test_country, tuple(FEATURES), period
    )
except Exception as e:
    st.error(t("err_prepare_data", e=e))
    st.stop()

if len(train_df) == 0 or len(test_df) == 0:
    warn_box(t("warn_not_enough_data"))
    st.stop()

hp_str = json.dumps(hp, sort_keys=True)
try:
    model = train_model_cached(csv_hash, TMP_PATH, test_country, tuple(FEATURES), period, algo_name, hp_str)
except Exception as e:
    st.error(t("err_train_model", e=e))
    st.stop()

X_test             = test_df[FEATURES]
y_test             = test_df["beta"]
y_pred             = model.predict(X_test)
test_df            = test_df.copy()
test_df["beta_pred"] = y_pred

# SHAP (solo modelos basados en árboles)
shap_available = supports_tree_shap(algo_name)
explainer = shap_values = None
if shap_available:
    try:
        explainer, shap_values = compute_shap_cached(
            model, X_test.values, tuple(FEATURES), algo_name
        )
    except Exception:
        shap_available = False

# ── Active config banner ──
period_icon = "✅" if period == "Pre-vaccination" else "📅"
success_box(t("active_config",
    icon=period_icon, algo_icon=algo_meta['icon'], algo_name=algo_name,
    n_train=len(train_countries), pred_color=C["pred"], test_country=test_country,
    n_test=len(test_df), n_rows=len(train_df), n_feats=len(FEATURES), period_tag=period_tag))

# ─────────────────────────────────────────────────────────────────────────────
# Constante GAMMA (usada en las secciones siguientes)
# ─────────────────────────────────────────────────────────────────────────────
GAMMA = 1 / 7

# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 1 — Rendimiento del modelo
# ─────────────────────────────────────────────────────────────────────────────
section_header("📊", t("sec1_title"),
               t("sec1_subtitle", algo_icon=algo_meta['icon'], algo_name=algo_name,
                 n_train=len(train_countries), test_country=test_country))

r2   = r2_score(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
corr = float(np.corrcoef(y_test, y_pred)[0, 1])

c1, c2, c3, c4 = st.columns(4)
with c1: metric_card(t("metric_r2"),  f"{r2:.3f}",   t("metric_r2_sub"))
with c2: metric_card("MSE",       f"{mse:.4f}",  t("metric_mse_sub"))
with c3: metric_card("MAE",       f"{mae:.4f}",  t("metric_mae_sub"))
with c4: metric_card(t("metric_pearson"), f"{corr:.3f}", t("metric_pearson_sub"))

col_ts2, col_sc = st.columns([3, 2])
with col_ts2:
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(x=test_df["date"], y=test_df["beta"], name=t("leg_beta_obs"),
        line=dict(color=C["real"], width=2, dash="dot")))
    fig_pred.add_trace(go.Scatter(x=test_df["date"], y=test_df["beta_pred"], name=t("leg_beta_pred"),
        line=dict(color=C["pred"], width=2.5)))
    fig_pred.add_hline(y=GAMMA, line_dash="dot", line_color=C["gold"], line_width=1.4,
        annotation_text=t("growth_threshold", gamma=GAMMA),
        annotation_font_color=C["gold"], annotation_position="bottom right")
    # Añadir anotación con las métricas
    fig_pred.add_annotation(
        x=0.01, y=0.97, xref="paper", yref="paper",
        text=f"R² = {r2:.3f}  |  MAE = {mae:.4f}  |  Pearson ρ = {corr:.3f}",
        showarrow=False, bgcolor="white", bordercolor=C["border"],
        font=dict(color=C["muted"], size=11), align="left",
        borderwidth=1, borderpad=4, xanchor="left", yanchor="top"
    )
    fig_pred.update_layout(title=t("title_obs_vs_pred", country=test_country),
        xaxis_title=t("axis_date"), yaxis_title=t("axis_beta"), height=360, **PLT)
    apply_theme(fig_pred); st.plotly_chart(fig_pred, use_container_width=True)

with col_sc:
    n_pts = len(y_test)
    lim = [min(float(y_test.min()), float(y_pred.min())) - 0.005,
           max(float(y_test.max()), float(y_pred.max())) + 0.005]
    fig_sc = go.Figure()
    fig_sc.add_trace(go.Scatter(
        x=y_test, y=y_pred, mode="markers",
        marker=dict(
            color=list(range(n_pts)), colorscale="Plasma",
            size=6, opacity=0.55, showscale=True,
            colorbar=dict(
                title=dict(text=t("colorbar_time"), font=dict(color=C["muted"], size=10)),
                tickvals=[0, n_pts//2, n_pts-1],
                ticktext=[t("tick_start"), t("tick_mid"), t("tick_end")],
                thickness=12,
            ),
        ), name=t("leg_predictions")))
    fig_sc.add_trace(go.Scatter(x=lim, y=lim, mode="lines",
        line=dict(color=C["real"], dash="dash", width=2), name=t("leg_perfect_fit")))
    fig_sc.add_annotation(
        x=0.05, y=0.92, xref="paper", yref="paper",
        text=f"R² = {r2:.3f}  |  ρ = {corr:.3f}<br>MAE = {mae:.4f}",
        showarrow=False, bgcolor="white", bordercolor=C["border"],
        font=dict(color=C["muted"], size=10), align="left",
        borderwidth=1, borderpad=4, xanchor="left", yanchor="top"
    )
    fig_sc.update_layout(title=t("title_real_vs_pred", country=test_country),
        xaxis_title=t("axis_real_beta"), yaxis_title=t("axis_pred_beta"), height=360, **PLT)
    apply_theme(fig_sc); st.plotly_chart(fig_sc, use_container_width=True)

_sample = ", ".join(train_countries[:5])
_ellipsis = "…" if len(train_countries) > 5 else ""
info_box(t("training_strategy", n_train=len(train_countries), sample=_sample,
           ellipsis=_ellipsis, test_country=test_country))

hr()

# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 2 — Importancia de variables / SHAP
# ─────────────────────────────────────────────────────────────────────────────
section_header("🔍", t("sec2_title"), t("sec2_subtitle"))

if shap_available and shap_values is not None:
    info_box(t("shap_explainer"))

    tab_bee, tab_bar_shap = st.tabs([t("tab_beeswarm"), t("tab_mean_shap")])

    with tab_bee:
        plt.style.use("default")
        fig_bee, ax = plt.subplots(figsize=(9, max(3.5, len(FEATURES)*0.65)))
        fig_bee.patch.set_facecolor("#FFFFFF"); ax.set_facecolor("#F8FAFC")
        shap.summary_plot(shap_values, X_test,
            feature_names=[FEAT_LABELS.get(f, f) for f in FEATURES],
            show=False, plot_size=None, color_bar=True)
        ax.tick_params(colors=C["text"])
        ax.spines[["top","right"]].set_visible(False)
        ax.spines[["bottom","left"]].set_color(C["border2"])
        plt.tight_layout(); st.image(shap_fig_to_bytes(), use_container_width=True)
        info_box(t("beeswarm_caption"))

    with tab_bar_shap:
        mean_shap = np.abs(shap_values).mean(axis=0)
        imp_df = pd.DataFrame({"Feature": [FEAT_LABELS.get(f,f) for f in FEATURES], "SHAP": mean_shap}).sort_values("SHAP", ascending=True)
        n_feats = len(imp_df)
        blues = [f"rgba({int(56 + 150*(i/max(n_feats-1,1)))},{int(100 + 100*(i/max(n_feats-1,1)))},{int(180 + 60*(i/max(n_feats-1,1)))},0.85)" for i in range(n_feats)]
        fig_bs = go.Figure(go.Bar(x=imp_df["SHAP"], y=imp_df["Feature"], orientation="h",
            marker_color=blues,
            text=[f"{v:.4f}" for v in imp_df["SHAP"]], textposition="outside",
            textfont=dict(color=C["muted"], size=11)))
        fig_bs.update_layout(title=t("title_global_importance", country=test_country),
            xaxis_title=t("axis_mean_shap"), height=max(280, len(FEATURES)*50), **PLT)
        apply_theme(fig_bs); st.plotly_chart(fig_bs, use_container_width=True)



hr()
# ════════════════════════════════════════════════════════════════
# Helper: detecta si una variable es categórica (niveles enteros)
# ════════════════════════════════════════════════════════════════
CONTINUOUS_FEATURES = {"mobility_transit_lag14", "stringency_index_lag14"}

def is_categorical(feat):
    return feat not in CONTINUOUS_FEATURES

def get_jitter(x_vals, jitter_strength=0.18):
    """Añade un pequeño jitter aleatorio a los valores categóricos del eje X para mejorar la legibilidad."""
    rng = np.random.default_rng(42)
    return x_vals + rng.uniform(-jitter_strength, jitter_strength, size=len(x_vals))

def find_best_interaction(sel_feat, shap_values, X_test, features):
    sel_idx  = features.index(sel_feat)
    sel_shap = shap_values[:, sel_idx]
    best_feat, best_corr = sel_feat, -1.0
    for f in features:
        if f == sel_feat:
            continue
        try:
            corr = abs(float(np.corrcoef(X_test[f].values, sel_shap)[0, 1]))
            if corr > best_corr:
                best_corr = corr
                best_feat = f
        except Exception:
            pass
    return best_feat, best_corr

# ══════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — Mini gráficos SHAP por variable (uno por cada variable seleccionada)
# ══════════════════════════════════════════════════════════════════════════
if shap_available and shap_values is not None:

    section_header("📊", t("sec3_title"), t("sec3_subtitle"))

    info_box(t("sec3_info"))

    n_feats_grid  = len(FEATURES)
    cols_per_row  = min(n_feats_grid, 3)
    n_rows_grid   = (n_feats_grid + cols_per_row - 1) // cols_per_row

    plt.style.use("default")
    fig_grid, axes = plt.subplots(
        n_rows_grid, cols_per_row,
        figsize=(cols_per_row * 4.2, n_rows_grid * 3.2),
        squeeze=False,
    )
    fig_grid.patch.set_facecolor("#FFFFFF")
    cmap_summary = plt.cm.RdBu_r

    for idx, feat in enumerate(FEATURES):
        row_i = idx // cols_per_row
        col_i = idx % cols_per_row
        ax    = axes[row_i][col_i]
        ax.set_facecolor("#F8FAFC")

        feat_vals  = X_test[feat].values.astype(float)
        shap_col_i = shap_values[:, idx]

        # Normalize feature values for colormap
        vmin_f, vmax_f = feat_vals.min(), feat_vals.max()
        norm_vals = (feat_vals - vmin_f) / (vmax_f - vmin_f + 1e-9)

        if is_categorical(feat):
            # ── Categórica: una fila por nivel entero, eje Y con jitter ──
            unique_levels = sorted(np.unique(feat_vals.astype(int)))
            level_labels  = [str(int(lv)) for lv in unique_levels]

            # Mapear niveles a posiciones del eje Y
            y_positions = {lv: i for i, lv in enumerate(unique_levels)}
            rng = np.random.default_rng(42)

            for lv in unique_levels:
                mask   = feat_vals.astype(int) == lv
                s_vals = shap_col_i[mask]
                n_lv   = mask.sum()
                y_jit  = y_positions[lv] + rng.uniform(-0.25, 0.25, size=n_lv)
                c_norm = norm_vals[mask]
                ax.scatter(s_vals, y_jit,
                           c=c_norm, cmap=cmap_summary, vmin=0, vmax=1,
                           s=18, alpha=0.65, linewidths=0, zorder=3)

            ax.set_yticks(list(y_positions.values()))
            ax.set_yticklabels(level_labels, fontsize=8, color=C["text"])
            ax.set_ylabel(t("axis_policy_level"), fontsize=8, color=C["muted"], labelpad=4)
            ax.axvline(0, color="#999999", linewidth=0.8, linestyle="--", alpha=0.6)
            ax.grid(axis="x", color=C["border"], linewidth=0.5, alpha=0.6)

        else:
            # ── Continua: scatter estilo beeswarm estándar ──
            # Ordenar por valor de la variable para que el degradado de color sea suave
            sort_idx  = np.argsort(feat_vals)
            rng2 = np.random.default_rng(42)
            y_jit = rng2.uniform(-0.4, 0.4, size=len(feat_vals))

            ax.scatter(shap_col_i[sort_idx], y_jit[sort_idx],
                       c=norm_vals[sort_idx], cmap=cmap_summary, vmin=0, vmax=1,
                       s=18, alpha=0.65, linewidths=0, zorder=3)
            ax.axvline(0, color="#999999", linewidth=0.8, linestyle="--", alpha=0.6)
            ax.set_yticks([])
            ax.grid(axis="x", color=C["border"], linewidth=0.5, alpha=0.6)

        ax.set_title(FEAT_LABELS.get(feat, feat), fontsize=9,
                     color=C["text"], fontweight="600", pad=5)
        ax.set_xlabel(t("axis_shap_value"), fontsize=8, color=C["muted"], labelpad=3)
        ax.tick_params(axis="x", labelsize=8, colors=C["text"])
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color(C["border2"])

    # Ocultar ejes no usados
    for idx in range(n_feats_grid, n_rows_grid * cols_per_row):
        axes[idx // cols_per_row][idx % cols_per_row].set_visible(False)

    # Barra de color compartida
    sm  = plt.cm.ScalarMappable(cmap=cmap_summary, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar_ax = fig_grid.add_axes([1.01, 0.15, 0.015, 0.7])
    cb_grid = fig_grid.colorbar(sm, cax=cbar_ax)
    cb_grid.set_label(t("colorbar_feat_value"), fontsize=8, color=C["text"], labelpad=8)
    cb_grid.ax.tick_params(labelsize=7, colors=C["text"])
    cb_grid.set_ticks([0, 0.5, 1])
    cb_grid.set_ticklabels([t("tick_low"), t("tick_mid2"), t("tick_high")])
    cb_grid.outline.set_edgecolor(C["border2"])

    fig_grid.tight_layout(pad=1.2)
    st.image(shap_fig_to_bytes(), use_container_width=True)

    hr()
    
# ══════════════════════════════════════════════════════════════════
# SECCIÓN 4 — Dependence Plot detallado con coloreado por interacción
# ══════════════════════════════════════════════════════════════════

if shap_available and shap_values is not None:

    section_header("📉", t("sec4_title"), t("sec4_subtitle"))

    info_box(t("sec4_info"))

    # ── Controles ──
    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 1])

    with ctrl1:
        sel_feat = st.selectbox(
            t("primary_feature_label"),
            options=FEATURES,
            format_func=lambda f: f"{ALL_FEATURES_META[f]['icon']} {FEAT_LABELS[f]}",
        )

    auto_interact_feat, auto_corr = find_best_interaction(sel_feat, shap_values, X_test, FEATURES)

    with ctrl2:
        _auto_label = t("auto_best_match")
        interact_options = [_auto_label] + [
            f"{ALL_FEATURES_META[f]['icon']} {FEAT_LABELS[f]}"
            for f in FEATURES if f != sel_feat
        ]
        interact_feat_display = st.selectbox(
            t("color_by_label"),
            options=interact_options,
            index=0,
        )
        if interact_feat_display == _auto_label:
            color_feat = auto_interact_feat
            color_mode = "auto"
        else:
            color_feat = next(
                f for f in FEATURES
                if f != sel_feat
                and f"{ALL_FEATURES_META[f]['icon']} {FEAT_LABELS[f]}" == interact_feat_display
            )
            color_mode = "manual"

    with ctrl3:
        dot_size = st.slider(t("dot_size_label"), 10, 80, 30, 5)

    # ── Badge ──
    badge_txt   = t("auto_detected") if color_mode == "auto" else t("manual")
    corr_suffix = f" · |corr|={auto_corr:.2f}" if color_mode == "auto" else ""
    _cdim    = C["dim"]
    _cpurple = C["purple"]
    _badge_html = (
        "<div style='margin:0.2rem 0 0.7rem 0;'>"
        + f"<span class='feat-chip'>X: {FEAT_LABELS[sel_feat]}</span>"
        + f"<span style='color:{_cdim};font-size:0.8rem;'>{t('colored_by')}</span>"
        + f"<span class='feat-chip' style='background:rgba(123,47,190,0.08);"
          f"border-color:rgba(123,47,190,0.25);color:{_cpurple};'>"
          f"🎨 {FEAT_LABELS[color_feat]}</span>"
        + f"<span style='color:{_cdim};font-size:0.72rem;margin-left:0.4rem;'>"
          f"({badge_txt}{corr_suffix})</span>"
        + "</div>"
    )
    st.markdown(_badge_html, unsafe_allow_html=True)

    col_dep, col_dep_r = st.columns([3, 1])

    with col_dep:
        from matplotlib.gridspec import GridSpec
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        plt.style.use("default")
        fig_dep = plt.figure(figsize=(8, 6.2))
        fig_dep.patch.set_facecolor("#FFFFFF")

        gs = GridSpec(2, 1, height_ratios=[1, 4], hspace=0.10, figure=fig_dep)
        ax_top  = fig_dep.add_subplot(gs[0])
        ax_main = fig_dep.add_subplot(gs[1])

        x_vals   = X_test[sel_feat].values.astype(float)
        y_vals   = shap_values[:, FEATURES.index(sel_feat)]
        c_vals   = X_test[color_feat].values.astype(float)
        cmap_dep = plt.cm.RdBu_r

        cv_min, cv_max = c_vals.min(), c_vals.max()
        c_norm = (c_vals - cv_min) / (cv_max - cv_min + 1e-9)

        # Panel superior
        ax_top.set_facecolor("#FAFBFC")
        feat_norm_top = (x_vals - x_vals.min()) / (x_vals.max() - x_vals.min() + 1e-9)
        rng_top = np.random.default_rng(42)
        y_top   = rng_top.uniform(0.1, 0.9, size=len(x_vals))
        ax_top.scatter(y_vals, y_top,
                       c=feat_norm_top, cmap=cmap_dep, vmin=0, vmax=1,
                       s=dot_size * 0.7, alpha=0.55, linewidths=0, zorder=3)
        ax_top.axvline(0, color="#999999", linewidth=0.8, linestyle="-", alpha=0.5)
        ax_top.set_yticks([])
        ax_top.set_xticks([])
        ax_top.spines[["top","right","left","bottom"]].set_visible(False)
        ax_top.set_title(FEAT_LABELS[sel_feat], fontsize=10,
                         color=C["text"], fontweight="600", pad=5)

        # Panel principal
        ax_main.set_facecolor("#F8FAFC")

        if is_categorical(sel_feat):
            x_plot = get_jitter(x_vals)
            unique_levels = sorted(np.unique(x_vals.astype(int)))
        else:
            x_plot = x_vals
            unique_levels = None

        sc = ax_main.scatter(
            x_plot, y_vals,
            c=c_norm, cmap=cmap_dep, vmin=0, vmax=1,
            s=dot_size, alpha=0.72,
            linewidths=0.2, edgecolors="white", zorder=3,
        )
        ax_main.axhline(0, color="#AAAAAA", linewidth=0.8, linestyle="--", alpha=0.5)

        if is_categorical(sel_feat) and unique_levels is not None:
            ax_main.set_xticks(unique_levels)
            ax_main.set_xticklabels([f"{int(lv)}" for lv in unique_levels],
                                    fontsize=9, color=C["text"])
        else:
            ax_main.tick_params(axis="x", labelsize=9, colors=C["text"])

        ax_main.set_xlabel(FEAT_LABELS[sel_feat], color=C["text"], fontsize=11, labelpad=6)
        ax_main.set_ylabel(t("axis_shap_value_for", feat=FEAT_LABELS[sel_feat]),
                           color=C["text"], fontsize=10, labelpad=6)
        ax_main.tick_params(axis="y", labelsize=9, colors=C["text"])
        ax_main.spines[["top","right"]].set_visible(False)
        ax_main.spines[["bottom","left"]].set_color(C["border2"])
        ax_main.grid(axis="y", color=C["border"], linewidth=0.5, alpha=0.6)

        divider = make_axes_locatable(ax_main)
        cax = divider.append_axes("right", size="3%", pad=0.12)
        cb  = fig_dep.colorbar(sc, cax=cax)
        c_label = FEAT_LABELS[color_feat]
        cb.set_label(c_label, fontsize=9, color=C["text"], rotation=270, labelpad=14)

        if is_categorical(color_feat):
            cb_levels = sorted(np.unique(c_vals.astype(int)))
            cb_ticks  = [(lv - cv_min) / (cv_max - cv_min + 1e-9) for lv in cb_levels]
            cb.set_ticks(cb_ticks)
            cb.set_ticklabels([str(int(lv)) for lv in cb_levels])
        cb.ax.tick_params(labelsize=8, colors=C["text"])
        cb.outline.set_edgecolor(C["border2"])

        ax_top.set_xlim(ax_main.get_ylim())

        plt.tight_layout(pad=0.6)
        st.image(shap_fig_to_bytes(), use_container_width=True)

    with col_dep_r:
        fi_val    = np.abs(shap_values[:, FEATURES.index(sel_feat)]).mean()
        shap_col  = shap_values[:, FEATURES.index(sel_feat)]
        direction = (
            t("dir_higher_increases")
            if np.corrcoef(x_vals, shap_col)[0, 1] > 0
            else t("dir_higher_reduces")
        )
        feat_type_lbl = t("feat_type_categorical") if is_categorical(sel_feat) else t("feat_type_continuous")

        metric_card(t("metric_mean_shap"), f"{fi_val:.4f}", t("metric_global_importance"))

        interact_corr = abs(float(np.corrcoef(X_test[color_feat].values, shap_col)[0, 1]))
        int_label = FEAT_LABELS[color_feat]
        _with_label = t("metric_with", feat=int_label[:16] + "…" if len(int_label) > 16 else int_label)
        metric_card(t("metric_interaction"), f"{interact_corr:.3f}", _with_label)

        _cdim3 = C["dim"]
        info_box(t("feat_type_label", val=feat_type_lbl, dir=direction,
                   color_feat=FEAT_LABELS[color_feat], dim=_cdim3))

    hr()

# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 5 — SHAP local por fecha (solo para modelos de árboles)
# ─────────────────────────────────────────────────────────────────────────────
if shap_available and shap_values is not None:
    section_header("📅", t("sec5_title", country=test_country), t("sec5_subtitle"))

    info_box(t("sec5_info"))

    available_dates = sorted(test_df["date"].dt.date.unique())
    sel_date = st.select_slider(
        t("select_date"),
        options=available_dates,
        value=available_dates[len(available_dates)//2]
    )

    idx_list = test_df.index[test_df["date"].dt.date == sel_date].tolist()

    if not idx_list:
        warn_box(t("no_data_for_date"))

    else:
        row_pos = test_df.index.get_loc(idx_list[0])
        row = test_df.iloc[row_pos]

        c_meta, c_wf = st.columns([1, 2])

        with c_meta:
            bv, pv = row["beta"], row["beta_pred"]

            trend = t("growing") if bv > 1/7 else t("declining")

            metric_card(
                t("metric_date"),
                str(sel_date),
                t("metric_selected_day")
            )

            metric_card(
                t("metric_real_beta"),
                f"{bv:.4f}",
                trend
            )

            metric_card(
                t("metric_predicted_beta"),
                f"{pv:.4f}",
                t("metric_error", err=abs(bv-pv))
            )

            st.markdown(
                f"<p style='color:{C['pred']};font-size:0.78rem;font-weight:600;margin:0.7rem 0 0.3rem 0;'>"
                f"{t('values_this_day')}</p>",
                unsafe_allow_html=True
            )

            for f in FEATURES:
                st.markdown(
                    f"<small style='color:{C['muted']};'>"
                    f"{FEAT_LABELS[f]}: "
                    f"<b style='color:{C['text']};'>{row[f]:.2f}</b>"
                    f"</small>",
                    unsafe_allow_html=True
                )


        with c_wf:

            plt.style.use("default")

            fig_wf = plt.figure(
                figsize=(8, max(3.5, len(FEATURES)*0.7))
            )

            fig_wf.patch.set_facecolor("#FFFFFF")


            # ─────────────────────────────────────────────
            # Compatibilidad SHAP local / Streamlit Cloud
            # ─────────────────────────────────────────────

            # Algunas versiones de SHAP devuelven lista
            if isinstance(shap_values, list):
                shap_values = shap_values[0]


            # Algunas versiones devuelven ndarray en expected_value
            base_value = explainer.expected_value

            if isinstance(base_value, np.ndarray):
                base_value = base_value.flatten()[0]


            shap.plots.waterfall(
                shap.Explanation(
                    values=shap_values[row_pos],
                    base_values=base_value,
                    data=X_test.iloc[row_pos],
                    feature_names=[
                        FEAT_LABELS.get(f, f)
                        for f in FEATURES
                    ]
                ),
                show=False
            )


            plt.tight_layout()

            st.image(
                shap_fig_to_bytes(),
                use_container_width=True
            )

    hr()
# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 6 — Simulador de Políticas
# ─────────────────────────────────────────────────────────────────────────────
section_header("🎛️", t("sec6_title", country=test_country), t("sec6_subtitle"))

info_box(t("sec6_info", country=test_country, algo_icon=algo_meta['icon'],
           algo_name=algo_name, n_feats=len(FEATURES)))

n_feats      = len(FEATURES)
cols_per_row = min(n_feats, 4)
slider_cols  = st.columns(cols_per_row)
user_vals    = {}

for i, feat in enumerate(FEATURES):
    lo, hi, step = FEAT_RANGES[feat]
    med = float(X_test[feat].median())
    with slider_cols[i % cols_per_row]:
        st.markdown(f"<small style='color:{C['pred']};font-weight:600;'>{ALL_FEATURES_META[feat]['icon']} {FEAT_LABELS[feat]}</small>", unsafe_allow_html=True)
        user_vals[feat] = st.slider(feat, float(lo), float(hi),
            float(round(med/step)*step) if step else float(med),
            float(step), label_visibility="collapsed")
    if (i+1) % cols_per_row == 0 and (i+1) < n_feats:
        slider_cols = st.columns(cols_per_row)

X_user    = pd.DataFrame([user_vals], columns=FEATURES)
beta_user = float(model.predict(X_user)[0])
beta_mean = float(y_pred.mean())
thresh    = 1 / 7

c1r, c2r, c3r = st.columns(3)
with c1r: metric_card(t("metric_predicted_beta_scenario"), f"{beta_user:.4f}", t("metric_custom_scenario"))
with c2r:
    if beta_user > thresh: metric_card(t("metric_epidemic_status"), t("growing"),   f"β > {thresh:.3f}")
    else:                  metric_card(t("metric_epidemic_status"), t("declining"), f"β ≤ {thresh:.3f}")
with c3r:
    chg  = ((beta_user - beta_mean) / beta_mean) * 100
    sign = "+" if chg >= 0 else ""
    metric_card(t("metric_vs_historical"), f"{sign}{chg:.1f}%", t("metric_avg_beta", beta=beta_mean))

if shap_available and explainer is not None:
    shap_user = explainer.shap_values(X_user)
    plt.style.use("default")
    fig_u = plt.figure(figsize=(8, max(3.5, n_feats*0.7)))
    fig_u.patch.set_facecolor("#FFFFFF")
    shap.plots.waterfall(shap.Explanation(
        values=shap_user[0], base_values=explainer.expected_value,
        data=X_user.iloc[0], feature_names=[FEAT_LABELS[f] for f in FEATURES]), show=False)
    plt.tight_layout(); st.image(shap_fig_to_bytes(), caption=t("shap_caption_scenario"), use_container_width=True)

hr()

# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 7 — Comparación de Escenarios SEIR
# -----------------------------------------------------------------------------
section_header(
    "⚙️", t("sec7_title", country=test_country),
    t("sec7_subtitle"),
)

info_box(t("sec7_info"))

# ─── Selector de compartimento compartido ───
COMP_OPTIONS = {
    t("comp_active_infectious"): "I",
    t("comp_new_infections"):    "new_infected",
}
COMP_YLABEL = {
    "I":            t("comp_active_infectious"),
    "new_infected": t("comp_new_infections"),
}
COMP_DESC = {
    "I":            t("comp_desc_I"),
    "new_infected": t("comp_desc_new"),
}
sel_comp_label = st.selectbox(
    t("comp_select_label"),
    options=list(COMP_OPTIONS.keys()), index=0,
)
sel_comp = COMP_OPTIONS[sel_comp_label]
info_box(f"<b>{sel_comp_label}:</b> {COMP_DESC[sel_comp]}")

# ─── Condiciones iniciales SEIR (comunes) ───
population_all = float(test_df["population"].iloc[0]) if "population" in test_df.columns else 47_000_000

# ══════════════════════════════════════════════════════════════════════════════
# ESCENARIO 1 — Periodo epidémico completo · tres enfoques de β
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style='background:linear-gradient(90deg,rgba(0,119,182,0.07),rgba(0,119,182,0.01));
border-left:3px solid {C["pred"]};border-radius:0 8px 8px 0;
padding:0.6rem 1.2rem;margin:1.5rem 0 0.8rem 0;'>
<span style='font-size:1rem;font-weight:600;color:{C["text"]};'>
{t("scenario1_banner_title")}</span><br>
<span style='font-size:0.82rem;color:{C["muted"]};'>
{t("scenario1_banner_sub")}</span>
</div>""", unsafe_allow_html=True)

info_box(t("scenario1_info"))

# Rango de fechas: inicio = primera fecha en el dataset para este país, fin = seleccionable por el usuario
all_dates_s1 = sorted(test_df["date"].dt.date.unique())
s1_start_fixed = all_dates_s1[7]

s1c1, s1c2, s1c_param = st.columns([1, 1, 1])
with s1c1:
    s1_start = st.date_input(
        t("s1_start_label"),
        value=s1_start_fixed,
        min_value=all_dates_s1[0],
        max_value=all_dates_s1[-1],
        key="s1_start",
        help=t("s1_start_help"),
    )
with s1c2:
    s1_end = st.date_input(
        t("s1_end_label"),
        value=all_dates_s1[-1],
        min_value=all_dates_s1[min(7, len(all_dates_s1)-1)],
        max_value=all_dates_s1[-1],
        key="s1_end",
    )

mask_s1 = (test_df["date"].dt.date >= s1_start) & (test_df["date"].dt.date <= s1_end)
data_s1  = test_df[mask_s1].copy()

if len(data_s1) < 7:
    warn_box(t("s1_not_enough_data"))
else:
    I0_s1 = float(data_s1["I_real"].iloc[0]) if "I_real" in data_s1.columns else 1000.0
    E0_s1 = float(data_s1["E_real"].iloc[0]) if "E_real" in data_s1.columns else 2000.0
    dates_s1 = data_s1["date"].values

    beta_mean_s1 = float(data_s1["beta"].dropna().mean())
    beta_min_s1  = float(data_s1["beta"].dropna().quantile(0.05))
    beta_max_s1  = float(data_s1["beta"].dropna().quantile(0.95))

    with s1c_param:
        st.markdown(f"<p style='color:{C['muted']};font-size:0.78rem;font-weight:600;"
                    f"margin:0.3rem 0 0.2rem 0;'>{t('s1_const_beta_label')}</p>",
                    unsafe_allow_html=True)
        beta_constant = st.slider(
            "β constant", label_visibility="collapsed",
            min_value=round(max(0.005, beta_min_s1 - 0.03), 4),
            max_value=round(beta_max_s1 + 0.03, 4),
            value=round(beta_mean_s1, 4),
            step=0.001, format="%.4f", key="s1_beta_const",
        )
        metric_card(t("s1_mean_beta"), f"{beta_mean_s1:.4f}", f"{s1_start_fixed} → {s1_end}")

    # Series β
    beta_s1_const = np.full(len(data_s1), beta_constant)
    beta_s1_pred  = model.predict(data_s1[FEATURES])
    beta_s1_estim = data_s1["beta"].values

    traj_s1_const = simulate_seir(beta_s1_const, population_all, E0_s1, I0_s1)
    traj_s1_pred  = simulate_seir(beta_s1_pred,  population_all, E0_s1, I0_s1)
    traj_s1_estim = simulate_seir(beta_s1_estim, population_all, E0_s1, I0_s1)

    # Gráfico comparativo de β
    fig_s1_beta = go.Figure()
    fig_s1_beta.add_trace(go.Scatter(x=dates_s1, y=beta_s1_const,
        name=t("leg_beta_const", val=beta_constant),
        line=dict(color=C["neutral2"], width=2, dash="dot")))
    fig_s1_beta.add_trace(go.Scatter(x=dates_s1, y=beta_s1_pred,
        name=t("leg_beta_pred_ml"),
        line=dict(color=C["pred"], width=2, dash="dash")))
    fig_s1_beta.add_trace(go.Scatter(x=dates_s1, y=beta_s1_estim,
        name=t("leg_beta_estimated"),
        line=dict(color=C["real"], width=2.5)))
    fig_s1_beta.add_hline(y=1/7, line_dash="dot", line_color=C["gold"], line_width=1.2,
        annotation_text=t("growth_threshold", gamma=1/7),
        annotation_font_color=C["gold"], annotation_position="bottom right")
    fig_s1_beta.update_layout(
        title=t("title_s1_beta_series", country=test_country),
        xaxis_title=t("axis_date"), yaxis_title=t("axis_beta"), height=280, **PLT)
    apply_theme(fig_s1_beta)
    st.plotly_chart(fig_s1_beta, use_container_width=True)

    # Gráfico de compartimento SEIR
    fig_s1 = go.Figure()
    fig_s1.add_trace(go.Scatter(x=dates_s1, y=traj_s1_const[sel_comp],
        name=t("leg_classical_seir", val=beta_constant),
        line=dict(color=C["neutral2"], width=2, dash="dot")))
    fig_s1.add_trace(go.Scatter(x=dates_s1, y=traj_s1_pred[sel_comp],
        name=t("leg_dynamic_pred"),
        line=dict(color=C["pred"], width=2.2, dash="dash")))
    fig_s1.add_trace(go.Scatter(x=dates_s1, y=traj_s1_estim[sel_comp],
        name=t("leg_dynamic_estim"),
        line=dict(color=C["real"], width=2.8)))
    fig_s1.update_layout(
        title=t("title_s1_seir", comp=sel_comp_label, country=test_country),
        xaxis_title=t("axis_date"), yaxis_title=COMP_YLABEL[sel_comp], height=420, **PLT)
    apply_theme(fig_s1)
    st.plotly_chart(fig_s1, use_container_width=True)

    # Tarjetas resumen de picos
    _k_classical = t("scen_classical")
    _k_pred      = t("scen_pred_ml")
    _k_estim     = t("scen_estimated")
    s1_scens = {
        _k_classical: (traj_s1_const, C["neutral2"]),
        _k_pred:      (traj_s1_pred,  C["pred"]),
        _k_estim:     (traj_s1_estim, C["real"]),
    }
    s1_cols = st.columns(3)
    s1_peaks = {}
    for ci, (sc_name, (traj, col_c)) in enumerate(s1_scens.items()):
        vals = traj[sel_comp]
        s1_peaks[sc_name] = safe_peak(vals)
        with s1_cols[ci]:
            st.markdown(
                f"<div style='background:{C['white']};border:1px solid {C['border']};"
                f"border-top:3px solid {col_c};border-radius:10px;padding:0.75rem 0.9rem;"
                f"text-align:center;box-shadow:0 2px 8px {C['shadow']};margin-top:0.5rem;'>"
                f"<div style='color:{C['muted']};font-size:0.68rem;text-transform:uppercase;"
                f"letter-spacing:0.09em;font-weight:600;'>{sc_name}</div>"
                f"<div style='color:{col_c};font-size:1.45rem;font-weight:700;"
                f"font-family:DM Mono,monospace;margin:0.2rem 0;'>{s1_peaks[sc_name]:,}</div>"
                f"<div style='color:{C['dim']};font-size:0.7rem;'>"
                f"{t('peak_day', day=safe_peak_day(vals), final=safe_final(vals))}</div>"
                f"</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:0.7rem;'></div>", unsafe_allow_html=True)
    diff_pred  = s1_peaks[_k_pred]  - s1_peaks[_k_estim]
    diff_const = s1_peaks[_k_classical] - s1_peaks[_k_estim]
    _dp = f"{'+' if diff_pred >= 0 else ''}{diff_pred:,}"
    _dc = f"{'+' if diff_const >= 0 else ''}{diff_const:,}"
    info_box(t("s1_interpretation", comp=sel_comp_label, diff_const=_dc, diff_pred=_dp))

# ══════════════════════════════════════════════════════════════════════════════
# ESCENARIO 2 — Ventana de política estable · impacto del nivel de restricción
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style='background:linear-gradient(90deg,rgba(10,158,110,0.07),rgba(10,158,110,0.01));
border-left:3px solid {C["strict"]};border-radius:0 8px 8px 0;
padding:0.6rem 1.2rem;margin:2rem 0 0.8rem 0;'>
<span style='font-size:1rem;font-weight:600;color:{C["text"]};'>
{t("scenario2_banner_title")}</span><br>
<span style='font-size:0.82rem;color:{C["muted"]};'>
{t("scenario2_banner_sub")}
</span></div>""", unsafe_allow_html=True)

info_box(t("scenario2_info"))

col_s2d1, col_s2d2 = st.columns(2)
with col_s2d1:
    s2_start = st.date_input(
        t("s2_start_label"),
        value=pd.Timestamp("2020-09-06").date(),
        min_value=all_dates_s1[0],
        max_value=all_dates_s1[-1],
        key="s2_start",
    )
with col_s2d2:
    s2_end = st.date_input(
        t("s2_end_label"),
        value=pd.Timestamp("2020-09-26").date(),
        min_value=all_dates_s1[0],
        max_value=all_dates_s1[-1],
        key="s2_end",
    )

mask_s2 = (test_df["date"].dt.date >= s2_start) & (test_df["date"].dt.date <= s2_end)
data_s2  = test_df[mask_s2].copy()

if len(data_s2) < 5:
    warn_box(t("s2_not_enough_data"))
else:
    I0_s2    = float(data_s2["I_real"].iloc[0]) if "I_real" in data_s2.columns else 1000.0
    E0_s2    = float(data_s2["E_real"].iloc[0]) if "E_real" in data_s2.columns else 2000.0
    dates_s2 = data_s2["date"].values

    # ─── Políticas activas en esta ventana ───
    st.markdown(
        f"<p style='color:{C['pred']};font-size:0.82rem;font-weight:600;"
        f"text-transform:uppercase;letter-spacing:0.07em;margin:1rem 0 0.4rem 0;'>"
        f"{t('active_policies')}</p>",
        unsafe_allow_html=True,
    )
    policy_cols = st.columns(min(len(FEATURES), 4))
    for pi, feat in enumerate(FEATURES):
        if feat in data_s2.columns:
            mean_val = float(data_s2[feat].mean())
            std_val  = float(data_s2[feat].std())
            lo, hi, _ = FEAT_RANGES[feat]
            intensity = (mean_val - lo) / (hi - lo + 1e-9)
            card_color = C["strict"] if intensity > 0.6 else (C["loose"] if intensity < 0.3 else C["pred"])
            with policy_cols[pi % min(len(FEATURES), 4)]:
                st.markdown(
                    f"<div style='background:{C['white']};border:1px solid {C['border']};"
                    f"border-top:3px solid {card_color};border-radius:10px;"
                    f"padding:0.6rem 0.8rem;text-align:center;"
                    f"box-shadow:0 2px 6px {C['shadow']};margin-bottom:0.4rem;'>"
                    f"<div style='color:{C['muted']};font-size:0.65rem;text-transform:uppercase;"
                    f"letter-spacing:0.08em;'>{ALL_FEATURES_META[feat]['icon']} {FEAT_LABELS[feat]}</div>"
                    f"<div style='color:{card_color};font-size:1.3rem;font-weight:700;"
                    f"font-family:DM Mono,monospace;'>{mean_val:.2f}</div>"
                    f"<div style='color:{C['dim']};font-size:0.65rem;'>±{std_val:.2f} std</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # ─── Selector de caso base ───
    s2_base_choice = st.radio(
        t("base_case_source"),
        options=["Estimated β", "Predicted β (ML model)"],
        index=0, key="s2_base_radio", horizontal=True,
        format_func=lambda v: t("base_estimated") if v == "Estimated β" else t("base_predicted_ml"),
    )

    # ─── Editable NPI scenario parameters ───
    SCENARIO_FEATS = [f for f in FEATURES if f != "people_fully_vaccinated_per_hundred"]

    STRICT_DEFAULTS = {
        "School_closing_lag14": 3, "Facial_Coverings_lag14": 4,
        "mobility_transit_lag14": -60.0, "Stay_at_home_lag14": 2,
        "Restrictions_on_gatherings_lag14": 4, "Workplace_closing_lag14": 3,
        "stringency_index_lag14": 90.0,
    }
    LOOSE_DEFAULTS = {
        "School_closing_lag14": 0, "Facial_Coverings_lag14": 1,
        "mobility_transit_lag14": -10.0, "Stay_at_home_lag14": 0,
        "Restrictions_on_gatherings_lag14": 1, "Workplace_closing_lag14": 0,
        "stringency_index_lag14": 20.0,
    }

    with st.expander(t("expand_high_restr"), expanded=False):
        strict_cols = st.columns(min(len(SCENARIO_FEATS), 4))
        STRICT_VALS = {}
        for i, f in enumerate(SCENARIO_FEATS):
            lo, hi, step = FEAT_RANGES[f]
            dv = max(float(lo), min(float(hi), float(STRICT_DEFAULTS.get(f, hi))))
            with strict_cols[i % min(len(SCENARIO_FEATS), 4)]:
                st.markdown(
                    f"<small style='color:{C['strict']};font-weight:600;'>"
                    f"{ALL_FEATURES_META[f]['icon']} {FEAT_LABELS[f]}</small>",
                    unsafe_allow_html=True)
                STRICT_VALS[f] = st.slider(
                    f"s2_strict_{f}", float(lo), float(hi),
                    float(round(dv / step) * step) if step else dv,
                    float(step), label_visibility="collapsed", key=f"s2strict_{f}")
        for f in FEATURES:
            if f not in STRICT_VALS:
                STRICT_VALS[f] = float(STRICT_DEFAULTS.get(f, float(data_s2[f].median())))

    with st.expander(t("expand_low_restr"), expanded=False):
        loose_cols = st.columns(min(len(SCENARIO_FEATS), 4))
        LOOSE_VALS = {}
        for i, f in enumerate(SCENARIO_FEATS):
            lo, hi, step = FEAT_RANGES[f]
            dv = max(float(lo), min(float(hi), float(LOOSE_DEFAULTS.get(f, lo))))
            with loose_cols[i % min(len(SCENARIO_FEATS), 4)]:
                st.markdown(
                    f"<small style='color:{C['loose']};font-weight:600;'>"
                    f"{ALL_FEATURES_META[f]['icon']} {FEAT_LABELS[f]}</small>",
                    unsafe_allow_html=True)
                LOOSE_VALS[f] = st.slider(
                    f"s2_loose_{f}", float(lo), float(hi),
                    float(round(dv / step) * step) if step else dv,
                    float(step), label_visibility="collapsed", key=f"s2loose_{f}")
        for f in FEATURES:
            if f not in LOOSE_VALS:
                LOOSE_VALS[f] = float(LOOSE_DEFAULTS.get(f, float(data_s2[f].median())))

    # ─── Series β para el escenario 2 ───
    if s2_base_choice == "Estimated β":
        beta_s2_base = data_s2["beta"].values
        base_label   = t("base_estimated")
        base_color   = C["real"]
    else:
        beta_s2_base = model.predict(data_s2[FEATURES])
        base_label   = t("base_predicted_ml")
        base_color   = C["pred"]

    sc2_strict = data_s2.copy()
    sc2_loose  = data_s2.copy()
    for f in FEATURES:
        sc2_strict[f] = STRICT_VALS[f]
        sc2_loose[f]  = LOOSE_VALS[f]

    beta_s2_strict = model.predict(sc2_strict[FEATURES])
    beta_s2_loose  = model.predict(sc2_loose[FEATURES])

    traj_s2_base   = simulate_seir(beta_s2_base,   population_all, E0_s2, I0_s2)
    traj_s2_strict = simulate_seir(beta_s2_strict,  population_all, E0_s2, I0_s2)
    traj_s2_loose  = simulate_seir(beta_s2_loose,   population_all, E0_s2, I0_s2)

    # Gráfico comparativo de β
    fig_s2_beta = go.Figure()
    fig_s2_beta.add_trace(go.Scatter(x=dates_s2, y=beta_s2_base,
        name=t("leg_beta_base", label=base_label),
        line=dict(color=base_color, width=2.5)))
    fig_s2_beta.add_trace(go.Scatter(x=dates_s2, y=beta_s2_strict,
        name=t("leg_beta_high_restr"),
        line=dict(color=C["strict"], width=1.8, dash="dash")))
    fig_s2_beta.add_trace(go.Scatter(x=dates_s2, y=beta_s2_loose,
        name=t("leg_beta_low_restr"),
        line=dict(color=C["loose"], width=1.8, dash="dash")))
    fig_s2_beta.add_hline(y=1/7, line_dash="dot", line_color=C["gold"], line_width=1.2,
        annotation_text=t("growth_threshold", gamma=1/7),
        annotation_font_color=C["gold"], annotation_position="bottom right")
    fig_s2_beta.update_layout(
        title=t("title_s2_beta_series", country=test_country, start=s2_start, end=s2_end),
        xaxis_title=t("axis_date"), yaxis_title=t("axis_beta"), height=280, **PLT)
    apply_theme(fig_s2_beta)
    st.plotly_chart(fig_s2_beta, use_container_width=True)

    # Gráfico de compartimento SEIR
    _es_keywords = ["estimado", "Estimated"]
    base_icon = "🔴" if any(k in base_label for k in _es_keywords) else "🔵"
    fig_s2 = go.Figure()
    fig_s2.add_trace(go.Scatter(x=dates_s2, y=traj_s2_base[sel_comp],
        name=t("leg_base_case", icon=base_icon, label=base_label),
        line=dict(color=base_color, width=2.8)))
    fig_s2.add_trace(go.Scatter(x=dates_s2, y=traj_s2_strict[sel_comp],
        name=t("leg_high_restr"),
        line=dict(color=C["strict"], width=2, dash="dash")))
    fig_s2.add_trace(go.Scatter(x=dates_s2, y=traj_s2_loose[sel_comp],
        name=t("leg_low_restr"),
        line=dict(color=C["loose"], width=2, dash="dash")))
    fig_s2.update_layout(
        title=t("title_s2_seir", comp=sel_comp_label, country=test_country),
        xaxis_title=t("axis_date"), yaxis_title=COMP_YLABEL[sel_comp], height=420, **PLT)
    apply_theme(fig_s2)
    st.plotly_chart(fig_s2, use_container_width=True)

    # Tarjetas resumen de picos
    _k_base  = t("scen_base", label=base_label)
    _k_high  = t("scen_high_restr")
    _k_low   = t("scen_low_restr")
    s2_scens = {
        _k_base:  (traj_s2_base,   base_color),
        _k_high:  (traj_s2_strict,  C["strict"]),
        _k_low:   (traj_s2_loose,   C["loose"]),
    }
    s2_cols = st.columns(3)
    s2_peaks = {}
    for ci, (sc_name, (traj, col_c)) in enumerate(s2_scens.items()):
        vals = traj[sel_comp]
        s2_peaks[sc_name] = safe_peak(vals)
        with s2_cols[ci]:
            st.markdown(
                f"<div style='background:{C['white']};border:1px solid {C['border']};"
                f"border-top:3px solid {col_c};border-radius:10px;padding:0.75rem 0.9rem;"
                f"text-align:center;box-shadow:0 2px 8px {C['shadow']};margin-top:0.5rem;'>"
                f"<div style='color:{C['muted']};font-size:0.68rem;text-transform:uppercase;"
                f"letter-spacing:0.09em;font-weight:600;'>{sc_name}</div>"
                f"<div style='color:{col_c};font-size:1.45rem;font-weight:700;"
                f"font-family:DM Mono,monospace;margin:0.2rem 0;'>{s2_peaks[sc_name]:,}</div>"
                f"<div style='color:{C['dim']};font-size:0.7rem;'>"
                f"{t('peak_day', day=safe_peak_day(traj[sel_comp]), final=safe_final(traj[sel_comp]))}</div>"
                f"</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:0.7rem;'></div>", unsafe_allow_html=True)
    base_peak   = s2_peaks[_k_base]
    strict_peak = s2_peaks[_k_high]
    loose_peak  = s2_peaks[_k_low]
    diff_s = strict_peak - base_peak
    diff_l = loose_peak  - base_peak
    _ds = f"{'+' if diff_s >= 0 else ''}{diff_s:,}"
    _dl = f"{'+' if diff_l >= 0 else ''}{diff_l:,}"
    info_box(t("s2_interpretation", comp=sel_comp_label, start=s2_start, end=s2_end,
               label=base_label, base_peak=base_peak,
               diff_s=_ds, strict_peak=strict_peak,
               diff_l=_dl, loose_peak=loose_peak))
hr()

# ─────────────────────────────────────────────────────────────────────────────
# PIE DE PÁGINA
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='text-align:center;padding:1.4rem 0;color:{C["dim"]};font-size:0.75rem;
     font-family:DM Mono,monospace;border-top:1px solid {C["border"]};margin-top:0.8rem;'>
    {t("footer_text")}<br>
    <span style='color:{C["border2"]};'>Streamlit · scikit-learn · SHAP · Plotly</span>
    &nbsp;·&nbsp;
    <span style='color:{C["pred"]};font-weight:600;'>{test_country}</span>
    &nbsp;·&nbsp;
    <span style='color:{C["pred"]};'>{algo_meta["icon"]} {algo_name}</span>
    &nbsp;·&nbsp;
    <span style='color:{period_color};font-weight:600;'>{period_tag}</span>
</div>
""", unsafe_allow_html=True)

