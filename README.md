# 🦠 Epidemiología Explicable: XAI para la Propagación de Enfermedades

TFM — Máster en Sistemas Inteligentes, Universidad de Salamanca (2026) — Nota: 9/10

## 🎯 Resumen
Sistema que estima β(t) (tasa de transmisión) con Theil-Sen + linealización SEIR,
la predice con modelos de ML en esquema zero-shot (Extra Trees, R²=0.840, ρ=0.922),
la explica con SHAP, y simula escenarios de política con un modelo SEIR.

## 🧠 Componentes
- Estimación robusta de β(t) (Theil-Sen sobre ventanas de 14 días)
- ML zero-shot: entrenado en 8 países europeos, evaluado en España
- Explicabilidad SHAP (global, local, dependencia)
- Simulador SEIR con escenarios contrafactuales de política
- Dashboard interactivo en Streamlit (ES/EN)

## 📊 Resultados clave
- Mascarillas > movilidad transporte > restricciones reuniones (importancia SHAP)
- Escenario restricciones altas: -66.222 infectados vs base (21 días)
- Escenario restricciones bajas: +109.738 infectados vs base

## 🚀 Demo
https://epidemiologia-explicable-covid19-jkq4jkvbsqg6djpax2cl48.streamlit.app/ 

## 🛠️ Stack
Python · scikit-learn · XGBoost · SHAP · Streamlit · Plotly

## 📄 Documento completo
[TFM_HamzaBenali_EpidemiologiaExplicable.pdf](docs/TFM_HamzaBenali_EpidemiologiaExplicable.pdf)

## ⚙️ Cómo ejecutarlo localmente
\`\`\`bash
git clone ...
pip install -r requirements.txt
streamlit run app.py
\`\`\`

## 🎓 Contexto académico
Dirigido por Dra. Belén Pérez Lancho y Dr. Javier Prieto Tejedor.
Con el apoyo del Proyecto Cátedra Internacional en IA Fiable y Reto Demográfico
(ref. TSI-100933-2023-0).
