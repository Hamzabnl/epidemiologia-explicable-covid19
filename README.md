# 🦠 Epidemiología Explicable: XAI para la Propagación de Enfermedades

TFM — Máster en Sistemas Inteligentes, Universidad de Salamanca (2026) — Nota: 9/10

> 📄 Autor: **Hamza Benali** · Tutora: Dra. Belén Pérez Lancho · Co-tutor: Dr. Javier Prieto Tejedor
> Departamento de Informática y Automática, Universidad de Salamanca

---

## 🎯 Motivación

Los modelos epidemiológicos clásicos (SIR/SEIR) dependen de una tasa de transmisión β que es difícil de estimar en tiempo real y que cambia constantemente en función de las políticas públicas y el comportamiento social. Este proyecto propone un pipeline completo que:

1. **Estima** β(t) de forma robusta a partir de datos reales de incidencia.
2. **Predice** β(t) mediante machine learning a partir de intervenciones no farmacéuticas (NPIs) y movilidad, entrenando en varios países y evaluando en uno nunca visto (*zero-shot*).
3. **Explica** qué políticas influyen más en la transmisión y cómo, usando SHAP.
4. **Simula** escenarios contrafactuales de política pública con un modelo SEIR.

El objetivo no es maximizar la precisión predictiva, sino la **interpretabilidad epidemiológica**: ofrecer a los responsables de salud pública una herramienta que explique *por qué* sube o baja la transmisión, no solo que la prediga.

---

## 🧩 Arquitectura del sistema

```
Datos (OWID + OxCGRT + Google Mobility)
        │
        ▼
Estimación de β(t)  ──►  Theil-Sen + linealización del modelo SEIR
        │
        ▼
Modelado ML (zero-shot)  ──►  Extra Trees entrenado en 8 países UE, evaluado en España
        │
        ▼
Explicabilidad XAI  ──►  SHAP global / local / gráficos de dependencia
        │
        ▼
Simulación SEIR  ──►  Escenarios contrafactuales de política
        │
        ▼
Aplicación interactiva (Streamlit, ES/EN)
```

---

## 📊 Resultados clave

### Estimación de β(t)
Serie temporal validada externamente frente a `R(t) × γ` (Our World in Data), sin haber usado esos datos en la estimación. Alta concordancia en momentos clave (confinamientos, oleadas).

### Modelado Machine Learning (zero-shot: entrenado en 8 países, evaluado en España)

| Modelo | R² | MAE | Pearson ρ |
|---|---|---|---|
| XGBoost | 0.802 | 0.0699 | 0.907 |
| Gradient Boosting | 0.814 | 0.0687 | 0.921 |
| Random Forest | 0.831 | 0.0672 | 0.927 |
| **Extra Trees (elegido)** | **0.840** | **0.0690** | **0.922** |

Extra Trees fue seleccionado no solo por sus métricas, sino porque ofrece **explicaciones SHAP más estables y coherentes**, algo imprescindible cuando el objetivo del trabajo es la interpretabilidad.

### Explicabilidad SHAP — importancia global de las NPIs

1. 😷 Uso obligatorio de mascarillas
2. 🚇 Movilidad en transporte público
3. 👥 Restricciones a reuniones
4. 🏫 Cierre de centros escolares
5. 🏠 Confinamiento domiciliario

Los gráficos de dependencia revelan **rendimientos decrecientes**: las primeras etapas de una medida (p. ej. obligar a las mascarillas) generan el mayor efecto protector, mientras que endurecerla más allá aporta beneficios marginales cada vez menores.

### Simulación SEIR — impacto contrafactual (España, 21 días)

| Escenario | Infectados activos (final) | Diferencia vs. base |
|---|---|---|
| Restricciones bajas | 206.160 | +109.738 |
| Base (política real) | 96.422 | — |
| Restricciones altas | 30.200 | −66.222 |

El SEIR parametrizado con β predicho por ML se ajusta mucho mejor a la realidad que el SEIR clásico con β constante (pico de 511.658 vs. 1.278.351 infectados), demostrando el valor añadido del enfoque.

---

## 🛠️ Stack tecnológico

- **Python** · pandas · NumPy
- **Machine Learning**: scikit-learn (Random Forest, Extra Trees, Gradient Boosting), XGBoost
- **Explicabilidad**: SHAP (TreeSHAP)
- **Simulación**: modelo SEIR con integración por diferencias finitas
- **Visualización**: Matplotlib, Plotly
- **Aplicación interactiva**: Streamlit (bilingüe ES/EN)

---

## 📁 Estructura del repositorio

```
├── app.py                      # Dashboard interactivo (Streamlit)
├── docs/
│   └── TFM_HamzaBenali_EpidemiologiaExplicable.pdf   # Memoria completa
├── figures/                    # Capturas de la aplicación
├── requirements.txt
└── README.md
```

---

## 🚀 Demo en vivo

👉 **[Probar la aplicación]([https://TU-APP.streamlit.app](https://epidemiologia-explicable-covid19-jioqoctudgkuvoovfrpdat.streamlit.app/))** 

## ⚙️ Ejecución local

```bash
git clone https://github.com/Hamzabnl/epidemiologia-explicable-covid19.git
cd epidemiologia-explicable-covid19
pip install -r requirements.txt
streamlit run app.py
```

---

## 📚 Fuentes de datos

- [Our World in Data — COVID-19](https://github.com/owid/covid-19-data)
- [Oxford COVID-19 Government Response Tracker (OxCGRT)](https://www.bsg.ox.ac.uk/research/publications/variation-government-responses-covid-19)
- [Google COVID-19 Community Mobility Reports](https://www.google.com/covid19/mobility/)

## 🔍 Metodología en detalle

Para la derivación matemática completa (linealización del SEIR, estimador de Theil-Sen, teoría de Shapley/TreeSHAP, criterios de selección de modelo, limitaciones y trabajo futuro), consulta la **[memoria completa del TFM](docs/TFM_HamzaBenali_EpidemiologiaExplicable.pdf)**.

## 🎓 Agradecimientos

Este TFM ha recibido el apoyo del **Proyecto Cátedra Internacional en Inteligencia Artificial Fiable y Reto Demográfico**, dentro de la Estrategia Nacional de Inteligencia Artificial (ENIA), en el marco del Plan de Recuperación, Transformación y Resiliencia Europeo (ref. TSI-100933-2023-0).


*¿Preguntas o sugerencias? Abre un issue o contacta vía [LinkedIn](https://www.linkedin.com/in/hamza-benali-51013819a/).*
