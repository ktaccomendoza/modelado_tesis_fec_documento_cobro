<h1 align="center" id="title">Script de modelado - Tesis de clasificación de documentos de cobro válidos para el factor de efectividad de cobranza de los asesores de seguro en una compañía aseguradora</h1>

<p id="description">Muestra la rama principal del script de modelamiento para la clasificación de los documentos de cobro validos para los asesore de seguro de una compañia de seguros.</p>

  
  
<h2>🧐 Features</h2>

Modelamiento realizado con iteración de múltiples parámetros de los algoritmos en mención: RL, Arbol CART, Naive Bayes, XGBoost. Se realizó validación de nulos, correlación, evaluación del p-value y ingeniería de características. Principales métricas evaluadas para los modelos: 

* Sensibilidad
* Especifidad
* GINI
* AUC
* Log Loss 
* KS
* F1 - Score
* Precisión
* Tiempo de entrenamiento y calculo de métricas (por algoritmo y expresado en minutos)
  
<h2>💻 Built with</h2>

Tecnologías usadas en el proyecto:

*   Regresión Logística
*   Arbol CART
*   Naive Bayes
*   XGBoost
*   R Studio

<h2>🛡️ Resultados finales</h2>

El algoritmo XGBoost, Regresión logística y Arbol CART son los que tuvieron mejores resultados en cuestión de sensibilidad siendo XGBoost el mejor con 97.61%. A nivel de AUC de igual forma están bastante proximos entre sí siendo el de Arbol CART el mejor con 96.12%.