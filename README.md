# Lab Semana 2

## Actividad 1 — 5.1 Varianza retenida vs Número de componentes (k)

Se calcula, para explained_variance = [0.99, 0.75, 0.3], el número de
componentes principales k que retiene esa proporción de varianza, y se
grafica k vs varianza retenida.

Los resultados con el dataset de rostros (13233 imágenes de 64x64 = 4096
features) fueron: k=2 para retener el 30% de la varianza, k=22 para el 75%, y
k=577 para el 99%. Lo que más llama la atención es lo rápido que crece esa
curva al principio y lo mucho que se aplana después: con apenas 2 componentes
(de 4096 posibles) ya se agarra un 30% de toda la información, pero para subir
del 75% al 99% hay que meter otros 555 componentes más. Esto tiene sentido
porque todas las caras se parecen bastante entre sí en su estructura general
(dos ojos, nariz, boca, más o menos en el mismo lugar), entonces las primeras
componentes capturan ese "patrón promedio de cara" y con pocas ya se explica
mucho. Los detalles finos y particulares de cada persona (arrugas, textura,
iluminación) son los que obligan a sumar cientos de componentes más para subir
esos últimos puntos de porcentaje.

## Actividad 5.2 — Calidad vs. Componente (imagen propia)

Se tomó una selfie propia Img1.jpeg, se convirtió a escala de grises, se
redimensionó a 512x512 y se le aplicó PCA reconstruyendo con 70% y 99% de
varianza retenida.

Con el 70% de varianza bastaron solo 2 componentes (de 512 posibles), pero la
imagen reconstruida se ve bastante mal, como borrosa/pixelada, prácticamente
irreconocible en detalle aunque se nota la silueta general. Con el 99% de
varianza ya se necesitaron 46 componentes, y ahí la reconstrucción es casi
idéntica a la original a simple vista. La diferencia entre 2 y 46 componentes
(de 512) es enorme en términos de calidad visual pero mínima en cuánto "pesa"
la representación comprimida, lo cual deja bien claro por qué PCA es tan
usado para compresión: con menos del 10% de las componentes originales (46/512)
se recupera casi toda la calidad visual de la imagen.

## Actividad 2 — Uso de DataSet Genérico

Se aplicó PCA sobre el dataset Student Performance
(UCI id=320, 649 estudiantes, 30 variables entre notas de portugués/matemáticas,
datos demográficos y hábitos de estudio)

Se descarga el dataset con ucimlrepo, se convierten las variables
categóricas con one-hot encoding (quedan 39 columnas), se estandarizan y se
aplica PCA. Con 2 componentes (para poder graficar) solo se retiene el 15.3%
de la varianza total; para llegar al 80% se necesitan 24 componentes, y al 95%
se necesitan 33 de 39.

## ¿El PCA fue útil en este dataset?

La verdad, no mucho. Aquí PCA no ayudó tanto como en el caso de las imágenes o
del dataset de vinos. El problema es que este dataset tiene muy pocas variables
numéricas "fuertes" y un montón de variables categóricas (trabajo de los
padres, si tiene internet, si quiere ir a la universidad, etc.) que al pasarlas
por one-hot encoding se convierten en columnas binarias sueltas, cada una
aportando un poquito de varianza pero ninguna dominando. Por eso con solo 2
componentes apenas se explica un 15% de la información, y hay que usar casi
todas las columnas (24 de 39) para llegar a un 80%. Si el objetivo fuera
comprimir el dataset, PCA no lo comprime bien aquí; en cambio en las imágenes
sí funcionó excelente porque ahí sí había mucha redundancia entre pixeles
vecinos.

## ¿Se lograron patrones visibles (agrupaciones, separaciones)?

No, la verdad no se ve ninguna agrupación clara. El scatter de PC1 vs PC2 es
básicamente una sola nube de puntos concentrada en el centro, sin separaciones
ni clusters definidos. Mirando los loadings, PC1 está dominado por el nivel
educativo de los padres (Medu, Fedu), si quiere estudiar en la universidad
(higher_yes) y si ha reprobado antes (failures); PC2 está más relacionado
con el consumo de alcohol (Walc, Dalc) y salir con amigos (goout). O sea
que sí hay una interpretación razonable de qué representa cada eje (uno más
"académico/familiar", el otro más "social"), pero como no hay grupos separados
en el gráfico, no podemos decir que existan subgrupos claros de estudiantes;
más bien es un espectro continuo donde cada estudiante cae en algún punto
intermedio.

## Actividad 2.1 — Mejora del análisis anterior

Como en la Actividad 2 el PCA no dio resultados muy claros, se hizo una
segunda versión para intentar mejorarlo.

Los dos cambios principales:
1. Solo variables numéricas (age, Medu, Fedu, traveltime, studytime,
   failures, famrel, freetime, goout, Dalc, Walc, health,
   absences — 13 en total), sin el one-hot de las categóricas, para no
   diluir la varianza en un montón de columnas binarias.
2. Colorear el scatter por la nota final (G3) para poder buscar visualmente
   si el PCA separa a los estudiantes por rendimiento académico.

Comparativa numérica:

|                      | Actividad 2 (con one-hot)   | Actividad 2.1 (solo numéricas) |
|----------------------|-----------------------------|--------------------------------|
| Columnas usadas      |           39                |              13                |
| Varianza con PC1+PC2 |           15.3%             |              31.8%             |
| Componentes para 80% |           24                |              9                 |
| Componentes para 95% |           33                |              12                |

¿Mejoró el análisis?

Sí, bastante. Al quitar las categóricas one-hot y quedarnos solo con las
variables numéricas, con los mismos 2 componentes ahora se explica el doble de
varianza (31.8% vs 15.3%), y para retener el 80% ya no se necesitan 24
componentes sino solo 9. Tiene sentido: las columnas binarias de one-hot
(trabajo del papá, si tiene internet, etc.) tienen muy poca varianza cada una
y ninguna se parece a las demás, entonces el PCA no encuentra dirección
dominante; en cambio las variables numéricas sí covarían entre sí (por
ejemplo Dalc y Walc, o Medu y Fedu), que es justo lo que PCA necesita
para comprimir bien.

Ahora, sobre el gráfico coloreado por la nota final: honestamente tampoco se
ve una separación clara por color. Los puntos con nota muy baja (morado
oscuro, cerca de 0, que en este dataset suelen ser estudiantes que no se
presentaron a los exámenes) están un poco más dispersos hacia la derecha del
gráfico, pero se mezclan con el resto y no forman un grupo aparte. O sea que
el rendimiento académico (G3) no depende principalmente de estas variables de
hábitos/demografía combinadas linealmente en 2 componentes; probablemente
influyen más las notas parciales (G1, G2) que ni siquiera se usaron aquí como
feature, o simplemente la relación no es lineal. En resumen: se mejoró la
"calidad" del PCA como técnica de compresión (menos componentes, más varianza
explicada), pero seguimos sin encontrar clusters visibles de estudiantes.

