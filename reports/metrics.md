# Métricas do classificador de landmarks

- 27855 amostras após remover 3854 linhas duplicadas; 21 letras: A, B, C, D, E, F, G, I, L, M, N, O, P, Q, R, S, T, U, V, W, Y
- Sem amostras suficientes (fora do estudo): 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, H, J, K, X, Z
- Validação cruzada em 5 partes por grupos de pose (10 grupos k-means por letra): cada parte de teste só tem poses que o modelo não viu

## Comparação dos modelos (média ± desvio nas partes)

|               | Acurácia      | F1 macro      |   Treino (s) |
|:--------------|:--------------|:--------------|-------------:|
| Random Forest | 0.871 ± 0.059 | 0.870 ± 0.046 |         10.7 |
| KNN           | 0.849 ± 0.084 | 0.851 ± 0.079 |          0.4 |
| SVM (RBF)     | 0.884 ± 0.072 | 0.875 ± 0.073 |          3.2 |
| MLP           | 0.870 ± 0.076 | 0.860 ± 0.067 |          3   |

## Por letra — SVM (RBF)

```
              precision    recall  f1-score   support

           A      0.998     0.809     0.894      1583
           B      0.997     1.000     0.998      1662
           C      0.814     0.943     0.874      1546
           D      1.000     0.818     0.900      1651
           E      0.922     0.799     0.856      1500
           F      0.732     0.807     0.767      1499
           G      0.800     0.800     0.800      1499
           I      1.000     0.751     0.858      1202
           L      0.800     1.000     0.889      1202
           M      0.993     0.720     0.835      1202
           N      0.740     0.996     0.849      1251
           O      0.732     0.600     0.659      1004
           P      0.997     0.998     0.998      1193
           Q      0.998     0.916     0.955      1202
           R      0.997     0.998     0.998      1201
           S      0.837     1.000     0.911      1202
           T      0.778     0.693     0.733      1448
           U      0.993     0.997     0.995      1202
           V      1.000     0.995     0.997      1202
           W      1.000     1.000     1.000      1202
           Y      0.706     1.000     0.828      1202

    accuracy                          0.886     27855
   macro avg      0.897     0.888     0.885     27855
weighted avg      0.898     0.886     0.885     27855
```

## Referência: separação aleatória

Com uma separação aleatória comum (80/20), o mesmo modelo chega a 0.999 de acurácia. Esse número é otimista: muitas amostras de teste têm uma quase-cópia no treino.
