# Convex Optimization

Private repository for algorithms implemented for my master's degrees class. All algorithms implemented based on Boyd and Vandenberghe's Convex Optimization book and Practical Optimization Algorithms and Engineering Applications from Antonious and Lu.

```
CPE 773 - Convex Optimization
Professor: Wallace A. Martins
Student: Pedro Bandeira de Mello Martins
COPPE/UFRJ - Brazil
2018/3
```

All optimization algorithms are implemented on [models/optimizers](models/optimizers) folder. A [function](functions/functionObj.py) object should always be passed to keep track of all its iterations, evaluations and gradients.
For some examples, please check [tests folder](tests/)

Exercises made for this class (might be in brazilian portuguese):
1. [Lista 1](Exercises/Lista1/ExerciciosLista1.ipynb)
2. [Lista 2](Exercises/Lista2/Lista2.ipynb)
3. [Lista 3](Exercises/Lista3/Lista%203.ipynb)

------
## TODO LIST
1. Lista 4.
2. Final project of the subject.
3. Redo Lista 2.
4. Separate module line searchs from optimizers.
5. Fix Autograd ArrayBoxes bug.
6. Debug Fletcher's Inexact Line Search
7. Update all line searchs to expect the same inputs and return the same outputs.
8. Clean up code.
9. Update code to follow proper PEP8.