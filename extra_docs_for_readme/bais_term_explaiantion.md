In the equation:

\[
0 = ax + by + c
\]

the **bias term** is:

### 👉 \( c \)

---

### 💡 Why is \( c \) the bias?

In machine learning — particularly in models like logistic regression or linear classifiers — the **bias** is the constant added to the weighted sum of the inputs:

\[
z = w1x + w2y + b
\]

This is **exactly** the same as:

\[
0 = w1x + w2y + b
→ 0 = ax + by + c
\]

So, in this general line equation:
- \( a \) and \( b \) are the weights (or coefficients) for the input features
- \( c \) is the **bias**

---

### 🧠 Bias term ( \( c \) ) controls:
- The **position** of the line/hyperplane
- But **not its orientation (slope)**

Changing \( c \) shifts the line **up/down or left/right** without rotating it.

---

