
---

### 🔢 What Are **Numerical Vectors** in This Context?

In **machine learning models** like **logistic regression** and **neural networks**, data needs to be **numerical** and usually in the form of **vectors** — which are just arrays of numbers.

For example, if you have three features:

| Age | Salary | Has Degree |
|-----|--------|------------|
| 25  | 50000  | Yes        |

After preprocessing:
```python
[25, 50000, 1]  # ← this is a numerical vector (list of numbers)
```

A **numerical vector** in this context just means:
- A list (or array) of real-valued numbers
- Ready to be used in mathematical operations like dot products, matrix multiplication, and gradient calculations

---

### ❌ Why Can’t We Directly Use **Categorical Data**?

Let’s say we have a feature like:

| Favorite Color |
|----------------|
| Red            |
| Green          |
| Blue           |

You **can’t feed this directly** into a model, because:

#### 1. **Math Doesn't Work on Strings**
Models like logistic regression and neural networks perform **mathematical operations** like:
- Matrix multiplications
- Gradients and derivatives
- Dot products

These operations don’t make sense for strings like `"Red"` or `"Green"`.

#### 2. **No Meaningful Order**
Even if you try to **assign numbers** like:
- Red = 1
- Green = 2
- Blue = 3

This implies an order or distance (e.g., that "Blue" is greater than "Red") which **doesn’t exist**. That could **confuse** the model.

---

### ✅ Solution: Convert Categorical Data into Numerical Vectors

We use **encoding techniques**:

#### - One-Hot Encoding
```text
"Red"    → [1, 0, 0]
"Green"  → [0, 1, 0]
"Blue"   → [0, 0, 1]
```

#### - Label Encoding (for ordinal data where order matters)
```text
"Low" = 0
"Medium" = 1
"High" = 2
```

#### - Embeddings (especially in deep learning)
Used for high-cardinality categories like words or product IDs.

---

### 🚀 Summary

- **Numerical vectors** = arrays of numbers your model can do math on
- **Categorical data** can't be directly used because models don’t understand strings or unordered categories
- We must **convert** categories into numerical formats using encoding techniques

