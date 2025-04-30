

> \( \vec{x} = [x_0, x_1, x_2, \dots, x_n] \) are the **input features** (like age, salary, etc.)

So what are \( \vec{w} = [w_0, w_1, w_2, \dots, w_n] \)?

![Equation](https://latex.codecogs.com/png.latex?\vec{w}\cdot\vec{x})
---

### ✅ **\( \vec{w} \) are the model's learned parameters — the weights**

Here’s the breakdown:

| Term        | Meaning                             |
|-------------|--------------------------------------|
| \( x_i \)   | The **i-th feature** (input value)   |
| \( w_i \)   | The **weight** for feature \( x_i \) |
| \( w_0 \)   | The **bias term** (or intercept)     |

Together, the model computes:

\[
h(x) = w_0 x_0 + w_1 x_1 + \dots + w_n x_n = \vec{w} \cdot \vec{x}
\]

Where \( x_0 = 1 \), so \( w_0 \) acts as the bias.

---

### 🧠 Intuition:

- Each **feature** \( x_i \) represents some measurable input (like age, education, etc.)
- Each **weight** \( w_i \) tells the model **how important** that feature is.
- **Positive \( w_i \)**: increases the prediction
- **Negative \( w_i \)**: decreases the prediction
- The model **learns** these \( w_i \)'s during training by minimizing some loss function

---

### 🧮 Example:

Let’s say:

- \( \vec{x} = [1, 25, 50000] \) → [bias term, age, salary]
- \( \vec{w} = [-3, 0.4, 0.0001] \)

Then:

\[
h(x) = -3(1) + 0.4(25) + 0.0001(50000) = -3 + 10 + 5 = 12
\]

So the model prediction is **12** (before applying an activation function like sigmoid or softmax, if needed).

---

