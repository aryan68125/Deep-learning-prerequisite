
---

## 📌 Why introduce a dummy variable \( x_0 = 1 \)?

We want to write the hypothesis function like this:

\[
h(x) = w0 + w1x1 + w2x2 + ... + wnxn
\]

But this is annoying to write because:
- \( w0 \) (the **bias** term) doesn’t multiply any feature — it just gets added in.

So instead, we define:

> **\( x0 = 1 \)** → a constant dummy feature

Now, we can **combine everything into a dot product**:

\[
h(x) = w0x0 + w1x1 + w2x2 + ... + wnxn
\]

---

### ✅ Vector form:

![Vector representation](util_pictures_for_explaination_README/explaiantion/vector.png)

This makes the math:
- Shorter
- Cleaner for gradients and matrix operations
- Easier for generalization (works in any dimension)

---

### 🧠 Example:

If your original model is:

![Same thing but easier](util_pictures_for_explaination_README/explaiantion/same_thing_but_easier.png)

Same thing — but easier to work with computationally.

---

### 📈 Bonus:

- This trick is heavily used in **linear regression**, **logistic regression**, and **neural networks**.
- Frameworks like TensorFlow and PyTorch usually **include the bias as a separate parameter**, but mathematically it's equivalent to this dummy-feature trick.

---

