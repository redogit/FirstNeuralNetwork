Sigmoid activation function:

df / dt = -k₁ *(1 / (1 + exp(-f(t)))) + k₂ *(1 / (1 + exp(-(1 - f(t)))))

ReLU(Rectified Linear Unit) activation function:

df / dt = -k₁ *max(0, f(t)) + k₂ *max(0, 1 - f(t))

Tanh(hyperbolic tangent) activation function:

df / dt = -k₁ *tanh(f(t)) + k₂ *tanh(1 - f(t))

Cosine activation function:

df / dt = -k₁ *cos(f(t)) + k₂ *cos(1 - f(t))

Sine activation function:

df / dt = -k₁ *sin(f(t)) + k₂ *sin(1 - f(t))

Square root activation function:

df / dt = -k₁ *sqrt(abs(f(t))) + k₂ *sqrt(abs(1 - f(t)))

