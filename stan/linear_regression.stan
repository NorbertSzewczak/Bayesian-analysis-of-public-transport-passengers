data {
  int<lower=0> N;          // Number of samples to generate
  int<lower=0> K;          // Number of features
  matrix[N, K] X;          // Feature matrix
}
parameters {
  vector[K] beta;          // Regression coefficients
  real beta0;              // Intercept
  real<lower=0> sigma;     // Noise standard deviation
}
model {
  // Priors
  beta0 ~ normal(80, 30);
  beta ~ normal(0, 10);
  sigma ~ student_t(3, 0, 5);
}
generated quantities {
  vector[N] y;             // Generated target variable
  array[K] vector[N] y_per_feature;  // Per-feature prior predictive
  for (n in 1:N) {
    y[n] = normal_rng(X[n] * beta + beta0, sigma);
    for (k in 1:K) {
      y_per_feature[k, n] = normal_rng(X[n, k] * beta[k] + beta0, sigma);
    }
  }
}   