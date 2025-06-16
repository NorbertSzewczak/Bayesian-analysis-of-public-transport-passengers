data {
  int<lower=0> N;          // Number of training samples
  int<lower=0> K;          // Number of features
  matrix[N, K] X;          // Feature matrix
  vector[N] y;             // Target variable
  int<lower=0> N_new;      // Number of test samples
  matrix[N_new, K] X_new;  // Test feature matrix
}
parameters {
  vector[K] beta;          // Regression coefficients
  real beta0;              // Intercept
  real<lower=0> sigma;     // Noise standard deviation
}
model {
  // Priors
  beta0 ~ normal(50, 20);
  beta ~ normal(0, 5);
  sigma ~ student_t(3, 0, 10);

  // Likelihood
  y ~ normal(X * beta + beta0, sigma);
}
generated quantities {
  vector[N_new] y_pred;    // Predictions for test data
  for (n in 1:N_new) {
    y_pred[n] = normal_rng(X_new[n] * beta + beta0, sigma);
  }
}