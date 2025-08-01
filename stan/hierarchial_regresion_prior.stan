data {
  int<lower=0> N;          // Number of samples to generate
  int<lower=0> K;          // Number of features
  matrix[N, K] X;          // Feature matrix
  int<lower=0> num_dow_indices; // Number of day_of_week features
  array[num_dow_indices] int<lower=1, upper=K> dow_indices;  // Indices of day_of_week features
}
parameters {
  real alpha;              // Intercept
  vector[K] beta;          // Feature coefficients
  real mu_dow;             // Mean of day_of_week coefficients
  real<lower=0> sigma_dow; // SD of day_of_week coefficients
  real<lower=0> sigma;     // Noise standard deviation
}
model {
  alpha ~ normal(0, 2.0);
  for (k in 1:K) {
    int is_dow = 0;
    for (i in 1:size(dow_indices)) {
      if (k == dow_indices[i]) {
        is_dow = 1;
      }
    }

    if (is_dow == 1) {
      beta[k] ~ normal(mu_dow, sigma_dow);
    } else {
      beta[k] ~ normal(0, 0.2);
    }
  }
  mu_dow ~ normal(0, 0.5);
  sigma_dow ~ normal(0, 0.5);
  sigma ~ normal(0, 0.5);
}
generated quantities {
  vector[N] y_pred;
  for (n in 1:N) {
    y_pred[n] = normal_rng(X[n] * beta + alpha, sigma);
  }
}