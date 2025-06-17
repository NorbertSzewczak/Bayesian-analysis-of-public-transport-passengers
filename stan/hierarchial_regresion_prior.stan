data {
  int<lower=0> N;          // Number of samples to generate
  int<lower=0> K;          // Number of features
  matrix[N, K] X;          // Feature matrix
  int<lower=0> num_dow_indices; // Number of day_of_week features
  array[num_dow_indices] int<lower=1, upper=K> dow_indices;  // Indices of day_of_week features
}
parameters {
  real beta0;              // Intercept
  vector[K] beta;          // Feature coefficients
  real mu_dow;             // Mean of day_of_week coefficients
  real<lower=0> sigma_dow; // SD of day_of_week coefficients
  real<lower=0> sigma;     // Noise standard deviation
}
model {
  // Priors
  beta0 ~ normal(0, 0.9);
  for (k in 1:K) {
    // Check if the current index k is one of the dow_indices
    int is_dow = 0; // Flag to indicate if k is a dow index
    for (i in 1:size(dow_indices)) { // Iterate through dow_indices
      if (k == dow_indices[i]) {
        is_dow = 1; // Set flag if match found
      }
    }

    if (is_dow == 1) { // Use the flag in the conditional
      beta[k] ~ normal(mu_dow, sigma_dow);
    } else {
      beta[k] ~ normal(0, 1);
    }
  }
  mu_dow ~ normal(0, 0.3);
  sigma_dow ~ normal(0, 0.3);
  sigma ~ normal(0, 0.3);
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