data {
  int<lower=0> N;          // Number of training samples
  int<lower=0> K;          // Number of features
  matrix[N, K] X;          // Feature matrix
  vector[N] y;             // Target variable
  int<lower=0> num_dow_indices; // Number of day_of_week features (Added for loop)
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
  vector[N] mu;
  // Priors
  alpha ~ normal(0, 2.0);
  for (k in 1:K) {
    // Check if the current index k is one of the dow_indices
    int is_dow = 0; // Flag to indicate if k is a dow index
    for (i in 1:num_dow_indices) { // Iterate through dow_indices
      if (k == dow_indices[i]) {
        is_dow = 1; // Set flag if match found
        break; // Exit the inner loop once a match is found
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

  // Likelihood
  for (n in 1:N) {
    mu[n] = alpha + dot_product(X[n], beta);
  }
  y ~ normal(mu, sigma);
}
generated quantities {
  vector[N] y_pred;        // Posterior predictive checks
  vector[N] log_lik;       // Log likelihood for model evaluation
  for (n in 1:N) {
    y_pred[n] = normal_rng(X[n] * beta + alpha, sigma);
    log_lik[n] = normal_lpdf(y[n] | X[n] * beta + alpha, sigma);
  }
}