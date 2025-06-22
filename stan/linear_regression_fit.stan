data {
  int<lower=0> N;
  int<lower=0> K;
  matrix[N, K] X;
  vector[N] y;
}
parameters {
  vector[K] beta;          // Regression coefficients
  real alpha;              // Intercept
  real<lower=0> sigma;     // Noise standard deviation
}
model {
  // Priors
  alpha ~ normal(0, 0.5);
  beta ~ normal(0, 0.2);
  sigma ~ student_t(4,0,1);

  // Likelihood
  y ~ normal(X * beta + alpha, sigma);
}
generated quantities {
  vector[N] y_pred;                    
  vector[N] log_lik;
  
  for (n in 1:N) {
    y_pred[n] = normal_rng(dot_product(X[n], beta) + alpha, sigma);
    log_lik[n] = normal_lpdf(y[n] | dot_product(X[n], beta) + alpha, sigma);
  }
}