data {
  int<lower=0> N;          // Liczba próbek treningowych
  int<lower=0> K;          // Liczba cech
  matrix[N, K] X;          // Macierz cech
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
}
generated quantities {
  vector[N] y;                    

  for (n in 1:N) {
    y[n] = normal_rng(dot_product(X[n], beta) + alpha, sigma);
  }
}