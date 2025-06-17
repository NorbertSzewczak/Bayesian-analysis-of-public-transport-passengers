data {
  int<lower=0> N;          // Liczba próbek treningowych
  int<lower=0> K;          // Liczba cech
  matrix[N, K] X;          // Macierz cech
  vector[N] y;             // Zmienna celu
}
parameters {
  vector[K] beta;          
  real beta0;              
  real<lower=0> sigma;     
}
model {
  // Priory
  beta0 ~ normal(0, 0.5);
  beta ~ normal(0, 0.2);
  sigma ~ student_t(4,0,1);

  // Likelihood
  y ~ normal(X * beta + beta0, sigma);
}
generated quantities {
  vector[N] y_pred;                    
  array[K] vector[N] y_per_feature;    

  for (n in 1:N) {
    y_pred[n] = normal_rng(dot_product(X[n], beta) + beta0, sigma);
    for (k in 1:K) {
      y_per_feature[k, n] = normal_rng(X[n, k] * beta[k] + beta0, sigma);
    }
  }
}