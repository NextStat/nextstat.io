functions {
  real conc_1cpt_oral(real dose, real cl, real v, real ka, real t) {
    real v_safe = fmax(v, 1e-8);
    real ke = fmax(cl / v_safe, 1e-10);
    real ka_safe = fmax(ka, 1e-10);
    real denom = ka_safe - ke;
    if (abs(denom) < 1e-6) {
      denom = (denom >= 0) ? 1e-6 : -1e-6;
    }
    return fmax(
      dose * ka_safe / (v_safe * denom) * (exp(-ke * t) - exp(-ka_safe * t)),
      1e-12
    );
  }
}

data {
  int<lower=1> N;
  int<lower=1> S;
  array[N] int<lower=1, upper=S> sid;
  vector<lower=0>[N] time;
  vector<lower=0>[N] y;
  real<lower=0> dose;
  real<lower=0> sigma;
}

parameters {
  real<lower=log(0.01), upper=log(2.0)> tcl;
  real<lower=log(0.5), upper=log(50.0)> tv;
  real<lower=log(0.05), upper=log(10.0)> tka;
  vector<lower=1e-8>[3] omega;          // IIV std devs
  array[S] vector[3] z;
}

transformed parameters {
  array[S] vector[3] eta;
  for (s in 1:S) {
    eta[s, 1] = omega[1] * z[s, 1];
    eta[s, 2] = omega[2] * z[s, 2];
    eta[s, 3] = omega[3] * z[s, 3];
  }
}

model {
  tcl ~ normal(log(0.134), 0.4);
  tv ~ normal(log(8.0), 0.4);
  tka ~ normal(log(1.0), 0.4);
  omega[1] ~ normal(0.20, 0.05);
  omega[2] ~ normal(0.15, 0.05);
  omega[3] ~ normal(0.25, 0.05);
  for (s in 1:S) {
    z[s] ~ std_normal();
  }

  for (n in 1:N) {
    int s = sid[n];
    real cl = exp(tcl + eta[s, 1]);
    real v = exp(tv + eta[s, 2]);
    real ka = exp(tka + eta[s, 3]);
    real mu = conc_1cpt_oral(dose, cl, v, ka, time[n]);
    y[n] ~ normal(mu, sigma);
  }
}
