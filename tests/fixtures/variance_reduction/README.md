Approved reference fixtures for shared CUPED/CURE validation.

These cases are deterministic, committed, and intended to exercise the stable
variance-reduction surface across:

- binary CUPED
- revenue CURE
- ratio-style CURE
- low-conversion CURE
- multi-channel CURE
- ridge-fallback CURE under collinearity

Each fixture stores aligned inputs plus reference outputs from the current
shared product surface. Rust integration tests and Python public-surface tests
must both remain green against these fixtures.
