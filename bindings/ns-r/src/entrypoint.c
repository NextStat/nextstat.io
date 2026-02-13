#include <R.h>
#include <R_ext/Rdynload.h>
#include <Rinternals.h>

// The Rust static library (libnsr.a) provides:
// - `R_init_nextstat` (registration alias)
// - extendr-generated `wrap__*` entry points.
//
// When linking a static library into a shared library, the linker may drop all
// objects from the archive if no symbols are referenced. We create a dummy
// references to exported symbols to ensure the Rust objects get pulled in
// without requiring platform-specific whole-archive linker flags.
extern void R_init_nextstat(DllInfo *dll);
void (*nextstat_force_link_r_init_nextstat)(DllInfo *) __attribute__((used)) = R_init_nextstat;

// extendr wrappers (exported via `.Call("wrap__...")`)
extern SEXP wrap__ns_normal_logpdf(SEXP, SEXP, SEXP);
extern SEXP wrap__ns_ols_fit(SEXP, SEXP, SEXP);
extern SEXP wrap__nextstat_fit(SEXP);
extern SEXP wrap__nextstat_hypotest(SEXP, SEXP);
extern SEXP wrap__nextstat_upper_limit(SEXP, SEXP, SEXP, SEXP);
extern SEXP wrap__nextstat_glm_logistic(SEXP, SEXP, SEXP);
extern SEXP wrap__nextstat_glm_poisson(SEXP, SEXP, SEXP);
extern SEXP wrap__nextstat_glm_negbin(SEXP, SEXP, SEXP);
extern SEXP wrap__nextstat_kalman(SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP wrap__nextstat_garch(SEXP);
extern SEXP wrap__nextstat_sv(SEXP, SEXP);

void *nextstat_force_link_wrap_syms[] __attribute__((used)) = {
    (void *)&wrap__ns_normal_logpdf,
    (void *)&wrap__ns_ols_fit,
    (void *)&wrap__nextstat_fit,
    (void *)&wrap__nextstat_hypotest,
    (void *)&wrap__nextstat_upper_limit,
    (void *)&wrap__nextstat_glm_logistic,
    (void *)&wrap__nextstat_glm_poisson,
    (void *)&wrap__nextstat_glm_negbin,
    (void *)&wrap__nextstat_kalman,
    (void *)&wrap__nextstat_garch,
    (void *)&wrap__nextstat_sv,
};
