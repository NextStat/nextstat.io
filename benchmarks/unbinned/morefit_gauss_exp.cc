// MoreFit single-fit benchmark runner for gauss_exp case.
// Reads data from a text file (one value per line), fits Gaussian+Exponential
// (fraction-based, not extended), and writes JSON result with timing.
//
// Build: add to CMakeLists.txt and `make morefit_gauss_exp`
//
// Usage:
//   ./morefit_gauss_exp <data.txt> <lo> <hi> [repeats]

#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <numeric>

#include <CL/cl.h>
#include "morefit.hh"

int main(int argc, char* argv[])
{
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <data.txt> <lo> <hi> [repeats]" << std::endl;
        return 1;
    }

    std::string data_path = argv[1];
    double lo = std::atof(argv[2]);
    double hi = std::atof(argv[3]);
    int repeats = argc > 4 ? std::atoi(argv[4]) : 3;

    std::vector<double> raw_data;
    {
        std::ifstream infile(data_path);
        if (!infile.is_open()) {
            std::cerr << "ERROR: cannot open " << data_path << std::endl;
            return 1;
        }
        double val;
        while (infile >> val) {
            raw_data.push_back(val);
        }
    }
    if (raw_data.empty()) {
        std::cerr << "ERROR: no data read from " << data_path << std::endl;
        return 1;
    }
    unsigned int nevents = raw_data.size();

    typedef double kernelT;
    typedef double evalT;

    morefit::compute_options compute_opts;
    compute_opts.llvm_nthreads = 1;
    compute_opts.llvm_vectorization = true;
    compute_opts.llvm_vectorization_width = 4;
    compute_opts.print_level = 0;

    typedef morefit::LLVMBackend backendT;
    typedef morefit::LLVMBlock<kernelT, evalT> blockT;
    morefit::LLVMBackend backend(&compute_opts);

    morefit::dimension<evalT> m("m", "mass", lo, hi, false);
    morefit::parameter<evalT> mu_sig("mu_sig", "mu_sig", 91.0, 85.0, 95.0, 0.01, false);
    morefit::parameter<evalT> sigma_sig("sigma_sig", "sigma_sig", 2.5, 0.5, 10.0, 0.01, false);
    morefit::parameter<evalT> fsig("fsig", "fsig", 0.25, 0.0, 1.0, 0.01, false);
    morefit::parameter<evalT> alpha("alpha", "lambda_bkg", -0.03, -0.1, -0.001, 0.001, false);

    morefit::GaussianPDF<kernelT, evalT> gaus(&m, &mu_sig, &sigma_sig);
    morefit::ExponentialPDF<kernelT, evalT> expo(&m, &alpha);
    morefit::SumPDF<kernelT, evalT> sum(&gaus, &expo, &fsig);
    std::vector<morefit::parameter<evalT>*> params({&mu_sig, &sigma_sig, &fsig, &alpha});

    morefit::fitter_options fit_opts;
    fit_opts.minuit_printlevel = 0;
    fit_opts.minimizer = morefit::fitter_options::minimizer_type::Minuit2;
    fit_opts.optimize_dimensions = false;
    fit_opts.optimize_parameters = true;
    fit_opts.analytic_gradient = false;
    fit_opts.analytic_hessian = false;
    fit_opts.kahan_on_accelerator = true;
    fit_opts.print_level = 0;
    fit_opts.analytic_fisher = false;
    fit_opts.postrun_hesse = true;

    std::vector<morefit::dimension<evalT>*> dims = {&m};

    morefit::EventVector<kernelT, evalT> ev(dims, nevents, false, 0);
    for (unsigned int i = 0; i < nevents; i++) {
        ev(i, (unsigned int)0) = static_cast<kernelT>(raw_data[i]);
    }

    morefit::fitter<kernelT, evalT, backendT, blockT> fit(&fit_opts, &backend);

    std::vector<double> fit_times;
    for (int r = 0; r < repeats; r++) {
        mu_sig.set_value(91.0);
        sigma_sig.set_value(2.5);
        fsig.set_value(0.25);
        alpha.set_value(-0.03);

        auto t0 = std::chrono::high_resolution_clock::now();
        bool ok = fit.fit(&sum, params, &ev, r > 0);
        auto t1 = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
        fit_times.push_back(dt);
    }

    double mean_ms = std::accumulate(fit_times.begin(), fit_times.end(), 0.0) / fit_times.size();
    double min_ms = *std::min_element(fit_times.begin(), fit_times.end());

    std::cout << std::setprecision(17);
    std::cout << "{\n";
    std::cout << "  \"tool\": \"morefit\",\n";
    std::cout << "  \"backend\": \"llvm_cpu_1t\",\n";
    std::cout << "  \"n_events\": " << nevents << ",\n";
    std::cout << "  \"repeats\": " << repeats << ",\n";
    std::cout << "  \"mean_ms\": " << mean_ms << ",\n";
    std::cout << "  \"min_ms\": " << min_ms << ",\n";
    std::cout << "  \"times_ms\": [";
    for (unsigned int i = 0; i < fit_times.size(); i++) {
        if (i > 0) std::cout << ", ";
        std::cout << fit_times[i];
    }
    std::cout << "],\n";
    std::cout << "  \"mu_sig\": " << mu_sig.get_value() << ",\n";
    std::cout << "  \"sigma_sig\": " << sigma_sig.get_value() << ",\n";
    std::cout << "  \"fsig\": " << fsig.get_value() << ",\n";
    std::cout << "  \"lambda_bkg\": " << alpha.get_value() << "\n";
    std::cout << "}\n";

    return 0;
}
