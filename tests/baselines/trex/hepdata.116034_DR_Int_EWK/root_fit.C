#include <TFile.h>
#include <TKey.h>
#include <TSystem.h>
#include <memory>
#include <stdexcept>
#include <RooWorkspace.h>
#include <RooAbsPdf.h>
#include <RooAbsData.h>
#include <RooAbsReal.h>
#include <RooFit.h>
#include <RooArgSet.h>
#include <RooRealVar.h>
#include <RooMinimizer.h>
#include <RooFitResult.h>
#include <RooStats/ModelConfig.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <list>
#include <string>
#include <vector>

static RooWorkspace* find_workspace(TFile& f) {
  RooWorkspace* w = nullptr;
  TIter next(f.GetListOfKeys());
  while (auto* k = (TKey*)next()) {
    auto cname = std::string(k->GetClassName());
    if (cname.find("RooWorkspace") == std::string::npos) continue;
    w = (RooWorkspace*)k->ReadObj();
    if (w) return w;
  }
  return nullptr;
}

static RooStats::ModelConfig* find_model_config(RooWorkspace& w) {
  if (auto* o = w.obj("ModelConfig")) return dynamic_cast<RooStats::ModelConfig*>(o);
  if (auto* o = w.obj("modelConfig")) return dynamic_cast<RooStats::ModelConfig*>(o);
  return nullptr;
}

static RooAbsData* find_data(RooWorkspace& w) {
  if (auto* d = w.data("obsData")) return d;
  if (auto* d = w.data("data")) return d;
  return nullptr;
}

static int minimize_nll(RooMinimizer& m) {
  int status = m.minimize("Minuit2", "Migrad");
  if (status != 0) {
    m.setStrategy(2);
    status = m.minimize("Minuit2", "Migrad");
  }
  if (status != 0) {
    (void)m.minimize("Minuit2", "Simplex");
    status = m.minimize("Minuit2", "Migrad");
  }
  if (status == 0) {
    int h = m.hesse();
    if (h != 0) status = h;
  }
  return status;
}

static std::string escape_json(const std::string& s) {
  std::string out;
  out.reserve(s.size()+8);
  for (char c : s) {
    if (c == '\\') out += "\\\\";
    else if (c == '"') out += "\\\"";
    else if (c == '\n') out += "\\n";
    else out += c;
  }
  return out;
}

void root_fit() {
  gSystem->Load("libRooFit");
  gSystem->Load("libRooStats");

  const char* root_path = "tests/baselines/trex/hepdata.116034_DR_Int_EWK/histfactory_stage/channels/results_combined_NormalMeasurement_model.root";
  const char* out_path = "tests/baselines/trex/hepdata.116034_DR_Int_EWK/root_fit.json";

  TFile f(root_path, "READ");
  if (f.IsZombie()) {
    throw std::runtime_error(std::string("Failed to open ROOT file: ") + root_path);
  }

  RooWorkspace* w = find_workspace(f);
  if (!w) {
    throw std::runtime_error("No RooWorkspace found in file.");
  }
  auto* mc = find_model_config(*w);
  if (!mc) {
    throw std::runtime_error("ModelConfig not found.");
  }
  RooAbsPdf* pdf = mc->GetPdf();
  if (!pdf) {
    throw std::runtime_error("ModelConfig has no PDF.");
  }
  RooAbsData* data = find_data(*w);
  if (!data) {
    throw std::runtime_error("Data not found in workspace.");
  }

  // Some exports leave POI ranges effectively unbounded, which can cause Minuit2 to
  // run away to absurd values and return status=-1. Tighten obviously-insane ranges.
  if (auto* poi_set = mc->GetParametersOfInterest()) {
    if (poi_set->getSize() > 0) {
      auto* poi = dynamic_cast<RooRealVar*>(poi_set->first());
      if (poi) {
        double lo = poi->getMin();
        double hi = poi->getMax();
        bool insane = !std::isfinite(lo) || !std::isfinite(hi) || (hi - lo) > 1e6 || std::fabs(lo) > 1e6 || std::fabs(hi) > 1e6;
        if (insane) {
          poi->setRange(-100.0, 100.0);
        }
        if (poi->getVal() < poi->getMin() || poi->getVal() > poi->getMax()) {
          poi->setVal(0.0);
        }
      }
    }
  }

  RooArgSet empty;
  const RooArgSet* nuis = mc->GetNuisanceParameters();
  const RooArgSet* globs = mc->GetGlobalObservables();
  if (!nuis) nuis = &empty;
  if (!globs) globs = &empty;

  std::unique_ptr<RooAbsReal> nll(pdf->createNLL(
      *data,
      RooFit::Extended(true),
      RooFit::Constrain(*nuis),
      RooFit::GlobalObservables(*globs)
  ));

  RooMinimizer m(*nll);
  m.setPrintLevel(-1);
  m.setStrategy(1);
  m.setEps(1e-12);
  m.setMaxFunctionCalls(200000);
  m.setMaxIterations(200000);
  m.setOffsetting(true);
  m.optimizeConst(2);
  int status = minimize_nll(m);
  double nll_hat = nll->getVal();

  RooFitResult* fr = nullptr;
  try { fr = m.save(); } catch (...) { fr = nullptr; }

  struct P { std::string name; double val; double err; };
  std::vector<P> ps;
  if (fr) {
    auto vars = fr->floatParsFinal();
    for (int i = 0; i < vars.getSize(); i++) {
      auto* v = dynamic_cast<RooRealVar*>(vars.at(i));
      if (!v) continue;
      ps.push_back(P{v->GetName(), v->getVal(), v->getError()});
    }
  }
  std::sort(ps.begin(), ps.end(), [](const P& a, const P& b) { return a.name < b.name; });

  std::ofstream out(out_path);
  out << std::setprecision(16);
  out << "{\n";
  out << "  \"status\": " << status << ",\n";
  out << "  \"nll_hat\": " << nll_hat << ",\n";
  out << "  \"twice_nll\": " << (2.0*nll_hat) << ",\n";
  out << "  \"parameters\": [\n";
  for (size_t i = 0; i < ps.size(); i++) {
    out << "    {\"name\": \"" << escape_json(ps[i].name) << "\", \"value\": " << ps[i].val << ", \"uncertainty\": " << ps[i].err << "}";
    if (i+1 < ps.size()) out << ",";
    out << "\n";
  }
  out << "  ]\n";
  out << "}\n";
  out.close();
}
