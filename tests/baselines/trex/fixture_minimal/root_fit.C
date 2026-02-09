
#include <TFile.h>
#include <TKey.h>
#include <TSystem.h>

#include <memory>
#include <stdexcept>

#include <RooWorkspace.h>
#include <RooAbsPdf.h>
#include <RooAbsData.h>
#include <RooAbsReal.h>
#include <RooArgSet.h>
#include <RooRealVar.h>
#include <RooMinimizer.h>
#include <RooFitResult.h>

#include <RooStats/ModelConfig.h>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

static RooWorkspace* find_workspace(TFile& f) {{
  RooWorkspace* w = nullptr;
  TIter next(f.GetListOfKeys());
  while (auto* k = (TKey*)next()) {{
    auto cname = std::string(k->GetClassName());
    if (cname.find("RooWorkspace") == std::string::npos) continue;
    w = (RooWorkspace*)k->ReadObj();
    if (w) return w;
  }}
  return nullptr;
}}

static RooStats::ModelConfig* find_model_config(RooWorkspace& w) {{
  if (auto* o = w.obj("ModelConfig")) return dynamic_cast<RooStats::ModelConfig*>(o);
  if (auto* o = w.obj("modelConfig")) return dynamic_cast<RooStats::ModelConfig*>(o);
  return nullptr;
}}

static RooAbsData* find_data(RooWorkspace& w) {{
  if (auto* d = w.data("obsData")) return d;
  if (auto* d = w.data("data")) return d;
  auto all = w.allData();
  if (all.getSize() == 0) return nullptr;
  auto* obj = all.first();
  if (!obj) return nullptr;
  return w.data(obj->GetName());
}}

static int minimize_nll(RooAbsReal& nll) {{
  RooMinimizer m(nll);
  m.setPrintLevel(-1);
  m.setStrategy(0);
  m.optimizeConst(2);
  int status = m.minimize("Minuit2", "Migrad");
  return status;
}}

static std::string escape_json(const std::string& s) {{
  std::string out;
  out.reserve(s.size()+8);
  for (char c : s) {{
    if (c == '\\\\') out += "\\\\\\\\";
    else if (c == '\"') out += "\\\\\"";
    else if (c == '\\n') out += "\\\\n";
    else out += c;
  }}
  return out;
}}

void fit() {{
  gSystem->Load("libRooFit");
  gSystem->Load("libRooStats");

  const char* root_path = "tests/fixtures/histfactory/results_combined_NominalMeasurement_model.root";
  const char* out_path = "tests/baselines/trex/fixture_minimal/root_fit.json";

  TFile f(root_path, "READ");
  if (f.IsZombie()) {{
    throw std::runtime_error(std::string("Failed to open ROOT file: ") + root_path);
  }}

  RooWorkspace* w = find_workspace(f);
  if (!w) {{
    throw std::runtime_error("No RooWorkspace found in file.");
  }}
  auto* mc = find_model_config(*w);
  if (!mc) {{
    throw std::runtime_error("ModelConfig not found (expected 'ModelConfig').");
  }}
  RooAbsPdf* pdf = mc->GetPdf();
  if (!pdf) {{
    throw std::runtime_error("ModelConfig has no PDF.");
  }}
  RooAbsData* data = find_data(*w);
  if (!data) {{
    throw std::runtime_error("Data not found in workspace.");
  }}

  std::unique_ptr<RooAbsReal> nll(pdf->createNLL(*data));
  int status = minimize_nll(*nll);
  double nll_hat = nll->getVal();

  RooFitResult* fr = nullptr;
  try {{
    RooMinimizer m(*nll);
    m.setPrintLevel(-1);
    m.setStrategy(0);
    m.optimizeConst(2);
    m.minimize("Minuit2", "Migrad");
    fr = m.save();
  }} catch (...) {{
    fr = nullptr;
  }}

  struct P {{ std::string name; double val; double err; }};
  std::vector<P> ps;
  if (fr) {{
    auto vars = fr->floatParsFinal();
    for (int i = 0; i < vars.getSize(); i++) {{
      auto* v = dynamic_cast<RooRealVar*>(vars.at(i));
      if (!v) continue;
      ps.push_back(P{{v->GetName(), v->getVal(), v->getError()}});
    }}
  }}
  std::sort(ps.begin(), ps.end(), [](const P& a, const P& b) {{ return a.name < b.name; }});

  std::ofstream out(out_path);
  out << std::setprecision(16);
  out << "{\\n";
  out << "  \\\"status\\\": " << status << ",\\n";
  out << "  \\\"nll_hat\\\": " << nll_hat << ",\\n";
  out << "  \\\"twice_nll\\\": " << (2.0*nll_hat) << ",\\n";
  out << "  \\\"parameters\\\": [\\n";
  for (size_t i = 0; i < ps.size(); i++) {{
    out << "    {\\\"name\\\": \\\"" << escape_json(ps[i].name) << "\\\", \\\"value\\\": " << ps[i].val << ", \\\"uncertainty\\\": " << ps[i].err << "}";
    if (i+1 < ps.size()) out << ",";
    out << "\\n";
  }}
  out << "  ]\\n";
  out << "}\\n";
  out.close();
}}
