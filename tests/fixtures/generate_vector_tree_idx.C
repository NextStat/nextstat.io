#include <vector>

#include "TFile.h"
#include "TTree.h"

// Generate a small ROOT TTree fixture with:
// - an unsplit std::vector<float> branch `jet_pt` (jagged per entry),
// - scalar int branches `idx` and `njet`,
// used to validate dynamic indexing expressions like `jet_pt[idx]`.
//
// Usage:
//   root -l -b -q tests/fixtures/generate_vector_tree_idx.C
void generate_vector_tree_idx() {
  TFile f("tests/fixtures/vector_tree_idx.root", "RECREATE");
  TTree t("events", "events");

  std::vector<float> jet_pt;
  int idx = 0;
  int njet = 0;

  t.Branch("jet_pt", &jet_pt, 32000, 0);
  t.Branch("idx", &idx, "idx/I");
  t.Branch("njet", &njet, "njet/I");

  std::vector<std::vector<float>> rows = {
      {10.0f, 11.0f},
      {20.0f},
      {},
      {30.0f, 31.0f, 32.0f},
      {40.0f},
      {},
      {50.0f, 51.0f},
      {60.0f},
  };

  std::vector<int> idxs = {0, 0, 0, 1, 1, 1, 1, 0};

  for (size_t i = 0; i < rows.size(); ++i) {
    jet_pt = rows[i];
    idx = idxs[i];
    njet = static_cast<int>(rows[i].size());
    t.Fill();
  }

  t.Write();
  f.Close();
}

