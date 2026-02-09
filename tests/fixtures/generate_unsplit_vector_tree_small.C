#include <vector>

#include "TFile.h"
#include "TTree.h"

// Generate a tiny ROOT TTree fixture with an unsplit std::vector<float> branch.
//
// This fixture is intentionally small (3 entries) to exercise ROOT's
// `fEntryOffsetLen` values that are not a fixed 16/32/64 bit width. For streamed
// `TBranchElement` vectors, ROOT encodes `fEntryOffsetLen` as the *byte length*
// of the per-basket offset table (typically `n_entries_in_basket * 4`).
//
// Usage:
//   root -l -b -q tests/fixtures/generate_unsplit_vector_tree_small.C
void generate_unsplit_vector_tree_small() {
  TFile f("tests/fixtures/unsplit_vector_tree_small.root", "RECREATE");
  TTree t("events", "events");

  std::vector<float> jet_pt;
  // Unsplit streamer branch (TBranchElement) for std::vector<T>.
  t.Branch("jet_pt", &jet_pt, 32000, 0);

  std::vector<std::vector<float>> rows = {
      {1.0f, 2.0f},
      {},
      {3.0f},
  };

  for (const auto& r : rows) {
    jet_pt = r;
    t.Fill();
  }

  t.Write();
  f.Close();
}

