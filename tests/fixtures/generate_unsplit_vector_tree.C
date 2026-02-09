#include <vector>

#include "TFile.h"
#include "TTree.h"

// Generate a small ROOT TTree fixture with an unsplit std::vector<float> branch.
//
// Usage:
//   root -l -b -q tests/fixtures/generate_unsplit_vector_tree.C
void generate_unsplit_vector_tree() {
  TFile f("tests/fixtures/unsplit_vector_tree.root", "RECREATE");
  TTree t("events", "events");

  std::vector<float> jet_pt;
  // Force unsplit streamer branch (TBranchElement) for std::vector<T>.
  auto* b = t.Branch("jet_pt", &jet_pt, 32000, 0);
  // Disable entry-offset table so the payload is length-prefixed per entry.
  // This matches the "unsplit std::vector<T>" layout exercised by ns-root's
  // best-effort decoder when fEntryOffsetLen == 0.
  b->SetEntryOffsetLen(0, true);

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

  for (const auto& r : rows) {
    jet_pt = r;
    t.Fill();
  }

  t.Write();
  f.Close();
}
