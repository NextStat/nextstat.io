#include "Compression.h"
#include "TFile.h"
#include "TTree.h"

// Re-write the existing `simple_tree.root` fixture using ROOT's ZSTD compression
// (algorithm tag "ZS"), so ns-root/ns-zstd can be validated against real ROOT ZSTD blocks.
//
// Usage:
//   root -l -b -q tests/fixtures/generate_simple_tree_zstd.C
void generate_simple_tree_zstd() {
  // Input fixture is committed to the repo (generated via uproot).
  TFile fin("tests/fixtures/simple_tree.root", "READ");
  if (fin.IsZombie()) {
    Error("generate_simple_tree_zstd", "failed to open input fixture simple_tree.root");
    return;
  }

  TTree* in_tree = nullptr;
  fin.GetObject("events", in_tree);
  if (!in_tree) {
    Error("generate_simple_tree_zstd", "missing TTree 'events' in input fixture");
    return;
  }

  // ROOT compression settings are encoded as algo*100 + level.
  // ZSTD is algorithm id 5 in ROOT::RCompressionSetting::EAlgorithm (kZSTD).
  const int zstd_settings =
      ROOT::CompressionSettings(ROOT::RCompressionSetting::EAlgorithm::kZSTD, 6);

  TFile fout("tests/fixtures/simple_tree_zstd.root", "RECREATE");
  fout.SetCompressionSettings(zstd_settings);
  fout.cd();

  // ROOT's CloneTree copies only *active* branches; be explicit.
  in_tree->SetBranchStatus("*", 1);

  // Clone structure and copy entries. This will (re)write baskets using the output
  // file's compression settings.
  TTree* out_tree = in_tree->CloneTree(0);
  if (!out_tree) {
    Error("generate_simple_tree_zstd", "CloneTree() failed");
    return;
  }
  out_tree->CopyEntries(in_tree);
  out_tree->Write();
  fout.Close();
  fin.Close();
}
