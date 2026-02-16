#include <ROOT/RNTupleReader.hxx>
#include <chrono>
#include <cstdint>
#include <iostream>

void rntuple_root_read_primitive_bench(const char *path, int iters = 5) {
  double total_ms = 0.0;
  std::uint64_t entries = 0;

  for (int i = 0; i < iters; ++i) {
    auto ntuple = ROOT::RNTupleReader::Open("Events", path);
    auto view_pt = ntuple->GetView<float>("pt");
    auto view_n = ntuple->GetView<std::int32_t>("n");
    entries = static_cast<std::uint64_t>(ntuple->GetNEntries());

    volatile double sink = 0.0;
    const auto t0 = std::chrono::steady_clock::now();
    for (std::uint64_t row = 0; row < entries; ++row) {
      sink += static_cast<double>(view_pt(row));
      sink += static_cast<double>(view_n(row));
    }
    const auto t1 = std::chrono::steady_clock::now();
    total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (sink == -1.0) {
      std::cerr << "impossible" << std::endl;
    }
  }

  const double avg_ms = total_ms / static_cast<double>(iters);
  const double entries_per_sec = static_cast<double>(entries) / (avg_ms / 1000.0);
  std::cout << "root_rntuple_bench entries=" << entries << " iters=" << iters
            << " avg_ms=" << avg_ms << " entries_per_sec=" << entries_per_sec
            << std::endl;
}
