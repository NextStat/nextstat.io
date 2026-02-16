#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleWriter.hxx>
#include <TSystem.h>

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace {

std::string FixturePath(const char *file_name)
{
  return std::string(gSystem->pwd()) + "/tests/fixtures/" + file_name;
}

void WriteSimpleFixture(const std::string &path)
{
  auto model = ROOT::RNTupleModel::Create();

  auto f_pt = model->MakeField<float>("pt");
  auto f_n = model->MakeField<std::int32_t>("n");

  auto ntuple = ROOT::RNTupleWriter::Recreate(std::move(model), "Events", path);
  for (std::int32_t i = 0; i < 8; ++i) {
    *f_pt = 0.5f * static_cast<float>(i);
    *f_n = i;
    ntuple->Fill();
  }
}

void WriteComplexFixture(const std::string &path)
{
  auto model = ROOT::RNTupleModel::Create();

  auto f_pt = model->MakeField<float>("pt");
  auto f_n = model->MakeField<std::int32_t>("n");
  auto f_arr_fixed = model->MakeField<std::array<float, 3>>("arr_fixed");
  auto f_arr_var = model->MakeField<std::vector<std::int32_t>>("arr_var");
  auto f_pair = model->MakeField<std::pair<float, float>>("pair_ff");

  auto ntuple = ROOT::RNTupleWriter::Recreate(std::move(model), "Events", path);
  *f_pt = 42.5f;
  *f_n = 7;
  *f_arr_fixed = {1.0f, 2.0f, 3.0f};
  *f_arr_var = {10, 20, 30};
  *f_pair = {4.0f, 5.0f};
  ntuple->Fill();
}

void WriteMultiClusterFixture(const std::string &path)
{
  auto model = ROOT::RNTupleModel::Create();

  auto f_pt = model->MakeField<float>("pt");
  auto f_n = model->MakeField<std::int32_t>("n");

  auto ntuple = ROOT::RNTupleWriter::Recreate(std::move(model), "Events", path);
  for (std::int32_t i = 0; i < 6; ++i) {
    *f_pt = 0.25f + static_cast<float>(i);
    *f_n = i * 10;
    ntuple->Fill();
    ntuple->CommitCluster(true);
  }
}

void WritePairScalarVariableFixture(const std::string &path)
{
  auto model = ROOT::RNTupleModel::Create();

  auto f_pt = model->MakeField<float>("pt");
  auto f_pair = model->MakeField<std::pair<float, std::vector<std::int32_t>>>("pair_f_vec");

  auto ntuple = ROOT::RNTupleWriter::Recreate(std::move(model), "Events", path);
  *f_pt = 11.5f;
  *f_pair = {3.0f, {10, 20}};
  ntuple->Fill();
  ntuple->CommitCluster(true);
}

void WritePairVariableScalarFixture(const std::string &path)
{
  auto model = ROOT::RNTupleModel::Create();

  auto f_pt = model->MakeField<float>("pt");
  auto f_pair = model->MakeField<std::pair<std::vector<std::int32_t>, float>>("pair_vec_f");

  auto ntuple = ROOT::RNTupleWriter::Recreate(std::move(model), "Events", path);
  *f_pt = 12.5f;
  *f_pair = {{10, 20}, 6.5f};
  ntuple->Fill();
  ntuple->CommitCluster(true);
}

void WritePairVariableVariableFixture(const std::string &path)
{
  auto model = ROOT::RNTupleModel::Create();

  auto f_pt = model->MakeField<float>("pt");
  auto f_pair =
      model->MakeField<std::pair<std::vector<std::int32_t>, std::vector<float>>>("pair_vec_vec");

  auto ntuple = ROOT::RNTupleWriter::Recreate(std::move(model), "Events", path);
  *f_pt = 13.5f;
  *f_pair = {{1, 2}, {30.0f, 40.5f}};
  ntuple->Fill();
  ntuple->CommitCluster(true);
}

void WriteUnsupportedNestedFixture(const std::string &path)
{
  auto model = ROOT::RNTupleModel::Create();

  auto f_pt = model->MakeField<float>("pt");
  auto f_pair = model->MakeField<std::pair<float, std::pair<float, float>>>("pair_f_pair");

  auto ntuple = ROOT::RNTupleWriter::Recreate(std::move(model), "Events", path);
  *f_pt = 21.5f;
  *f_pair = {6.0f, {7.0f, 8.0f}};
  ntuple->Fill();
  ntuple->CommitCluster(true);
}

void WriteSchemaEvolutionFixture(const std::string &path)
{
  auto model = ROOT::RNTupleModel::Create();
  auto f_pt = model->MakeField<float>("pt");

  auto ntuple = ROOT::RNTupleWriter::Recreate(std::move(model), "Events", path);

  for (std::int32_t i = 0; i < 2; ++i) {
    *f_pt = 1.0f + static_cast<float>(i);
    ntuple->Fill();
  }
  ntuple->CommitCluster(true);

  auto updater = ntuple->CreateModelUpdater();
  updater->BeginUpdate();
  auto f_n = updater->MakeField<std::int32_t>("n");
  updater->CommitUpdate();

  for (std::int32_t i = 0; i < 2; ++i) {
    *f_pt = 10.0f + static_cast<float>(i);
    *f_n = 100 + i;
    ntuple->Fill();
  }
  ntuple->CommitCluster(true);
}

} // namespace

void make_rntuple_fixtures()
{
  WriteSimpleFixture(FixturePath("rntuple_simple.root"));
  WriteComplexFixture(FixturePath("rntuple_complex.root"));
  WriteMultiClusterFixture(FixturePath("rntuple_multicluster.root"));
  WritePairScalarVariableFixture(FixturePath("rntuple_pair_scalar_variable.root"));
  WritePairVariableScalarFixture(FixturePath("rntuple_pair_variable_scalar.root"));
  WritePairVariableVariableFixture(FixturePath("rntuple_pair_variable_variable.root"));
  WriteUnsupportedNestedFixture(FixturePath("rntuple_unsupported_nested.root"));
  WriteSchemaEvolutionFixture(FixturePath("rntuple_schema_evolution.root"));
}
