#include "NumericExecutor.hpp"
#include <tbb/parallel_for.h>

using namespace std;

namespace ie {
NumericExecutor::NumericExecutor() {}

NumericExecutor::NumericExecutor(Eigen::SparseMatrix<ie::NumericType, Eigen::RowMajor>& R, size_t gap)
{
    this->gap = gap;
    this->th = NumericVisitorTreeHashing(R.nonZeros());
    tbb::parallel_for(size_t(0), size_t(R.nonZeros()), [&](size_t i) {
        R.valuePtr()[i].accept(this->th, i);
    });
    // this->choice = choice;
    // NumericType::clear_pool();

    // for grouped function
    cout << "Group by tree type method chosen\n";
    this->tg = TreeToFileVisitorGroupByFunction(gap);
    this->th.accept(this->tg);
    this->tg.compile_file();
    this->tg.link_functions();
}
NumericExecutor::NumericExecutor(Eigen::SparseMatrix<ie::NumericType, Eigen::ColMajor>& R, size_t gap)
{
    this->gap = gap;
    this->th = NumericVisitorTreeHashing(R.nonZeros());
    tbb::parallel_for(size_t(0), size_t(R.nonZeros()), [&](size_t i) {
        R.valuePtr()[i].accept(this->th, i);
    });
    // this->choice = choice;
    // NumericType::clear_pool();

    // for grouped function
    cout << "Group by tree type method chosen\n";
    this->tg = TreeToFileVisitorGroupByFunction(gap);
    this->th.accept(this->tg);
    this->tg.compile_file();
    this->tg.link_functions();
}



void NumericExecutor::ExecuteSingle(const vector<vector<double>>& data, vector<double>& result)
{
    this->tg.Execute_single(this->tg.reordered_data_id, this->tg.reordered_result_position, data, result);
}

void NumericExecutor::ExecuteMulti(const vector<vector<double>>& data, vector<double>& result)
{
    this->tg.Execute_multi(this->tg.reordered_data_id, this->tg.reordered_result_position, data, result);
}
}