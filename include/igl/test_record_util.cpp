#include "test_record_util.h"

namespace igl
{
IGL_INLINE void write_to_file(std::ofstream &file, std::string message, double time, bool first_call)
{
    if (!first_call)
    {
        file << message << ": " << time << "\n";
    }
}
} // namespace igl