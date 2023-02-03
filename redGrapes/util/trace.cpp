
#include <redGrapes/util/trace.hpp>


namespace redGrapes
{
namespace trace
{
#if REDGRAPES_ENABLE_TRACE

thread_local TraceBuf trace_buf = TraceBuf();

std::optional<TimePoint> begin;
std::optional<TimePoint> end;

#endif
}
}

