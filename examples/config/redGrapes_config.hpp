#pragma once

#define REDGRAPES_ENABLE_TRACE 0

#include <redGrapes/task/property/label.hpp>
#include <redGrapes/scheduler/tag_match.hpp>

enum SchedulerTags
{
    SCHED_MPI,
    SCHED_CUDA
};

#define REDGRAPES_TASK_PROPERTIES \
    redGrapes::LabelProperty, \
    redGrapes::scheduler::SchedulingTagProperties< SchedulerTags >





template <>
struct fmt::formatter< SchedulerTags >
{
    constexpr auto parse( format_parse_context& ctx )
    {
        return ctx.begin();
    }

    template < typename FormatContext >
    auto format(
        SchedulerTags const & tag,
        FormatContext & ctx
    )
    {
        switch(tag)
        {
        case SCHED_MPI: return fmt::format_to(ctx.out(), "\"MPI\"");
        case SCHED_CUDA: return fmt::format_to(ctx.out(), "\"CUDA\"");
        default: return fmt::format_to(ctx.out(), "\"undefined\"");
        }
    }
};


