
#pragma once

// #include <redGrapes_config.hpp>

#ifndef REDGRAPES_ENABLE_TRACE
#    define REDGRAPES_ENABLE_TRACE 0
#endif

#if REDGRAPES_ENABLE_TRACE

#    include <perfetto.h>

#    include <memory>

PERFETTO_DEFINE_CATEGORIES(
    perfetto::Category("Worker"),
    perfetto::Category("Scheduler"),
    perfetto::Category("Event"),
    perfetto::Category("TaskSpace"),
    perfetto::Category("Graph"),
    perfetto::Category("Task"),
    perfetto::Category("Allocator"),
    perfetto::Category("CondVar"),
    perfetto::Category("ChunkedList"),
    perfetto::Category("ResourceUser"));

std::shared_ptr<perfetto::TracingSession> StartTracing();
void StopTracing(std::shared_ptr<perfetto::TracingSession> tracing_session);

#else

#    undef TRACE_EVENT
#    define TRACE_EVENT

#    undef TRACE_EVENT_BEGIN
#    define TRACE_EVENT_BEGIN

#    undef TRACE_EVENT_END
#    define TRACE_EVENT_END

#endif
