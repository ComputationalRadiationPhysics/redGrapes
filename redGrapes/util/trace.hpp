
#pragma once

#include <redGrapes_config.hpp>
#include <chrono>
#include <array>
#include <optional>
#include <spdlog/spdlog.h>

#ifndef REDGRAPES_ENABLE_TRACE
#define REDGRAPES_ENABLE_TRACE 0
#endif

#include <perfetto.h>

#if REDGRAPES_ENABLE_TRACE

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
                           perfetto::Category("ResourceUser")
);


#else


#undef TRACE_EVENT
#define TRACE_EVENT

#undef TRACE_EVENT_BEGIN
#define TRACE_EVENT_BEGIN

#undef TRACE_EVENT_END
#define TRACE_EVENT_END


#endif

std::unique_ptr<perfetto::TracingSession> StartTracing();
void StopTracing(std::unique_ptr<perfetto::TracingSession> tracing_session);

