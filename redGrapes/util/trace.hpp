
#pragma once

#include <redGrapes_config.hpp>
#include <chrono>
#include <array>
#include <optional>
#include <spdlog/spdlog.h>

#ifndef REDGRAPES_ENABLE_TRACE
#define REDGRAPES_ENABLE_TRACE 1
#endif

#include <perfetto.h>

PERFETTO_DEFINE_CATEGORIES(
                           perfetto::Category("Worker"),
                           perfetto::Category("Scheduler"),
                           perfetto::Category("Event"),
                           perfetto::Category("TaskSpace"),
                           perfetto::Category("Graph")
);

std::unique_ptr<perfetto::TracingSession> StartTracing();
void StopTracing(std::unique_ptr<perfetto::TracingSession> tracing_session);

