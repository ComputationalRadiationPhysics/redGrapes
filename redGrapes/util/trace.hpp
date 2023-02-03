
#pragma once

#include <redGrapes_config.hpp>
#include <chrono>
#include <array>
#include <optional>
#include <spdlog/spdlog.h>
#include <fmt/os.h>

#ifndef REDGRAPES_ENABLE_TRACE
#define REDGRAPES_ENABLE_TRACE 1
#endif

#ifndef REDGRAPES_TRACE_SIZE
#define REDGRAPES_TRACE_SIZE 0x10000
#endif

#if REDGRAPES_ENABLE_TRACE

namespace redGrapes
{

namespace dispatch {
namespace thread {
extern thread_local uint16_t current_waker_id;
}
}

namespace trace
{

using Clock = std::chrono::high_resolution_clock;
using TimePoint = typename Clock::time_point;

enum RangeType
{
    WORKER_EXECUTE_TASK = 0,
    WORKER_SLEEP,
    WORKER_SCHEDULE,
    TASK_INIT_GRAPH,
    TASK_DELETE_FROM_RUL,
    GRAPH_NOTIFY_EVENT,
    GRAPH_NOTIFY_FOLLOWERS,
    SCHEDULER_ALLOC_WORKER,
    TASK_SUBMIT,
    TASK_DELETE
};

constexpr char color_table[][7] = {
    "00cc00",
    "0000cc",
    "ee22ee",
    "eec0ee",
    "eeccaa",
    "cc0000",
    "ff3333",
    "c3d2ff",
    "aaaaaa",
    "aaaaaa"
};

struct TraceEntry
{
    bool present;
    enum RangeType type;
    TimePoint begin;
    TimePoint end;
};

extern std::optional<TimePoint> begin;
extern std::optional<TimePoint> end;

struct TraceBuf
{
    unsigned next_id;
    std::array< TraceEntry, REDGRAPES_TRACE_SIZE > entries;

    TraceBuf()
        :next_id(0)
    {
    }

    ~TraceBuf()
    {
        if( ! end )
            end = Clock::now();
        write();
    }

    unsigned start( enum RangeType type )
    {
        unsigned id = next_id ++;
        if ( id > REDGRAPES_TRACE_SIZE )
            spdlog::error("trace limit reached");

        if( ! begin )
            begin = Clock::now();

        entries[id].type = type;
        entries[id].begin = Clock::now();

        return id;
    }

    void stop( unsigned id )
    {
        entries[id].end = Clock::now();
        entries[id].present = true;
    }

    void write()
    {
        auto out = fmt::output_file(fmt::format("trace_thread_{:d}.svg", dispatch::thread::current_waker_id));

        double width = std::chrono::duration_cast<std::chrono::nanoseconds>(*end - *begin).count() / 100.0;
        double height = width / 10.0;

        out.print("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n");
        out.print("<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\" \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n");
        out.print("<svg width=\"{}\" height=\"{}\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">\n", width, height);

        // background
        out.print("<rect fill=\"#fff\" x=\"0\" y=\"0\" width=\"{}\" height=\"{}\"/>\n", width, height);

        for( int i = 0 ; i < REDGRAPES_TRACE_SIZE; ++i )
        {
            if( entries[i].present )
            {
                int r = 200;
                int g = 0;
                int b = 0;

                double x = std::chrono::duration_cast<std::chrono::nanoseconds>(entries[i].begin - *begin).count() / 100.0;
                double y = 0;

                double w = std::chrono::duration_cast<std::chrono::nanoseconds>(entries[i].end - entries[i].begin).count() / 100.0;
;
                double h = height;

                //out.print("<rect fill=\"#{:02x}{:02x}{:02x}\" stroke=\"#000\" x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\"/>\n", r,g,b, x,y, w, h);
                out.print("<rect fill=\"#{}\" stroke=\"#000\" x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\"/>\n", color_table[entries[i].type], x,y, w, h);
            }
        }

        out.print("</svg>");
    }
};

extern thread_local TraceBuf trace_buf;

} // namespace trace
} // namespace redGrapes


#define REDGRAPES_TRACE_START( type ) \
    ::redGrapes::trace::trace_buf.start( type )

#define REDGRAPES_TRACE_STOP( handle ) \
    ::redGrapes::trace::trace_buf.stop( handle )

#else

#define REDGRAPES_TRACE_START( type ) 0
#define REDGRAPES_TRACE_STOP( handle )

#endif

